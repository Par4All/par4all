/* HPFC module by Fabien COELHO
 *
 * This file provides functions used by directives.c to deal with 
 * dynamic mappings (re*). It includes keeping track of variables 
 * tagged as dynamic, and managing the static synonyms introduced
 * to deal with them in HPFC.
 *
 * $RCSfile: dynamic.c,v $ version $Revision$
 * ($Date: 1995/10/05 11:32:25 $, )
 */

#include "defines-local.h"

#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/*  DYNAMIC MANAGEMENT
 *
 * the synonyms of a given array are stored in a entities.
 * What I intend as a synonym is a version of the array or template
 * which is distributed or aligned in a different way.
 * the renamings are associated to the remapping statements here.
 * - dynamic_hpf: keeps track of entities declared as dynamic.
 * - primary_entity: primary entity of an entity, when synonyms are 
 *   introduced to handle remappings.
 * - renamings: remappings associated to a statement.
 */
GENERIC_GLOBAL_FUNCTION(dynamic_hpf, entity_entities)
GENERIC_GLOBAL_FUNCTION(primary_entity, entitymap)
GENERIC_GLOBAL_FUNCTION(renamings, statement_renamings)

entity safe_load_primary_entity(entity e)
{
    if (!bound_dynamic_hpf_p(e))
	pips_user_error("%s is not dynamic\n", entity_local_name(e));
   
    return load_primary_entity(e);
}

#define primary_entity_p(a) (a==load_primary_entity(a))

/*   DYNAMIC STATUS management.
 */
void init_dynamic_status()
{
    init_dynamic_hpf();
    init_primary_entity();
    init_renamings();
}

void reset_dynamic_status()
{
    reset_dynamic_hpf();
    reset_primary_entity();
    reset_renamings();
}

dynamic_status get_dynamic_status()
{
    return make_dynamic_status(get_dynamic_hpf(), 
			       get_primary_entity(), 
			       get_renamings());
}

void set_dynamic_status(d)
dynamic_status d;
{
    set_dynamic_hpf(dynamic_status_dynamics(d));
    set_primary_entity(dynamic_status_primary(d));
    set_renamings(dynamic_status_renamings(d));
}

void close_dynamic_status()
{
    close_dynamic_hpf();
    close_primary_entity();
    close_renamings();
}

/*  a new dynamic entity is stored.
 *  HPF allows arrays and templates as dynamic.
 *  ??? could be asserted, but not here. should be checked afterward.
 */
void set_entity_as_dynamic(e)
entity e;
{
    if (!bound_dynamic_hpf_p(e))
    {
	store_dynamic_hpf(e, make_entities(CONS(ENTITY, e, NIL)));
	store_primary_entity(e, e);
    }
    /* else the entity was already declared as dynamic... */
}

/*  as expected, TRUE if entity e is dynamic. 
 *  it is just a function name nicer than bound_...
 */
bool (*dynamic_entity_p)(entity) = bound_dynamic_hpf_p;

/* what: new_e is stored as a synonym of e.
 */
static void add_dynamic_synonym(new_e, e)
entity new_e, e;
{
    entities es = load_dynamic_hpf(e);

    pips_debug(3, "%s as %s synonyms\n", entity_name(new_e), entity_name(e));

    pips_assert("dynamicity", dynamic_entity_p(e) && !dynamic_entity_p(new_e));

    entities_list(es) = CONS(ENTITY, new_e, entities_list(es));
    store_dynamic_hpf(new_e, es);
    store_primary_entity(new_e, load_primary_entity(e));
}

/*   NEW ENTITIES FOR MANAGING DYNAMIC ARRAYS
 */

/*  builds a synonym for entity e. The name is based on e, plus
 *  an underscore and a number added. May be used for templates and arrays.
 *  the new synonym is added as a synonym of e.
 */
static entity new_synonym(e)
entity e;
{
    int n = gen_length(entities_list(load_dynamic_hpf(e))); /* syn. number */
    entity primary = load_primary_entity(e), new_e;
    string module = entity_module_name(e);
    char new_name[100];	
    
    sprintf(new_name, "%s_%x", entity_local_name(primary), (unsigned int) n);

    pips_debug(5, "building entity %s\n", new_name);

    new_e = FindOrCreateEntityLikeModel(module, new_name, primary);
    AddEntityToDeclarations(new_e, get_current_module_entity());

    add_dynamic_synonym(new_e, e);
    return new_e;
}

/*  builds a new synonym for array a, the alignment of which 
 *  will be al. The new array is set as distributed.
 */
static entity new_synonym_array(a, al)
entity a;
align al;
{
    entity new_a = new_synonym(a);
    set_array_as_distributed(new_a);
    store_entity_align(new_a, al);
    return new_a;
}

/*  builds a new synonym for template t, the distribution of which
 *  will be di. the new entity is set as a template.
 */
static entity new_synonym_template(t, di)
entity t;
distribute di;
{
    entity new_t = new_synonym(t);
    set_template(new_t);
    store_entity_distribute(new_t, di);
    return new_t;
}

/*  comparison of DISTRIBUTE.
 */
static bool same_distribute_p(d1, d2)
distribute d1, d2;
{
    list /* of distributions */ l1 = distribute_distribution(d1),
                                l2 = distribute_distribution(d2);

    if (distribute_processors(d1)!=distribute_processors(d2)) return FALSE;
    
    pips_assert("valid distribution", gen_length(l1)==gen_length(l2));

    for(; !ENDP(l1); POP(l1), POP(l2))
    {
	distribution i1 = DISTRIBUTION(CAR(l1)),
	             i2 = DISTRIBUTION(CAR(l2));
	style s1 = distribution_style(i1),
	      s2 = distribution_style(i2);
	tag t1 = style_tag(s1);

	if (t1!=style_tag(s2)) return FALSE;
	if (t1!=is_style_none &&
	    !expression_equal_p(distribution_parameter(i1),
				distribution_parameter(i2)))
	    return FALSE;
    }

    return TRUE;
}

/*  comparison of ALIGN.
 */
static bool same_alignment_in_list_p(a, l)
alignment a;
list /* of alignments */ l;
{
    int adim = alignment_arraydim(a),
        tdim = alignment_templatedim(a);

    MAP(ALIGNMENT, b,
    {
	if (adim==alignment_arraydim(b) && tdim==alignment_templatedim(b))
	    return expression_equal_p(alignment_rate(a), 
				      alignment_rate(b)) &&
		   expression_equal_p(alignment_constant(a), 
				      alignment_constant(b));
    },
	l);

    return FALSE;
}

static bool same_align_p(a1, a2)
align a1, a2;
{
    list /* of alignments */ l1 = align_alignment(a1),
                             l2 = align_alignment(a2);

    if (align_template(a1)!=align_template(a2)) return FALSE;

    MAP(ALIGNMENT, a,
	if (!same_alignment_in_list_p(a, l2)) return FALSE,
	l1);

    return TRUE;
}

/* entity array_synonym_aligned_as(array, a)
 * entity array;
 * align a;
 *
 * what: finds or creates a new entity aligned as needed.
 * input: an array (which *must* be dynamic) and an align
 * output: returns an array aligned as specified by align a
 * side effects:
 *  - creates a new entity if necessary. 
 *  - this entity is stored as a synonym of array, and tagged as dynamic.
 *  - the align is freed if not used.
 * bugs or features:
 */
entity array_synonym_aligned_as(array, a)
entity array;
align a;
{
    MAP(ENTITY, ar,
    {
	if (same_align_p(load_entity_align(ar), a))
	{
	    free_align(a);
	    return ar;    /* the one found is returned */
	}
    },
	entities_list(load_dynamic_hpf(array)));

    /*  else no compatible array does exist, so one must be created
     */
    return new_synonym_array(array, a);
}

align new_align_with_template(a, t)
align a;
entity t;
{
    align b = copy_align(a);
    align_template(b) = t;
    return b;
}

/* entity template_synonym_distributed_as(temp, d)
 * entity temp;
 * distribute d;
 *
 * what: finds or creates a new entity distributed as needed.
 * input: an template (which *must* be dynamic) and a distribute
 * output: returns a template distributed as specified by d
 * side effects:
 *  - creates a new entity if necessary. 
 *  - this entity is stored as a synonym of array, and tagged as dynamic.
 *  - the distribute is freed if not used.
 */
entity template_synonym_distributed_as(temp, d)
entity temp;
distribute d;
{
    MAP(ENTITY, t,
    {
	if (same_distribute_p(load_entity_distribute(t), d))
	{
	    free_distribute(d);
	    return t;    /* the one found is returned */
	}
    },
	entities_list(load_dynamic_hpf(temp)));

    /*  else no compatible template does exist, so one must be created
     */
    return new_synonym_template(temp, d);
}

/* DYNAMIC LOCAL DATA
 *
 * these static functions are used to store the remapping graph
 * while it is built, or when optimizations are performed on it.
 *
 * - alive_synonym: used when building the remapping graph. synonym of 
 *   a primary entity that has reached a given remapping statement. Used 
 *   for both arrays and templates. 
 * - used_dynamics: from a remapping statement, the remapped arrays that 
 *   are actually referenced in their new shape.
 * - remapping_graph: the remapping graph, based on the control domain.
 *   the control_statement is the remapping statement in the code.
 *   predecessors and successors are the possible remapping statements 
 *   for the arrays remapped at that vertex.
 * - reaching_mappings: the mappings that may reached a vertex.
 * - leaving_mappings: the mappings that may leave the vertex. 
 *   (simplification assumption: only one per array)
 * - remapped: the (primary) arrays remapped at the vertex.
 */
GENERIC_LOCAL_FUNCTION(alive_synonym, statement_entities)
GENERIC_LOCAL_FUNCTION(used_dynamics, statement_entities)
GENERIC_LOCAL_FUNCTION(remapping_graph, controlmap)
GENERIC_LOCAL_FUNCTION(reaching_mappings, statement_entities)
GENERIC_LOCAL_FUNCTION(leaving_mappings, statement_entities)
GENERIC_LOCAL_FUNCTION(remapped, statement_entities)

void init_dynamic_locals()
{
    init_alive_synonym();
    init_used_dynamics();
    init_reaching_mappings();
    init_leaving_mappings();
    init_remapped();
    init_remapping_graph();
}

void close_dynamic_locals()
{
    close_alive_synonym();
    close_used_dynamics();
    close_reaching_mappings();
    close_leaving_mappings();
    close_remapped();

    /*  can't close it directly...
     */
    CONTROLMAP_MAP(s, c, 
      {
	  what_stat_debug(9, s);

	  control_statement(c) = statement_undefined;
	  gen_free_list(control_successors(c)); control_successors(c) = NIL;
	  gen_free_list(control_predecessors(c)); control_predecessors(c) = NIL;
      },
		   get_remapping_graph());

    close_remapping_graph();
}

/* void propagate_synonym(s, old, new)
 * statement s;
 * entity old, new;
 *
 * what: propagates a new array/template synonym (old->new) from statement s.
 * how: travels thru the control graph till the next remapping.
 * input: the starting statement, plus the two entities.
 * output: none.
 * side effects:
 *  - uses the crtl_graph travelling.
 *  - set some static variables for the continuation decisions and switch.
 * bugs or features:
 *  - not very efficient. Could have done something to deal with 
 *    several synonyms at the same time...
 *  - what is done on an "incorrect" code is not clear.
 */

static entity 
    old_variable = entity_undefined, /* entity to be replaced */
    new_variable = entity_undefined; /* replacement */
static bool 
    array_propagation, /* TRUE if an array is propagated, FALSE if template */
    array_used;        /* TRUE if the array was actually used */
static statement 
    initial_statement = statement_undefined; /* starting point */

/*  initialize both the remapping graph and the used dynamics for s
 */
static void lazy_initialize_for_statement(s)
statement s;
{
    if (!bound_remapping_graph_p(s))
	store_remapping_graph(s, make_control(s, NIL, NIL));

    if (!bound_used_dynamics_p(s))
	store_used_dynamics(s, make_entities(NIL));
}

static void add_as_a_closing_statement(s)
statement s;
{
    control 
	c = load_remapping_graph(initial_statement),
	n = (lazy_initialize_for_statement(s), load_remapping_graph(s));

    what_stat_debug(6, s);

    control_successors(c) = gen_once(load_remapping_graph(s),
				     control_successors(c));
    control_predecessors(n) = gen_once(load_remapping_graph(initial_statement), 
				       control_predecessors(n));
}

static void add_as_a_used_variable(e)
entity e;
{
    entities es = load_used_dynamics(initial_statement);
    entities_list(es) = gen_once(load_primary_entity(e), entities_list(es));
}

static void add_alive_synonym(s, a)
statement s;
entity a;
{
    entities es;

    /*   lazy initialization 
     */
    if (!bound_alive_synonym_p(s))
	store_alive_synonym(s, make_entities(NIL));

    es = load_alive_synonym(s);
    entities_list(es) = gen_once(a, entities_list(es));
}

static void ref_rwt(r)
reference r;
{
    if (reference_variable(r)==old_variable)
	reference_variable(r) = new_variable,
	array_used = TRUE;
}

static void 
simple_switch_old_to_new(statement s)
{
    gen_multi_recurse
	(statement_instruction(s),
	 statement_domain,    gen_false, gen_null, /* STATEMENT */
	 unstructured_domain, gen_false, gen_null, /* UNSTRUCTURED ? */
	 reference_domain,    gen_true,  ref_rwt,  /* REFERENCE */
	 NULL);
}

/*  TRUE if not a remapping for old. 
 *  if it is a remapping, operates the switch.
 */
static bool 
continue_propagation_p(statement s)
{
    instruction i = statement_instruction(s);

    what_stat_debug(8, s);

    if (!instruction_call_p(i)) 
    {
	pips_debug(8, "not a call\n"); return TRUE;
    }
    else
    {
	call c = instruction_call(i);
	entity fun = call_function(c);

	if (realign_directive_p(fun) && array_propagation)
	{
	    DEBUG_STAT(8, "realign directive", s);

	    MAP(EXPRESSION, e,
	    {
		reference r = expression_to_reference(e);
		entity var = reference_variable(r);

		if (entity_template_p(var)) /* up to the template, stop */
		    return TRUE;
		
		if (safe_load_primary_entity(var)==old_variable)
		{
		    /*  the variable is realigned.
		     */
		    reference_variable(r) = new_variable;
		    add_as_a_closing_statement(s);
		    return FALSE;
		}
	    },
		call_arguments(c));
	}
	else if (redistribute_directive_p(fun))
	{
	    entity t = array_propagation ?
		align_template(load_entity_align(new_variable)) : old_variable;
		
	    MAP(EXPRESSION, e,
	    {
		entity v = expression_to_entity(e);

		if (!entity_template_p(v)) /* up to the processor, stop */
		    return TRUE;

		/*   if template t is redistributed...
		 */
		if (safe_load_primary_entity(v)==t)
		{
		    if (array_propagation)
			/*  then the new_variable is the alive one.
			 */
			add_alive_synonym(s, new_variable);
		    else
			reference_variable
			    (syntax_reference(expression_syntax(e))) = 
				new_variable;
		    
		    add_as_a_closing_statement(s);
		    return FALSE;
		}
	    },
		call_arguments(c));
	}

	return TRUE;
    }
}

void 
propagate_synonym(
    statement s,
    entity old, 
    entity new)
{
    statement current;

    what_stat_debug(3, s);
    pips_debug(3, "%s -> %s\n", entity_name(old), entity_name(new));

    old_variable = safe_load_primary_entity(old), new_variable = new, 
    array_propagation = array_distributed_p(old),
    array_used = FALSE,
    initial_statement = s;

    lazy_initialize_for_statement(s);

    init_ctrl_graph_travel(s, continue_propagation_p);

    while (next_ctrl_graph_travel(&current))
	simple_switch_old_to_new(current);

    if (array_used) add_as_a_used_variable(old);

    old_variable = entity_undefined, new_variable = entity_undefined;
    close_ctrl_graph_travel();

    pips_debug(4, "out\n");
}

/*   REMAPPING GRAPH REMAPS "SIMPLIFICATION"
 */

/* for statement s
 * - remapped arrays
 * - reaching mappings (1 per array, or more)
 * - leaving mappings (1 per remapped array, to simplify)
 */
static void initialize_reaching_propagation(s)
statement s;
{
    list /* of entities */ le = NIL, lp = NIL, ll = NIL;
    entity old, new;

    what_stat_debug(4, s);

    MAP(RENAMING, r,
    {
	old = renaming_old(r);
	new = renaming_new(r);
	
	le = CONS(ENTITY, old, le);
	lp = gen_once(load_primary_entity(old), lp);
	ll = CONS(ENTITY, new, ll);
    },	 
	load_renamings(s));

    store_reaching_mappings(s, make_entities(le));
    store_remapped(s, make_entities(lp));
    store_leaving_mappings(s, make_entities(ll));
}

static void remove_not_remapped_leavings(s)
statement s;
{
    entities leaving  = load_leaving_mappings(s);
    list /* of entities */ 
	ll = entities_list(leaving),
	ln = gen_copy_seq(ll),
	lr = entities_list(load_remapped(s)),
	lu = entities_list(load_used_dynamics(s));
    entity primary;

    MAP(ENTITY, array,
    {
	primary = load_primary_entity(array);
	
	if (gen_in_list_p(primary, lr) && /* REMAPPED and */
	    !gen_in_list_p(primary, lu))  /* NOT USED */
	{
	    what_stat_debug(4, s);
	    pips_debug(4, "removing %s\n", entity_name(array));
	    gen_remove(&ln, array);	    /* => NOT LEAVED */
	}
    },
	ll);

    gen_free_list(ll), entities_list(leaving) = ln;
}

static void reinitialize_reaching_mappings(s)
statement s;
{
    entities er = load_reaching_mappings(s);
    list /* of entity */ newr = NIL, /* new reachings */
                         lrem = entities_list(load_remapped(s));

    gen_free_list(entities_list(er));

    MAP(CONTROL, c,
    {
	MAP(ENTITY, e,
	{
	    if (gen_in_list_p(load_primary_entity(e), lrem))
		newr = gen_once(e, newr);
	},
	    entities_list(load_leaving_mappings(control_statement(c))));
    },
	control_predecessors(load_remapping_graph(s)));

    entities_list(er) = newr;
}

static list /* of statements */ propagate_used_arrays(s, ls)
statement s;
list /* of statements */ ls;
{
    bool touched = FALSE;
    entities er = load_reaching_mappings(s);
    list /* of entity */
	lrem = entities_list(load_remapped(s)),
	lrea = entities_list(er);
    control current = load_remapping_graph(s);

    MAP(CONTROL, c,
    {
	statement sc = control_statement(c);
	list /* of entity */ lp_rem = entities_list(load_remapped(sc));
	list /* idem      */ lp_use = entities_list(load_used_dynamics(sc));

	MAP(ENTITY, e,
	{
	    entity prim = load_primary_entity(e);

	    if (gen_in_list_p(prim, lrem) && 
		gen_in_list_p(prim, lp_rem) &&
		!gen_in_list_p(prim, lp_use) &&
		!gen_in_list_p(e, lrea))
	    {
		lrea = CONS(ENTITY, e, lrea);
		touched = TRUE;
	    }
	},
	    entities_list(load_reaching_mappings(sc)));
    },
	control_predecessors(current));

    /* if Reachings(current) was modified, the successors of current
     * may have to be recomputed at the next stage. quite rough.
     */
    if (touched)
    {
	entities_list(er) = lrea;
	MAP(CONTROL, c, 
	    ls = gen_once(control_statement(c), ls), 
	    control_successors(current));
    }

    return ls;
}

static void remove_from_entities(primary, es)
entity primary;
entities es;
{
    list /* of entity(s) */ le = gen_copy_seq(entities_list(es));

    MAP(ENTITY, array,
    {
	if (load_primary_entity(array)==primary)
	    gen_remove(&entities_list(es), array);
    },
	le);

    gen_free_list(le);
}

static void remove_unused_remappings(s)
statement s;
{
    entities remapped = load_remapped(s),
             reaching = load_reaching_mappings(s),
             leaving = load_leaving_mappings(s);
    list /* of entity(s) */ le = gen_copy_seq(entities_list(remapped)),
                            lu = entities_list(load_used_dynamics(s));

    MAP(ENTITY, primary,
    {
	if (!gen_in_list_p(primary, lu))
	{
	    remove_from_entities(primary, remapped);
	    remove_from_entities(primary, reaching);
	    remove_from_entities(primary, leaving);
	}
    },
	le);

    gen_free_list(le);
}

/* regenerate the renaming structures after the optimizations performed on 
 * the remapping graph. 
 */
static void regenerate_renamings(s)
statement s;
{
    list /* of entity(s) */
	lr = entities_list(load_reaching_mappings(s)),
	ll = entities_list(load_leaving_mappings(s)),
	ln = NIL;
    
    what_stat_debug(4, s);

    MAP(ENTITY, target,
    {
	entity primary = load_primary_entity(target);

	MAP(ENTITY, source,
	{
	    if (load_primary_entity(source)==primary && source!=target)
	    {
		pips_debug(4, "%s -> %s\n", 
			   entity_name(source), entity_name(target));
		ln = CONS(RENAMING, make_renaming(source, target), ln);
	    }
	},
	    lr);
    },
	ll);

    {
	list /* of renaming(s) */ l = load_renamings(s);
	gen_map(gen_free, l), gen_free_list(l); /* ??? */
	update_renamings(s, ln);
    }
}

static list /* of statements */ 
list_of_remapping_statements()
{
    list /* of statements */ l = NIL;
    CONTROLMAP_MAP(s, c,
    {
	pips_debug(9, "0x%x -> 0x%x\n", (unsigned int) s, (unsigned int) c);
	l = CONS(STATEMENT, s, l);
    },
		   get_remapping_graph());
    return l;
}

/* functions used for debug.
 */
static void print_control_ordering(c)
control c;
{
    register int so = statement_ordering(control_statement(c));
    fprintf(stderr, "(%d,%d), ", ORDERING_NUMBER(so), ORDERING_STATEMENT(so));
}

static void dump_remapping_graph_info(s)
statement s;
{
    control c = load_remapping_graph(s);

    what_stat_debug(1, s);

    fprintf(stderr, "predecessors: ");
    gen_map(print_control_ordering, control_predecessors(c));
    fprintf(stderr, "\nsuccessors: ");
    gen_map(print_control_ordering, control_successors(c));
    fprintf(stderr, "\n");

    DEBUG_ELST(1, "remapped", entities_list(load_remapped(s)));
    DEBUG_ELST(1, "used", entities_list(load_used_dynamics(s)));
    DEBUG_ELST(1, "reaching", entities_list(load_reaching_mappings(s)));
    DEBUG_ELST(1, "leaving", entities_list(load_leaving_mappings(s)));
}

/* void simplify_remapping_graph()
 *
 * what: simplifies the remapping graph.
 * how: propagate unused reaching mappings to the next remappings,
 *      and remove unnecessary remappings.
 * input: none.
 * output: none.
 * side effects: all is there!
 *  - the remapping graph remappings are modified.
 *  - some static structures are used.
 *  - current module statement needed.
 * bugs or features:
 *  - special treatment of the current module statement to model the 
 *    initial mapping at the entry of the module. The initial mapping may
 *    be modified by the algorithm if it is remapped before being used.
 *  - ??? the convergence and correctness are to be proved.
 *  - expected complexity: n vertices (small!), q nexts, m arrays, r remaps.
 *    assumes fast sets instead of lists, with O(1) tests/add/del...
 *    closure: O(n^2*vertex_operation) (if it is a simple propagation...)
 *    map: O(n*vertex_operation)
 *    C = n^2 q m r
 */
void simplify_remapping_graph()
{
    list /* of statements */ ls = list_of_remapping_statements();
    statement root = get_current_module_statement();
    what_stat_debug(4, root);

    gen_map(initialize_reaching_propagation, ls);

    ifdebug(4) gen_map(dump_remapping_graph_info, ls);

    gen_map(remove_not_remapped_leavings, ls);
    gen_map(reinitialize_reaching_mappings, ls);
    gen_closure(propagate_used_arrays, ls);

    ifdebug(4) gen_map(dump_remapping_graph_info, ls);

    if (bound_remapped_p(root))	remove_unused_remappings(root);

    gen_map(regenerate_renamings, ls);

    gen_map(gen_free, load_renamings(root)),
    gen_free_list(load_renamings(root)), 
    (void) delete_renamings(root);

    gen_free_list(ls);
}

/* list alive_arrays(s, t);
 * statement s;
 * entity t;
 * 
 * what: returns the list of alive arrays for statement s and template t.
 * how: uses the alive_synonym, and computes the defaults.
 * input: statement s and template t which are of interest.
 * output: a list of entities which is allocated.
 * side effects: none. 
 */
list /* of entities */ 
alive_arrays(
    statement s,
    entity t)
{
    list /* of entities */ l = NIL, lseens = NIL; /* to tag seen primaries. */

    pips_assert("template", entity_template_p(t));

    /*   first the alive list is scanned.
     */

    /* ??? well, it is not necessarily initialized, I guess...
     */
    if (!bound_alive_synonym_p(s))
	store_alive_synonym(s, make_entities(NIL));

    MAP(ENTITY, array,
    {
	if (align_template(load_entity_align(array))==t)
	    l = CONS(ENTITY, array, l);

	pips_debug(8, "adding %s as alive\n", entity_name(array));
	
	lseens = CONS(ENTITY, load_primary_entity(array), lseens);
    },
	entities_list(load_alive_synonym(s)));

    /*   second the defaults are looked for. namely the primary entities.
     */
    MAP(ENTITY, array,
    {
	if (primary_entity_p(array) && !gen_in_list_p(array, lseens))
	{
	    if (align_template(load_entity_align(array))==t)
		l = CONS(ENTITY, array, l);
	    
	    pips_debug(8, "adding %s as default\n", entity_name(array));
	
	    lseens = CONS(ENTITY, array, lseens);
	}
    },
	list_of_distributed_arrays());

    DEBUG_ELST(9, "seen arrays", lseens);
    gen_free_list(lseens); 

    DEBUG_ELST(7, "returned alive arrays", l);
    return l;
}

/* statement generate_copy_loop_nest(src, trg)
 * entity src, trg;
 *
 * what: generates a parallel loop nest that copies src in trg.
 * how: by building the corresponding AST.
 * input: the two entities, which should be arrays with the same shape.
 * output: a statement containing the loop nest.
 * side effects:
 *  - adds a few new variables for the loop indexes.
 * bugs or features:
 *  - could be more general?
 */
statement generate_copy_loop_nest(src, trg)
entity src, trg;
{
    type t = entity_type(src);
    list /* of entities */    indexes = NIL,
         /* of expressions */ idx_expr,
         /* of dimensions */  dims;
    statement current;
    entity module;
    int ndims, i;

    pips_assert("valid arguments",
		array_distributed_p(src) && array_distributed_p(trg) &&
		type_variable_p(t) &&
		load_primary_entity(src)==load_primary_entity(trg)); /* ??? */

    dims = variable_dimensions(type_variable(t));
    ndims = gen_length(dims);
    
    /*  builds the set of indexes needed to scan the dimensions.
     */
    for(module=get_current_module_entity(), i=ndims; i>0; i--)
      indexes = CONS(ENTITY, hpfc_new_variable(module, is_basic_int), indexes);

    idx_expr = entity_list_to_expression_list(indexes);

    /*  builds the assign statement to put in the body.
     *  TRG(indexes) = SRC(indexes)
     */
    current = make_assign_statement
	(reference_to_expression(make_reference(trg, idx_expr)),
	 reference_to_expression(make_reference(src, gen_copy_seq(idx_expr))));

    /*  builds the copy loop nest
     */
    for(; ndims>0; POP(dims), POP(indexes), ndims--)
    {
	dimension d = DIMENSION(CAR(dims));
	
	current = loop_to_statement
	    (make_loop(ENTITY(CAR(indexes)),
		       make_range(copy_expression(dimension_lower(d)),
				  copy_expression(dimension_upper(d)),
				  int_to_expression(1)),
		       current,
		       entity_empty_label(),
		       make_execution(is_execution_parallel, UU),
		       NIL));
    }

    DEBUG_STAT(7, concatenate(entity_name(src), " -> ", entity_name(trg), NULL),
	       current);

    return current;
}

/* that is all
 */
