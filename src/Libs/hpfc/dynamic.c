/*
 * HPFC module by Fabien COELHO
 *
 * This file provides functions used by directives.c to deal with 
 * dynamic mappings (re*). It includes keeping track of variables 
 * tagged as dynamic, and managing the static synonyms introduced
 * to deal with them in HPFC.
 *
 * $RCSfile: dynamic.c,v $ ($Date: 1995/04/21 14:50:04 $, )
 * version $Revision$
 */

#include "defines-local.h"

#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/*--------------------------------------------------
 *
 *        UTILITIES
 */

/*  DYNAMIC MANAGEMENT
 *
 * the synonyms of a given array are stored in a entities.
 * What I intend as a synonym is a version of the array or template
 * which is distributed or aligned in a different way.
 * the renamings are associated to the remapping statements here.
 */
GENERIC_GLOBAL_FUNCTION(dynamic_hpf, entity_entities);
GENERIC_GLOBAL_FUNCTION(primary_entity, entitymap);
GENERIC_GLOBAL_FUNCTION(renamings, statement_renamings);

#define primary_entity_p(a) (a==load_primary_entity(a))

/*   DYNAMIC STATUS
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
    return(make_dynamic_status(get_dynamic_hpf(),
			       get_primary_entity(),
			       get_renamings()));
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
    /* else the entity was already declared as dynamic */
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

    debug(3, "add_dynamic_synonym", "%s added to %s synonyms\n",
	  entity_name(new_e), entity_name(e));

    assert(dynamic_entity_p(e) && !dynamic_entity_p(new_e));

    entities_list(es) = CONS(ENTITY, new_e, entities_list(es));
    store_dynamic_hpf(new_e, es);
    store_primary_entity(new_e, load_primary_entity(e));
}

/*------------------------------------------------------------------
 *
 *   NEW ENTITIES FOR MANAGING DYNAMIC ARRAYS
 *
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

    debug(5, "new_synonym", "building entity %s\n", new_name);

    new_e = FindOrCreateEntityLikeModel(module, new_name, primary);
    AddEntityToDeclarations(new_e, get_current_module_entity());

    add_dynamic_synonym(new_e, e);
    return(new_e);
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
    return(new_a);
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
    return(new_t);
}

/*  comparison of DISTRIBUTE.
 */
static bool same_distribute_p(d1, d2)
distribute d1, d2;
{
    list /* of distributions */ l1 = distribute_distribution(d1),
                                l2 = distribute_distribution(d2);

    if (distribute_processors(d1)!=distribute_processors(d2)) return(FALSE);
    
    assert(gen_length(l1)==gen_length(l2));

    for(; !ENDP(l1); POP(l1), POP(l2))
    {
	distribution i1 = DISTRIBUTION(CAR(l1)),
	             i2 = DISTRIBUTION(CAR(l2));
	style s1 = distribution_style(i1),
	      s2 = distribution_style(i2);
	tag t1 = style_tag(s1);

	if (t1!=style_tag(s2)) return(FALSE);
	if (t1!=is_style_none &&
	    !expression_equal_p(distribution_parameter(i1),
				distribution_parameter(i2)))
	    return(FALSE);
    }

    return(TRUE);
}

/*  comparison of ALIGN.
 */
static bool same_alignment_in_list_p(a, l)
alignment a;
list /* of alignments */ l;
{
    int adim = alignment_arraydim(a),
        tdim = alignment_templatedim(a);

    MAPL(ca,
     {
	 alignment b = ALIGNMENT(CAR(ca));
	 
	 if (adim==alignment_arraydim(b) && tdim==alignment_templatedim(b))
	     return(expression_equal_p(alignment_rate(a), 
				       alignment_rate(b)) &&
		    expression_equal_p(alignment_constant(a), 
				       alignment_constant(b)));
     },
	 l);

    return(FALSE);
}

static bool same_align_p(a1, a2)
align a1, a2;
{
    list /* of alignments */ l1 = align_alignment(a1),
                             l2 = align_alignment(a2);

    if (align_template(a1)!=align_template(a2)) return(FALSE);

    MAPL(ca,
     {
	 if (!same_alignment_in_list_p(ALIGNMENT(CAR(ca)), l2))
	     return(FALSE);
     },
	 l1);

    return(TRUE);
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
    entities es = load_dynamic_hpf(array);
    list /* of entities */ l = entities_list(es);

    for (; !ENDP(l); POP(l))
    {
	entity ar = ENTITY(CAR(l));

	if (same_align_p(load_entity_align(ar), a))
	{
	    free_align(a);
	    return(ar);    /* the one found is returned */
	}
    }

    /*  else no compatible array does exist, so one must be created
     */
    return(new_synonym_array(array, a));
}

align new_align_with_template(a, t)
align a;
entity t;
{
    align b = copy_align(a);
    align_template(b) = t;
    return(b);
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
    entities es = load_dynamic_hpf(temp);
    list /* of entities */ l = entities_list(es);

    for (; !ENDP(l); POP(l))
    {
	entity t = ENTITY(CAR(l));

	if (same_distribute_p(load_entity_distribute(t), d))
	{
	    free_distribute(d);
	    return(t);    /* the one found is returned */
	}
    }

    /*  else no compatible template does exist, so one must be created
     */
    return(new_synonym_template(temp, d));
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
list /* of entities */ alive_arrays(s, t)
statement s;
entity t;
{
    entity array;
    list /* of entities */ 
	l = NIL,
	lseens = NIL; /* primary entities already seen. just to tag them. */

    assert(entity_template_p(t));

    /*   first the alive list is scanned.
     */
    MAPL(ce,
     {
	 array = ENTITY(CAR(ce));
	 
	 if (align_template(load_entity_align(array))==t)
	     l = CONS(ENTITY, array, l);

	 lseens = CONS(ENTITY, load_primary_entity(array), l);
     },
	 entities_list(load_alive_synonym(s)));

    /*   second the defaults are looked for. namely the primary entities.
     */
    MAPL(ce,
     {
	 array = ENTITY(CAR(ce));

	 if (primary_entity_p(array) && !gen_in_list_p(array, lseens))
	 {
	     if (align_template(load_entity_align(array))==t)
		 l = CONS(ENTITY, array, l);

	     lseens = CONS(ENTITY, array, l);
	 }
     },
	 list_of_distributed_arrays());

    gen_free_list(lseens); return(l);
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
GENERIC_GLOBAL_FUNCTION(alive_synonym, statement_entities);

static entity 
    old_variable = entity_undefined,
    new_variable = entity_undefined;
static bool array_propagation;

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
	reference_variable(r) = new_variable;
}

static void simple_switch_old_to_new(s)
statement s;
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
 *  ??? just check for realign right now.
 */
static bool continue_propagation_p(s)
statement s;
{
    instruction i = statement_instruction(s);

    if (!instruction_call_p(i)) 
	return(TRUE);
    else
    {
	call c = instruction_call(i);
	entity fun = call_function(c);

	if (realign_directive_p(fun) && array_propagation)
	{
	    MAPL(ce,
	     {
		 reference r = expression_to_reference(EXPRESSION(CAR(ce)));

		 if (reference_variable(r)==old_variable)
		 {
		     /*  the variable is realigned.
		      */
		     reference_variable(r) = new_variable;
		     return(FALSE);
		 }
	     },
		 call_arguments(c));
	}
	else if (redistribute_directive_p(fun))
	{
	    entity t = array_propagation ?
		align_template(load_entity_align(new_variable)) : old_variable;
	    expression e;
		
	    MAPL(ce,
	     {
		 e = EXPRESSION(CAR(ce));

		 /*   if template t is redistributed...
		  */
		 if (expression_to_entity(e)==t)
		 {
		     if (array_propagation)
			 /*  then the new_variable is the alive one.
			  */
			 add_alive_synonym(s, new_variable);
		     else
			 reference_variable
			     (syntax_reference(expression_syntax(e))) = 
				 new_variable;
		     
		     return(FALSE);
		 }
	     },
		 call_arguments(c));
	}

	return(TRUE);
    }
}

void propagate_synonym(s, old, new)
statement s;
entity old, new;
{
    statement current;

    debug(3, "propagate_array_synonym", "%s -> %s from statement 0x%x\n",
	  entity_name(old), entity_name(new), (unsigned int) s);

    old_variable = old, new_variable = new, 
    array_propagation = array_distributed_p(old);

    init_ctrl_graph_travel(s, continue_propagation_p);

    while (next_ctrl_graph_travel(&current))
	simple_switch_old_to_new(current);

    old_variable = entity_undefined, new_variable = entity_undefined;
    close_ctrl_graph_travel();
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
    int ndims, i;

    assert(array_distributed_p(src) &&
	   array_distributed_p(trg) &&
	   type_variable_p(t) &&
	   load_primary_entity(src)==load_primary_entity(trg)); /* ??? */

    dims = variable_dimensions(type_variable(t));
    ndims = gen_length(dims);
    
    /*  builds the set of indexes needed to scan the dimensions.
     */
    for(i=ndims; i>0; i--)
	indexes = 
	    CONS(ENTITY, 
		 hpfc_new_variable(get_current_module_entity(), is_basic_int),
		 indexes);

    idx_expr = entity_list_to_expression_list(indexes);

    /*  builds the assign statement to put in the body.
     *  TRG(indexes) = SRC(indexes)
     */
    current = make_assign_statement
	(reference_to_expression(make_reference(trg, idx_expr)),
	 reference_to_expression(make_reference(src, gen_copy_seq(idx_expr))));

    /*  builds the loop nest
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

    return(current);
}

/* that is all
 */
