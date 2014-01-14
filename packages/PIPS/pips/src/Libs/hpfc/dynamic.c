/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* HPFC module by Fabien COELHO
 *
 * This file provides functions used by directives.c to deal with 
 * dynamic mappings (re*). It includes keeping track of variables 
 * tagged as dynamic, and managing the static synonyms introduced
 * to deal with them in HPFC.
 */

#include "defines-local.h"

#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

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
 * - maybeuseful_mappings: to be kept for a statement. 
 * - similar_mapping: copies with similar mappings...
 */
GENERIC_GLOBAL_FUNCTION(dynamic_hpf, entity_entities)
GENERIC_GLOBAL_FUNCTION(primary_entity, entitymap)
GENERIC_GLOBAL_FUNCTION(renamings, statement_renamings)
GENERIC_GLOBAL_FUNCTION(maybeuseful_mappings, statement_entities)
GENERIC_GLOBAL_FUNCTION(similar_mapping, entitymap)

entity safe_load_primary_entity(entity e)
{
    if (!bound_dynamic_hpf_p(e))
	pips_user_error("%s is not dynamic\n", entity_local_name(e));
   
    return load_primary_entity(e);
}

bool
same_primary_entity_p(entity e1, entity e2)
{
    if (bound_primary_entity_p(e1) && bound_primary_entity_p(e2))
	return load_primary_entity(e1)==load_primary_entity(e2);
    else
	return false;
}


/*   DYNAMIC STATUS management.
 */
void init_dynamic_status()
{
    init_dynamic_hpf();
    init_primary_entity();
    init_renamings();
    init_maybeuseful_mappings();
    init_similar_mapping();
}

void reset_dynamic_status()
{
    reset_dynamic_hpf();
    reset_primary_entity();
    reset_renamings();
    reset_maybeuseful_mappings();
    /* reset_similar_mapping(); */
}

dynamic_status get_dynamic_status()
{
    return make_dynamic_status(get_dynamic_hpf(), 
			       get_primary_entity(), 
			       get_renamings(),
			       get_maybeuseful_mappings());
    /* get_similar_mapping() */
}

void set_dynamic_status(dynamic_status d)
{
    set_dynamic_hpf(dynamic_status_dynamics(d));
    set_primary_entity(dynamic_status_primary(d));
    set_renamings(dynamic_status_renamings(d));
    set_maybeuseful_mappings(dynamic_status_tokeep(d));
    /* set_similar_mapping(...) */
}

void close_dynamic_status()
{
    close_dynamic_hpf();
    close_primary_entity();
    close_renamings();
    close_maybeuseful_mappings();
    close_similar_mapping();
}

/*  a new dynamic entity is stored.
 *  HPF allows arrays and templates as dynamic.
 *  ??? could be asserted, but not here. should be checked afterward.
 */
void set_entity_as_dynamic(entity e)
{
    if (!bound_dynamic_hpf_p(e))
    {
	store_dynamic_hpf(e, make_entities(CONS(ENTITY, e, NIL)));
	store_primary_entity(e, e);

	store_similar_mapping(e, e);
    }
    /* else the entity was already declared as dynamic... */
}

/*  as expected, true if entity e is dynamic. 
 *  it is just a function name nicer than bound_...
 */
bool (*dynamic_entity_p)(entity) = bound_dynamic_hpf_p;

/* what: new_e is stored as a synonym of e.
 */
static void add_dynamic_synonym(
    entity new_e, 
    entity e)
{
    entities es = load_dynamic_hpf(e);
    pips_debug(3, "%s as %s synonyms\n", entity_name(new_e), entity_name(e));
    pips_assert("dynamicity", dynamic_entity_p(e) && !dynamic_entity_p(new_e));
    entities_list(es) = CONS(ENTITY, new_e, entities_list(es));
    store_dynamic_hpf(new_e, es);
    store_primary_entity(new_e, load_primary_entity(e));
}

void 
set_similar_mappings_for_updates(void)
{
    /* ??? for final update after compilation! hummm....
     */

    MAP(ENTITY, array,
    {
	pips_debug(7, "considering array %s\n", entity_name(array));

	/* ??? should not deal with the same array twice... 
	 */
	if (dynamic_entity_p(array) && 
	    bound_similar_mapping_p(array) && 
	    bound_new_node_p(array)) /* may be in another module */
	{
	    entity n = load_new_node(array);
	    entity s = load_similar_mapping(array);
	    entity ns = load_new_node(s);

	    pips_debug(7, "array %s\n", entity_name(array));
	    
	    if (!bound_new_node_p(n)) {
		store_new_node_variable(ns, n);
		store_new_node_variable(ns, array);
	    }
	}
    },
        list_of_distributed_arrays());
}

/******************************** NEW ENTITIES FOR MANAGING DYNAMIC ARRAYS */

/*  builds a synonym for entity e. The name is based on e, plus
 *  an underscore and a number added. May be used for templates and arrays.
 *  the new synonym is added as a synonym of e.
 */
static entity new_synonym(entity e)
{
    int n = gen_length(entities_list(load_dynamic_hpf(e))); /* syn. number */
    entity primary = load_primary_entity(e), new_e;
    const char* module = entity_module_name(e);
    char new_name[100];	
    
    sprintf(new_name, "%s_%x", entity_local_name(primary), (unsigned int) n);

    pips_debug(5, "building entity %s\n", new_name);

    new_e = FindOrCreateEntityLikeModel(module, new_name, primary);

    if (storage_formal_p(entity_storage(new_e)))
    { /* the new entity is rather local! */
	storage s = entity_storage(new_e);
	entity sub = get_current_module_entity();
	entity_storage(new_e) = 
	    make_storage(is_storage_ram,
	      make_ram(sub,
	        FindEntity(entity_local_name(sub),
				      DYNAMIC_AREA_LOCAL_NAME), 0, NIL));
	free_storage(s);
    }

    AddEntityToDeclarations(new_e, get_current_module_entity());

    add_dynamic_synonym(new_e, e);
    return new_e;
}

void static
check_for_similarity(
    entity a, /* array a to be compared against its similars */
    list /* of entity */ others)
{
    bool similar_found = false;

    pips_debug(3, "of %s\n", entity_name(a));

    if (!array_distributed_p(a)) /* no templates! */
	return;

    if (bound_similar_mapping_p(a)) /* job already done */
	return;

    /* look for copies with similar mappings for latter update
     */
    MAP(ENTITY, e,
    {
	pips_debug(7, "%s against %s\n", entity_name(a), entity_name(e));
	if (a!=e && array_distribution_similar_p(e, a))
	{
	    /* a -> init[e] for storage purposes 
	     */
	    pips_debug(8, "found similar: %s -> %s\n", 
		       entity_name(a), entity_name(e));
	    store_similar_mapping(a, load_similar_mapping(e));
	    similar_found = true;
	    break;
	}
    },
	others);

    if (!similar_found) store_similar_mapping(a, a);

    return;
}

/* check all *dynamic* arrays for some similars...
 */
void hpfc_check_for_similarities(list /* of entity */ le)
{
    MAP(ENTITY, array,
    {
	if (array_distributed_p(array))
	{
	    list /* of entity */ 
		seens = CONS(ENTITY, array, NIL); /* similar bases */
	    
	    pips_assert("is a primary!", primary_entity_p(array));
	    
	    check_for_similarity(array, NIL);
	    
	    MAP(ENTITY, a,
	    {
		if (!gen_in_list_p(a, seens))
		{
		    entity similar;
		    check_for_similarity(a, seens);
		    similar = load_similar_mapping(a);
		    pips_debug(1, "%s similar to %s\n",
			       entity_name(a), entity_name(similar));
		    
		    seens = gen_once(similar, seens);
		}
	    },
		entities_list(load_dynamic_hpf(array)));
	    
	    gen_free_list(seens);
	}
    },
	le);
}

/*  builds a new synonym for array a, the alignment of which 
 *  will be al. The new array is set as distributed.
 */
static entity new_synonym_array(
    entity a,
    align al)
{
    entity new_a = new_synonym(a);
    set_array_as_distributed(new_a);
    store_hpf_alignment(new_a, al);
    return new_a;
}

/*  builds a new synonym for template t, the distribution of which
 *  will be di. the new entity is set as a template, and/or as an array
 */
static entity new_synonym_template(
    entity t,
    distribute di)
{
    entity new_t = new_synonym(t);
    set_template(new_t);
    store_hpf_distribution(new_t, di);
    return new_t;
}

static bool conformant_entities_p(
    entity e1,
    entity e2)
{
    int ndim;
    if (e1==e2) return true;

    ndim = NumberOfDimension(e1);

    if (ndim!=NumberOfDimension(e2)) return false;

    for (; ndim>0; ndim--)
    {
	int l1, u1, l2, u2;
	get_entity_dimensions(e1, ndim, &l1, &u1);
	get_entity_dimensions(e2, ndim, &l2, &u2);
	if (l1!=l2 || u1!=u2) return false;
    }

    return true;
}

/*  comparison of DISTRIBUTE.
 */
static bool same_distribute_p(
    distribute d1, 
    distribute d2)
{
    list /* of distributions */ l1 = distribute_distribution(d1),
                                l2 = distribute_distribution(d2);

    if (!conformant_entities_p(distribute_processors(d1),
			       distribute_processors(d2))) return false;
    
    pips_assert("valid distribution", gen_length(l1)==gen_length(l2));

    for(; !ENDP(l1); POP(l1), POP(l2))
    {
	distribution i1 = DISTRIBUTION(CAR(l1)),
	             i2 = DISTRIBUTION(CAR(l2));
	style s1 = distribution_style(i1),
	      s2 = distribution_style(i2);
	tag t1 = style_tag(s1);

	if (t1!=style_tag(s2)) return false;
	if (t1!=is_style_none &&
	    !expression_equal_p(distribution_parameter(i1),
				distribution_parameter(i2)))
	    return false;
    }

    return true;
}

/*  comparison of ALIGN.
 */
static bool same_alignment_in_list_p(
    alignment a,
    list /* of alignments */ l)
{
    intptr_t adim = alignment_arraydim(a),
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

    return false;
}

bool conformant_templates_p(
    entity t1,
    entity t2)
{
    if (t1==t2) 
	return true;
    else if (!conformant_entities_p(t1, t2))
	return false;
    else
	return same_distribute_p
	    (load_hpf_distribution(t1), load_hpf_distribution(t2));
}

static bool same_align_p(
    align a1,
    align a2)
{
    list /* of alignments */ l1 = align_alignment(a1),
                             l2 = align_alignment(a2);

    if (!conformant_templates_p(align_template(a1),align_template(a2)) ||
	(gen_length(l1)!=gen_length(l2))) 
	return false;

    MAP(ALIGNMENT, a,
	if (!same_alignment_in_list_p(a, l2)) return false,
	l1);

    return true;
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
entity array_synonym_aligned_as(
    entity array,
    align a)
{
    ifdebug(8)
    {
	pips_debug(8, "array %s\n", entity_name(array));
	print_align(a);
    }

    MAP(ENTITY, ar,
    {
	if (same_align_p(load_hpf_alignment(ar), a))
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

align new_align_with_template(
    align a,
    entity t)
{
    align b = copy_align(a);
    align_template(b) = t;
    return b;
}

/* what: finds or creates a new entity distributed as needed.
 * input: an template (which *must* be dynamic) and a distribute
 * output: returns a template distributed as specified by d
 * side effects:
 *  - creates a new entity if necessary. 
 *  - this entity is stored as a synonym of array, and tagged as dynamic.
 *  - the distribute is freed if not used.
 */
entity template_synonym_distributed_as(
    entity temp,
    distribute d)
{
    MAP(ENTITY, t,
    {
	if (same_distribute_p(load_hpf_distribution(t), d))
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

/******************************************************** SIMILAR MAPPPINGS */
/* array_distribution_similar_p
 *
 * returns whether a1 and a2 are similar, i.e. even if 
 * distributed differently, the resulting mapping is similar.
 *
 * e.g. align a1(i,j) with T(i,j), distribute T(*,block) 
 * and  align a2(i,j) with T(j,i), distribute T(block,*)
 *
 * impact: the same area can be used for holding both array versions
 * but the version number must be accurate.
 */
static bool
same_distribution_p(distribution d1, distribution d2)
{
    return 
	style_tag(distribution_style(d1))==style_tag(distribution_style(d2))
	&& expression_equal_p(distribution_parameter(d1),
			      distribution_parameter(d2));
}

#define RETAL(msg, res) \
  { pips_debug(7, "%d because %s\n", res, msg); return res;}

static bool 
same_alignment_p(entity e1, entity t1, alignment a1, 
		 entity e2, entity t2, alignment a2)
{
    int b1, l1, b2, l2;
    bool b;

    if (alignment_undefined_p(a1) || alignment_undefined_p(a2))
    {
        b=alignment_undefined_p(a1) && alignment_undefined_p(a2);
	RETAL("some undefined", b);
    }

    pips_debug(7, "considering %s[dim=%"PRIdPTR"] and %s[dim=%"PRIdPTR"]\n",
	       entity_name(e1), alignment_arraydim(a1),
	       entity_name(e2), alignment_arraydim(a2));

    /* compares the alignments if any 
     */
    if (alignment_arraydim(a1)!=alignment_arraydim(a2)) 
	RETAL("diff. array dim", false);

    if (!expression_equal_p(alignment_rate(a1), alignment_rate(a2)))
	RETAL("different rate", false);

    if (SizeOfIthDimension(t1, alignment_templatedim(a1))!=
	SizeOfIthDimension(t2, alignment_templatedim(a2)))
	RETAL("different template size", false);

    b1 = HpfcExpressionToInt(alignment_constant(a1));
    b2 = HpfcExpressionToInt(alignment_constant(a2));
    l1 = HpfcExpressionToInt
	(dimension_lower(FindIthDimension(t1,alignment_templatedim(a1))));
    l2 = HpfcExpressionToInt
	(dimension_lower(FindIthDimension(t2,alignment_templatedim(a2))));

    b=(b1-l1)==(b2-l2);
    RETAL("shift", b);
}

#define RET(msg, what) \
  { pips_debug(6, "not similar because %s\n", msg); return what;}

bool 
array_distribution_similar_p(entity a1, entity a2)
{
    align 
	al1 = load_hpf_alignment(a1),
	al2 = load_hpf_alignment(a2);
    entity 
	t1 = align_template(al1),
	t2 = align_template(al2);
    distribute 
	d1 = load_hpf_distribution(t1),
	d2 = load_hpf_distribution(t2);
    entity
	p1 = distribute_processors(d1),
	p2 = distribute_processors(d2);
    int i, td1, td2,
	ndimdist = NumberOfDimension(p1);

    pips_assert("same primary", 
		load_primary_entity(a1)==load_primary_entity(a2));

    pips_debug(6, "comparing %s and %s\n", entity_name(a1), entity_name(a2));

    /* conformant processors 
     * ??? could assume that P(1:1,1:n) is conformant to P'(1:n)?
     */
    if (!conformant_entities_p(p1, p2)) 
	RET("different processors", false);
    
    for (i=1; i<=ndimdist; i++) /* considering ith dim of proc */
    {
	/* conformant distributions
	 */
	distribution 
	    x1 = FindDistributionOfProcessorDim
	      (distribute_distribution(d1), i, &td1),
	    x2 = FindDistributionOfProcessorDim
	      (distribute_distribution(d2), i, &td2);
	alignment at1, at2;
	int dsize = SizeOfIthDimension(p1,i);
	
	/* if the size is 1, whatever the distribution it is ok! */
	if (dsize!=1)
	{
	    if (!same_distribution_p(x1, x2)) 
		RET("different distribution", false);
	    
	    /* conformant alignments for that pe dim
	     * !!! the HPF mapping "semantics" insure that the corresponding 
	     * dimension is distributed!
	     */
	    at1 = FindAlignmentOfTemplateDim(align_alignment(al1), td1);
	    at2 = FindAlignmentOfTemplateDim(align_alignment(al2), td2);

	    if (!same_alignment_p(a1,t1,at1,a2,t2,at2)) 
		RET("different alignment", false);
	}
    }

    pips_debug(6, "similar distributions!\n");
    return true;
}

/**************************************************** MAPPING OF ARGUMENTS */

/* whether call c inplies a distributed argument 
 */
bool 
hpfc_call_with_distributed_args_p(call c)
{
    entity f = call_function(c);
    int len = gen_length(call_arguments(c));

    /* no intrinsics */
    if (value_intrinsic_p(entity_initial(f)) ||
	hpf_directive_entity_p(f)) return false;
    
    /* else checks for distributed arguments */
    for (; len>0; len--)
    {
	entity arg = find_ith_parameter(f, len);
	if (array_distributed_p(arg)) return true;
    }

    return false;
}

static list /* of renaming */ 
add_once_to_renaming_list(
    list /* of renaming */ l,
    entity o,
    entity n)
{
    MAP(RENAMING, r,
    {
	if (renaming_old(r)==o)
	{
	    pips_assert("same required mapping", renaming_new(r)==n);
	    return l;
	}
    },
	l);

    return CONS(RENAMING, make_renaming(o, n), l);
}

/* ??? only simple calls are handled. imbrication may cause problems.
 * ??? should recurse thru all the calls at a call instruction...
 */
void
hpfc_translate_call_with_distributed_args(
    statement s, /* the statement the call belongs to */
    call c)      /* the call. (not necessarily an instruction) */
{
    list /* of remapping */ lr = NIL,
         /* of expression */ args;
    int len, i;
    entity f;

    f = call_function(c);
    args = call_arguments(c);
    len = gen_length(args);

    pips_debug(1, "function %s\n", entity_name(f));

    for (i=1; i<=len; i++, POP(args))
    {
	entity arg = find_ith_parameter(f, i);
	pips_debug(7, "considering argument %d\n", i);

	if (array_distributed_p(arg))
	{
	    align al;
	    entity passed, copy;
	    expression e = EXPRESSION(CAR(args));

	    passed = expression_to_entity(e);

	    pips_debug(8, "passed:%s, arg:%s\n", 
		       entity_name(passed), entity_name(arg));
	    pips_assert("distributed and conformant",
			array_distributed_p(passed) &&
			conformant_entities_p(passed, arg));

	    set_entity_as_dynamic(passed);
	    add_a_dynamic(passed);

	    al = copy_align(load_hpf_alignment(arg));
	    copy = array_synonym_aligned_as(passed, al);

	    pips_debug(3, "%s (arg %d) %s -> %s\n", entity_name(arg), i, 
		       entity_name(passed), entity_name(copy));

	    /* the substitution in the call will be performed at the 
	     * propagation phase of dynamic arrays, later on.
	     */
	    /* add the renaming in the list.
	     * ??? should be added only once! what about call(A,A)...
	     */
	    lr = add_once_to_renaming_list(lr, passed, copy);
	}
    }

    if (lr) /* should always be the case */
    {
	list /* of statement */ lpre = NIL, lpos = NIL;
	entity rename = hpfc_name_to_entity(HPF_PREFIX RENAME_SUFFIX);
	
	lpre = CONS(STATEMENT, 
		    instruction_to_statement(statement_instruction(s)), 
		    NIL);

	MAP(RENAMING, r,
	{
	    entity passed = renaming_old(r);
	    entity copied = renaming_new(r);

	    lpre = CONS(STATEMENT, call_to_statement(make_call(rename,
			CONS(EXPRESSION, entity_to_expression(passed),
			CONS(EXPRESSION, entity_to_expression(copied), NIL)))),
			lpre);

	    lpos = CONS(STATEMENT, call_to_statement(make_call(rename,
			CONS(EXPRESSION, entity_to_expression(copied),
			CONS(EXPRESSION, entity_to_expression(passed), NIL)))),
			lpos);
	},
	    lr);

	statement_instruction(s) = 
	    make_instruction_block(gen_nconc(lpre, lpos));
	/* Do not forget to move forbidden information associated with
	   block: */
	fix_sequence_statement_attributes(s);	
	
	DEBUG_STAT(3, "out", s);
    }
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
 * - modified_dynamics: from a remapping statement, the remapped arrays 
 *   that may be modified in their new shape.
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
GENERIC_LOCAL_FUNCTION(modified_dynamics, statement_entities)
GENERIC_LOCAL_FUNCTION(remapping_graph, controlmap)
GENERIC_LOCAL_FUNCTION(reaching_mappings, statement_entities)
GENERIC_LOCAL_FUNCTION(leaving_mappings, statement_entities)
GENERIC_LOCAL_FUNCTION(remapped, statement_entities)

void init_dynamic_locals()
{
    init_alive_synonym();
    init_used_dynamics();
    init_modified_dynamics();
    init_reaching_mappings();
    init_leaving_mappings();
    init_remapped();
    init_remapping_graph();
}

void close_dynamic_locals()
{
    close_alive_synonym();
    close_used_dynamics();
    close_modified_dynamics();
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
    old_variable = entity_undefined, /* entity to be replaced, the primary? */
    new_variable = entity_undefined; /* replacement */
static bool 
    array_propagation, /* true if an array is propagated, false if template */
    array_used,        /* true if the array was actually used */
    array_modified;    /* true if the array may be modified... */
static statement 
    initial_statement = statement_undefined; /* starting point */

/*  initialize both the remapping graph and the used dynamics for s
 */
static void lazy_initialize_for_statement(statement s)
{
    if (!bound_remapping_graph_p(s))
	store_remapping_graph(s, make_control(s, NIL, NIL));

    if (!bound_used_dynamics_p(s))
	store_used_dynamics(s, make_entities(NIL));

    if (!bound_modified_dynamics_p(s))
	store_modified_dynamics(s, make_entities(NIL));
}

static void add_as_a_closing_statement(statement s)
{
    control 
	c = load_remapping_graph(initial_statement),
	n = (lazy_initialize_for_statement(s), load_remapping_graph(s));

    what_stat_debug(6, s);

    control_successors(c) = 
	gen_once(load_remapping_graph(s), control_successors(c));
    control_predecessors(n) = 
	gen_once(load_remapping_graph(initial_statement),
		 control_predecessors(n));
}

static void add_as_a_used_variable(entity e)
{
    entities es = load_used_dynamics(initial_statement);
    entities_list(es) = gen_once(load_primary_entity(e), entities_list(es));
}

void 
add_as_a_used_dynamic_to_statement(statement s, entity e)
{
    entities es = load_used_dynamics(s);
    entities_list(es) = gen_once(load_primary_entity(e), entities_list(es));
}

static void add_as_a_modified_variable(entity e)
{
    entities es = load_modified_dynamics(initial_statement);
    entities_list(es) = gen_once(load_primary_entity(e), entities_list(es));
}

static void add_alive_synonym(
    statement s,
    entity a)
{
    entities es;

    /*   lazy initialization 
     */
    if (!bound_alive_synonym_p(s))
	store_alive_synonym(s, make_entities(NIL));

    es = load_alive_synonym(s);
    entities_list(es) = gen_once(a, entities_list(es));
}

static void ref_rwt(reference r)
{
    entity var = reference_variable(r);
    if (var==old_variable || var==new_variable)
    {
	reference_variable(r) = new_variable;
	array_used = true;
    }
}

static void simple_switch_old_to_new(statement s)
{
    DEBUG_STAT(9, "considering", s);

    /* looks for direct references in s and switch them
     */
    gen_multi_recurse
	(statement_instruction(s),
	 statement_domain,    gen_false, gen_null, /* STATEMENT */
	 unstructured_domain, gen_false, gen_null, /* UNSTRUCTURED ? */
	 reference_domain,    gen_true,  ref_rwt,  /* REFERENCE */
	 NULL);

    /* whether the array may be written... by scanning the proper effects of s.
     * (caution, was just switched to the new_variable!)
     */
    pips_debug(8, "statement %p, array %s, rw proper %d\n", 
	       s, entity_name(new_variable), bound_proper_rw_effects_p(s));

    if (!array_modified && bound_proper_rw_effects_p(s))
    {
	FOREACH(EFFECT, e, load_proper_rw_effects_list(s))
	  {
	    if(store_effect_p(e)) {
	      entity v = reference_variable(effect_any_reference(e));
	      if (same_primary_entity_p(v,new_variable) && effect_write_p(e))
		{
		  pips_debug(9, "%s W in %p\n", entity_name(new_variable), s);
		  array_modified = true;
		  return;
		}
	    }
	  }
    }
}

static bool 
rename_directive_p(entity f)
{
    return same_string_p(entity_local_name(f), HPF_PREFIX RENAME_SUFFIX);
}

/*  true if not a remapping for old. 
 *  if it is a remapping, operates the switch.
 */
#define ret(why, what) \
  { pips_debug(9, "ret %d because %s\n", what, why); return what; }

static bool 
continue_propagation_p(statement s)
{
    instruction i = statement_instruction(s);

    what_stat_debug(8, s);
    DEBUG_STAT(9, "current", s);

    if (!instruction_call_p(i)) 
    {
	ret("not a call", true);
    }
    else
    {
	call c = instruction_call(i);
	entity fun = call_function(c);
	
	if (rename_directive_p(fun) && array_propagation)
	{
	    entity
		primary = safe_load_primary_entity(old_variable),
		array;

	    DEBUG_STAT(8, "rename directive", s);

	    array = expression_to_entity(EXPRESSION(CAR(call_arguments(c))));

	    if (safe_load_primary_entity(array)==primary)
	    {
		add_alive_synonym(s, new_variable);
		add_as_a_closing_statement(s);
		ret("rename to the same", false);
	    }
	}
	else if (realign_directive_p(fun) && array_propagation)
	{
	    entity primary = safe_load_primary_entity(old_variable);
	    int nbofargs = gen_length(call_arguments(c));

	    DEBUG_STAT(8, "realign directive", s);

	    MAP(EXPRESSION, e,
	    {
		nbofargs--;
		if (expression_reference_p(e)) 
		{
		    reference r = expression_to_reference(e);
		    entity var = reference_variable(r);
		    
		    if (nbofargs==0) /* up to the template! */
			ret("template of realign", true);
		
		    if (safe_load_primary_entity(var)==primary)
		    {
			/*  the variable is realigned.
			 */
			add_alive_synonym(s, new_variable);
			add_as_a_closing_statement(s);
			ret("realign array",  false);
		    }
		}
		/* else it may be a call because of ALIGN () WITH T()::X...
		 */
	    },
		call_arguments(c));
	}
	else if (redistribute_directive_p(fun))
	{
	    entity t = safe_load_primary_entity(array_propagation ?
		align_template(load_hpf_alignment(new_variable)):old_variable);
		
	    DEBUG_STAT(8, "redistribute directive", s);

	    MAP(EXPRESSION, e,
	    {
		entity v = expression_to_entity(e);

		if (!entity_template_p(v)) /* up to the processor, stop */
		    ret("processors of distribute", true);

		/*   if template t is redistributed...
		 */
		if (safe_load_primary_entity(v)==t)
		{
		    if (array_propagation)
			add_alive_synonym(s, new_variable);
		    add_as_a_closing_statement(s);

		    ret("redistribute template", false);
		}
	    },
		call_arguments(c));
	}
	else if (dead_fcd_directive_p(fun) && array_propagation)
	{
	    entity primary = safe_load_primary_entity(old_variable);
	    
	    MAP(EXPRESSION, e, 
		if (primary==expression_to_entity(e)) ret("dead array", false),
		call_arguments(c)); 
	}

	ret("default", true);
    }
}

void 
propagate_synonym(
    statement s,   /* starting statement for the propagation */
    entity old,    /* entity to be replaced */
    entity new,    /* replacement for the entity */
    bool is_array  /* true if array, false if template */)
{
    statement current;

    what_stat_debug(3, s);
    pips_debug(3, "%s -> %s (%s)\n", entity_name(old), entity_name(new),
	       is_array? "array": "template");
    DEBUG_STAT(7, "before propagation", get_current_module_statement());

    old_variable = safe_load_primary_entity(old), new_variable = new, 
    array_propagation = is_array;
    array_used = false;
    array_modified = false;
    initial_statement = s;

    lazy_initialize_for_statement(s);

    init_ctrl_graph_travel(s, continue_propagation_p);

    while (next_ctrl_graph_travel(&current))
	simple_switch_old_to_new(current);

    close_ctrl_graph_travel();

    if (array_used) add_as_a_used_variable(old);
    if (array_modified) add_as_a_modified_variable(old);

    old_variable = entity_undefined, 
    new_variable = entity_undefined;

    DEBUG_STAT(7, "after propagation", get_current_module_statement());

    pips_debug(4, "out\n");
}

/********************************* REMAPPING GRAPH REMAPS "SIMPLIFICATION" */

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
	
	le = gen_once(old, le);
	lp = gen_once(load_primary_entity(old), lp);
	ll = gen_once(new, ll);
    },	 
	load_renamings(s));

    store_reaching_mappings(s, make_entities(le));
    store_remapped(s, make_entities(lp));
    store_leaving_mappings(s, make_entities(ll));
}

static void remove_not_remapped_leavings(statement s)
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

/* must be called after useless leaving mappings removal */
static void initialize_maybeuseful_mappings(statement s)
{
    store_maybeuseful_mappings(s, copy_entities(load_leaving_mappings(s)));    
}

static void reinitialize_reaching_mappings(statement s)
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

/* more options, such as may or must? */
static list /* of statement */
propagation_on_remapping_graph(
    statement s,
    list /* of statement */ ls,
    entities (*built)(statement),
    entities (*condition)(statement),
    bool local, /* local or remote condition */
    bool forward) /* backward or forward */
{
    bool modified = false;
    entities built_set = built(s);
    list /* of entity */
	lrem = entities_list(load_remapped(s)),
	lbuilt = entities_list(built_set);
    control current = load_remapping_graph(s);

    MAP(CONTROL, c,
    {
	statement sc = control_statement(c);
	list /* of entity */ lp_rem = entities_list(load_remapped(sc));
	list /* idem      */ lp_cond = entities_list(condition(local?s:sc));

	MAP(ENTITY, e,
	{
	    entity prim = load_primary_entity(e);

	    if (gen_in_list_p(prim, lrem) && 
		gen_in_list_p(prim, lp_rem) &&
		!gen_in_list_p(prim, lp_cond) &&
		!gen_in_list_p(e, lbuilt))
	    {
		what_stat_debug(5, s);
		pips_debug(5, "adding %s from\n", entity_local_name(e));
		what_stat_debug(5, sc);

		lbuilt = CONS(ENTITY, e, lbuilt);
		modified = true;
	    }
	},
	    entities_list(built(sc)));
    },
	forward ? control_predecessors(current):control_successors(current));

    /* if the built set was modified, must propagate at next step... */
    if (modified)
    {
	entities_list(built_set) = lbuilt;
	MAP(CONTROL, c, ls = gen_once(control_statement(c), ls), 
	  forward?control_successors(current):control_predecessors(current));
    }

    return ls;
}

static list /* of statements */
propagate_used_arrays(
    statement s,
    list /* of statements */ ls)
{
    return propagation_on_remapping_graph
      (s, ls, load_reaching_mappings, load_used_dynamics, false, true);
}

static list /* of statements */
propagate_maybeuseful_mappings(
    statement s,
    list /* of statement */ ls)
{
    return propagation_on_remapping_graph
      (s, ls, load_maybeuseful_mappings, load_modified_dynamics, true, false);
}

static void remove_from_entities(
    entity primary,
    entities es)
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

static void remove_unused_remappings(statement s)
{
    entities remapped = load_remapped(s),
             reaching = load_reaching_mappings(s),
             leaving = load_leaving_mappings(s);
    list /* of entity(s) */ le = gen_copy_seq(entities_list(remapped)),
                            lu = entities_list(load_used_dynamics(s));

    MAP(ENTITY, primary,
    {
	pips_debug(5, "considering array %s\n", entity_name(primary));

	if (!storage_formal_p(entity_storage(primary)) &&
	    !gen_in_list_p(primary, lu))
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
static void regenerate_renamings(statement s)
{
    list /* of entity(s) */
	lr = entities_list(load_reaching_mappings(s)),
	ll = entities_list(load_leaving_mappings(s)),
	ln = NIL;
    
    what_stat_debug(4, s);

    MAP(ENTITY, target,
    {
	entity primary = load_primary_entity(target);
	bool some_source_found = false;

	MAP(ENTITY, source,
	{
	    if (load_primary_entity(source)==primary && source!=target)
	    {
		pips_debug(4, "%s -> %s\n", 
			   entity_name(source), entity_name(target));
		ln = CONS(RENAMING, make_renaming(source, target), ln);
		some_source_found = true;
	    }
	},
	    lr);

	/* ensures some remapping to enforce an update of the status,
	 * which may be necessary, for instance if KILL was used.
	 */
	if (!some_source_found)
	{
	    pips_debug(7, "no source found for %s\n", entity_name(target));
	    ln = CONS(RENAMING, make_renaming(target, target), ln);
	}
    },
	ll);

    {
	list /* of renaming(s) */ l = load_renamings(s);
	gen_map((gen_iter_func_t)gen_free, l), gen_free_list(l); /* ??? */
	update_renamings(s, ln);
    }
}

static list /* of statements */ 
list_of_remapping_statements()
{
    list /* of statements */ l = NIL;
    CONTROLMAP_MAP(s, c,
    {
	pips_debug(9, "%p -> %p\n", s, c);
	l = CONS(STATEMENT, s, l);
    },
		   get_remapping_graph());
    return l;
}

/* functions used for debug.
 */
static void print_control_ordering(control c)
{
    statement s = control_statement(c);
    int so = statement_ordering(s);
    fprintf(stderr, "(%d:%d:%" PRIdPTR "), ", 
	    ORDERING_NUMBER(so), ORDERING_STATEMENT(so),
	    statement_ordering(s));
}

#define elst_ifdef(what, name, s) \
 if (bound_##name##_p(s)){DEBUG_ELST(1, what, entities_list(load_##name(s)));}\
 else pips_debug(1, "no " what "\n");

static void dump_remapping_graph_info(statement s)
{
    control c = load_remapping_graph(s);

    what_stat_debug(1, s);

    fprintf(stderr, "predecessors: ");
    gen_map((gen_iter_func_t)print_control_ordering, control_predecessors(c));
    fprintf(stderr, "\nsuccessors: ");
    gen_map((gen_iter_func_t)print_control_ordering, control_successors(c));
    fprintf(stderr, "\n");

    elst_ifdef("remapped", remapped, s);
    elst_ifdef("used", used_dynamics, s);
    elst_ifdef("modified", modified_dynamics, s);
    elst_ifdef("reaching", reaching_mappings, s);
    elst_ifdef("leaving", leaving_mappings, s);
    elst_ifdef("maybe useful", maybeuseful_mappings, s);
}

static void 
dump_remapping_graph(
    string when,
    list /* of statement */ ls)
{
    fprintf(stderr, "[dump_remapping_graph] for %s\n", when);
    gen_map((gen_iter_func_t)dump_remapping_graph_info, ls);
    fprintf(stderr, "[dump_remapping_graph] done\n");
}

void dump_current_remapping_graph(string when)
{
    dump_remapping_graph(when, list_of_remapping_statements());
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
void simplify_remapping_graph(void)
{
    list /* of statements */ ls = list_of_remapping_statements();
    statement root = get_current_module_statement();
    what_stat_debug(4, root);

    ifdebug(8) dump_remapping_graph("0", ls);

    gen_map((gen_iter_func_t)initialize_reaching_propagation, ls);
    gen_map((gen_iter_func_t)remove_not_remapped_leavings, ls);
    gen_map((gen_iter_func_t)initialize_maybeuseful_mappings, ls);
    gen_map((gen_iter_func_t)reinitialize_reaching_mappings, ls);

    ifdebug(4) dump_remapping_graph("1", ls);

    pips_debug(4, "used array propagation\n");
    gen_closure(propagate_used_arrays, ls);

    pips_debug(4, "may be useful mapping propagation\n");
    gen_closure(propagate_maybeuseful_mappings, ls);

    ifdebug(4) dump_remapping_graph("2", ls);

    if (bound_remapped_p(root))	remove_unused_remappings(root);

    gen_map((gen_iter_func_t)regenerate_renamings, ls);

    gen_free_list(ls);
}

/* what: returns the list of alive arrays for statement s and template t.
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

    pips_assert("template", entity_template_p(t) && primary_entity_p(t));
    pips_debug(7, "for template %s\n", entity_name(t));

    /* ??? well, it is not necessarily initialized, I guess...
     */
    if (!bound_alive_synonym_p(s))
	store_alive_synonym(s, make_entities(NIL));

    /*   first the alive list is scanned.
     */
    MAP(ENTITY, array,
    {
	entity ta = align_template(load_hpf_alignment(array));

	if (safe_load_primary_entity(ta)==t)
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
	    if (align_template(load_hpf_alignment(array))==t)
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
statement generate_copy_loop_nest(
    entity src,
    entity trg)
{
    type t = entity_type(src);
    list /* of entities */    indexes = NIL,
         /* of expressions */ idx_expr,
         /* of dimensions */  dims;
    statement current;
    entity module;
    int ndims, i;

    if (src==trg) return make_empty_statement();

    pips_assert("valid arguments",
		array_distributed_p(src) && array_distributed_p(trg) &&
		type_variable_p(t) &&
		load_primary_entity(src)==load_primary_entity(trg)); /* ??? */

    dims = variable_dimensions(type_variable(t));
    ndims = gen_length(dims);
    
    /*  builds the set of indexes needed to scan the dimensions.
     */
    for(module=get_current_module_entity(), i=ndims; i>0; i--)
      indexes = CONS(ENTITY, 
		     hpfc_new_variable(module, MakeBasic(is_basic_int)),
		     indexes);

    idx_expr = entities_to_expressions(indexes);

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
