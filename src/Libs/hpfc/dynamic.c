/*
 * HPFC module by Fabien COELHO
 *
 * $RCSfile: dynamic.c,v $ ($Date: 1995/04/12 15:49:30 $, )
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
 */
GENERIC_GLOBAL_FUNCTION(dynamic_hpf, entity_entities);
GENERIC_GLOBAL_FUNCTION(primary_entity, entitymap);

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

/*  as expected, TRUE if entity e is dynamic. 
 */
bool dynamic_entity_p(e)
entity e;
{
    return(bound_dynamic_hpf_p(e));
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
    int n = gen_length(entities_list(load_dynamic_hpf(e))); /* number */
    entity primary = load_primary_entity(e), new_e;
    string module = entity_module_name(e);
    char new_name[100];	
    
    sprintf(new_name, "%s_%x", entity_local_name(primary), (unsigned int) n);

    debug(5, "new_synonym", "building entity %s\n", new_name);

    new_e = FindOrCreateEntityLikeModel(module, new_name, primary);
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

/*  as expected, TRUE if d1 and d2 describe the same mapping
 */
static bool same_distribute_p(d1, d2)
distribute d1, d2;
{
    pips_error("same_distribute_p", "not implemented yet");
    return(TRUE);
}

/* idem for align
 */
static bool same_align_p(a1, a2)
align a1, a2;
{
    pips_error("same_align_p", "not implemented yet");
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

/* that is all
 */
