/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: unary_operators.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains generic unary operators for effects and lists of effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "effects-generic.h"

void 
effects_map(list l_eff, void (*apply)(effect))
{
    MAP(EFFECT, eff, {apply(eff);}, l_eff);      
}

list
effects_to_effects_map(list l_eff, effect (*pure_apply)(effect))
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	{l_new = gen_nconc(l_new, CONS(EFFECT, pure_apply(eff),NIL));},
	l_eff);
    return l_new;
}

void
effects_filter_map(list l_eff, bool (*filter)(effect), void (*apply)(effect))
{
    MAP(EFFECT, eff, {if (filter(eff)) apply(eff);}, l_eff);      
}

list
effects_to_effects_filter_map(list l_eff , bool (*filter)(effect),
			      effect (*pure_apply)(effect))
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	{
	    if (filter(eff))
		l_new = gen_nconc(l_new, CONS(EFFECT, pure_apply(eff),NIL));},
	l_eff);
    return l_new;
}


list 
effects_add_effect(list l_eff, effect eff)
{
    return gen_nconc(l_eff, CONS(EFFECT, eff, NIL));
}



list
effects_read_effects(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	{
	    if (effect_read_p(eff))
		l_new = gen_nconc(l_new, CONS(EFFECT, eff,NIL));},
	l_eff);
    return l_new;
}

list
effects_write_effects(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	{
	    if (effect_write_p(eff))
		l_new = gen_nconc(l_new, CONS(EFFECT, eff,NIL));},
	l_eff);
    return l_new;
}

list
effects_read_effects_dup(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	{
	    if (effect_read_p(eff))
		l_new =
		    gen_nconc(l_new, CONS(EFFECT, (*effect_dup_func)(eff), NIL));},
	l_eff);
    return l_new;
}

list
effects_write_effects_dup(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
    {
	if (effect_write_p(eff))
	    l_new = gen_nconc(l_new, 
			      CONS(EFFECT, (*effect_dup_func)(eff), NIL));
    },
	l_eff);
    return l_new;
}

effect
effect_nop(effect eff)
{
    return eff;
}

list
effects_nop(list l_eff)
{
    return l_eff;
}

void
effect_to_may_effect(effect eff)
{
    effect_approximation_tag(eff) = is_approximation_may;
}

void
effects_to_may_effects(list l_eff)
{
    effects_map(l_eff, effect_to_may_effect);
}

void
effect_to_write_effect(effect eff)
{
    effect_action_tag(eff) = is_action_write;
}

void
effects_to_write_effects(list l_eff)
{
    effects_map(l_eff, effect_to_write_effect);
}

void
array_effects_to_may_effects(list l_eff)
{
    MAP(EFFECT, eff, 
	{
	    if (!effect_scalar_p(eff))
		effect_to_may_effect(eff);
	}, 
	l_eff);      

}

list
effects_dup_without_variables(list l_eff, list l_var)
{
    list l_res = NIL;
    
    MAP(EFFECT, eff,
    {
	if (gen_find_eq(effect_entity(eff), l_var) == entity_undefined)
	    l_res = effects_add_effect(l_res, (*effect_dup_func)(eff));
        else
	    pips_debug(7, "Effect on variable %s removed\n",
		       entity_local_name(effect_entity(eff)));
    }, l_eff);
    return(l_res);
}

effect 
effect_dup(effect eff)
{
    return((*effect_dup_func)(eff));
}

list
effects_dup(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	{l_new = gen_nconc(l_new, CONS(EFFECT, (*effect_dup_func)(eff), NIL));},
	l_eff);
    return l_new;
}

void
effect_free(effect eff)
{
    (*effect_free_func)(eff);
}

void
effects_free(list l_eff)
{
    MAP(EFFECT, eff,
	{(*effect_free_func)(eff);},
	l_eff);
    gen_free_list(l_eff);
}

/* list effect_to_nil_list(effect eff)
 * input    : an effect
 * output   : an empty list of effects
 * modifies : nothing
 * comment  : 
 */
list effect_to_nil_list(effect eff)
{
    return(NIL);
}

/* list effects_to_nil_list(eff)
 * input    : an effect
 * output   : an empty list of effects
 * modifies : nothing
 * comment  : 	
 */
list effects_to_nil_list(effect eff1, effect eff2)
{
    return(NIL);
}

list effect_to_list(effect eff)
{
    return(CONS(EFFECT,eff,NIL));
}

list effect_to_may_effect_list(effect eff)
{
    effect_approximation_tag(eff) = is_approximation_may;
    return(CONS(EFFECT,eff,NIL));
}


/***********************************************************************/
/* UNDEFINED UNARY OPERATORS                                           */
/***********************************************************************/

/* Composition with transformers */

list
effects_undefined_composition_with_transformer(list l_eff, transformer trans)
{
    return list_undefined;
}

list
effects_composition_with_transformer_nop(list l_eff, transformer trans)
{
    return l_eff;
}




/* Composition with preconditions */

list
effects_undefined_composition_with_preconditions(list l_eff, transformer trans)
{
    return list_undefined;
}

list
effects_composition_with_preconditions_nop(list l_eff, transformer trans)
{
    return l_eff;
}

/* Union over a range */

descriptor
loop_undefined_descriptor_make(loop l)
{
    return descriptor_undefined;
}

list 
effects_undefined_union_over_range(
    list l_eff, entity index, range r, descriptor d)
{
    return list_undefined;
}

list 
effects_union_over_range_nop(list l_eff, entity index, range r, descriptor d)
{
    return l_eff;
}


list
effects_undefined_descriptors_variable_change(list l_eff, entity orig_ent,
					      entity new_ent)
{
    return list_undefined;
}

list
effects_descriptors_variable_change_nop(list l_eff, entity orig_ent,
					      entity new_ent)
{
    return l_eff;
}


descriptor
effects_undefined_vector_to_descriptor(Pvecteur v)
{
    return descriptor_undefined;
}

list 
effects_undefined_loop_normalize(list l_eff, entity index, range r,
				 entity *new_index, descriptor range_descriptor,
				 bool descriptor_update_p)
{
    return list_undefined; 
}

list 
effects_loop_normalize_nop(list l_eff, entity index, range r,
			   entity *new_index, descriptor range_descriptor,
			   bool descriptor_update_p)
{
    return l_eff; 
}

list /* of nothing */
db_get_empty_list(string name)
{
    pips_debug(5, "getting nothing for %s\n", name);
    return NIL;
}
