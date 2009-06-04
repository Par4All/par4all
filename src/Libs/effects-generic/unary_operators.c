/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
	l_new = CONS(EFFECT, pure_apply(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
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
	if (filter(eff)) l_new = CONS(EFFECT, pure_apply(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
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
	if (effect_read_p(eff)) l_new = CONS(EFFECT, eff, l_new),
	l_eff);
    return gen_nreverse(l_new);
}

list
effects_write_effects(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (effect_write_p(eff)) l_new = CONS(EFFECT, eff, l_new),
	l_eff);
    return gen_nreverse(l_new);
}

/* At least one of the effects in l_eff is a write */
bool effects_write_at_least_once_p(list l_eff)
{
  bool write_once_p = FALSE;

  FOREACH(EFFECT, eff, l_eff) {
    if (effect_write_p(eff)) {
      write_once_p = TRUE;
      break;
    }
  }
  return write_once_p;
}

list
effects_read_effects_dup(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (effect_read_p(eff))
           l_new = CONS(EFFECT, (*effect_dup_func)(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
}

list
effects_write_effects_dup(list l_eff)
{
    list l_new = NIL;
    MAP(EFFECT, eff,
	if (effect_write_p(eff))
  	   l_new = CONS(EFFECT, (*effect_dup_func)(eff), l_new),
	l_eff);
    return gen_nreverse(l_new);
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
	{
	  l_res = CONS(EFFECT, (*effect_dup_func)(eff), l_res);
	}
        else
	    pips_debug(7, "Effect on variable %s removed\n",
		       entity_local_name(effect_entity(eff)));
    }, l_eff);
    return gen_nreverse(l_res);
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
    list ec = list_undefined;

    ifdebug(8) {
      effects e = make_effects(l_eff);
      pips_assert("input effects are consistent", effects_consistent_p(e));
      effects_effects(e) = NIL;
      free_effects(e);
    }

    for(ec = l_eff; !ENDP(ec); POP(ec)) {
      effect eff = EFFECT(CAR(ec));

      /* build last to first */
      l_new = CONS(EFFECT, (*effect_dup_func)(eff), l_new);
    }

    /* and the order is reversed */
    l_new =  gen_nreverse(l_new);

    ifdebug(8) {
      effects e = make_effects(l_new);
      pips_assert("input effects are consistent", effects_consistent_p(e));
      effects_effects(e) = NIL;
      free_effects(e);
    }

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
list effect_to_nil_list(effect eff __attribute__((__unused__)))
{
    return(NIL);
}

/* list effects_to_nil_list(eff)
 * input    : an effect
 * output   : an empty list of effects
 * modifies : nothing
 * comment  : 	
 */
list effects_to_nil_list(effect eff1  __attribute__((__unused__)), effect eff2 __attribute__((__unused__))   )
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
effects_undefined_composition_with_transformer(list l_eff __attribute__((__unused__)), transformer trans __attribute__((__unused__)))
{
    return list_undefined;
}


list effects_composition_with_transformer_nop(list l_eff,
					      transformer trans __attribute__((__unused__)))
{
  return l_eff;
}




/* Composition with preconditions */

list
effects_undefined_composition_with_preconditions(list l_eff __attribute__((__unused__)), transformer trans __attribute__((__unused__)))
{
    return list_undefined;
}

list
effects_composition_with_preconditions_nop(list l_eff, transformer trans __attribute__((__unused__)))
{
  return l_eff;
}

/* Union over a range */

descriptor
loop_undefined_descriptor_make(loop l __attribute__((__unused__)))
{
    return descriptor_undefined;
}

list 
effects_undefined_union_over_range(
    list l_eff __attribute__((__unused__)), entity index __attribute__((__unused__)), range r __attribute__((__unused__)), descriptor d __attribute__((__unused__)))
{
    return list_undefined;
}

list effects_union_over_range_nop(list l_eff,
				  entity index __attribute__((__unused__)),
				  range r __attribute__((__unused__)),
				  descriptor d __attribute__((__unused__)))
{
  return l_eff;
}


list
effects_undefined_descriptors_variable_change(list l_eff __attribute__((__unused__)),
					      entity orig_ent __attribute__((__unused__)),
					      entity new_ent __attribute__((__unused__)))
{
    return list_undefined;
}

list
effects_descriptors_variable_change_nop(list l_eff, entity orig_ent __attribute__((__unused__)),
					      entity new_ent __attribute__((__unused__)))
{
    return l_eff;
}


descriptor
effects_undefined_vector_to_descriptor(Pvecteur v __attribute__((__unused__)))
{
    return descriptor_undefined;
}

list 
effects_undefined_loop_normalize(list l_eff __attribute__((__unused__)),
				 entity index __attribute__((__unused__)),
				 range r __attribute__((__unused__)),
				 entity *new_index, 
				 descriptor range_descriptor __attribute__((__unused__)),
				 bool descriptor_update_p __attribute__((__unused__)))
{
  *new_index = entity_undefined;
  return list_undefined; 
}

list 
effects_loop_normalize_nop(list l_eff, 
			   entity index __attribute__((__unused__)), 
			   range r __attribute__((__unused__)),
			   entity *new_index __attribute__((__unused__)), 
			   descriptor range_descriptor __attribute__((__unused__)),
			   bool descriptor_update_p __attribute__((__unused__)))
{
    return l_eff; 
}

list /* of nothing */
db_get_empty_list(string name)
{
    pips_debug(5, "getting nothing for %s\n", name);
    return NIL;
}


/* adding special dimensions */

/* void effect_add_dereferencing_dimension(effect eff)
 * input    : an effect
 * output   : nothing
 * modifies : the effect eff.
 * comment  : adds a last dimension  to represent an effect on the memory
 *            location pointed by the reference of the initial effect.
 *            also modifies the descriptor if there is one	
 */
void effect_add_dereferencing_dimension(effect eff)
{

  expression deref_exp = MakeIntegerConstantExpression("0");

  (*effect_add_expression_dimension_func)(eff, deref_exp);
  free_expression(deref_exp);

  return;
}

/* void effect_add_dereferencingfield_dimension(effect eff, int rank)
 * input    : an effect
 * output   : nothing
 * modifies : the effect eff.
 * comment  : adds a last dimension to represent an effect on the field
 *            of rank "rank" of the actual reference represented by the 
 *            effect reference.	The dimension is added at the end of the
 *            effect reference dimensions.Also modifies the descriptor
 *            if the representation includes one.
 */
void effect_add_field_dimension(effect eff, int rank)
{

  expression rank_exp = int_to_expression(rank);
  (*effect_add_expression_dimension_func)(eff, rank_exp);
  free_expression(rank_exp);
  return;
}

