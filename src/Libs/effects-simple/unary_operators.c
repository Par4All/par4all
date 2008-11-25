/* package simple effects :  Be'atrice Creusillet 6/97
 *
 * File: unary_operators.c
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the intanciation of the generic functions necessary 
 * for the computation of all types of simple effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "properties.h"

#include "effects-generic.h"
#include "effects-simple.h"


/*********************************************************************************/
/* REFERENCE EFFECTS                                                             */
/*********************************************************************************/

effect
reference_effect_dup(effect eff)
{
    return (copy_effect(eff));
}

void
reference_effect_free(effect eff)
{
    free_effect(eff);
}

effect
reference_to_simple_effect(reference ref, action ac)
{
/* It shoulb be this one, but it does not work. Maybe there is a clash with
   old uses of make_simple_effects. Or persistancy is not properly handled? 
   - bc.
*/
  /* FI: this should be revisited now that I have cleaned up a lot of
     the effects library to handled the two kinds of cells. */
  /* cell cell_ref = make_cell(is_cell_reference, copy_reference(ref)); */
  cell cell_ref = make_cell_preference(make_preference(ref));
  addressing ad = make_addressing_index();
  approximation ap = make_approximation_must();
  effect eff = make_effect(cell_ref, ac, ad, ap, make_descriptor_none());  
  return eff;
}

/*********************************************************************************/
/* SIMPLE EFFECTS                                                                */
/*********************************************************************************/

/* A persistant reference wss forced. */
effect
simple_effect_dup(effect eff)
{
  effect new_eff = effect_undefined;

  new_eff = copy_effect(eff);

  if(cell_preference_p(effect_cell(new_eff))) {
    /* FI: memory leak? we allocate something and put it behind a persistent pointer */
    effect_reference(new_eff) = reference_dup(effect_reference(new_eff));
  }

  ifdebug(8) pips_assert("the new effect is consistent", effect_consistent_p(new_eff));

  return(new_eff);
}

/* use free_effect() instead? persistent reference assumed */
void
simple_effect_free(effect eff)
{
  if(cell_preference_p(effect_cell(eff))) {
    free_reference(effect_reference(eff));
    effect_reference(eff) = reference_undefined;
  }
  else {
    free_reference(effect_any_reference(eff));
    cell_reference(effect_cell(eff)) = reference_undefined;
  }
  free_effect(eff);
}

/* In fact, reference to persistant reference, preference */
 effect
 reference_to_reference_effect(reference ref, action ac)
 {
   cell cell_ref = make_cell(is_cell_preference, make_preference(ref));
   addressing ad = make_addressing_index();
   approximation ap = make_approximation(is_approximation_must, UU);
   effect eff;
    
   eff = make_effect(cell_ref, ac, ad, ap, make_descriptor(is_descriptor_none,UU));  
   return(eff);
 }


list simple_effects_union_over_range(list l_eff,
				     entity i,
				     range r,
				     descriptor d __attribute__ ((unused)))
{
  /* FI: effects in index and in in range must be taken into account. it
     would be easier to have the loop proper effects as argument instead
     of recomputing it. */
  if(FALSE) {
    list c_eff = list_undefined;
    reference ref = make_reference(i, NIL);
    cell c = make_cell_reference(ref);
    effect i_eff = make_effect(c, make_action_write(), make_addressing_index(),
			       make_approximation_must(), make_descriptor_none());

    list r_eff_l = proper_effects_of_range(r);
    list h_eff_l = CONS(EFFECT, i_eff, r_eff_l);
    list ch_eff = list_undefined;

    for(ch_eff=h_eff_l; !ENDP(ch_eff); POP(ch_eff)) {
      effect h_eff = EFFECT(CAR(ch_eff));

      for(c_eff = l_eff; !ENDP(c_eff); POP(c_eff)) {
	effect eff = EFFECT(CAR(c_eff));

	eff = effect_interference(eff, h_eff);

	EFFECT_(CAR(c_eff)) = eff;
      }
    }

    //gen_full_free_list(h_eff_l);
  }
 if (!get_bool_property("ONE_TRIP_DO"))
    {
      effects_to_may_effects(l_eff);
    }
  return l_eff;
}


/* FI: instead of simply getting rid of indices, I preserve constant
   indices for the semantics analysis. Instead of stripping the
   indices, they are replaced by unbounded expressions to keep the
   difference between p and p[*] when p is a pointer. 

   No memory allocation, side effects on eff? Side effects on the
   reference too in spite of the persistant reference?
*/
list 
effect_to_store_independent_sdfi_list(effect eff, bool force_may_p)
{
  reference r = effect_any_reference(eff);
  list ind = reference_indices(r);
  list cind = list_undefined;
  bool may_p = FALSE;

  for(cind = ind; !ENDP(cind); POP(cind)) {
    expression se = EXPRESSION(CAR(cind));

    if(!extended_integer_constant_expression_p(se)) {
      if(!unbounded_expression_p(se)) {
	expression nse = make_unbounded_expression();
	may_p = TRUE;
	free_expression(se);
	CAR(cind).p = (void *) nse;
      }
    }
  }

  /* FI: Why is MAY always forced? Because of the semantics of the function! */
  if(may_p || force_may_p)
    effect_approximation_tag(eff) = is_approximation_may;

  return(CONS(EFFECT,eff,NIL));
}

list 
effect_to_may_sdfi_list(effect eff)
{
  return effect_to_store_independent_sdfi_list(eff, TRUE);
}

/* FI: instead of simpy getting rid of indices, I preserve cosntant
   indices for the semantics analysis. */
list 
effect_to_sdfi_list(effect eff)
{
  return effect_to_store_independent_sdfi_list(eff, FALSE);
}

void
simple_effects_descriptor_normalize(list l_eff __attribute__ ((unused)))
{
  return;
}
