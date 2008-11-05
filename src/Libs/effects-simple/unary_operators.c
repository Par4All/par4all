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


list 
simple_effects_union_over_range(list l_eff,
				entity i __attribute__ ((unused)),
				range r __attribute__ ((unused)),
				descriptor d __attribute__ ((unused)))
{
  if (!get_bool_property("ONE_TRIP_DO"))
    {
      effects_to_may_effects(l_eff);
    }
  return l_eff;
}


list 
effect_to_may_sdfi_list(effect eff)
{
  if (!ENDP(reference_indices(effect_any_reference(eff)))) {
    /* FI: a persistant reference is forced here */
    pips_assert("reference is a persistant reference", cell_preference_p(effect_cell(eff)));
    effect_reference(eff) = make_reference(effect_entity(eff), NIL);
  }
  effect_approximation_tag(eff) = is_approximation_may;
  return(CONS(EFFECT,eff,NIL));
}

list 
effect_to_sdfi_list(effect eff)
{
  if (!ENDP(reference_indices(effect_any_reference(eff))))
    {
      /* FI: persistant reference assumed */
      pips_assert("reference is a persistant reference", cell_preference_p(effect_cell(eff)));
      effect_reference(eff) = make_reference(effect_entity(eff), NIL);
      effect_approximation_tag(eff) = is_approximation_may;
    }
  return(CONS(EFFECT,eff,NIL));
}

void
simple_effects_descriptor_normalize(list l_eff __attribute__ ((unused)))
{
  return;
}
