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
  /* cell cell_ref = make_cell(is_cell_reference, copy_reference(ref)); */
  cell cell_ref = make_cell(is_cell_preference, make_preference(ref));
  approximation ap = make_approximation(is_approximation_must, UU);
  effect eff = make_effect(cell_ref, ac, ap, make_descriptor_none());  
  return eff;
}

/*********************************************************************************/
/* SIMPLE EFFECTS                                                                */
/*********************************************************************************/

effect
simple_effect_dup(effect eff)
{
    effect new_eff;
    new_eff = copy_effect(eff);
    effect_reference(new_eff) = reference_dup(effect_reference(new_eff));
    return(new_eff);
}

void
simple_effect_free(effect eff)
{
    free_reference(effect_reference(eff));
    effect_reference(eff) = reference_undefined;
    free_effect(eff);
}

effect
reference_to_reference_effect(reference ref, action ac)
{
    cell cell_ref = make_cell(is_cell_preference, make_preference(ref));
    approximation ap = make_approximation(is_approximation_must, UU);
    effect eff;
    
    eff = make_effect(cell_ref, ac, ap, make_descriptor(is_descriptor_none,UU));  
    return(eff);
}


list 
simple_effects_union_over_range(list l_eff, entity i, range r, descriptor d)
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
    if (!ENDP(reference_indices(effect_reference(eff))))
    {
	effect_reference(eff) = make_reference(effect_entity(eff), NIL);
    }
    effect_approximation_tag(eff) = is_approximation_may;
    return(CONS(EFFECT,eff,NIL));
}

list 
effect_to_sdfi_list(effect eff)
{
    if (!ENDP(reference_indices(effect_reference(eff))))
    {
	effect_reference(eff) = make_reference(effect_entity(eff), NIL);
	effect_approximation_tag(eff) = is_approximation_may;
    }
    return(CONS(EFFECT,eff,NIL));
}

void
simple_effects_descriptor_normalize(list l_eff)
{
    return;
}
