/* package effect: new version by Beatrice Creusillet 
 *
 * This File contains several functions to combine effects
 *
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"


#include "properties.h"

#include "transformer.h"
#include "semantics.h"

#include "effects.h"

#include "pipsdbm.h"
#include "resources.h"



/* list EffectsMayUnion(list l1, list l2, union_combinable_p)
 * input    : two lists of effects
 * output   : a list of effects, may union of the two initial lists
 * modifies : l1 and l2 and their effects. Effects that are not reused in
 *            the output list of effects are freed.nothing (no sharing introduced).
 */
list EffectsMayUnion(list l1, list l2,
		     boolean (*union_combinable_p)(effect, effect))
{
    list lr;

    lr = list_of_effects_generic_binary_op(l1, l2,
					   union_combinable_p,
					   effects_may_union,
					   effect_to_may_sdfi_list,
					   effect_to_may_sdfi_list);
    return(lr);
}


/* list EffectsMustUnion(list l1, list l2, union_combinable_p)
 * input    : two lists of effects
 * output   : a list of effects, must union of the two initial lists
 * modifies : l1 and l2 and their effects. Effects that are not reused in
 *            the output list of effects are freed.
 */
list EffectsMustUnion(list l1, list l2,
		      boolean (*union_combinable_p)(effect, effect)) 
{
    list lr;

    lr = list_of_effects_generic_binary_op(l1, l2,
					   union_combinable_p,
					   effects_must_union,
					   effect_to_sdfi_list,
					   effect_to_sdfi_list);
    return(lr);
}

list effects_may_union(effect eff1, effect eff2)
{
    list l_res = NIL;
    l_res = CONS(EFFECT, effect_may_union(eff1,eff2), NIL);
    return(l_res);
}

list effects_must_union(effect eff1, effect eff2)
{
    list l_res = NIL;
    l_res = CONS(EFFECT, effect_must_union(eff1,eff2), NIL);
    return(l_res);
}

effect effect_may_union(effect eff1, effect eff2)
{
    effect eff;
    
    if (effect_scalar_p(eff1))
    {
	tag app1 = effect_approximation_tag(eff1);
	tag app2 = effect_approximation_tag(eff2);
	
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(approximation_and(app1,app2), UU));
    }
    else
    {
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(is_approximation_may, UU));
    }
    return(eff);
}

effect effect_must_union(effect eff1, effect eff2)
{
    effect eff;
    if (effect_scalar_p(eff1))
    {
	tag app1 = effect_approximation_tag(eff1);
	tag app2 = effect_approximation_tag(eff2);
	
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(approximation_or(app1,app2), UU));
    }
    else
    {
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(is_approximation_may, UU));
    }
    return(eff);
}



/*********************************************************************************/
/* PROPER EFFECTS TO SUMMARY EFFECTS                                             */
/*********************************************************************************/

/* list proper_to_summary_effects(list l_effects)
 * input    : a list of proper effects: there may be several effects for 
 *            a given array with a given action.
 * output   : a list of summary effects, with the property that is only
 *            one effect for each array per action.
 * modifies : the input list. Some effects are freed. 
 * comment  : The computation scans the first list (the "base" list);
 *            From this element, the next elements are scanned (the "current"
 *            list). If the current effect is combinable with the base effect,
 *            they are combined. Both original effects are freed, and the base
 *            is replaced by the union. 
 */
list proper_to_summary_effects(list l_effects)
{
        return(proper_effects_combine(l_effects, FALSE));
}

effect proper_to_summary_effect(effect eff)
{
    if (!effect_scalar_p(eff))
    {
	effect_reference(eff) = make_reference(effect_entity(eff), NIL);
	effect_approximation_tag(eff) = is_approximation_may;
    }
    return(eff);
}

/* list proper_effects_contract(list l_effects)
 * input    : a list of proper effects
 * output   : a list of proper effects in which there is no two identical 
 *            scalar effects. 
 * modifies : the input list. 
 * comment  : This is used to reduce the number of dependence tests.
 */

list proper_effects_contract(list l_effects)
{
    return(proper_effects_combine(l_effects, TRUE));
}


/* list proper_effects_combine(list l_effects, bool scalars_only_p)
 * input    : a list of proper effects, and a boolean to know on which
 *            elements to perform the union.
 * output   : a list of effects, in which the selected elements have been merged.
 * modifies : the input list.
 * comment  :
 */
list proper_effects_combine(list l_effects, bool scalars_only_p)
{
    list base, current, pred;

    ifdebug(6)
    {
	pips_debug(6, "proper effects: \n");
	print_effects(l_effects);	
    }

    base = l_effects;
    /* scan the list of effects */
    while(!ENDP(base) )
    {	
	/* scan the next elements to find effects combinable
	 * with the effects of the base element.
	 */
	current = CDR(base);
	pred = base;
	while (!ENDP(current))
	{
	    effect eff_base = EFFECT(CAR(base));
	    effect eff_current = EFFECT(CAR(current));
	    

	    /* Both effects are about the same scalar variable, 
	       with the same action 
	       */ 
	    if ((!scalars_only_p || effect_scalar_p(eff_base)) &&
		effects_same_action_p(eff_base, eff_current) )  
	    {
		list tmp;
		effect new_eff_base;
				
		/* compute their union */
		new_eff_base = effect_must_union
		    (proper_to_summary_effect(eff_base),
		     proper_to_summary_effect(eff_current)); 
		/* free the original effects: no memory leak */
		free_effect(eff_base);
		free_effect(eff_current);
		
		/* replace the base effect by the new effect */
		EFFECT(CAR(base)) = new_eff_base;
		
		/* remove the current list element from the global list */
		tmp = current;	    
		current = CDR(current);
		CDR(pred) = current;
		free(tmp);	    
	    }
	    else
	    {
		pred = current;
		current = CDR(current);
	    }
	}
	base = CDR(base);
    }

    ifdebug(6)
    {
	pips_debug(6, "summary effects: \n");
	print_effects(l_effects);	
    }
    return(l_effects);    
}
