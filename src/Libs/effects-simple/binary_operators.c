/* package effect: new version by Beatrice Creusillet 
 *
 * This File contains several functions to combine effects
 *
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
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

#include "effects-generic.h"
#include "effects-simple.h"

#include "pipsdbm.h"
#include "resources.h"


list
ReferenceUnion(list l1, list l2,
	       boolean (*union_combinable_p)(effect, effect))
{
    return gen_nconc(l1,l2);
}

list
ReferenceTestUnion(list l1, list l2,
		   boolean (*union_combinable_p)(effect, effect))
{
    list l_res;

    /* somthing more clever should be done here - bc. */
    l_res = gen_nconc(l1,l2);
    effects_to_may_effects(l_res);
    return l_res;
}



/* list EffectsMayUnion(list l1, list l2, union_combinable_p)
 * input    : two lists of effects
 * output   : a list of effects, may union of the two initial lists
 * modifies : l1 and l2 and their effects. Effects that are not reused in
 *            the output list of effects are freed.nothing (no sharing introduced).
 */
list 
EffectsMayUnion(list l1, list l2,
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
list 
EffectsMustUnion(list l1, list l2,
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

list 
effects_may_union(effect eff1, effect eff2)
{
    list l_res = NIL;
    l_res = CONS(EFFECT, effect_may_union(eff1,eff2), NIL);
    return(l_res);
}

list 
effects_must_union(effect eff1, effect eff2)
{
    list l_res = NIL;
    l_res = CONS(EFFECT, effect_must_union(eff1,eff2), NIL);
    return(l_res);
}

effect 
effect_may_union(effect eff1, effect eff2)
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

effect 
effect_must_union(effect eff1, effect eff2)
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


static list
effect_sup_difference(/* const */ effect eff1, /* const */ effect eff2)
{
    list l_res = NIL;
    if (effect_must_p(eff2))
	l_res = NIL;
    else
	l_res = effect_to_may_effect_list(effect_dup(eff1));
    return(l_res);
}

static list
effect_inf_difference(effect eff1, effect eff2)
{
    list l_res = NIL;
    return(l_res);
}



/* list EffectsSupDifference(list l1, l2)
 * input    : two lists of effects
 * output   : a list of effect, representing the sup_difference of the 
 *            initial effects.
 * modifies : the effects of l2 may be freed.
 * comment  : we keep the effects of l1 that are not combinable with those
 *            of l2, but we don't keep the effects of l2 that are not 
 *            combinable with those of l_reg1.	
 */
list 
EffectsSupDifference(list l1, list l2,
			  boolean (*difference_combinable_p)(effect, effect))
{
    list l_res = NIL;

    debug(3, "EffectsSupDifference", "begin\n");
    l_res = list_of_effects_generic_binary_op(l1, l2,
					   difference_combinable_p,
					   effect_sup_difference,
					   effect_to_list,
					   effect_to_nil_list);
    debug(3, "EffectsSupDifference", "end\n");

    return l_res;
}

/* list EffectsInfDifference(list l1, l2)
 * input    : two lists of effects
 * output   : a list of effect, representing the inf_difference of the 
 *            initial effects.
 * modifies : the effects of l2 may be freed.
 * comment  : we keep the effects of l1 that are not combinable with those
 *            of l2, but we don't keep the effects of l2 that are not 
 *            combinable with those of l_reg1.	
 */
list 
EffectsInfDifference(list l1, list l2,
			  boolean (*difference_combinable_p)(effect, effect))
{
    list l_res = NIL;

    debug(3, "EffectsInfDifference", "begin\n");
    l_res = list_of_effects_generic_binary_op(l1, l2,
					   difference_combinable_p,
					   effect_inf_difference,
					   effect_to_list,
					   effect_to_nil_list);
    debug(3, "EffectsInfDifference", "end\n");

    return l_res;
}



effect 
proper_to_summary_simple_effect(effect eff)
{
    if (!effect_scalar_p(eff))
    {
	entity e = effect_entity(eff);
	free_reference(effect_reference(eff));
	effect_reference(eff) = make_reference(e, NIL);
	effect_approximation_tag(eff) = is_approximation_may;
    }
    return(eff);
}
