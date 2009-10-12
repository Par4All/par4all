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
	       boolean (*union_combinable_p)(effect, effect) __attribute__ ((unused)))
{
    return gen_nconc(l1,l2);
}

list
ReferenceTestUnion(list l1, list l2,
		   boolean (*union_combinable_p)(effect, effect) __attribute__ ((unused)))
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

/* Preserve store independent information as long as you can. I should
   have some order on references to absorb, for instance, x[1] by
   x[*]. */
effect effect_may_union(effect eff1, effect eff2)
{
    effect eff;
    tag app1 = effect_approximation_tag(eff1);
    tag app2 = effect_approximation_tag(eff2);
    
    if (anywhere_effect_p(eff1) || anywhere_effect_p(eff2))
      eff = make_anywhere_effect(copy_action(effect_action(eff1)));
    
    else if (effect_scalar_p(eff1))
    {
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(approximation_and(app1,app2), UU));
    }
    else
    {
      /*
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(is_approximation_may, UU));
      */
      eff = copy_effect(eff1);
      approximation_tag(effect_approximation(eff)) = approximation_and(app1,app2);
    }
    return(eff);
}

/* FI: this very simple function assumed that the two effect are fully comparable */
/* FI: memoy management is unclear to me, but a new sharing free effect is produced */
effect effect_must_union(effect eff1, effect eff2)
{
    effect eff;
    tag app1 = effect_approximation_tag(eff1);
    tag app2 = effect_approximation_tag(eff2);
	
    if (anywhere_effect_p(eff1) || anywhere_effect_p(eff2))
      eff = make_anywhere_effect(copy_action(effect_action(eff1)));
    else if (effect_scalar_p(eff1))
    {
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(approximation_or(app1,app2), UU));
    }
    else
    {
      /* FI: I do change this a lot, because the keys used in
	 proper_effects_combine() are now much more extensive for
	 C. */
      /* we might have must effects for struct or array elements */
      /*pips_assert("The tags for non scalar effects are may",
	app1==app2 && app1 == is_approximation_may); */
	/*
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL), 
			  make_action(action_tag(effect_action(eff1)), UU), 
			  make_approximation(is_approximation_may, UU));
	*/
	eff = copy_effect(eff1);
	effect_approximation_tag(eff) = approximation_or(app1,app2);
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
effect_inf_difference(effect eff1 __attribute__ ((unused)),
		      effect eff2 __attribute__ ((unused)))
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

/* FI: the goal is to get rid of array subscripts to handle the arrays
 * atomically.
 *
 * This is not possible with pointer indexing. For instance, p[0] is
 * reduced the Fortran way into p, which is wrong. It could be reduced
 * to p[*] and lead to an anywhere effects. Or we must preserve
 * constant indices (store independent), which is the way to go with C
 * since we transform lots of scalar accesses into array accesses.
 *
 * FI: I do not understand the mix of side effects on eff, free and
 * alloc conditionnaly for some fields. To be checked.
 */
effect proper_to_summary_simple_effect(effect eff)
{
  if (!effect_scalar_p(eff)) {
    //cell c = effect_cell(eff);
    reference r = effect_any_reference(eff);
    //entity e = reference_variable(r);
    //type ut = ultimate_type(entity_type(e));
    list inds = reference_indices(r);
    list cind = list_undefined;
    bool may_p = FALSE;

    ifdebug(8) {
      pips_debug(8, "Proper effect %p with reference %p: %s\n", eff, r,
		 words_to_string(words_effect(eff)));
    }

    for(cind = inds; !ENDP(cind); POP(cind)) {
      expression se = EXPRESSION(CAR(cind));

      ifdebug(8) {
	pips_debug(8, "Subscript expression :\n");
	print_expression(se);
      }

      if(!extended_integer_constant_expression_p(se)) {
	if(!unbounded_expression_p(se)) {
	  may_p = TRUE;
	  EXPRESSION_(CAR(cind)) = make_unbounded_expression();
	}
      }
    }

    if(may_p)
      effect_approximation_tag(eff) = is_approximation_may;

    ifdebug(8) {
      pips_debug(8, "Summary simple effect %p with reference %p: %s\n", eff, r,
		 words_to_string(words_effect(eff)));
    }

    /*
    if(!reference_with_constant_indices_p(r)) {
      reference nr = pointer_type_p(ut)?
	make_reference(e, CONS(EXPRESSION, make_unbounded_expression(), NIL))
	: make_reference(e, NIL);
      free_reference(effect_any_reference(eff));
      if(cell_preference_p(c)) {
	preference p = cell_preference(c);
	preference_reference(p) = nr;
      }
      else {
	cell_reference(c) = nr;
      }
    }
    */
  }
  return(eff);
}
