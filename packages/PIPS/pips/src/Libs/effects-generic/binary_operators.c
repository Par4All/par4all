/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: binary_operators.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains generic binary operators for effects and lists of them.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "properties.h"
#include "text.h"
#include "text-util.h"

#include "effects-generic.h"


/******************************************** GENERIC BINARY OPERATORS ON EFFECTS */


/**
   @brief returns the contribution of the union of eff1 and eff2 to
   the result list of the generic binary operator.
   beware: may modify eff1 and eff2

   @param[in] eff1_abstract_location_p is true if eff1 is an abstract location
          ( put as a parameter because testing abstract locations is costly and is
	  better done outside loops when possible)
   @param[in] eff2_abstract_location_p is true if eff2 is an abstract location
   @param[out] eff1_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff1 may still be
	  combinable with other effects in the calling function
   @param[out] eff2_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff2 may still be
	  combinable with other effects in the calling function
   @param[in] concrete_effects_union_op computes the union of two effects on
          concrete locations.
 */
static list effects_generic_union_op(effect eff1, effect eff2,
				     bool eff1_abstract_location_p,
				     bool eff2_abstract_location_p,
				     bool *eff1_still_combinable_p,
				     bool *eff2_still_combinable_p,
				     list (*concrete_effects_union_op)(effect,effect))
{
  list l_res = NIL;

  ifdebug(1)
    pips_assert("eff1 and eff2 must be still combinable",
		*eff1_still_combinable_p && *eff2_still_combinable_p);

  pips_debug_effect(8, "eff1: \n", eff1);
  pips_debug_effect(8, "eff2: \n", eff2);

  if (eff1_abstract_location_p && eff2_abstract_location_p)
    {
      entity al1 = effect_entity(eff1);
      entity al2 = effect_entity(eff2);

      if (same_entity_p(al1, al2)) /* currently most common case */
	{
	  *eff1_still_combinable_p = false;
	  *eff2_still_combinable_p = false;
	  l_res = CONS(EFFECT, (*effect_dup_func)(eff1), l_res);
	}
      else
	{
	  // the result is the max of the two abstract locations, which is one
	  // of them; we remove the lowest from it's list, and keep the
	  // highest for potential combination with another effect.
	  // There is no need to add it in the result list, since
	  // it will belong to the remnants of one of the initial list
	  // and as such be appended to the final result list.

	  entity al_max = abstract_locations_max(al1, al2);
	  if (same_entity_p(al1, al_max))
	    {
	      /* *eff1_still_combinable_p = true; (redundant) */
	      *eff2_still_combinable_p = false;
	    }
	  else
	    {
	      ifdebug(1) pips_assert("combinable abstract locations: one of them is the max of both",
				     same_entity_p(al2, al_max));
	      *eff1_still_combinable_p = false;
	      /* *eff2_still_combinable_p = true; (redundant) */
	    }
	}
    }
  else if (eff1_abstract_location_p) /* and eff2 concrete location */
    {
      /* *eff1_still_combinable_p = true;  (redundant) */
      *eff2_still_combinable_p = false;
    }
  else if (eff2_abstract_location_p) /* and eff1 concrete location */
    {
      *eff1_still_combinable_p = false;
      /* *eff2_still_combinable_p = true;  (redundant) */
    }
  else /* two concrete locations */
    {
      l_res = gen_nconc((*concrete_effects_union_op)(eff1,eff2), l_res);
      *eff1_still_combinable_p = false;
      *eff2_still_combinable_p = false;
    }

  pips_debug_effects(8, "returning:\n", l_res);
  return l_res;
}

/**
   @brief returns the contribution of the intersection of eff1 and eff2 to
   the result list of the generic binary operator.
   beware: may modify eff1 and eff2

   @param[in] eff1_abstract_location_p is true if eff1 is an abstract location
          ( put as a parameter because testing abstract locations is costly and is
	  better done outside loops when possible)
   @param[in] eff2_abstract_location_p is true if eff2 is an abstract location
   @param[out] eff1_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff1 may still be
	  combinable with other effects in the calling function
   @param[out] eff2_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff2 may still be
	  combinable with other effects in the calling function
   @param[in] concrete_effects_intersection_op computes the union of two effects on
          concrete locations.
 */
static list effects_generic_intersection_op(effect eff1, effect eff2,
				     bool eff1_abstract_location_p,
				     bool eff2_abstract_location_p,
				     bool *eff1_still_combinable_p,
				     bool *eff2_still_combinable_p,
				     list (*concrete_effects_intersection_op)(effect,effect))
{
  list l_res = NIL;

  ifdebug(1)
    pips_assert("eff1 and eff2 must be still combinable",
		*eff1_still_combinable_p && *eff2_still_combinable_p);

  pips_debug_effect(8, "eff1: \n", eff1);
  pips_debug_effect(8, "eff2: \n", eff2);


  if (eff1_abstract_location_p && eff2_abstract_location_p)
    {
      entity al1 = effect_entity(eff1);
      entity al2 = effect_entity(eff2);

      if (same_entity_p(al1, al2)) /* currently most common case */
	{
	  effect r = (*effect_dup_func)(eff1);
	  *eff1_still_combinable_p = false;
	  *eff2_still_combinable_p = false;
	  l_res = CONS(EFFECT, r, l_res);
	}
      else
	{
	  entity al_max = abstract_locations_max(al1, al2);

	  if (same_entity_p(al1, al_max))
	    {
	      effect r = (*effect_dup_func)(eff2);
	      /* *eff1_still_combinable_p = true; (redundant) */
	      *eff2_still_combinable_p = false;
	      l_res = CONS(EFFECT, r, l_res);
	    }
	  else
	    {
	      ifdebug(1) pips_assert("combinable abstract locations: one of them is the max of both",
				     same_entity_p(al2, al_max));
	      effect r = (*effect_dup_func)(eff1);
	      *eff1_still_combinable_p = false;
	      /* *eff2_still_combinable_p = true; (redundant) */
	      l_res = CONS(EFFECT, r, l_res);
	    }
	}
    }
  else if (eff1_abstract_location_p) /* and r2 concrete location */
    {
      effect r = (*effect_dup_func)(eff2);
      effect_to_may_effect(r);
      /* *eff1_still_combinable_p = true;  (redundant) */
      *eff2_still_combinable_p = false;
      l_res = CONS(EFFECT, r, l_res);
    }
  else if (eff2_abstract_location_p) /* and r1 concrete location */
    {
      effect r = (*effect_dup_func)(eff1);
      effect_to_may_effect(r);
      *eff1_still_combinable_p = false;
      /* *eff2_still_combinable_p = true;  (redundant) */
      l_res = CONS(EFFECT, r, l_res);
    }
  else /* two concrete locations */
    {
      l_res = gen_nconc((*concrete_effects_intersection_op)(eff1,eff2), l_res);
      *eff1_still_combinable_p = false;
      *eff2_still_combinable_p = false;
    }

  pips_debug_effects(8, "returning:\n", l_res);
  return l_res;
}


/**
   @brief returns the contribution of the sup difference of eff1 and eff2 to
   the result list of the generic binary operator.
   beware: may modify eff1 and eff2

   @param[in] eff1_abstract_location_p is true if eff1 is an abstract location
          ( put as a parameter because testing abstract locations is costly and is
	  better done outside loops when possible)
   @param[in] eff2_abstract_location_p is true if eff2 is an abstract location
   @param[out] eff1_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff1 may still be
	  combinable with other effects in the calling function
   @param[out] eff2_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff2 may still be
	  combinable with other effects in the calling function
   @param[in] concrete_effects_sup_difference_op computes the union of two effects on
          concrete locations.
 */
static list effects_generic_sup_difference_op(effect eff1, effect eff2,
				     bool eff1_abstract_location_p,
				     bool eff2_abstract_location_p,
				     bool *eff1_still_combinable_p,
				     bool *eff2_still_combinable_p,
				     list (*concrete_effects_sup_difference_op)(effect,effect))
{
  list l_res = NIL;

  ifdebug(1)
    pips_assert("eff1 and eff2 must be still combinable",
		*eff1_still_combinable_p && *eff2_still_combinable_p);

  pips_debug_effect(8, "eff1: \n", eff1);
  pips_debug_effect(8, "eff2: \n", eff2);

  if (eff1_abstract_location_p || eff2_abstract_location_p)
    {
      effect_to_may_effect(eff1);

      if (eff1_abstract_location_p && eff2_abstract_location_p)
	{
	  entity al1 = effect_entity(eff1);
	  entity al2 = effect_entity(eff2);
	  if (same_entity_p(al1, al2))
	    *eff1_still_combinable_p = *eff2_still_combinable_p = false;
	  else
	    {
	      entity al_max = abstract_locations_max(al1, al2);
	      if (same_entity_p(al1, al_max)) /* r2 is strictly less than r1 */
		{
		  /* *eff1_still_combinable_p = true; redundant */
		  *eff2_still_combinable_p = false;
		}
	      else /* r1 is strictly less than r2 */
		{
		  ifdebug(1) pips_assert("combinable abstract locations: one of them is the max of both",
					 same_entity_p(al2, al_max));
		  *eff1_still_combinable_p = false;
		  /* *eff2_still_combinable_p = true; redundant */
		}
	    }
	}
      else
	{
	  *eff1_still_combinable_p = eff1_abstract_location_p;
	  *eff2_still_combinable_p = eff2_abstract_location_p;
	}
      if (!*eff1_still_combinable_p)
	{
	  l_res = CONS(EFFECT, (*effect_dup_func)(eff1), l_res); /* add it now to the result */
	}
      /* else it will be added later, since it is still combinable */
    }
  else /* two concrete locations */
    {
      l_res = gen_nconc((*concrete_effects_sup_difference_op)(eff1,eff2), l_res);
      *eff1_still_combinable_p = false;
      *eff2_still_combinable_p = false;
    }

  pips_debug_effects(8, "returning:\n", l_res);
  return l_res;
}

/**
   @brief returns the contribution of the inf difference of eff1 and eff2 to
   the result list of the generic binary operator.
   beware: may modify eff1 and eff2

   @param[in] eff1_abstract_location_p is true if eff1 is an abstract location
          ( put as a parameter because testing abstract locations is costly and is
	  better done outside loops when possible)
   @param[in] eff2_abstract_location_p is true if eff2 is an abstract location
   @param[out] eff1_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff1 may still be
	  combinable with other effects in the calling function
   @param[out] eff2_still_combinable_p is a pointer to a bool which must be true
          when entering the function, and stays true if eff2 may still be
	  combinable with other effects in the calling function
   @param[in] concrete_effects_sup_difference_op computes the union of two effects on
          concrete locations.
 */
static list effects_generic_inf_difference_op(effect eff1, effect eff2,
				     bool eff1_abstract_location_p,
				     bool eff2_abstract_location_p,
				     bool *eff1_still_combinable_p,
				     bool *eff2_still_combinable_p,
				     list (*concrete_effects_inf_difference_op)(effect,effect))
{
  list l_res = NIL;

  ifdebug(1)
    pips_assert("eff1 and eff2 must be still combinable",
		*eff1_still_combinable_p && *eff2_still_combinable_p);

  pips_debug_effect(8, "eff1: \n", eff1);
  pips_debug_effect(8, "eff2: \n", eff2);

  if (eff1_abstract_location_p || eff2_abstract_location_p)
    {
      /* return a NIL list */
      /* *eff1 is not combinable anymore, because we have already found out that something
	 must be removed from it
      */
      *eff1_still_combinable_p = false;

      if (eff1_abstract_location_p && eff2_abstract_location_p)
	{
	  entity al1 = effect_entity(eff1);
	  entity al2 = effect_entity(eff2);
	  if (same_entity_p(al1, al2))
	    *eff2_still_combinable_p = false;
	  else
	    {
	      entity al_max = abstract_locations_max(al1, al2);
	      if (same_entity_p(al1, al_max)) /* r2 is strictly less than r1 */
		{
		  /* *eff1_still_combinable_p = true; redundant */
		  *eff2_still_combinable_p = false;
		}
	      else /* r1 is strictly less than r2 */
		{
		  ifdebug(1) pips_assert("combinable abstract locations: one of them is the max of both",
					 same_entity_p(al2, al_max));
		  /* *eff1_still_combinable_p = false; redundant */
		  /* *eff2_still_combinable_p = true; redundant */
		}
	    }
	}
      else
	{
	  /* *eff1_still_combinable_p = eff1_abstract_location_p; redundant */
	  *eff2_still_combinable_p = eff2_abstract_location_p;
	}
    }
  else /* two concrete locations */
    {
      l_res = gen_nconc((*concrete_effects_inf_difference_op)(eff1,eff2), l_res);
      *eff1_still_combinable_p = false;
      *eff2_still_combinable_p = false;
    }

  pips_debug_effects(8, "returning:\n", l_res);
  return l_res;
}

/*********************************** GENERIC BINARY OPERATORS ON LISTS OF EFFECTS */

/**
   beware : modifies l1, l2 and their effects

  @param l1 and l2 are two lists of effects.
  @param  r1_r2_combinable_p is a bool function that takes two
          individual effects as arguments and renders true when they are
          considered as combinable ;
  @param  r1_r2_generic_binary_op is a binary operator that combines two
          individual effects whatever their path may be;
  @param  r1_r2_binary_op is a binary operator that combines two
          individual effects on concrete paths; it is called by the previous
	  parameter;
  @param  r1_unary_op is a unary operators that deal with the remnants of l1,
          that is those effects that are not combinable with any effect of l2;
  @param  r2_unary_op is a unary operators that deal with the remnants of l2,
          that is those effects that are not combinable with any effect of l1;

  @return a list of effects, combination of l1 and l2.

  There is a strong assumption on l1 and l2 effects: two effects of l1
  (resp. l2) are not combinable wrt r1_r2_combinable_p.  The algorithm
  relies on this assumption to avoid unecessary comparisons of effects
  when possible, for performance reasons ((the asymptotic complexity is
  o(n1*n2), but taking into account this assumption reduces the average
  complexity).

  The algorithm takes into account the fact that if an effect of l1 (resp. l2)
  is an abstract location, it may be combinable with several effects
  of l2 (resp. l1).
  We strongly rely on the abstract locations lattice properties and on the
  list properties to avoid to recursively compute the union of two
  individual effects with the current resulting list.

*/
list
list_of_effects_generic_binary_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_generic_binary_op)(effect,effect, bool, bool, bool*, bool*, list (*)(effect, effect) ),
    list (*r1_r2_concrete_binary_op)(effect,effect),
    list (*r1_unary_op)(effect),
    list (*r2_unary_op)(effect))
{
    list l_res = NIL;
    bool fortran_p = fortran_module_p(get_current_module_entity()); // no need to test for abstract locations

    debug_on("EFFECTS_OPERATORS_DEBUG_LEVEL");

    pips_debug_effects(1, "Initial effects : \n\t l1 :\n", l1);
    pips_debug_effects(1, "\t l2 :\n", l2);


    /* we first deal with the effects of l1 : those that are combinable with
     * the effects of l2, and the others, which we call the remnants of l1 */
    FOREACH(EFFECT, r1, l1)
      {
	list lr2 = l2;
	list prec_lr2 = NIL;
	bool r1_still_combinable_p = true;
	bool r1_abstract_location_p = fortran_p? false : effect_abstract_location_p(r1);

	pips_debug_effect(8, "r1: %s\n", r1);

	while(r1_still_combinable_p && !ENDP(lr2))
	  {
	    effect r2 = EFFECT(CAR(lr2));
	    bool r2_still_combinable_p = true;
	    bool r2_abstract_location_p = fortran_p? false : effect_abstract_location_p(r2);

	    pips_debug_effect(8, "r2: \n", r2);

	    if ( (*r1_r2_combinable_p)(r1,r2) )
	      {
		pips_debug(8, "combinable\n");

		list l_tmp = (*r1_r2_generic_binary_op)(r1, r2,
							r1_abstract_location_p,
							r2_abstract_location_p,
							&r1_still_combinable_p,
							&r2_still_combinable_p,
							r1_r2_concrete_binary_op);
		l_res = gen_nconc(l_tmp, l_res);
		if (!r2_still_combinable_p) /* remove it from l2 */
		  {
		    list new_lr2 = CDR(lr2);
		    /* gen_remove(&l2, EFFECT(CAR(lr2))); */
		    if (prec_lr2 != NIL)
		      CDR(prec_lr2) = CDR(lr2);
		    else
		      l2 = CDR(lr2);
		    free_effect(r2); r2=effect_undefined;
		    CDR(lr2) = NIL;
		    free(lr2);
		    lr2 = new_lr2;
		  }
		else
		  {
		    prec_lr2 = lr2;
		    lr2 = CDR(lr2);
		  }
	      } /* if (*r1_r2_combinable_p)() */
	    else
	      {
		pips_debug(8,"not combinable\n");
		prec_lr2 = lr2;
		lr2 = CDR(lr2);
	      }
	  } /* while (r1_still_combinable_p && !ENDP(lr2))*/

	if (r1_still_combinable_p)
	  l_res = gen_nconc((*r1_unary_op)(r1),l_res);
	else
	  free_effect(r1); /* r1 won't be used anymore */
      }

    /* we must finally add the remaining effects of l2 */
    FOREACH(EFFECT, r2, l2) {
      l_res = gen_nconc((*r2_unary_op)(r2),l_res);
    }

    // necessary to avoid flip-flops in list order
    // when there are successive invocations, in particular
    // with proper_effects_combine
    l_res = gen_nreverse(l_res);

    pips_debug_effects(1, "final effects:\n", l_res);

    /* no memory leaks: l1 and l2 won't be used anymore */
    gen_free_list(l1);
    gen_free_list(l2);

    debug_off();

    return l_res;
}

/**
   @brief computes the union of the two input lists of effects.
   beware : modifies/frees l1, l2 and their effects

  @param l1 and l2 are two lists of effects.
  @param  r1_r2_combinable_p is a bool function that takes two
          individual effects as arguments and renders true when they are
          considered as combinable ;
  @param  r1_r2_union_op is a union operator that combines two
          individual effects;
  @param  r_unary_op is an operator applied to the effects from one
          orignal list that are not combinable with the effects of the
	  other list;

  @return a list of effects, combination of l1 and l2.

  There is a strong assumption on l1 and l2 effects: two effects of l1
  (resp. l2) are not combinable wrt r1_r2_combinable_p.  The algorithm
  relies on this assumption to avoid unecessary comparisons of effects
  when possible, for performance reasons ((the asymptotic complexity is
  o(n1*n2), but taking into account this assumption reduces the average
  complexity).

  The algorithm takes into account the fact that if an effect of l1 (resp. l2)
  is an abstract location, it may be combinable with several effects
  of l2 (resp. l1).
  We strongly rely on the abstract locations lattice properties and on the
  list properties to avoid to recursively compute the union of two
  individual effects with the current resulting list.

*/
list
list_of_effects_generic_union_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_concrete_union_op)(effect,effect),
    list (*r_unary_op)(effect))
{
  list l_res = list_of_effects_generic_binary_op(l1, l2, r1_r2_combinable_p,
						 effects_generic_union_op,
						 r1_r2_concrete_union_op,
						 r_unary_op, r_unary_op);
  return l_res;
}



/**
   @brief computes the intersection of two lists of effects/regions
   *beware*: modifies l1, l2 and their effects.

  @param  l1 and l2 are two lists of effects.
  @param  r1_r2_combinable_p is a bool function that takes two
          individual effects as arguments and renders true when they are
          considered as combinable;
  @param  r1_r2_intersection_op is a binary operator that combines two
          individual effects on concrete locations;

  @return a list of effects, intersection of l1 and l2.


  There is a strong assumption on l1 and l2 effects: two effects of l1
  (resp. l2) are not combinable wrt r1_r2_combinable_p.  The algorithm
  relies on this assumption to avoid unecessary comparisons of effects
  when possible, for performance reasons (the asymptotic complexity is
  o(n1*n2), but taking into account this assumption reduces the average
  complexity).

*/
list list_of_effects_generic_intersection_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_concrete_intersection_op)(effect,effect))
{
  list l_res = list_of_effects_generic_binary_op(l1, l2, r1_r2_combinable_p,
						 effects_generic_intersection_op,
						 r1_r2_concrete_intersection_op,
						 effect_to_nil_list_and_free,
						 effect_to_nil_list_and_free);

  return l_res;
}

/**
   @brief computes the intersection of two lists of effects/regions
   *beware*: modifies l1, l2 and their effects.

  @param  l1 and l2 are two lists of effects.
  @param  r1_r2_combinable_p is a bool function that takes two
          individual effects as arguments and renders true when they are
          considered as combinable;
  @param  r1_r2_intersection_op is a binary operator that combines two
          individual effects on concrete locations;

  @return a list of effects, intersection of l1 and l2.


  There is a strong assumption on l1 and l2 effects: two effects of l1
  (resp. l2) are not combinable wrt r1_r2_combinable_p.  The algorithm
  relies on this assumption to avoid unecessary comparisons of effects
  when possible, for performance reasons (the asymptotic complexity is
  o(n1*n2), but taking into account this assumption reduces the average
  complexity).

*/
list list_of_effects_generic_cells_intersection_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_concrete_cells_intersection_op)(effect,effect))
{
  list l_res = list_of_effects_generic_binary_op(l1, l2, r1_r2_combinable_p,
						 effects_generic_intersection_op,
						 r1_r2_concrete_cells_intersection_op,
						 effect_to_nil_list_and_free,
						 effect_to_nil_list_and_free);

  return l_res;
}


/**
   @brief computes an over-approximate of the difference of two lists
   of effects/regions. *beware*: modifies l1, l2 and their effects.

  @param  l1 and l2 are two lists of effects.
  @param  r1_r2_combinable_p is a bool function that takes two
          individual effects as arguments and renders true when they are
          considered as combinable;
  @param  r1_r2_difference_op is a binary operator that combines two
          individual effects on concrete locations;

  @return a list of effects, over-approximating the difference of l1 and l2.


  There is a strong assumption on l1 and l2 effects: two effects of l1
  (resp. l2) are not combinable wrt r1_r2_combinable_p.  The algorithm
  relies on this assumption to avoid unecessary comparisons of effects
  when possible, for performance reasons (the asymptotic complexity is
  o(n^2), but taking into account this assumption reduces the average
  complexity).

*/
list list_of_effects_generic_sup_difference_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_concrete_sup_difference_op)(effect,effect))
{

  list l_res = list_of_effects_generic_binary_op(l1, l2, r1_r2_combinable_p,
						 effects_generic_sup_difference_op,
						 r1_r2_concrete_sup_difference_op,
						 effect_to_list,
						 effect_to_nil_list_and_free);

  return l_res;
}

/**
   @brief computes an undef-approximate of the difference of two lists
   of effects/regions. *beware*: modifies l1, l2 and their effects.

  @param  l1 and l2 are two lists of effects.
  @param  r1_r2_combinable_p is a bool function that takes two
          individual effects as arguments and renders true when they are
          considered as combinable;
  @param  r1_r2_concrete_inf_difference_op is a binary operator that combines two
          individual effects on concrete locations;

  @return a list of effects, over-approximating the difference of l1 and l2.


  There is a strong assumption on l1 and l2 effects: two effects of l1
  (resp. l2) are not combinable wrt r1_r2_combinable_p.  The algorithm
  relies on this assumption to avoid unecessary comparisons of effects
  when possible, for performance reasons (the asymptotic complexity is
  o(n^2), but taking into account this assumption reduces the average
  complexity).

*/
list list_of_effects_generic_inf_difference_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_concrete_inf_difference_op)(effect,effect))
{

  list l_res = list_of_effects_generic_binary_op(l1, l2, r1_r2_combinable_p,
						 effects_generic_inf_difference_op,
						 r1_r2_concrete_inf_difference_op,
						 effect_to_list,
						 effect_to_nil_list_and_free);

  return l_res;
}

/**
   @brief computes an undef-approximate of the difference of two lists
   of effects/regions. *beware*: modifies l1, l2 and their effects.

  @param  l1 and l2 are two lists of effects.
  @param  r1_r2_combinable_p is a bool function that takes two
          individual effects as arguments and renders true when they are
          considered as combinable;
  @param  r1_r2_concrete_inf_difference_op is a binary operator that combines two
          individual effects on concrete locations;

  @return a list of effects, over-approximating the difference of l1 and l2.


  There is a strong assumption on l1 and l2 effects: two effects of l1
  (resp. l2) are not combinable wrt r1_r2_combinable_p.  The algorithm
  relies on this assumption to avoid unecessary comparisons of effects
  when possible, for performance reasons (the asymptotic complexity is
  o(n^2), but taking into account this assumption reduces the average
  complexity).

*/
list list_of_effects_generic_cells_inf_difference_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_concrete_cells_inf_difference_op)(effect,effect))
{

  list l_res = list_of_effects_generic_binary_op(l1, l2, r1_r2_combinable_p,
						 effects_generic_inf_difference_op,
						 r1_r2_concrete_cells_inf_difference_op,
						 effect_to_list,
						 effect_to_nil_list_and_free);

  return l_res;
}

list proper_to_summary_effects(list l_effects)
{
    return proper_effects_combine(l_effects, false);
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
    return(proper_effects_combine(l_effects, true));
}

/**
   merges the selected elements of the input list
   @param l_eff is a list of effects, and is modified/freed
   @param scalars_only_p; when true, only scalar effects are merged
   @return a new list of effects
   @see effects_combinable_p
   @see old_proper_effects_combine

   Asymptotic complexity is o(n^2/2). Former version was o(n) but did not
   properly take abstract locations into account.
 */
list proper_effects_combine(list l_eff, bool scalars_only_p)
{
  list l_res = NIL;
  bool (*effects_comb_p)(effect eff1, effect eff2) =
    scalars_only_p ? effects_scalars_and_same_action_p : effects_same_action_p;

  pips_debug_effects(6, "input effects:", l_eff);

  // merge one effect at a time in the result list using the union
  // operator.
  FOREACH(EFFECT, eff, l_eff)
    {
      eff = (*proper_to_summary_effect_func)(eff);

      l_res = (*effects_union_op)(effect_to_list(eff), l_res, *effects_comb_p);
    }
  gen_free_list(l_eff);
  pips_debug_effects(6, "output effects:", l_res);
  return l_res;
}


/* OBSOLETE - however, I keep it because it may be adapted to handle
   abstract locations properly - But it is unclear to me whether
   it's better to have a o(n^2/2) algorithm or a o(n) with hash
   table keys generated from strings, as manipulating strings
   is always costly, and many performance problems in Pips have been
   solved by avoiding them. BC
*/
/* list proper_effects_combine(list l_effects, bool scalars_only_p)
 * input    : a list of proper effects, and a bool to know on which
 *            elements to perform the combination.
 * output   : a list of effects, in which the selected elements have been
 *            merged.
 * modifies : the input list.
 * comment  : the algorithm is in O(n) (was in (n^2)/2)
 *
 * we need "entity/action" -> consp to check for the
 * condition in the second loop directly.
 * or to simplify the hash management, two entity -> consp?
 * a generic multi key combination hash would help.
 * the list is modified IN PLACE, storing on the first effect encountered...
 *
 * Extensions for C make it more complex as an effect on p and on p[1]
 * are not related. It would be possible to eliminate syntactically
 * equal effects by converting them into strings, without information
 * about the approximation. As a first try, I keep the hash tables
 * based on names and actions, and I only check compatibility in
 * reference and addressing mode afterwards. But this is not working
 * as I do not get enough different entries in the hash table since
 * entity_name is the key.
 *
 * The syntactic elimination is implemented, but the scoping
 * information is left out because the function is shard with the
 * effect prettyprinter. Scoping information should be preserved in the key.
 *
 * Also, it might be useful to normalize the effects and to convert
 * the pointer-based effects into an anywhere effect as soon as the
 * corresponding pointer is written before it is
 * dereferenced. However, we can delay the conversion till needed, as
 * long as the concrete semantics of the effect lists is understoof by
 * the user. This gives us a chance to find later the value of the
 * written pointer and to substitue it where needed. Else, the effects
 * to substitute will be gone.
 */
static list __attribute__((unused)) old_proper_effects_combine(list l_effects, bool scalars_only_p)
{
  list cur, pred = NIL;
  /* entity name -> consp in effect list. */
  hash_table all_read_effects, all_write_effects;
  /* FI: at this level, it's pretty dangerous to combine effects with
     no constant addresses */

  ifdebug(6) {
    list refl = NIL;
    int i = 0;
    pips_debug(6, "Proper effects to combine: \n");
    (*effects_prettyprint_func)(l_effects);

    /* This is a pretty weak assert because it's performed on the
       addresses */
    pips_assert("The very same effect does not appear twice",
		gen_once_p(l_effects));

    FOREACH(EFFECT, eff, l_effects) {
	cell c = effect_cell(eff);
	/* FI: the cell tag is not checked here but later a posteriori! */
	reference ref = cell_reference(c);

	if(cell_reference_p(c)) {
	  refl = gen_nconc(refl, CONS(REFERENCE, ref, NIL));
	}
	i++;
	fprintf(stderr, "Effect %d: %p\tReference %d: %p (%spersistant)\n",
		i, eff, i, ref, cell_reference_p(c)? "not ":"");
    }

    pips_assert("The very same reference does not appear twice"
		" unless it is persistant", gen_once_p(refl));
  }

  all_read_effects = hash_table_make(hash_string, 10);
  all_write_effects = hash_table_make(hash_string, 10);
  //all_read_pre_effects = hash_table_make(hash_string, 10);
  //all_write_pre_effects = hash_table_make(hash_string, 10);
  //all_read_post_effects = hash_table_make(hash_string, 10);
  //all_write_post_effects = hash_table_make(hash_string, 10);

  cur = l_effects;
  /* scan the list of effects... the list is modified in place */
  while(!ENDP(cur))
  {
    effect lcurrent = EFFECT(CAR(cur));
    effect current = effect_undefined;
    string n;
    tag a;
    action_kind ak;
    bool may_combine, do_combine = false;
    list do_combine_item = NIL;
    list next = CDR(cur); /* now, as 'cur' may be removed... */

    /* Summarization before combination: let's be as store independent as possible */
    current = (*proper_to_summary_effect_func)(lcurrent);

    ak = action_to_action_kind(effect_action(current));
   // n = entity_name(effect_entity(current));
    list w;
    string rn = words_to_string
      (w=effect_words_reference(effect_any_reference(current)));
    gen_map(free,w);gen_free_list(w);
    /* Do not combine effects of different kinds: use the kind in the key */
    asprintf(&n,"%s %s",rn,action_kind_to_string(ak));
    free(rn);
    a = effect_action_tag(current);

    pips_debug(8,"key: \"\%s\"\n", n);

    /* may/do we have to combine ? */
    /* ??? FC this should be no big deal... anyway :
     * in the previous implementation, 'current' was not yet
     * passed thru proper_to_summary_effect_func when tested...
     *
     * FI: effect_scalar_p() should be redefined because the new key
     * used let us deal with complex effects.
     */
    may_combine = (!scalars_only_p || effect_scalar_p(current));
    //&& !store_effect_p(current);


    /* FI: addressing should be checked below against writing of the
       underlying pointer. No time right now. */

    if (may_combine)
    {
      /* did we see it at a previous iteration? */
      switch (a) {
      case is_action_write:
	if (hash_defined_p(all_write_effects, n))
	{
	  do_combine = true;
	  do_combine_item = hash_get(all_write_effects, n);
	}
	break;
      case is_action_read:
	if (hash_defined_p(all_read_effects, n))
	{
	  do_combine = true;
	  do_combine_item = hash_get(all_read_effects, n);
	}
	break;
      default: pips_internal_error("unexpected action tag %d", a);
      }
    }

    if (do_combine)
    {
      /* YES, me must combine */

      effect base = EFFECT(CAR(do_combine_item));
      if(effect_comparable_p(base, current)) {
	/* compute their union */
	effect combined = (*effect_union_op)(base, current);

	/* replace the base effect by the new effect */
	pips_assert("combined is a consistent effect",
		    effect_consistent_p(combined));
	EFFECT_(CAR(do_combine_item)) = combined;

	/* free the original effects: no memory leak */
	free_effect(base);
	//free_effect(current); SG: this free triggers a segfault ...

	/* remove the current list element from the global list */
	/* pred!=NIL as on the first items hash's are empty */
	CDR(pred) = next; /* pred is not changed! */
	free(cur);
      }
      else {
	do_combine = false;
      }
    }
    if(!do_combine)
    {
      /* NO, just store if needed... */
      EFFECT_(CAR(cur)) = current;
      if (may_combine)
      {
	/* if we do not combine. ONLY IF we test, we put... */
	switch (a) {
	case is_action_write:
	  hash_put(all_write_effects, strdup(n), cur);
	  break;
	case is_action_read:
	  hash_put(all_read_effects, strdup(n), cur);
	  break;
	default: pips_internal_error("unexpected action tag %d", a);
	}
      }
      pred = cur;
    }

    cur = next;
    free(n);
  }

  ifdebug(6){
    pips_debug(6, "summary effects: \n");
    (*effects_prettyprint_func)(l_effects);
  }

  /* The keys should be freed as well */
  hash_table_free(all_write_effects);
  hash_table_free(all_read_effects);

  return l_effects;
}



/******************************************************* BOOL(EAN) FUNCTIONS */


/**
   @param eff1 and eff2 are two effects
   @ return true if their entities are the same, and if their access path
            as described by their references are the same, or if one of
	    the effect is an anywhere effect.
	    false otherwise.
 */
bool effects_combinable_p(effect eff1, effect eff2)
{
  bool combinable_p = false;
  action_kind ak1 = effect_action_kind(eff1);
  action_kind ak2 = effect_action_kind(eff2);

  pips_debug_effect(1,"eff1 = ", eff1);
  pips_debug_effect(1,"eff2 = ", eff2);


  if(action_kind_tag(ak1)==action_kind_tag(ak2))
    combinable_p = cells_combinable_p(effect_cell(eff1), effect_cell(eff2));

  pips_debug(1, "-> %s combinable\n", combinable_p? "": "not");

  return combinable_p;
}

bool cells_combinable_p(cell c1, cell c2)
{
  bool combinable_p = true;
  pips_assert("cell c1 is not a GAP", !cell_gap_p(c1));
  pips_assert("cell c2 is not a GAP", !cell_gap_p(c2));

  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);

  entity e1 = reference_variable(r1);
  entity e2 = reference_variable(r2);

  if (same_entity_p(e1, e2))
    {
      /* let us check the indices */
      list l1 = reference_indices(r1);
      list l2 = reference_indices(r2);
      
      if (gen_length(l1) != gen_length(l2))
	combinable_p = false;

      bool finished_p = false;

      while(combinable_p && !finished_p)
	{
	  if (ENDP(l1) && ENDP(l2))
	    {
	      finished_p = true;
	    }
	  else if (ENDP(l1) || ENDP(l2))
	    {
	      combinable_p = false;
	      finished_p = true;
	    }
	  else
	    {
	      expression exp1 = EXPRESSION(CAR(l1));
	      syntax s1 = expression_syntax(exp1);
	      expression exp2 = EXPRESSION(CAR(l2));
	      syntax s2 = expression_syntax(exp2);

	      /* check if the index expressions are the same for regions
		 or are comparable for simple effects.
		 For instance a[1] and a[2] can be merged into a[*] even
		 if the indices are not the same. The same for a[*] and a[1].
	         former tests were first based on string comparisons;
		 then expression_equal_p was used to lessen comparison cost,
		 especially for regions, where expressions are
		 references to PHI entities.
		 However, a[1] and a[2] were then considered as not comparable,
		 which resulted in long lists of effects, and was not compatible
		 with the generic binary operators which assume that there is at
		 most one effect/region per path kind.
	      */

	      if (syntax_reference_p(s1))
		{
		  entity es1 = reference_variable(syntax_reference(s1));
		  if (syntax_reference_p(s2))
		    {
		      entity es2 = reference_variable(syntax_reference(s2));
		      if (es1 != es2)
			{
			  if (variable_phi_p(es1) || variable_phi_p(es2)
			      || entity_field_p(es1) /* and then necessarily entity_field_p(e2) is true */)
			    {
			      combinable_p = false;
			      finished_p = true;
			    }
			}
		    }
		  else
		    {
		      // it cannot be a region; it's a simple effect and the current indices are array
		      // indices or a pointer dimension
		      pips_assert( "the current index must be an array index or a pointer dimension",
				   (!variable_phi_p(es1)) && (!entity_field_p(es1)));
		      combinable_p = true;
		      finished_p = false;
		    }
		}
	      if (!finished_p)
		{
		  POP(l1);
		  POP(l2);
		}
	    }
	}

    }
  else
    {

      if (c_module_p(get_current_module_entity()))
	/* don't do unecessary and costly things for fortran 77 codes*/
	{
	  bool al1_p = entity_abstract_location_p(e1);
	  bool al2_p = entity_abstract_location_p(e2);
	  bool heap1_p = al1_p && entity_heap_location_p(e1);
	  bool heap2_p = al2_p && entity_heap_location_p(e2);
	  bool heap1_context_sensitive_p = heap1_p && entity_flow_or_context_sentitive_heap_location_p(e1);
	  bool heap2_context_sensitive_p = heap2_p && entity_flow_or_context_sentitive_heap_location_p(e2);

	  /* if one of them is an abstract location, they may be combinable */
	  if ( al1_p || al2_p)
	    {
	      /* this is only a temporary hack before we avoid generating anywhere effects
		 each time we lose some information
	      */
	      if (al1_p && (malloc_cell_p(c2) || io_cell_p(c2)
			    || (!get_bool_property("USER_EFFECTS_ON_STD_FILES")
				&& std_file_cell_p(c2))))
		{
		  pips_debug(8, "eff1 is an effect on an abstract location "
			     "and eff2 is a malloc or io effect\n");
		  combinable_p = false;
		}
	      else if (al2_p && (malloc_cell_p(c1) || io_cell_p(c1)
				 || (!get_bool_property("USER_EFFECTS_ON_STD_FILES")
				     && std_file_cell_p(c1))))
		{
		  pips_debug(8, "eff2 is an effect on an abstract location "
			     "and eff1 is a malloc or io effect\n");
		  combinable_p = false;
		}
	      else if ((al1_p && entity_all_locations_p(e1))
		    || (al2_p && entity_all_locations_p(e2)))
		  combinable_p = true;
	      else if ((al1_p && undefined_pointer_value_entity_p(e1))
		      || (al2_p && undefined_pointer_value_entity_p(e2)))
		combinable_p = false;
	      else if ((al1_p && entity_null_locations_p(e1))
		      || (al2_p && entity_null_locations_p(e2)))
		combinable_p = false;
	      else if ( (al1_p && al2_p)
		       || (al1_p && ENDP(reference_indices(r2)))
		       || (al2_p && ENDP(reference_indices(r1))))
		    /*two abstract locations or one abstract location and a scalar */
		    combinable_p = entities_may_conflict_p(e1,e2);
		  else
		    {
		      if (heap1_p || heap2_p)
			{
			  if (heap1_context_sensitive_p && heap2_context_sensitive_p)
			    {
			      // they conflict only if they are the same. however, this case
			      // has already been handled before
			      combinable_p = false;
			    }
			  else if (heap1_p && heap2_p)
			    {
			      // the two are heap entities but at most one of them is context sensitive
			      // they are combinable
			      combinable_p = true;
			    }
			  else 
			    {
			      // only one of them is a heap location
			      // if the other one is not an abstract location, then they are not
			      // combinable
			      if ((heap1_p && !al2_p) || (heap2_p && !al1_p))
				{
				  combinable_p = false;
				}
			      else
				{
				  /* here it is more difficult, to be handled later */
				  pips_internal_error("case not handled yet (c1 = %s, c2 = %s)\n",
						      effect_reference_to_string(r1),
						      effect_reference_to_string(r2));
				}
			    }
			}
		      else
			{
			  /* here we should consider the type of the non abstract location reference,
			     whether there is a dereferencement or not to guess the memory area, ... */
			  pips_internal_error("case not handled yet (c1 = %s, c2 = %s)\n",
					      effect_reference_to_string(r1),
					      effect_reference_to_string(r2));
			}
		    }
	    }

	  else combinable_p = false; /* two concrete location paths beginning from different entities
					are not combinable by binary operators.
					well this is true for constant path effects/regions...
					for pointer regions this is true for union
					I still have to check for intersection and difference. BC.
				     */
	}
      else combinable_p = false;
    }

  return combinable_p;
}

/* BC: same action, but also same variable and same indexing and
 * at least one of them is a scalar (abstract locations and concrete
 * are combined even if the concrete location is not a scalar)
 *
 * Checking the action kind may sometimes be useless or even wrong.
  */
bool effects_scalars_and_same_action_p(effect eff1, effect eff2)
{
  bool same_p = false;

  if (effect_undefined_p(eff1) || effect_undefined_p(eff2))
    same_p = true;
  else if (effect_action_tag(eff1)!=effect_action_tag(eff2))
    {
      same_p = false;
    }
  else {
    action_kind ak1 = action_to_action_kind(effect_action(eff1));
    action_kind ak2 = action_to_action_kind(effect_action(eff2));
    if(action_kind_tag(ak1)!=action_kind_tag(ak2))
      same_p = false;
    else
      same_p = (effect_scalar_p(eff1) || effect_scalar_p(eff2)) && effects_combinable_p(eff1, eff2);
  }

  return same_p;
}

/* FI: same action, but also same variable and same indexing
 *
 * Checking the action kind may sometimes be useless or even wrong.
  */
bool effects_same_action_p(effect eff1, effect eff2)
{
  bool same_p = false;

  if (effect_undefined_p(eff1) || effect_undefined_p(eff2))
    same_p = true;
  else if (effect_action_tag(eff1)!=effect_action_tag(eff2))
    {
      same_p = false;
    }
  else {
    action_kind ak1 = action_to_action_kind(effect_action(eff1));
    action_kind ak2 = action_to_action_kind(effect_action(eff2));
    if(action_kind_tag(ak1)!=action_kind_tag(ak2))
      same_p = false;
    else
      same_p = effects_combinable_p(eff1, eff2);
  }

  return same_p;
}

bool effects_same_variable_p(effect eff1, effect eff2)
{
    bool same_var = (effect_entity(eff1) == effect_entity(eff2));
    return(same_var);
}


bool r_r_combinable_p(effect eff1, effect eff2)
{
    bool combinable_p, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_read_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_read_p(eff1));

    combinable_p = effects_combinable_p(eff1,eff2);
    act_combinable = (effect_read_p(eff1) && effect_read_p(eff2));

    return(combinable_p && act_combinable);
}

bool w_w_combinable_p(effect eff1, effect eff2)
{
    bool combinable_p, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_write_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_write_p(eff1));

    combinable_p = effects_combinable_p(eff1,eff2);
    act_combinable = (effect_write_p(eff1) && effect_write_p(eff2));

    return(combinable_p && act_combinable);
}

bool r_w_combinable_p(effect eff1, effect eff2)
{
    bool combinable_p, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_write_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_read_p(eff1));

    combinable_p = effects_combinable_p(eff1,eff2);
    act_combinable = (effect_read_p(eff1) && effect_write_p(eff2));

    return(combinable_p && act_combinable);
}

bool w_r_combinable_p(effect eff1, effect eff2)
{
    bool combinable_p, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_read_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_write_p(eff1));

    combinable_p = effects_combinable_p(eff1,eff2);
    act_combinable = (effect_write_p(eff1) && effect_read_p(eff2));

    return(combinable_p && act_combinable);
}

/***********************************************************************/
/* UNDEFINED BINARY OPERATOR                                           */
/***********************************************************************/

list
effects_undefined_binary_operator(list l1, list l2,
				  bool (*effects_combinable_p)(effect, effect))
{
  pips_assert("unused arguments", l1==l1 && l2==l2 &&
	      effects_combinable_p==effects_combinable_p);
  return list_undefined;
}


/***********************************************************************/
/* SOME BINARY OPERATORS which do not depend on the representation     */
/***********************************************************************/

/* list effect_entities_intersection(effect eff1, effect eff2)
 * input    : two effects
 * output   : a mere copy of the first effect.
 * modifies : nothing.
 * comment  : We assume that both effects concern the same entity.
 */
static list
effect_entities_intersection(effect eff1, effect eff2)
{
  pips_assert("unused argument", eff2==eff2);
  return CONS(EFFECT, (*effect_dup_func)(eff1), NIL);
}

/* list effects_entities_intersection(list l1, list l2,
                           bool (*intersection_combinable_p)(effect, effect))
 * input    : two lists of effects.
 * output   : a list of effects containing all the effects of l1 that have
 *            a corresponding effect (i.e. same entity) in l2.
 * modifies : l1 and l2.
 * comment  :
 */
list
effects_entities_intersection(list l1, list l2,
			      bool (*intersection_combinable_p)(effect, effect))
{
    list l_res = NIL;

    pips_debug(3, "begin\n");
    l_res = list_of_effects_generic_cells_intersection_op(l1, l2,
					   intersection_combinable_p,
					   effect_entities_intersection);
    pips_debug(3, "end\n");

    return l_res;
}


/* list effects_entities_inf_difference(list l1, l2)
 * input    : two lists of effects
 * output   : a list of effects, such that: if there is a effect R concerning
 *            entity A in l1 and in l2, then R is removed from the result;
 *            if there is a effect R concerning array A in l1, but not in l2,
 *            then it is kept in l1, and in the result.
 * modifies : the effects of l2 may be freed.
 * comment  : we keep the effects of l1 that are not combinable with those
 *            of l2, but we don't keep the effects of l2 that are not
 *            combinable with those of l_reg1.
 */
list
effects_entities_inf_difference(
    list l1,
    list l2,
    bool (*difference_combinable_p)(effect, effect))
{
    list l_res = NIL;

    pips_debug(3, "begin\n");
    l_res = list_of_effects_generic_inf_difference_op(l1, l2,
					   difference_combinable_p,
					   effects_to_nil_list);
    pips_debug(3, "end\n");

    return l_res;
}

/* that is all
 */
