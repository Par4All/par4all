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
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"




#include "effects-generic.h"
#include "effects-simple.h"

#include "resources.h"


list
ReferenceUnion(list l1, list l2,
	       bool (*union_combinable_p)(effect, effect) __attribute__ ((unused)))
{
    return gen_nconc(l1,l2);
}

list
ReferenceTestUnion(list l1, list l2,
		   bool (*union_combinable_p)(effect, effect) __attribute__ ((unused)))
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
		     bool (*union_combinable_p)(effect, effect))
{
    list lr;

    lr = list_of_effects_generic_union_op(l1, l2,
					   union_combinable_p,
					   effects_may_union,
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
ProperEffectsMustUnion(list l1, list l2,
		      bool (*union_combinable_p)(effect, effect)) 
{
    list lr;

    lr = list_of_effects_generic_union_op(l1, l2,
					   union_combinable_p,
					   effects_must_union,
					   effect_to_list);
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
		      bool (*union_combinable_p)(effect, effect)) 
{
    list lr;

    lr = list_of_effects_generic_union_op(l1, l2,
					   union_combinable_p,
					   effects_must_union,
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

/* Preserve store independent information as long as you can. I should
   have some order on references to absorb, for instance, x[1] by
   x[*]. */

/**
    @brief computes the may union of two combinable effects
    @param[in] eff1 effect
    @param[in] eff2
    @see effects_combinable_p
    @return a new effect with no sharing with the input effects
 */
effect effect_may_union(effect eff1, effect eff2)
{
    effect eff;
    tag app1 = effect_approximation_tag(eff1);
    tag app2 = effect_approximation_tag(eff2);

    bool al1_p = effect_abstract_location_p(eff1);
    bool al2_p = effect_abstract_location_p(eff2);

    /* Abstract locations cases */
    /* In fact, we could have :
       if (al1_p || al_2_p)
       {
         entity e1 = effect_entity(e1);
	 entity e2 = effect_entity(e2);

	 new_ent = entity_locations_max(e1, e2);

	 eff = make_simple_effect(make_reference(new_ent, NIL),
			  copy_action(effect_action(eff1)),
			  make_approximation(approximation_and(app1,app2), UU));
       }

       but entity_locations_max involves string manipulations, which are always costly.
       So we treat apart the cases where (al1_p and ! al2_p) and (al2_p and ! al1_p) because
       we already know that the abstract location is the max of both locations
       (because they are combinable (see effects_combinable_p))

     */
    if (al1_p && al2_p)
      {
	entity e1 = effect_entity(eff1);
	entity e2 = effect_entity(eff2);

	entity new_ent = entity_locations_max(e1, e2);

	eff = make_simple_effect(make_reference(new_ent, NIL),
			  copy_action(effect_action(eff1)),
			  make_approximation(approximation_and(app1,app2), UU));
      }
    else if (al1_p)
      eff = (*effect_dup_func)(eff1);
    else if (al2_p)
      eff = (*effect_dup_func)(eff2);

    /* concrete locations cases */
    else if (effect_scalar_p(eff1))
    {
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL),
			  copy_action(effect_action(eff1)),
			  make_approximation(approximation_and(app1,app2), UU));
    }
    else
    {
      eff = (*effect_dup_func)(eff1);
      cell eff_c = effect_cell(eff);

      if (cell_preference_p(eff_c))
	{
	  /* it's a preference : we change for a reference cell */
	  pips_debug(8, "It's a preference\n");
	  reference ref = copy_reference(preference_reference(cell_preference(eff_c)));
	  free_cell(eff_c);
	  effect_cell(eff) = make_cell_reference(ref);
	}

      /* let us check the indices */
      list l1 = reference_indices(effect_any_reference(eff));
      list l2 = reference_indices(effect_any_reference(eff2));

      tag app_tag =  approximation_and(app1,app2);

      for(; !ENDP(l1); POP(l1), POP(l2))
	{
	  expression exp1 = EXPRESSION(CAR(l1));
	  expression exp2 = EXPRESSION(CAR(l2));
	  if (!expression_equal_p(exp1, exp2))
	    {
	      EXPRESSION_(CAR(l1)) =  make_unbounded_expression();
	      app_tag = is_approximation_may;
	    }
	}

      approximation_tag(effect_approximation(eff)) = app_tag;
    }
    return(eff);
}

/**
    @brief computes the must union of two combinable effects
    @param[in] eff1 effect
    @param[in] eff2
    @see effects_combinable_p
    @return a new effect with no sharing with the input effects
 */
effect effect_must_union(effect eff1, effect eff2)
{
    effect eff;
    tag app1 = effect_approximation_tag(eff1);
    tag app2 = effect_approximation_tag(eff2);

    bool al1_p = effect_abstract_location_p(eff1);
    bool al2_p = effect_abstract_location_p(eff2);

    /* Abstract locations cases */
    /* In fact, we could have :
       if (al1_p || al_2_p)
       {
         entity e1 = effect_entity(e1);
	 entity e2 = effect_entity(e2);

	 new_ent = entity_locations_max(e1, e2);

	 eff = make_simple_effect(make_reference(new_ent, NIL),
			  copy_action(effect_action(eff1)),
			  make_approximation(approximation_and(app1,app2), UU));
       }

       but entity_locations_max involves string manipulations, which are always costly.
       So we treat apart the cases where (al1_p and ! al2_p) and (al2_p and ! al1_p) because
       we already know that the abstract location is the max of both locations
       (because they are combinable (see effects_combinable_p))

     */
    if (al1_p && al2_p)
      {
	entity e1 = effect_entity(eff1);
	entity e2 = effect_entity(eff2);

	entity new_ent = entity_locations_max(e1, e2);

	eff = make_simple_effect(make_reference(new_ent, NIL),
			  copy_action(effect_action(eff1)),
			  make_approximation(approximation_and(app1,app2), UU));
      }
    else if (al1_p)
      eff = (*effect_dup_func)(eff1);
    else if (al2_p)
      eff = (*effect_dup_func)(eff2);

    /* concrete locations cases */
    else if (effect_scalar_p(eff1))
    {
	eff = make_simple_effect(make_reference(effect_entity(eff1), NIL),
			  copy_action(effect_action(eff1)),
			  make_approximation(approximation_or(app1,app2), UU));
    }
    else
    {
      eff = (*effect_dup_func)(eff1);
      cell eff_c = effect_cell(eff);
      if (cell_preference_p(eff_c))
	{
	  /* it's a preference : we change for a reference cell */
	  pips_debug(8, "It's a preference\n");
	  reference ref = copy_reference(preference_reference(cell_preference(eff_c)));
	  free_cell(eff_c);
	  effect_cell(eff) = make_cell_reference(ref);
	}

      /* let us check the indices */
      list l1 = reference_indices(effect_any_reference(eff));
      list l2 = reference_indices(effect_any_reference(eff2));

      tag app_tag = approximation_or(app1,app2);

      for(; !ENDP(l1); POP(l1), POP(l2))
	{
	  expression exp1 = EXPRESSION(CAR(l1));
	  expression exp2 = EXPRESSION(CAR(l2));
	  if (!expression_equal_p(exp1, exp2))
	    {
	      EXPRESSION_(CAR(l1)) =  make_unbounded_expression();
	      app_tag = is_approximation_may;
	    }
	}

      effect_approximation_tag(eff) = app_tag;
    }
    return(eff);
}


static list
effect_sup_difference(/* const */ effect eff1, /* const */ effect eff2)
{
    list l_res = NIL;
    reference ref1 = effect_any_reference(eff1);
    reference ref2 = effect_any_reference(eff2);

    /* We already know that effects are combinable and they are not abstract locations
       (or they are context_sensitive heap locations)
    */
    if (reference_equal_p(ref1, ref2)) // a[1] - a[1] or a[*] - a[*]_may
      {
	if (effect_may_p(eff2))
	  l_res = effect_to_may_effect_list((*effect_dup_func)(eff1));
	// else: empty list
      }
    else
      {
	if (effect_exact_p(eff1) && effect_exact_p(eff2)) // a[1] - a[2]
	  l_res = effect_to_list((*effect_dup_func)(eff1));
	else
	  {
	    // a[*] - a[1] or a[1] - a[*]_may for instance
	    l_res = effect_to_may_effect_list((*effect_dup_func)(eff1));
	  }
      }

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
			  bool (*difference_combinable_p)(effect, effect))
{
    list l_res = NIL;

    debug(3, "EffectsSupDifference", "begin\n");
    l_res = list_of_effects_generic_sup_difference_op(l1, l2,
					   difference_combinable_p,
					   effect_sup_difference);
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
			  bool (*difference_combinable_p)(effect, effect))
{
    list l_res = NIL;

    debug(3, "EffectsInfDifference", "begin\n");
    l_res = list_of_effects_generic_inf_difference_op(l1, l2,
					   difference_combinable_p,
					   effect_inf_difference);
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
    cell eff_c = effect_cell(eff);
    bool may_p = false;
    reference ref = reference_undefined;

    if (cell_preference_p(eff_c))
    {
      /* it's a preference : we change for a reference cell */
      pips_debug(8, "It's a preference\n");
      ref = copy_reference(preference_reference(cell_preference(eff_c)));
      free_cell(eff_c);
      effect_cell(eff) = make_cell_reference(ref);
    }
    else
    {
      /* it's a reference : let'us modify it */
      ref = cell_reference(eff_c);
    }

    ifdebug(8) {
      pips_debug(8, "Proper effect %p with reference %p: %s\n", eff, ref,
		 words_to_string(words_effect(eff)));
    }

    list inds = reference_indices(ref);
    list cind = list_undefined;
    for(cind = inds; !ENDP(cind); POP(cind)) {
      expression se = EXPRESSION(CAR(cind));

      ifdebug(8) {
	pips_debug(8, "Subscript expression :\n");
	print_expression(se);
      }

      if(!extended_integer_constant_expression_p(se)) {
	if(!unbounded_expression_p(se))
	  {
	    /* it may still be a field entity */
	    if (!(expression_reference_p(se) &&
		  entity_field_p(expression_variable(se))))
	    {
	      may_p = true;
	      free_expression(se);
	      EXPRESSION_(CAR(cind)) = make_unbounded_expression();
	    }
	  }
      }
    }

    if(may_p)
      effect_approximation_tag(eff) = is_approximation_may;

    ifdebug(8) {
      pips_debug(8, "Summary simple effect %p with reference %p: %s\n", eff, ref,
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


bool simple_cells_intersection_p(cell c1, descriptor __attribute__ ((__unused__)) d1,
				 cell c2, descriptor __attribute__ ((__unused__)) d2,
				 bool * exact_p)
{
  bool res = true; /* default safe result */
  bool concrete_locations_p = true;

  if (cells_combinable_p(c1, c2))
    {
      bool c1_abstract_location_p = cell_abstract_location_p(c1);
      bool c2_abstract_location_p = cell_abstract_location_p(c2);

      if (c1_abstract_location_p || c2_abstract_location_p)
	{
	  entity e1 = cell_entity(c1);
	  entity e2 = cell_entity(c2);
	  bool heap1_context_sensitive_p = c1_abstract_location_p && entity_flow_or_context_sentitive_heap_location_p(e1);
	  bool heap2_context_sensitive_p = c2_abstract_location_p && entity_flow_or_context_sentitive_heap_location_p(e2);

	  if (heap1_context_sensitive_p && heap2_context_sensitive_p)
	    {
	      concrete_locations_p = true;
	    }
	  else
	    {
	      concrete_locations_p = false;

	      res = true;
	      *exact_p = true;
	    }
	}

      if (concrete_locations_p)
	{
	  /* we have combinable concrete locations or assimilated (context sensitive heap locations) */
	  list l1 = cell_indices(c1);
	  list l2 = cell_indices(c2);

	  /* they intersect if their corresponding indices are unbounded or equal */
	  res = true; *exact_p = true;
	  while(res && !ENDP(l1))
	    {
	      expression exp1 = EXPRESSION(CAR(l1));
	      expression exp2 = EXPRESSION(CAR(l2));

	      if (unbounded_expression_p(exp1) || unbounded_expression_p(exp2))
		*exact_p = false;
	      else if (!expression_equal_p(exp1, exp2))
		{
		  res = false;
		  *exact_p = true;
		}
	      POP(l1);
	      POP(l2);
	    }
	}
    }
  else
    {
      res = false;
      *exact_p = true;
    }
  return res;
}

/* Inclusion test :
 */

/**
   returns true if c1 is included into c2, false otherwise.
   returns false if c1 may only be included into c2.

   @param exact_p target is set to true if the result is exact, false otherwise.

   In fact, this parameter would be useful only if there are overflows during
   the systems inclusion test. But it is not currently used.
 */
bool simple_cells_inclusion_p(cell c1, __attribute__ ((__unused__)) descriptor d1,
			      cell c2, __attribute__ ((__unused__)) descriptor d2,
			      bool * exact_p)
{
  bool res = true; /* default result */
  *exact_p = true;

  bool concrete_locations_p = true;

  if (cells_combinable_p(c1, c2))
    {
      bool c1_abstract_location_p = cell_abstract_location_p(c1);
      bool c2_abstract_location_p = cell_abstract_location_p(c2);

      if (c1_abstract_location_p || c2_abstract_location_p)
	{
	  entity e1 = cell_entity(c1);
	  entity e2 = cell_entity(c2);
	  bool heap1_context_sensitive_p = c1_abstract_location_p && entity_flow_or_context_sentitive_heap_location_p(e1);
	  bool heap2_context_sensitive_p = c2_abstract_location_p && entity_flow_or_context_sentitive_heap_location_p(e2);

	  if (heap1_context_sensitive_p && heap2_context_sensitive_p)
	    {
	      concrete_locations_p = true;
	    }
	  else
	    {
	      entity al_max = abstract_locations_max(e1, e2);
	      res = same_entity_p(e2, al_max);
	      concrete_locations_p = false;
	    }
	}

      if (concrete_locations_p)
	{
	  /* we have combinable concrete locations or assimilated (context sensitive heap locations) */
	  list inds1 = cell_indices(c1);
	  list inds2 = cell_indices(c2);


	  for(;!ENDP(inds1) && res == true; POP(inds1), POP(inds2))
	    {
	      expression exp1 = EXPRESSION(CAR(inds1));
	      expression exp2 = EXPRESSION(CAR(inds2));

	      if (unbounded_expression_p(exp1))
		{
		  pips_debug(8, "case 4.1\n");
		  if (!unbounded_expression_p(exp2))
		    {
		      pips_debug(8, "case 4.2\n");
		      res = false;
		    }
		}
	      else if (!unbounded_expression_p(exp2) && !expression_equal_p(exp1, exp2) )
		{
		  pips_debug(8, "case 4.3\n");
		  res = false;
		}
	    }
	}
    }
  else
    {
      res = false;
    }
  return res;
}
