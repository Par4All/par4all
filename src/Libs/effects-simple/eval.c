/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2009-2010 HPC Project

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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "text-util.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "pointer_values.h"

#include "effects-generic.h"
#include "effects-simple.h"

/*
  @param r1 and r2 are two path references
  @param strict_p is true if the path length of r1 must be strictly inferior
         to the path length of r2
  @param exact_p is a pointer towards a boolean, which is set to false
         is the result is an over-approximation, true if it's an exact answer.
  @return true if r1 path may be a predecessor of r2 path

  (we consider that p[1] is a predecessor of p[*], with *exact_p = false.)

 */
bool simple_cell_reference_preceding_p(reference r1, descriptor __attribute__ ((unused)) d1,
			     reference r2, descriptor __attribute__ ((unused)) d2,
			     transformer __attribute__ ((unused)) current_precondition,
		             bool strict_p,
			     bool * exact_p)
{
  bool res = true;
  entity e1 = reference_variable(r1);
  list ind1 = reference_indices(r1);
  size_t r1_path_length = gen_length(ind1);
  entity e2 = reference_variable(r2);
  list ind2 = reference_indices(r2);
  size_t r2_path_length = gen_length(ind2);

  pips_debug(8, "input references r1 : %s, r2: %s \n",
	words_to_string(words_reference(r1, NIL)),
	words_to_string(words_reference(r2, NIL)));

  *exact_p = true;
  if (same_entity_p(e1, e2)
      && ((r1_path_length < r2_path_length)
	  || (!strict_p && r1_path_length == r2_path_length)))
    {
      /* same entity and the path length of r1 is shorter than the path length of r2.
         we now have to check that each common index matches
      */
      pips_debug(8,"same entities, and r1 path is shorter than r2 path\n");
      while (res && !ENDP(ind1))
	{
	  expression exp1 = EXPRESSION(CAR(ind1));
	  expression exp2 = EXPRESSION(CAR(ind2));

	  if (unbounded_expression_p(exp1) || unbounded_expression_p(exp2))
	    {
	      res = true;
	      *exact_p = false;
	    }
	  else if(!expression_equal_p(exp1, exp2))
	    {
	      res = false;
	      *exact_p = true;
	    }

	  POP(ind1);
	  POP(ind2);
	}
    }
  else
    {
      res = false;
      *exact_p = true;
    }

  pips_debug(8, "end : r1 is %s a predecessor of r2 (%s exact)\n",
	     res ? "":"not", *exact_p ? "":"not");
  return res;
}


bool simple_cell_preceding_p(cell c1, descriptor d1,
			     cell c2, descriptor d2,
			     transformer current_precondition,
		             bool strict_p,
			     bool * exact_p)
{
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);

  return simple_cell_reference_preceding_p(r1, d1, r2, d2, current_precondition,
					   strict_p, exact_p);
}

bool path_preceding_p(effect eff1, effect eff2,
		      transformer current_precondition,
		      bool strict_p,
		      bool * exact_p)
{
  reference r1 = effect_any_reference(eff1);
  descriptor d1 = effect_descriptor(eff1);
  reference r2 = effect_any_reference(eff2);
  descriptor d2 = effect_descriptor(eff2);

  return simple_cell_reference_preceding_p(r1, d1, r2, d2, current_precondition,
					   strict_p, exact_p);
}

void simple_reference_to_simple_reference_conversion(reference ref,
						     reference * output_ref,
						     descriptor * output_desc)
{
  *output_ref = ref;
  *output_desc = make_descriptor(is_descriptor_none,UU);
}

void simple_cell_to_simple_cell_conversion(cell input_cell,
					   cell * output_cell,
					   descriptor * output_desc)
{
  *output_cell = input_cell;
  *output_desc = make_descriptor(is_descriptor_none,UU);
}

/*
  @param c is a the simple cell for which we look an equivalent constant path
  @param ptl is the list of points-to in which we search for constant paths
  @param  exact_p is a pointer towards a boolean. It is set to true if
         the result is exact, and to false if it is an approximation,
	 either because the matching points-to sources found in ptl are
	 over-approximations of the preceding path of input_ref or because
	 the matching points-to have MAY approximation tag.
  @return a list of constant path cells. It is a list because at a given
          program point the cell may correspond to several constant paths.


  original comment:
  goal: see if cell c can be shortened by replacing its indirections
  by their values when they are defined in ptl. For instance, p[0][0]
  and (p,q,EXACT) is reduced to q[0]. And if (q,i,EXACT) is also
  available, the reference is reduced to i. The reduced cell is less likely
  to be invalidated by a write effect. The function is called "eval"
  because its goal is to build an as constant as possible reference or
  gap.

  I currently assume that points-to sinks are constant paths because
  in the last example we should also have (p[0], i, EXACT) in the
  points-to list. (BC)

  This function is called by effects to see if a memory access path
  can be transformed into a constant. It should also be called by the
  points-to analysis to see if a sink or a source can be preserved in
  spite of write effects. This function should be called before
  points_to_filter_effects() to reduce the number of anywhere
  locations generated.
*/
list eval_simple_cell_with_points_to(cell c, descriptor __attribute__ ((unused)) d,
				     list ptl, bool *exact_p,
				     transformer __attribute__ ((unused)) t)
{

  return generic_eval_cell_with_points_to(c, descriptor_undefined, ptl, exact_p,
					  transformer_undefined,
					  simple_cell_reference_preceding_p,
					  simple_cell_reference_with_address_of_cell_reference_translation,
					  simple_reference_to_simple_reference_conversion);
}


list simple_effect_to_constant_path_effects_with_pointer_values(effect eff)
{
  list le = NIL;
  bool exact_p;
  reference ref = effect_any_reference(eff);

  if (effect_reference_dereferencing_p(ref, &exact_p))
    {
      pips_debug(8, "dereferencing case \n");
      bool exact_p = false;
      list l_pv = cell_relations_list( load_pv(effects_private_current_stmt_head()) );
      pv_context ctxt = make_simple_pv_context();
      list l_aliased = effect_find_aliased_paths_with_pointer_values(eff, l_pv, &ctxt);
      pips_debug_effects(8, "aliased effects\n", l_aliased);
      reset_pv_context(&ctxt);

      FOREACH(EFFECT, eff_alias, l_aliased)
	{
	  entity ent_alias = effect_entity(eff_alias);
	  if (undefined_pointer_value_entity_p(ent_alias)
	      || null_pointer_value_entity_p(ent_alias))
	    {
	      // currently interpret them as anywhere effects since these values
	      // are not yet well integrated in abstract locations lattice
	      // and in effects computations
	      // to be FIXED later.
	      le = CONS(EFFECT, make_anywhere_effect(copy_action(effect_action(eff_alias))), le);
	      free_effect(eff_alias);
	    }
	  else if (entity_abstract_location_p(effect_entity(eff_alias))
	      || !effect_reference_dereferencing_p(effect_any_reference(eff_alias), &exact_p))
	    le = CONS(EFFECT, eff_alias, le); /* it should be a union here.
						 However, we expect the caller
						 to perform the contraction afterwards. */
	  else
	    free_effect(eff_alias);
	}
      gen_free_list(l_aliased);
    }
  else
    le = CONS(EFFECT, (*effect_dup_func)(eff), le);
 return le;
}

list simple_effect_to_constant_path_effects_with_points_to(effect eff)
{
  list le = NIL;
  bool exact_p;
  reference ref = effect_any_reference(eff);

  pips_debug_effect(5, "input effect", eff);

  points_to_list ptl = load_pt_to_list(effects_private_current_stmt_head());
  if(!points_to_list_bottom(ptl)) {
    if (effect_reference_dereferencing_p(ref, &exact_p))
      {
	pips_debug(8, "dereferencing case \n");
	bool exact_p = false;
	transformer context;
	if (effects_private_current_context_empty_p())
	  context = transformer_undefined;
	else {
	  context = effects_private_current_context_head();
	}

	list l_eval = eval_simple_cell_with_points_to(effect_cell(eff), effect_descriptor(eff),
						      points_to_list_list(ptl),
						      &exact_p, context);
	if (ENDP(l_eval))
	  {
	    pips_debug(8, "no equivalent constant path found -> anywhere effect\n");
	    /* We have not found any equivalent constant path : it may point anywhere */
	    /* We should maybe contract these effects later. Is it done by the callers ? */
	    // le = CONS(EFFECT, make_anywhere_effect(copy_action(effect_action(eff))), le);
	    le = NIL; // A translation failure means an execution
	    // failure, at least according to the standard
	  }
	else
	  {
	    /* change the resulting effects action to the current effect action */
	    if (effect_read_p(eff))
	      effects_to_read_effects(l_eval);
	    if (effect_may_p(eff))
	      effects_to_may_effects(l_eval);
	    le = gen_nconc(l_eval,le);
	  }
      }
    else
      le = CONS(EFFECT, copy_effect(eff), le);
  }
  else {
    /* This is dead code: no effects, do not modify le */
    ;
  }
  return le;

}



/* for backward compatibility */

list eval_cell_with_points_to(cell c, list ptl, bool *exact_p)
{
  list l_eff = eval_simple_cell_with_points_to(c, descriptor_undefined, ptl, exact_p, transformer_undefined);
  list l = NIL;

  FOREACH(EFFECT,eff, l_eff)
    {
      l = CONS(CELL, effect_cell(eff), l);
      effect_cell(eff) = cell_undefined;
    }
  gen_full_free_list(l_eff);
  return l;
}
