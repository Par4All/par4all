/*

  $Id: eval.c 17426 2010-06-25 09:24:14Z creusillet $

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

#include "effects-generic.h"
#include "effects-convex.h"

/*
  @param r1 and r2 are two path references
  @param exact_p is a pointer towards a boolean, which is set to false
         is the result is an over-approximation, true if it's an exact answer.
  @return true if r1 path may be a predecessor of r2 path

  (we consider that p[1] is a predecessor of p[*], with *exact_p = false.)

 */
bool convex_cell_reference_preceding_p(reference r1, descriptor d1, 
				       reference r2, descriptor d2, 
<<<<<<< HEAD
				       transformer current_precondition, 
=======
>>>>>>> new phase to compute constant path regions using points-to.
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
      && (r1_path_length < r2_path_length))
    {
      /* same entity and the path length of r1 is shorter than the path length of r2.
         we now have to check that each common index matches
      */
      pips_debug(8,"same entities, and r1 path is shorter than r2 path\n");
      while (res && !ENDP(ind1))
	{
	  expression exp1 = EXPRESSION(CAR(ind1));
	  expression exp2 = EXPRESSION(CAR(ind2));
	  
	  if(!expression_equal_p(exp1, exp2))
	    {
	      res = false;
	      *exact_p = true;
	    }
	  
	  POP(ind1);
	  POP(ind2);
	}
      if (res)
	{
	  /* only matching reference indices have been found (phi variables or struct field entities).
	     we must now check the descriptors.
	  */
	  region reg1 = make_effect(make_cell(is_cell_reference, r1), make_action_write(), make_approximation_must(), d1);
	  region reg2 = make_effect(make_cell(is_cell_reference, r2), make_action_write(), make_approximation_must(), d2);

	  pips_debug_effect(6, "reg1 = \n", reg1);
	  pips_debug_effect(6, "reg2 = \n", reg1);
	  
	  list li = region_intersection(reg1, reg2);
	  if (ENDP(li))
	    {
	      res = false;
	      *exact_p = true;
	    }
	  else
	    {
<<<<<<< HEAD
	      
	      pips_debug_effect(8, "reg2 before eliminating phi variables: \n ", reg2);
	
	      effect reg2_dup = copy_effect(reg2);
	      list l_reg2 = CONS(EFFECT,reg2_dup,NIL);
	      list l_phi = phi_entities_list(r1_path_length+1,r2_path_length);    
	      project_regions_along_variables(l_reg2, l_phi);
	      gen_free_list(l_reg2);
	      gen_free_list(l_phi);
	      pips_debug_effect(8, "reg2_dup after elimination: \n ", reg2_dup);
	
	      effect reg1_dup = copy_effect(reg1);
	      if (!transformer_undefined_p(current_precondition))
		{
		  Psysteme sc_context = predicate_system(transformer_relation(current_precondition));
		  region_sc_append(reg1_dup, sc_context, FALSE);
		}
		
	      pips_debug_effect(8, "reg1_dup after adding preconditions: \n ", reg1_dup);
	      pips_debug_effect(8, "reg1 after adding preconditions: \n ", reg1);
	
	      list ld = region_sup_difference(reg1_dup, reg2_dup);
=======
	      list ld = region_sup_difference(reg1, reg2);
>>>>>>> new phase to compute constant path regions using points-to.
	      if (ENDP(ld))
		{
		  res = true;
		  *exact_p = true;
		}
	      else
		{
		  res = true;
		  *exact_p = false;
		}
	      gen_full_free_list(ld);	   
	    }
	  gen_full_free_list(li);
	      
	  cell_reference(effect_cell(reg1)) = reference_undefined;
	  effect_descriptor(reg1) = descriptor_undefined;
	  free_effect(reg1);

	  cell_reference(effect_cell(reg2)) = reference_undefined;
	  effect_descriptor(reg2) = descriptor_undefined;
	  free_effect(reg2);
	}
    }
  else
    {
      res = false;
<<<<<<< HEAD
      *exact_p = true;
=======
      *exact_p = false;
>>>>>>> new phase to compute constant path regions using points-to.
    }

  pips_debug(8, "end : r1 is %s a predecessor of r2 (%s exact)\n", res ? "":"not", *exact_p ? "":"not"); 
  return res;
}

void simple_reference_to_convex_reference_conversion(reference ref, reference * output_ref, descriptor * output_desc)
{

  effect reg = make_effect(make_cell(is_cell_reference, make_reference(reference_variable(ref), NIL)),
			   make_action_write(), make_approximation_must(),
			   make_descriptor(is_descriptor_convex, sc_new()));

  FOREACH(EXPRESSION, exp, reference_indices(ref))
    {
      if((expression_reference_p(exp) && entity_field_p(reference_variable(expression_reference(exp)))))
	{
	  entity e = reference_variable(expression_reference(exp));
	  effect_add_field_dimension(reg, e);
	}
      else
	convex_region_add_expression_dimension(reg, exp);
    }
  *output_ref = effect_any_reference(reg);
  *output_desc = effect_descriptor(reg);

  pips_debug_effect(6, "reg = \n", reg);

  cell_reference(effect_cell(reg)) = reference_undefined;
  effect_descriptor(reg) = descriptor_undefined;
  free_effect(reg);
}

/*
  @param c is a the convex cell for which we look an equivalent constant path
  @param ptl is the list of points-to in which we search for constant paths
  @param  exact_p is a pointer towards a boolean. It is set to true if
         the result is exact, and to false if it is an approximation, 
	 either because the matching points-to sources found in ptl are 
	 over-approximations of the preceding path of input_ref or because 
	 the matching points-to have MAY approximation tag.
  @return a list of constant path effects. It is a list because at a given
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

  This function is called by effects to see if a convex memory access path
  can be transformed into a constant one. 
*/
<<<<<<< HEAD
list eval_convex_cell_with_points_to(cell c, descriptor d, list ptl, bool *exact_p, transformer current_precondition)
{
  
  return generic_eval_cell_with_points_to(c, d, ptl, exact_p, current_precondition,
=======
list eval_convex_cell_with_points_to(cell c, descriptor d, list ptl, bool *exact_p)
{
  
  return generic_eval_cell_with_points_to(c, d, ptl, exact_p,
>>>>>>> new phase to compute constant path regions using points-to.
					  convex_cell_reference_preceding_p,
					  convex_cell_reference_with_address_of_cell_reference_translation,
					  simple_reference_to_convex_reference_conversion);
}


