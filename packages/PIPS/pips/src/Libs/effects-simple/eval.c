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

#include "effects-simple.h"

/*
  @param r1 and r2 are two path references
  @param exact_p is a pointer towards a boolean, which is set to false
         is the result is an over-approximation, true if it's an exact answer.
  @return true if r1 path may be a predecessor of r2 path

  (we consider that p[1] is not a predecessor of p[*], with *exact_p = false.)

  should be move elsewhere. There should be a cell library
 */
bool cell_reference_predeceding_p(reference r1, reference r2, bool * exact_p)
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
	  
	  if (unbounded_expression_p(exp1))
	    *exact_p = false;
	  else if (unbounded_expression_p(exp2))
	    {
	      res = false;
	      *exact_p = false;
	    }	    
	  else if(!expression_equal_p(exp1, exp2)) /*would it be more efficient to compare strings ?*/
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
      *exact_p = false;
    }

  pips_debug(8, "end : r1 is %s a predecessor of r2 (%s exact)\n", res ? "":"not", *exact_p ? "":"not"); 
  return res;
}


/*
  @brief tries to find in the points-to list ptl a points-to with a
       maximal length source path but shorter than the length of the
       input path represetned by input_ref. If it's an exact
       points-to, that's OK, if not we have to find other possible
       paths. (I don't know yet if we must iterate the process with
       found paths as suggested in the original comment of this
       function), because I don't know if points-to sinks can be
       non-constant paths.  I presently assume that all sinks are
       constant paths.
       
       There is no sharing between the returned list cells and the
       input reference.
       
       When the input reference contains no dereferencing dimension
       the result is an empty list. The caller should check this
       case before calling the function because an empty list when there 
       is a dereferencing dimension means that no target has been found
       and that the input path may point to anywhere.

  @param input_ref is a reference representing a memory access path.
  @param ptl is a list of points-to
  @param exact_p is a pointer towards a boolean. It is set to true if
         the result is exact, and to false if it is an approximation, 
	 either because the matching points-to sources found in ptl are 
	 over-approximations of the preceding path of input_ref or because 
	 the matching points-to have MAY approximation tag.

  From the original comment (FI): it is not clear if the decision to
  replace a dereferencement by a reference to anywhere should be taken
  here or not. I would rather postpone it as it depends on write
  effects unknown from this function.

  Reply (BC): the original function used to return a single
  reference. It now returns a list of cell. It's up to the caller to
  take the decision to turn this list into an anywhere effect or not,
  depending on the context.

 */

list eval_reference_with_points_to(reference input_ref, list ptl, bool *exact_p)
{
/* iterer sur le path p[0][1][2][0] et tester chaque fois si on peut
 * dereferencer le pointeur*/
  entity input_ent = reference_variable(input_ref);
  list input_indices = reference_indices(input_ref);
  size_t input_path_length = gen_length(input_indices);
  list l = NIL;
  list matching_ptl = NIL;

  pips_debug(8, "input reference  : %s\n",
	words_to_string(words_reference(input_ref, NIL)));
  ifdebug(8)
    {
      points_to_list ptll = make_points_to_list(ptl);
      fprintf(stderr, "%s\n", words_to_string(words_points_to_list("", ptll)));
      points_to_list_list(ptll) = NIL;
      free_points_to_list(ptll);
      
    }

  if (entity_abstract_location_p(input_ent))
    {
      /* If the input path is an abstract location, the result is the same abstract location */
      l = CONS(CELL, make_cell(is_cell_reference, make_reference(input_ent, NIL)), NIL);
      *exact_p = false;
    }
  else
    {
      if (input_path_length == 0) 
	{
	  /* simple scalar case. I think this should not happen, because there is no dereferencing dimension. */
	  l = NIL;
	  *exact_p = true;
	}
      else 
	{
	  /* first build a temporary list with matching points-to of maximum length */
	  size_t current_max_path_length = 0; /* the current maximum length */
	  list matching_list = NIL;
	  *exact_p = true; /* assume exactness */
	  FOREACH(POINTS_TO, pt, ptl)
	    {
	      cell source_cell = points_to_source(pt);
	      reference source_ref = reference_undefined;
	      
	      if (cell_reference_p(source_cell))
		source_ref = cell_reference(source_cell);
	      else if (cell_preference_p(source_cell))
		source_ref = preference_reference(cell_preference(source_cell));
	      else
		pips_internal_error("GAP case not implemented yet\n");

	      pips_debug(8, "considering point-to  : %s\n ",
			 words_to_string(word_points_to(pt)));

	      list source_indices = reference_indices(source_ref);	      
	      size_t source_path_length = gen_length(source_indices);
	      bool exact_prec = false;

	      /* eligible points-to candidates must have a path length
	         greater or equal to the current maximum length and
	         their path must be a predecessor of the input_ref
	         path.*/
	      if ( (source_path_length >= current_max_path_length)
		   && cell_reference_predeceding_p(source_ref, input_ref, &exact_prec))
		{		  
		  if (source_path_length > current_max_path_length )
		    {
		      /* if the candidate has a path length strictly greater than the current maximum lenght,
		         the list of previously found matching candidates must be cleared
		      */
		      if(!ENDP(matching_ptl))
			{
			  gen_free_list(matching_ptl);
			  matching_list = NIL;
			}
		      current_max_path_length = source_path_length;
		      *exact_p = exact_prec;
		    }
		  else
		    *exact_p = *exact_p && exact_prec;
		  /* I keep the whole points_to and not only the sink because I will need the approximation to further test
		     the exactness
		  */
		  matching_list = CONS(POINTS_TO, pt, matching_list); 		  
		}
	    } /* end of FOREACH */

	     
	  ifdebug(8)
	    {
	      points_to_list ptll = make_points_to_list(matching_list);
	      fprintf(stderr, "matching points-to list %s\n", 
		      words_to_string(words_points_to_list("", ptll)));
	      points_to_list_list(ptll) = NIL;
	      free_points_to_list(ptll);
	      
	    }
	  

	  /* Then build the return list with the points-to sinks to which we add/append the indices of input_ref 
	     which are not in the points-to source reference. This is comparable to an interprocedural translation
	     where the input_ref is a path from the formal parameter, the source the formal parameter and the
	     sink the actual parameter preceded by a & operator.
	     
	     Let the remaining indices be the indices from input_ref which are not in common with the sources of the points-to
	     (which by construction have all the same path length).
	     For each points-to, the new path is built from the sink reference. If the latter has indices, we add to 
	     the value of its last index the value of the first index of the remaining indices. Then we append to the new path 
	     the last indices of the reamining indices. If the sink reference has no indices, then the first index of the
	     remaining indices must be equal to zero, and be skipped. And we simply append to the new path 
	     the last indices of the reamining indices.

	     The part of the code inside of the FOREACH could certainely be shared with the interprocedural translation.
	     Besides, it should be made generic to handle convex input paths as well. Only simple paths are currently 
	     taken into account.
	  */
	  /* first compute the list of additional indices of input_ref */
	  for(int i = 0; i < (int) current_max_path_length; i++, POP(input_indices));
	    
	  /* Then transform each sink reference to add it in the return list */
	  FOREACH(POINTS_TO, pt, matching_list)
	    {
	      cell sink_cell = points_to_sink(pt);
	      reference sink_ref = reference_undefined;
	      reference build_ref = reference_undefined;
	      list build_indices = NIL;
	      
	      pips_debug(8, "considering point-to  : %s\n ",
			 words_to_string(word_points_to(pt)));

	      if (cell_reference_p(sink_cell))
		sink_ref = cell_reference(sink_cell);
	      else if (cell_preference_p(sink_cell))
		sink_ref = preference_reference(cell_preference(sink_cell));
	      else
		pips_internal_error("GAP case not implemented yet\n");

	      entity sink_ent = reference_variable(sink_ref);
	      if (entity_abstract_location_p(sink_ent) 
		  && ! entity_flow_or_context_sentitive_heap_location_p(sink_ent))
		{
		  /* Here, we should analyse the source remaining indices to know if there are remaining dereferencing dimensions.
		     This would imply keeping track of the indices types.
		     In case there are no more dereferencing dimensions, we would then reuse the sink abstract location.
		     Otherwise (which is presently the only default case), we return an all location cell
		  */
		  build_ref = make_reference(entity_all_locations(), NIL);
		  *exact_p = false;
		}
	      else
		{
	      

		  /* from here it's not generic, that is to say it does not work for convex effect references */
		  list input_remaining_indices = input_indices;
		  
		  build_ref = copy_reference(sink_ref);
		  build_indices = gen_last(reference_indices(build_ref));
		  
		  /* special case for the first remaning index: we must add it to the last index of build_ref */
		  if (!ENDP(build_indices))
		    {
		      expression last_build_indices_exp = EXPRESSION(CAR(build_indices)); 
		      expression first_input_remaining_exp = EXPRESSION(CAR(input_remaining_indices));
		      expression new_exp = expression_undefined;
		      /* adapted from the address_of case of c_simple_effects_on_formal_parameter_backward_translation 
			 this should maybe be put in another function
		      */
		      if(!unbounded_expression_p(last_build_indices_exp))
			{
			  if (expression_reference_p(last_build_indices_exp) && 
			      entity_field_p(expression_variable(last_build_indices_exp)))
			    {
			      if (!expression_equal_integer_p(first_input_remaining_exp, 0))
				{
				  pips_internal_error("potential memory overflow: should have been converted to anywhere before \n");
				}	
			      else
				new_exp = last_build_indices_exp;
			    }
			    
			  else if(!unbounded_expression_p(first_input_remaining_exp))
			    {
				
			      value v;
			      new_exp = MakeBinaryCall
				(entity_intrinsic(PLUS_OPERATOR_NAME),
				 copy_expression(last_build_indices_exp), copy_expression(first_input_remaining_exp));
			      /* Then we must try to evaluate the expression */
			      v = EvalExpression(new_exp);
			      if (! value_undefined_p(v) && 
				  value_constant_p(v))
				{
				  constant vc = value_constant(v);
				  if (constant_int_p(vc))
				    {				    
				      free_expression(new_exp);
				      new_exp = int_to_expression(constant_int(vc));
				    }
				}
			    }
			  else
			    {
			      new_exp = make_unbounded_expression();
			      *exact_p = false;
			    }
			  CAR(gen_last(reference_indices(build_ref))).p 
			    = (void *) new_exp;
			}
		      else
			{
			  *exact_p = false;
			}
		    }
		  else /* ENDP(build_indices) */
		    {
		      /* The sink is a scalar: the first remaning index must be equal to 0 */
		      expression first_input_remaining_exp = EXPRESSION(CAR(input_remaining_indices));
		      if (!expression_equal_integer_p(first_input_remaining_exp, 0))
			pips_internal_error("potential memory overflow: should have been converted to anywhere before \n");
		    }
	      
		  FOREACH(EXPRESSION, input_ind, CDR(input_remaining_indices))
		    {
		      reference_indices(build_ref) = gen_nconc(reference_indices(build_ref),
							       CONS(EXPRESSION, 
								    copy_expression(input_ind), 
								    NIL));
		    }
		  pips_debug(8, "adding reference %s\n",
			     words_to_string(words_reference(build_ref, NIL)));
		  l = CONS(CELL, make_cell(is_cell_reference, build_ref), l);
		  /* the approximation tag of the points-to is taken into account for the exactness of the result  */
		  *exact_p = *exact_p && approximation_exact_p(points_to_approximation(pt));
		} /* end of else branch of if (entity_abstract_location_p(sink_ent) 
		     && ! entity_flow_or_context_sentitive_heap_location_p(sink_ent)) */
	    } /* end of FOREACH(POINTS_TO,...) */
	} /* else branche of if (input_path_length == 0) */
    } /* else branch of if (entity_abstract_location_p(input_ent)) */

  ifdebug(8)
    {
      pips_debug(8, "resulting list before recursing:");
      FOREACH(CELL, c, l)
	{ 
	  reference ref = reference_undefined;
	  if(cell_reference_p(c)) 
	    ref = cell_reference(c);
	  else if (cell_preference_p(c))
	    ref = preference_reference(cell_preference(c));
	  else /* Should be the gap case */
	    pips_internal_error("GAPs not implemented yet\n");
	  fprintf(stderr, " %s", words_to_string(words_reference(ref, NIL)));
	}
      fprintf(stderr, "\n");
    }

  /* If the results contain dereferencing dimensions, we must eval them recursively */
  list l_tmp = l;
  l = NIL;
  FOREACH(CELL, c, l_tmp)
    {
      bool r_exact_p;
      reference ref = reference_undefined;
      if(cell_reference_p(c)) 
	ref = cell_reference(c);
      else if (cell_preference_p(c))
	ref = preference_reference(cell_preference(c));
      else /* Should be the gap case */
	pips_internal_error("GAPs not implemented yet\n");
      
      if ((!entity_abstract_location_p(reference_variable(ref)))
	  && effect_reference_dereferencing_p(ref, &r_exact_p))
	{
	  pips_debug(8, "recursing\n");
	  *exact_p = *exact_p && r_exact_p;
	  l = gen_nconc(eval_reference_with_points_to(ref, ptl, &r_exact_p), l);
	  *exact_p = *exact_p && r_exact_p;
	  free_cell(c);
	}
      else 
	{
	  pips_debug(8, "no need to recurse\n");
	  l = CONS(CELL, c, l);
	}
    }
  l = gen_nreverse(l);
  gen_free_list(l_tmp);

  ifdebug(8)
    {
      pips_debug(8, "resulting list after recursing:");
      FOREACH(CELL, c, l)
	{ 
	  reference ref = reference_undefined;
	  if(cell_reference_p(c)) 
	    ref = cell_reference(c);
	  else if (cell_preference_p(c))
	    ref = preference_reference(cell_preference(c));
	  else /* Should be the gap case */
	    pips_internal_error("GAPs not implemented yet\n");
	  fprintf(stderr, " %s", words_to_string(words_reference(ref, NIL)));
	}
      fprintf(stderr, "\n");
    }
  
  return l;
}

/*
  @param c is a the cell for which we look an equivalent constant path
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
list eval_cell_with_points_to(cell c, list ptl, bool *exact_p)
{
  list l = NIL;
  debug_on("EVAL_CELL_WITH_POINTS_TO_DEBUG_LEVEL");

  if(cell_reference_p(c)) {
    l = eval_reference_with_points_to(cell_reference(c), ptl, exact_p);
  }
  else if(cell_preference_p(c)) {
    reference r = preference_reference(cell_preference(c));
    l = eval_reference_with_points_to(r, ptl, exact_p);
  }
  else { /* Should be the gap case */
    pips_internal_error("GAPs not implemented yet\n");
  }
  debug_off();
  return l;
}



