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
#include "effects-generic.h"

/*

*/



/*
  @brief tries to find in the points-to list ptl a points-to with a
       maximal length source path but shorter than the length of the
       input path represented by input_ref. If it's an exact
       points-to, that's OK, if not we have to find other possible
       paths. 
       
       There is no sharing between the returned list cells and the
       input reference.
       
       When the input reference contains no dereferencing dimension
       the result is an empty list. The caller should check this
       case before calling the function because an empty list when there 
       is a dereferencing dimension means that no target has been found
       and that the input path may point to anywhere.


  @param c is a the cell for which we look an equivalent constant path
  @param ptl is the list of points-to in which we search for constant paths
  @param  exact_p is a pointer towards a boolean. It is set to true if
         the result is exact, and to false if it is an approximation, 
	 either because the matching points-to sources found in ptl are 
	 over-approximations of the preceding path of the input cell or because 
	 the matching points-to have MAY approximation tag.
  @return a list of constant path effect. It is a list because at a given
          program point the cell may correspond to several constant paths.

  From the original comment (FI): it is not clear if the decision to
  replace a dereferencement by a reference to anywhere should be taken
  here or not. I would rather postpone it as it depends on write
  effects unknown from this function.

  Reply (BC): the original function used to return a single
  reference. It now returns a list of cell. It's up to the caller to
  take the decision to turn this list into an anywhere effect or not,
  depending on the context.

  original comment:	  
  goal: see if cell c can be shortened by replacing its indirections
  by their values when they are defined in ptl. For instance, p[0][0]
  and (p,q,EXACT) is reduced to q[0]. And if (q,i,EXACT) is also
  available, the reference is reduced to i. The reduced cell is less likely
  to be invalidated by a write effect. The function is called "eval"
  because its goal is to build an as constant as possible reference or
  gap.

  This function is called by effects to see if a memory access path
  can be transformed into a constant. It should also be called by the
  points-to analysis to see if a sink or a source can be preserved in
  spite of write effects. This function should be called before
  points_to_filter_effects() to reduce the number of anywhere
  locations generated.


 */

list generic_eval_cell_with_points_to(cell input_cell, descriptor input_desc, list ptl, bool *exact_p,
				      transformer current_precondition,
				      bool (*cell_reference_preceding_p_func)(reference, descriptor,
									      reference, descriptor ,
									      transformer, bool * ),
				      void (*cell_reference_with_address_of_cell_reference_translation_func)(reference, descriptor,
													    reference, descriptor,
													    int,
													    reference *, descriptor *,
													     bool *),
				      void (*cell_reference_conversion_func)(reference, reference *, descriptor *))
{
  debug_on("EVAL_CELL_WITH_POINTS_TO_DEBUG_LEVEL");
  reference input_ref = reference_undefined;
  if(cell_reference_p(input_cell)) {
    input_ref = cell_reference(input_cell);
  }
  else if(cell_preference_p(input_cell)) {
    input_ref = preference_reference(cell_preference(input_cell));
  }
  else { /* Should be the gap case */
    pips_internal_error("GAPs not implemented yet\n");
  }

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
	      reference source_ref = reference_undefined, tmp_ref = reference_undefined;
	      descriptor source_desc = descriptor_undefined;
	      if (cell_reference_p(source_cell))
		tmp_ref = cell_reference(source_cell);
	      else if (cell_preference_p(source_cell))
		tmp_ref = preference_reference(cell_preference(source_cell));
	      else
		pips_internal_error("GAP case not implemented yet\n");

	      (*cell_reference_conversion_func)(tmp_ref, &source_ref, &source_desc);
	     

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
		   && (*cell_reference_preceding_p_func)(source_ref, source_desc, input_ref, input_desc, current_precondition, &exact_prec))
		{		  
		  pips_debug(8, "exact_prec is %s\n", exact_prec? "true":"false");
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
	      if(source_cell != points_to_source(pt)) free_cell(source_cell);
	    } /* end of FOREACH */

	     
	  ifdebug(8)
	    {
	      points_to_list ptll = make_points_to_list(matching_list);
	      fprintf(stderr, "matching points-to list %s\n", 
		      words_to_string(words_points_to_list("", ptll)));
	      points_to_list_list(ptll) = NIL;
	      free_points_to_list(ptll);
	      fprintf(stderr,"current_max_path_length = %d\n", (int) current_max_path_length);
	      fprintf(stderr, "*exact_p is %s\n", *exact_p? "true":"false");
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
	    
	  /* Transform each sink reference to add it in the return list */
	  FOREACH(POINTS_TO, pt, matching_list)
	    {
	      cell sink_cell = points_to_sink(pt);
	      reference sink_ref = reference_undefined, tmp_ref = reference_undefined;
	      descriptor sink_desc = descriptor_undefined;

	      if (cell_reference_p(sink_cell))
		tmp_ref = cell_reference(sink_cell);
	      else if (cell_preference_p(sink_cell))
		tmp_ref = preference_reference(cell_preference(sink_cell));
	      else
		pips_internal_error("GAP case not implemented yet\n");

	      (*cell_reference_conversion_func)(tmp_ref, &sink_ref, &sink_desc);
	      reference build_ref = reference_undefined;
	      
	      pips_debug(8, "considering point-to  : %s\n ",
			 words_to_string(word_points_to(pt)));

	      entity sink_ent = reference_variable(sink_ref);
	      if (entity_abstract_location_p(sink_ent) 
		  && ! entity_flow_or_context_sentitive_heap_location_p(sink_ent))
		{
		  /* Here, we should analyse the source remaining indices to know if there are remaining dereferencing dimensions.
		     This would imply keeping track of the indices types.
		     In case there are no more dereferencing dimensions, we would then reuse the sink abstract location.
		     Otherwise (which is presently the only default case), we return an all location cell
		  */
		  l = CONS(EFFECT, make_anywhere_effect(is_action_write), l);
		  *exact_p = false;
		}
	      else
		{
	      
		  descriptor build_desc = descriptor_undefined;
		  bool exact_translation_p;
		  
		  (*cell_reference_with_address_of_cell_reference_translation_func)(input_ref, input_desc, 
										   sink_ref, sink_desc,
										   current_max_path_length,
										   &build_ref, &build_desc, 
										   &exact_translation_p);
		  *exact_p = *exact_p && exact_translation_p; 
		  /* the approximation tag of the points-to is taken into account for the exactness of the result  */
		  *exact_p = *exact_p && approximation_exact_p(points_to_approximation(pt));
		  pips_debug(8, "adding reference %s\n",
			     words_to_string(words_reference(build_ref, NIL)));
		  l = CONS(EFFECT, make_effect(make_cell(is_cell_reference, build_ref),
					       make_action_write(),
					       make_approximation(*exact_p? is_approximation_must : is_approximation_may, UU),
					       build_desc), l);
		  		  
		} /* end of else branch of if (entity_abstract_location_p(sink_ent) 
		     && ! entity_flow_or_context_sentitive_heap_location_p(sink_ent)) */
	      if(sink_cell != points_to_sink(pt)) free_cell(sink_cell);
	    } /* end of FOREACH(POINTS_TO,...) */
	} /* else branche of if (input_path_length == 0) */
    } /* else branch of if (entity_abstract_location_p(input_ent)) */

  pips_debug_effects(8, "resulting list before recursing:", l);

  /* If the results contain dereferencing dimensions, we must eval them recursively */
  list l_tmp = l;
  l = NIL;
  FOREACH(EFFECT, eff, l_tmp)
    {
      bool r_exact_p;
      reference ref = effect_any_reference(eff);
     
      if ((!entity_abstract_location_p(reference_variable(ref)))
	  && effect_reference_dereferencing_p(ref, &r_exact_p))
	{
	  pips_debug(8, "recursing\n");
	  list l_eval = generic_eval_cell_with_points_to(effect_cell(eff), effect_descriptor(eff), ptl, &r_exact_p,
							current_precondition,
							 cell_reference_preceding_p_func,
							cell_reference_with_address_of_cell_reference_translation_func,
							cell_reference_conversion_func);
	  *exact_p = *exact_p && r_exact_p;
	  /* if eff is a may effect, the resulting effects must be may effects */
	  if (effect_may_p(eff) || !(*exact_p))
	    effects_to_may_effects(l_eval);
	  l = gen_nconc(l_eval, l);
	  free_effect(eff);
	}
      else 
	{
	  pips_debug(8, "no need to recurse\n");
	  l = CONS(EFFECT, eff, l);
	}
    }
  l = gen_nreverse(l);
  gen_free_list(l_tmp);

  pips_debug_effects(8, "resulting list after recursing:", l);

  debug_off();
  
  return l;
}



