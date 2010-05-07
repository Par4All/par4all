#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "properties.h"
#include "preprocessor.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "alias-classes.h"

/*
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
  A subcase of eval_cell_with_points_to(). It takes a reference and a
  list of points-to and try to evaluate the reference. Each time we
  dereference we test if it's a constant one by exploiting the
  points-to information.

  we dereference and eval until either found a constant one so return a
  list of reference to wich points r or return anywhere:anywhere

  it is not clear if the decision to replace a dereferencement by a
  reference to anywhere should be taken here or not. I would rather
  postpone it as it depends on write effects unknown from this function.
*/
/*
  BC : try to find in the points-to list a points-to with a maximal length source path but 
       shorter than the length of the input path. If it's an exact points-to, that's OK, if not
       we have to find other possible paths. I don't know yet if we must iterate the process 
       with found paths, because I don't know if points-to sinks can be non-constant paths.
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

  pips_debug(8, "input reference  : %s\n input points_to :",
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
      l = CONS(CELL, make_cell(is_cell_reference, make_reference(input_ent, NIL)), NIL);
      *exact_p = false;
    }
  else
    {
      if (input_path_length == 0) /* should this case arise ? */
	{
	  l = NIL;
	  *exact_p = true;
	}
      else 
	{
	  size_t current_max_path_length = 0;
	  /* first build a temporary list with matching points-to of maximum length */
	  list matching_list = NIL;
	  *exact_p = true; /* assume exactness */
	  while (!ENDP(ptl))
	    {
	      points_to pt = POINTS_TO(CAR(ptl));
	      cell source_cell = points_to_source(pt);
	      reference source_ref = reference_undefined;
	      
	      if (cell_reference_p(source_cell))
		source_ref = cell_reference(source_cell);
	      else if (cell_preference_p(source_cell))
		source_ref = preference_reference(cell_preference(source_cell));
	      else
		pips_internal_error("GAP case not implemented yet\n");

	      list source_indices = reference_indices(source_ref);	      
	      size_t source_path_length = gen_length(source_indices);
	      bool exact_prec = false;

	      if ( (source_path_length >= current_max_path_length)
		   && cell_reference_predeceding_p(source_ref, input_ref, &exact_prec))
		{
		  
		  if (source_path_length > current_max_path_length )
		    {
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
	      
	      POP(ptl);
	    } /* end of while */

	  /* Then build the return list with the points-to sinks to which we add/append the indices of input_ref 
	     which are not in the source reference 
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
	      
	      if (cell_reference_p(sink_cell))
		sink_ref = cell_reference(sink_cell);
	      else if (cell_preference_p(sink_cell))
		sink_ref = preference_reference(cell_preference(sink_cell));
	      else
		pips_internal_error("GAP case not implemented yet\n");

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
	      *exact_p = *exact_p && approximation_exact_p(points_to_approximation(pt));
	    } /* end of FOREACH(POINTS_TO,...) */
	} /* else branche of if (input_path_length == 0) */
    } /* else branch of if (entity_abstract_location_p(input_ent)) */
  

  

  return l;
}

/*
  @param c is a the cell for which we look a constant path
  @param ptl is the list of points-to in which we search for a constant path
  @return a list of constant path cells. It is a list because at a given
          program point the cell may correspond to several constant paths.


  goal: see if cell c can be shortened by replacing its indirections
  by their values when they are defined in ptl. For instance, p[0][0]
  and (p,q,EXACT) is reduced to q[0]. And if (q,i,EXACT) is also
  available, the reference is reduced to i. The reduced cell is less likely
  to be invalidated by a write effect. The function is called "eval"
  because its goal is to build an as constant as possible reference or
  gap.

  For the time being this function is never called...

  It should be called by effect to see if a memory access path can be
  transformed into a constant, and by the points-to analysis to see if
  a sink or a source can be preserved in spite of write effects. This
  function should be called before points_to_filter_effects() to
  reduce the number of anywhere locations generated.

  BC : I'm not yet sure, but we may also need an exactness information to keep exactness at the effect level.
*/
list eval_cell_with_points_to(cell c, list ptl, bool *exact_p)
{
  list l = NIL;

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

  return l;
}



