/*

  $Id$

  Copyright 1989-2014 MINES ParisTech
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

/* In case, the points-to information is not complete, use anywhere
 * locations to convert the reference.
 *
 * FI: I do not understand why it is always a write effect. I do not
 * understand why the type should always be lost.
 *
 */
static list use_default_sink_cell
(reference input_ref __attribute__ ((__unused__)),
 descriptor input_desc __attribute__ ((__unused__)),
 void (*cell_reference_with_address_of_cell_reference_translation_func)
 (reference, descriptor, reference, descriptor, int, reference *, descriptor *,
  bool *) __attribute__ ((__unused__)),
 void (*cell_reference_conversion_func)(reference, reference *, descriptor *) __attribute__ ((__unused__))
 )
{
  // FI: the anywhere effects could be typed
  // type t = points_to_reference_to_concrete_type(input_ref);
  list l = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), NIL);
  return l;
}

/* Build the return list with the points-to sinks to which we
 * add/append the indices of input_ref which are not in the points-to
 * source reference. This is comparable to an interprocedural
 * translation where the input_ref is a path from the formal
 * parameter, the source the formal parameter and the sink the actual
 * parameter preceded by a & operator.
 *
 * Let the remaining indices be the indices from input_ref which are
 * not in common with the sources of the points-to (which by
 * construction have all the same path length).  For each points-to,
 * the new path is built from the sink reference. If the latter has
 * indices, we add to the value of its last index the value of the
 * first index of the remaining indices. Then we append to the new
 * path the last indices of the reamining indices. If the sink
 * reference has no indices, then the first index of the remaining
 * indices must be equal to zero, and be skipped. And we simply append
 * to the new path the last indices of the remaining indices.
 *
 * The part of the code inside of the FOREACH could certainely be
 * shared with the interprocedural translation.  Besides, it should be
 * made generic to handle convex input paths as well. Only simple
 * paths are currently taken into account.
 *
 * This function has been outlined from
 * generic_eval_cell_with_points_to() to reduce its size to about one
 * page.
 */
static list generic_transform_sink_cells_from_matching_list
(list matching_list,
 size_t current_max_path_length,
 bool * exact_p,
 reference input_ref,
 descriptor input_desc,
 void (*cell_reference_with_address_of_cell_reference_translation_func)
 (reference, descriptor, reference, descriptor, int, reference *, descriptor *,
  bool *),
 void (*cell_reference_conversion_func)(reference, reference *, descriptor *)
 )
{
  list l = NIL; // returned list of write effects based on the transformed sink_cells

  /* Transform each sink reference to add it in the return list */
  FOREACH(POINTS_TO, pt, matching_list) {
    cell sink_cell = points_to_sink(pt);
    reference sink_ref = reference_undefined;
    reference tmp_ref = cell_to_reference(sink_cell);
    descriptor sink_desc = make_descriptor(is_descriptor_none,UU);

    (*cell_reference_conversion_func)(tmp_ref, &sink_ref, &sink_desc);
    reference build_ref = reference_undefined;

    pips_debug(8, "considering point-to  : %s\n ", words_to_string(word_points_to(pt)));

    entity sink_ent = reference_variable(sink_ref);
    if (entity_abstract_location_p(sink_ent)
	&& ! entity_flow_or_context_sentitive_heap_location_p(sink_ent)) {
      /* Here, we should analyse the source remaining indices to know
	 if there are remaining dereferencing dimensions.  This would
	 imply keeping track of the indices types.  In case there are
	 no more dereferencing dimensions, we would then reuse the
	 sink abstract location.  Otherwise (which is presently the
	 only default case), we return an all location cell
      */
      if (entity_null_locations_p(sink_ent)
	  && approximation_may_p(points_to_approximation(pt))) {
	pips_debug(5, "Null pointer, may approximation: ignore, assuming code is correct\n");
      }
      else {
	l = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), l);
	*exact_p = false;
      }
    }
    else {
      descriptor build_desc = make_descriptor(is_descriptor_none,UU);
      bool exact_translation_p;

      (*cell_reference_with_address_of_cell_reference_translation_func)
	(input_ref, input_desc,
	 sink_ref, sink_desc,
	 current_max_path_length,
	 &build_ref, &build_desc,
	 &exact_translation_p);
      *exact_p = *exact_p && exact_translation_p;
      /* the approximation tag of the points-to is taken into account
	 for the exactness of the result except if the matching list
	 has been reduced to one element and if the target is
	 atomic. */
      int mll = (int) gen_length(matching_list); // matching list length
      if(mll==1) {
	cell sink_c = points_to_sink(pt);
	reference sink_c_r = cell_any_reference(sink_c);
	*exact_p = *exact_p && generic_atomic_points_to_reference_p(sink_c_r, false);
      }
      else {
	/* It should be false as soon as mll>1... */
	*exact_p = *exact_p && approximation_exact_p(points_to_approximation(pt));
      }
      pips_debug(8, "adding reference %s\n",
		 effect_reference_to_string(build_ref));
      l = CONS(EFFECT, make_effect(make_cell(is_cell_reference, build_ref),
				   make_action_write_memory(),
				   make_approximation(*exact_p? is_approximation_exact : is_approximation_may, UU),
				   build_desc), l);

    } /* end of else branch of if (entity_abstract_location_p(sink_ent)
	 && ! entity_flow_or_context_sentitive_heap_location_p(sink_ent)) */
    if(sink_cell != points_to_sink(pt)) free_cell(sink_cell);
  } /* end of FOREACH(POINTS_TO,...) */
  return l;
}

/* To provide information when a but is encountered in the source file or within PIPS. */
int effects_statement_line_number(void)
{
  statement s = effects_private_current_stmt_head();
  return statement_number(s);
}

/* This function has been outlined from
 * generic_eval_cell_with_points_to() to reduce the size of a function
 * to about one page.
 *
 * It computes a list of points-to arcs whose source is compatible
 * with the input reference, "input_ref". It provides information about
 * the number of common indices, "p_current_max_path_length" and about the
 * approximation of the points-to informationx(?).
 */
list generic_reference_to_points_to_matching_list
(reference input_ref,
 descriptor input_desc,
 size_t * p_current_max_path_length,
 bool * exact_p,
 transformer current_precondition,
 list ptl,
 void (*cell_reference_conversion_func)(reference, reference *, descriptor *),
 bool (*cell_reference_preceding_p_func)(reference, descriptor,
					 reference, descriptor ,
					 transformer, bool, bool * )
 )
{
  list matching_list = NIL;
  *exact_p = true; /* assume exactness */
  // previous matching list I guess: the initial code looks wrong
  // since matching_ptl is itialized to NIL...
  // list matching_ptl = NIL;

  FOREACH(POINTS_TO, pt, ptl) {
    cell sink_cell = points_to_sink(pt);
    if(null_cell_p(sink_cell)  || nowhere_cell_p(sink_cell))
      ; // This points-to arc can be ignored since it would lead to a segfault
    else {
      cell source_cell = points_to_source(pt);
      reference source_ref = reference_undefined;
      reference tmp_ref = cell_to_reference(source_cell);
      descriptor source_desc = make_descriptor(is_descriptor_none,UU);

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
      if ( (source_path_length >= *p_current_max_path_length)
	   && (*cell_reference_preceding_p_func)(source_ref, source_desc, input_ref,
						 input_desc, current_precondition, true,
						 &exact_prec)) {
	pips_debug(8, "exact_prec is %s\n", exact_prec? "true":"false");
	if (source_path_length > *p_current_max_path_length ) {
	  /* if the candidate has a path length strictly greater
	   * than the current maximum length, the list of
	   * previously found matching candidates must be cleared
	   */
	  //if(!ENDP(matching_ptl)) {
	  //	  gen_free_list(matching_ptl);
	  //	  matching_list = NIL;
	  //}
	  if(!ENDP(matching_list)) {
	    gen_free_list(matching_list);
	    matching_list = NIL;
	  }
	  *p_current_max_path_length = source_path_length;
	  *exact_p = exact_prec;
	}
	else
	  *exact_p = *exact_p && exact_prec;
	/* I keep the whole points_to and not only the sink because I
	   will need the approximation to further test the exactness
	*/
	/* FI: I try adding the stripping... */
	points_to npt = points_to_with_stripped_sink(pt, effects_statement_line_number);
	matching_list = CONS(POINTS_TO, npt, matching_list);
      }
      if(source_cell != points_to_source(pt)) free_cell(source_cell);
    }
  } /* end of FOREACH */
  return matching_list;
}

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

  @param exact_p is a pointer towards a boolean. It is set to true if
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

list generic_eval_cell_with_points_to(
  cell input_cell, descriptor input_desc, list ptl, bool *exact_p,
  transformer current_precondition,
  bool (*cell_reference_preceding_p_func)(reference, descriptor,
                                          reference, descriptor ,
                                          transformer, bool, bool * ),
  void (*cell_reference_with_address_of_cell_reference_translation_func)
      (reference, descriptor,
       reference, descriptor,
       int,
       reference *, descriptor *,
       bool *),
  void (*cell_reference_conversion_func)(reference, reference *, descriptor *))
{
  debug_on("EVAL_CELL_WITH_POINTS_TO_DEBUG_LEVEL");
  reference input_ref = cell_to_reference(input_cell);

  /* iterer sur le path p[0][1][2][0] et tester chaque fois si on peut
   * dereferencer le pointeur*/
  entity input_ent = reference_variable(input_ref);
  list input_indices = reference_indices(input_ref);
  size_t input_path_length = gen_length(input_indices);
  list l = NIL; // result of this function: a list of effects
  char * effect_reference_to_string(reference);
  void print_points_to_relations(list);

  pips_debug(8, "input reference  : %s\n", effect_reference_to_string(input_ref));
  ifdebug(8) print_points_to_relations(ptl);

  if (entity_abstract_location_p(input_ent)) {
    /* If the input path is an abstract location, the result is the same abstract location */
    /*  l = CONS(CELL, make_cell(is_cell_reference, make_reference(input_ent, NIL)), NIL); */
    l = CONS(EFFECT, make_reference_simple_effect( make_reference(input_ent, NIL),
						   make_action_write_memory(),
						   make_approximation(is_approximation_may, UU)),
	     NIL);
    *exact_p = false;
  }
  else if (input_path_length == 0) {
    /* simple scalar case. I think this should not happen, because there is no dereferencing dimension. */
    l = NIL;
    *exact_p = true;
  }
  else {
    /* first build a temporary list with matching points-to of maximum length */
    size_t current_max_path_length = 0; /* the current maximum length */
    list matching_list = generic_reference_to_points_to_matching_list
      (input_ref, input_desc,
       &current_max_path_length,
       exact_p, current_precondition, ptl,
       cell_reference_conversion_func,
       cell_reference_preceding_p_func);

    ifdebug(8) {
      fprintf(stderr, "matching points-to list:\n");
      print_points_to_relations(matching_list);
      fprintf(stderr,"\ncurrent_max_path_length = %d\n", (int) current_max_path_length);
      fprintf(stderr, "*exact_p is %s\n", *exact_p? "true":"false");
    }

    if(ENDP(matching_list)) {
      pips_user_warning
	("NULL pointer dereferencing... or insufficient points-to information for reference \"%s\".\n",
	 reference_to_string(input_ref));
      l = use_default_sink_cell(input_ref, input_desc,
       cell_reference_with_address_of_cell_reference_translation_func,
       cell_reference_conversion_func);
      *exact_p = false;
    }
    else {
    l = generic_transform_sink_cells_from_matching_list
      (matching_list,
       current_max_path_length,
       exact_p,
       input_ref,
       input_desc,
       cell_reference_with_address_of_cell_reference_translation_func,
       cell_reference_conversion_func);
    }

  } /* else branche of if (input_path_length == 0) */

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



/**
   @brief find pointer_values in l_in which give (possible or exact) paths
          equivalent to eff.
   @param eff is the considered input path.
   @param l_in is the input pointer values list.
   @param exact_aliased_pv gives an exact equivalent path found in l_in if it exists.
   @param l_in_remnants contains the elemnts of l_in which are neither
          exact_aliased_pv nor in the returned list.

   @return a list of elements of l_in which give (possible or exact) paths
           equivalent to eff, excluding exact_aliased_pv if one exact equivalent
           path can be found in l_in.
 */
list
generic_effect_find_equivalent_simple_pointer_values(effect eff, list l_in,
					      cell_relation * exact_aliased_pv,
					      list * l_in_remnants,
					      bool (*cells_intersection_p_func)(cell, descriptor,
										cell, descriptor,
										bool *),
					      bool (*cells_inclusion_p_func)(cell, descriptor,
									     cell, descriptor,
									     bool*),
					      void (*simple_cell_conversion_func)(cell, cell *, descriptor *))
{

  pips_debug_pvs(1,"begin, l_in =", l_in);
  pips_debug_effect(1, "and eff:", eff);

  /* eff characteristics */
  cell eff_cell = effect_cell(eff);
  descriptor eff_desc = effect_descriptor(eff);
  /******/

 /* first, search for the (exact/possible) values of eff cell in l_in */
  /* we search for the cell_relations where ref appears
     as a first cell, or the exact value_of pointer_values where ref appears as
     a second cell. If an exact value_of relation is found, it is retained in
     exact_aliased_pv
  */
  *l_in_remnants = NIL;
  *exact_aliased_pv = cell_relation_undefined;
  list l_res = NIL;

  FOREACH(CELL_RELATION, pv_in, l_in)
    {
      cell first_cell_in = cell_relation_first_cell(pv_in);
      cell converted_first_cell_in = cell_undefined;
      descriptor first_cell_in_desc = descriptor_undefined;

      cell second_cell_in = cell_relation_second_cell(pv_in);
      cell converted_second_cell_in = cell_undefined;
      descriptor second_cell_in_desc = descriptor_undefined;

      bool intersection_test_exact_p = false;
      bool inclusion_test_exact_p = true;

      pips_debug_pv(4, "considering:", pv_in);

      (*simple_cell_conversion_func)(first_cell_in, &converted_first_cell_in, &first_cell_in_desc);
      (*simple_cell_conversion_func)(second_cell_in, &converted_second_cell_in, &second_cell_in_desc);

      if ((*cells_intersection_p_func)(eff_cell, eff_desc,
				  converted_first_cell_in, first_cell_in_desc,
				  &intersection_test_exact_p))
	{
	  pips_debug(4, "non empty intersection with first cell (%sexact)\n",
		     intersection_test_exact_p? "": "non ");
	  if (cell_relation_exact_p(pv_in)
	      && intersection_test_exact_p
	      && (*cells_inclusion_p_func)(eff_cell, eff_desc,
					   converted_first_cell_in, first_cell_in_desc,
					   &inclusion_test_exact_p)
	      && inclusion_test_exact_p)
	    {
	      if (cell_relation_undefined_p(*exact_aliased_pv))
		{
		  pips_debug(4, "exact value candidate found\n");
		  *exact_aliased_pv = pv_in;
		}
	      else if ((cell_relation_second_address_of_p(*exact_aliased_pv)
			&& cell_relation_second_value_of_p(pv_in))
		       || null_pointer_value_cell_p(cell_relation_second_cell(pv_in))
		       || undefined_pointer_value_cell_p(cell_relation_second_cell(pv_in)))
		{
		  pips_debug(4, "better exact value candidate found\n");
		  l_res = CONS(CELL_RELATION, *exact_aliased_pv, l_res);
		  *exact_aliased_pv = pv_in;
		}
	      else
		{
		  pips_debug(4, "not kept as exact candidate\n");
		  l_res = CONS(CELL_RELATION, pv_in, l_res);
		}
	    }
	  else
	    {
	      pips_debug(5, "potentially non exact value candidate found\n");
	      l_res = CONS(CELL_RELATION, pv_in, l_res);
	    }
	}
      else if(cell_relation_second_value_of_p(pv_in)
	      && (*cells_intersection_p_func)(eff_cell, eff_desc,
					     converted_second_cell_in, second_cell_in_desc,
					     &intersection_test_exact_p))
	{
	  pips_debug(4, "non empty intersection with second value_of cell "
		     "(%sexact)\n", intersection_test_exact_p? "": "non ");
	  if(cell_relation_exact_p(pv_in)
	     && intersection_test_exact_p
	     && (*cells_inclusion_p_func)(eff_cell, eff_desc,
					 second_cell_in, second_cell_in_desc,
					 &inclusion_test_exact_p)
	     && inclusion_test_exact_p)
	    {
	      if (cell_relation_undefined_p(*exact_aliased_pv))
		{
		  pips_debug(4, "exact value candidate found\n");
		  *exact_aliased_pv = pv_in;
		}
	       else if (cell_relation_second_address_of_p(*exact_aliased_pv))
		{
		  pips_debug(4, "better exact value candidate found\n");
		  l_res = CONS(CELL_RELATION, *exact_aliased_pv, l_res);
		  *exact_aliased_pv = pv_in;
		}
	      else
		{
		  pips_debug(4, "not kept as exact candidate\n");
		  l_res = CONS(CELL_RELATION, pv_in, l_res);
		}
	    }
	  else
	    {
	      pips_debug(5, "potentially non exact value candidate found\n");
	      l_res = CONS(CELL_RELATION, pv_in, l_res);
	    }
	}
      else
	{
	  pips_debug(4, "remnant\n");
	  *l_in_remnants = CONS(CELL_RELATION, pv_in, *l_in_remnants);
	}
      // here we should free the converted cells and descriptors if necessary
    }
  pips_debug_pvs(3, "l_in_remnants:", *l_in_remnants);
  pips_debug_pvs(3, "l_res:", l_res);
  pips_debug_pv(3, "*exact_aliased_pv:", *exact_aliased_pv);

  return l_res;
}


/*
  @brief tries to find in the input pointer values list aliases to
       the input cell and its descriptor.

       There is no sharing between the returned list cells and the
       input reference.

       When the input cell contains no dereferencing dimension
       the result is an empty list. The caller should check this
       case before calling the function because an empty list when there
       is a dereferencing dimension means that no target has been found
       and that the input path may point to anywhere (this has to evolve).


  @param c is a the cell for which we look equivalent paths
  @param l_pv is the list of pointer values in which we search for equivalent paths
  @param  exact_p is a pointer towards a boolean. It is set to true if
         the result is exact, and to false if it is an approximation,
	 either because the matching pointer values sources found in l_pv are
	 over-approximations of the preceding path of the input cell or because
	 the matching pointer values have MAY approximation tag.
  @return a list of equivalent path effects. It is a list because at a given
          program point the cell may correspond to several paths.


 Comments to generic_eval_cell_with_points_to may apply here to.
 */

list generic_effect_find_aliases_with_simple_pointer_values(
  effect eff, list l_pv, bool *exact_p,
  transformer current_precondition,
  bool (*cell_preceding_p_func)(cell, descriptor,
				cell, descriptor ,
				transformer, bool, bool * ),
  void (*cell_with_address_of_cell_translation_func)
      (cell, descriptor,
       cell, descriptor,
       int,
       cell *, descriptor *,
       bool *),
  void (*cell_with_value_of_cell_translation_func)
      (cell, descriptor,
       cell, descriptor,
       int,
       cell *, descriptor *,
       bool *),
  bool (*cells_intersection_p_func)(cell, descriptor,
				    cell, descriptor,
				    bool *),
  bool (*cells_inclusion_p_func)(cell, descriptor,
				 cell, descriptor,
				 bool*),
  void (*simple_cell_conversion_func)(cell, cell *, descriptor *))
{
  list l_res = NIL;
  list l_remnants = l_pv;
  cell eff_cell = effect_cell(eff);
  descriptor eff_desc= effect_descriptor(eff);
  //  reference eff_ref = effect_any_reference(eff);
  bool anywhere_p = false;

  pips_debug_effect(5, "begin with eff:", eff);
  pips_debug_pvs(5, "and l_pv:", l_pv);

  if (anywhere_effect_p(eff)
      || null_pointer_value_cell_p(effect_cell(eff))
      || undefined_pointer_value_cell_p(effect_cell(eff))) /* should it be turned into entity_abstract_location_p (?) */
    {
      pips_debug(5, "anywhere, undefined or null case\n");
      return (NIL);
    }
  else
    {
      /* first we must find in eff intermediary paths to pointers */
      list l_intermediary = effect_intermediary_pointer_paths_effect(eff);
      pips_debug_effects(5, "intermediary paths to eff:", l_intermediary);

      /* and find if this gives equivalent paths in l_pv */
      FOREACH(EFFECT, eff_intermediary, l_intermediary)
	{
	  pips_debug_effect(5, "considering intermediary path:", eff_intermediary);
	  list tmp_l_remnants = NIL;
	  cell_relation pv_exact = cell_relation_undefined;
	  list l_equiv =
	    generic_effect_find_equivalent_simple_pointer_values(eff_intermediary, l_remnants,
								 &pv_exact, &tmp_l_remnants,
								 cells_intersection_p_func,
								 cells_inclusion_p_func,
								 simple_cell_conversion_func);
	  if (!cell_relation_undefined_p(pv_exact))
	    {
	     l_equiv = CONS(CELL_RELATION, pv_exact, l_equiv);
	    }
	  l_remnants = tmp_l_remnants;
	  pips_debug_pvs(5, "list of equivalent pvs \n", l_equiv);

	  cell cell_intermediary = effect_cell(eff_intermediary);
	  reference ref_intermediary = effect_any_reference(eff_intermediary);
	  //entity ent_intermediary = reference_variable(ref_intermediary);
	  //descriptor d_intermediary = effect_descriptor(eff_intermediary);
	  int nb_common_indices =
	    (int) gen_length(reference_indices(ref_intermediary));

	  FOREACH(CELL_RELATION, pv_equiv, l_equiv)
	    {
	      cell c = cell_undefined;
	      descriptor d = descriptor_undefined;
	      bool exact_translation_p;
	      cell c1 = cell_relation_first_cell(pv_equiv);
	      cell c2 = cell_relation_second_cell(pv_equiv);

	      pips_debug_pv(5, "translating eff using pv: \n", pv_equiv);

	      if (undefined_pointer_value_cell_p(c1)
		  || undefined_pointer_value_cell_p(c2))
		{
		  pips_debug(5,"potential dereferencement of an undefined pointer -> returning undefined\n");
		  l_res = effect_to_list(make_undefined_pointer_value_effect(copy_action(effect_action(eff))));
		  if (cell_relation_may_p(pv_equiv))
		    effects_to_may_effects(l_res);
		  anywhere_p = true;
		}
	      else if (null_pointer_value_cell_p(c1)
		       || null_pointer_value_cell_p(c2))
		{
		      pips_debug(5,"potential dereferencement of a null pointer -> returning null\n");
		      l_res = effect_to_list(make_null_pointer_value_effect(copy_action(effect_action(eff))));
		      if (cell_relation_may_p(pv_equiv))
			effects_to_may_effects(l_res);
		      anywhere_p = true;
		}
	      else
		{
		  /* this is valid only if the first value_of corresponds
		     to eff_intermediary */

		  /* This test is valid here because by construction either c1 or c2 is an equivalent
		     for eff_intermediary
		  */
		  if (same_entity_p(cell_entity(cell_intermediary), cell_entity(c1))
		      && (gen_length(reference_indices(ref_intermediary)) == gen_length(cell_indices(c1))))
		    {
		      cell converted_c2 = cell_undefined;
		      descriptor converted_d2 = descriptor_undefined;
		      (*simple_cell_conversion_func)(c2, &converted_c2, &converted_d2);

		      /* use second cell as equivalent value for intermediary path */
		      if (cell_relation_second_value_of_p(pv_equiv))
			{
			  (*cell_with_value_of_cell_translation_func)
			    (eff_cell, eff_desc,
			     converted_c2, converted_d2,
			     nb_common_indices,
			     &c, &d, &exact_translation_p);
			}
		      else /* cell_relation_second_address_of_p is true */
			{
			  (*cell_with_address_of_cell_translation_func)
			    (eff_cell, eff_desc,
			     converted_c2, converted_d2,
			     nb_common_indices,
			     &c, &d, &exact_translation_p);
			}
		      // we should maybe free converted stuff here
		    }
		  else /* use first cell as equivalent value for intermediary path  */
		    {
		      pips_assert("pv_equiv must be value_of here\n",
				  cell_relation_second_value_of_p(pv_equiv));
		      cell converted_c1 = cell_undefined;
		      descriptor converted_d1 = descriptor_undefined;
		      (*simple_cell_conversion_func)(c1, &converted_c1, &converted_d1);

		      (*cell_with_value_of_cell_translation_func)
			(eff_cell, eff_desc,
			 converted_c1, converted_d1,
			 nb_common_indices,
			 &c, &d, &exact_translation_p);

		      // we should maybe free converted stuff here
		    }
		  exact_translation_p = effect_exact_p(eff_intermediary) && exact_translation_p && cell_relation_exact_p(pv_equiv);

		  effect eff_alias = make_effect(c,
						 copy_action(effect_action(eff_intermediary)),
						 exact_translation_p ?
						 make_approximation_exact()
						 : make_approximation_may(),
						 descriptor_undefined_p(d) ? make_descriptor_none() : d);

		  pips_debug_effect(5, "resulting effect \n", eff_alias);
		  // there we should perform a union...
		  if (anywhere_effect_p(eff_alias))
		    {
		      gen_full_free_list(l_res);
		      l_res = CONS(EFFECT, eff_alias, NIL);
		      anywhere_p = true;
		    }
		  else
		    {
		      l_res = CONS(EFFECT, eff_alias, l_res);
		    }
		}
	    } /* FOREACH */
	}

      if (!anywhere_p)
	{
	  pips_debug_effects(5, "l_res after first phase : \n", l_res);

	  /* Then we must find  if there are address_of second cells
	     which are preceding paths of eff path
	     in which case they must be used to generate other aliased paths
	  */
	  list l_remnants_2 = NIL;
	  FOREACH(CELL_RELATION, pv_remnant, l_remnants)
	    {
	      cell pv_remnant_second_cell =
		cell_relation_second_cell(pv_remnant);
	      bool exact_preceding_test = true;

	      pips_debug_pv(5, "considering pv: \n", pv_remnant);

	      cell pv_remnant_converted_cell = cell_undefined;
	      descriptor pv_remnant_converted_desc = descriptor_undefined;

	      // this (maybe costly) translation should take place after the first 3 tests
	      (*simple_cell_conversion_func)(pv_remnant_second_cell,
				      &pv_remnant_converted_cell,
				      &pv_remnant_converted_desc);

	      if (cell_relation_second_address_of_p(pv_remnant)
		  && same_entity_p(effect_entity(eff),
				   cell_entity(pv_remnant_second_cell))
		  && (gen_length(cell_indices(eff_cell))
		      >= gen_length(cell_indices(pv_remnant_second_cell)))
		  && (*cell_preceding_p_func)(pv_remnant_converted_cell, pv_remnant_converted_desc,
					     eff_cell, eff_desc,
					     transformer_undefined,
					     true,
					     &exact_preceding_test))
		{
		  cell c;
		  descriptor d = descriptor_undefined;
		  bool exact_translation_p;

		  pips_debug(5, "good candidate (%sexact)\n",exact_preceding_test? "":"non ");
		  /* for the translation, add a dereferencing_dimension to pv_remnant_first_cell */
		  reference new_ref = copy_reference
		    (cell_reference(cell_relation_first_cell(pv_remnant)));
		  int nb_common_indices = (int) gen_length(cell_indices(pv_remnant_second_cell));
		  reference_indices(new_ref) = gen_nconc(reference_indices(new_ref),
							 CONS(EXPRESSION,
							      int_to_expression(0),
							      NIL));
		  cell new_cell = make_cell_reference(new_ref);
		  cell new_converted_cell = cell_undefined;
		  descriptor new_converted_desc = descriptor_undefined;
		  (*simple_cell_conversion_func)(new_cell,
				      &new_converted_cell,
				      &new_converted_desc);

		  (*cell_with_value_of_cell_translation_func)
		    (eff_cell, eff_desc, 
		     new_converted_cell, new_converted_desc, 
		     nb_common_indices,
		     &c, &d, &exact_translation_p);

		  exact_translation_p = exact_translation_p && cell_relation_exact_p(pv_remnant);

		  effect eff_alias = make_effect(c,
						 make_action_write_memory(),
						 exact_translation_p && exact_preceding_test ?
						 make_approximation_exact()
						 : make_approximation_may(),
						 descriptor_undefined_p(d)? make_descriptor_none() : d);
		  free_cell(new_cell);
		  // we should also free new_converted_cell and new_converted_desc if they have actually been translated
		  pips_debug_effect(5, "resulting effect \n", eff_alias);
		  l_res = CONS(EFFECT, eff_alias, l_res);

		}
	      else
		{
		  l_remnants_2 = CONS(CELL_RELATION, pv_remnant, l_remnants_2);
		}
	    } /* FOREACH */

	  l_remnants = l_remnants_2;
	} /* if (!anywhere_p)*/
      if (!ENDP(l_remnants))
	{
	  pips_debug(5, "recursing to find aliases to aliased effect...\n");
	  pips_debug_effects(5, "l_res before recursing : \n", l_res);
	  list l_recurs = NIL;
	  FOREACH(EFFECT, eff_alias, l_res)
	    {
	      l_recurs = gen_nconc(l_recurs,
				   generic_effect_find_aliases_with_simple_pointer_values(eff_alias,
											  l_remnants,
											  exact_p,
											  current_precondition,
											  cell_preceding_p_func,
											  cell_with_address_of_cell_translation_func,
											  cell_with_value_of_cell_translation_func,
											  cells_intersection_p_func,
											  cells_inclusion_p_func,
											  simple_cell_conversion_func ));
	    }
	  l_res = gen_nconc(l_recurs, l_res);
	}
    } /* else branche of if (anywhere_effect_p(eff))*/

  pips_debug_effects(5, "returning : \n", l_res);
  return l_res;
}
