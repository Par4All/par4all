/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/* Processing of loops for transformers, preconditions and
   transformer lists*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
 /* semantic analysis: processing of loops
  */
#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
/* #include "database.h" */
#include "linear.h"
#include "ri.h"
#include "effects.h"
/*
#include "text.h"
#include "text-util.h"
*/
#include "ri-util.h"
#include "effects-util.h"
/* #include "constants.h" */
/* #include "control.h" */
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "vecteur.h"
#include "contrainte.h"
/*
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
*/

#include "transformer.h"

#include "semantics.h"

/* General handling of structured loops: while, repeat, for, do.
 *
 * Because the functions for "do" loops pre-existed this framework,
 * they have not (yet) been reimplemented with it. Their
 * implementations are likely to be more CPU efficient because only
 * useful objects are computed whereas the general case requires the
 * computation and use of neutral and absorbent elements such as
 * tranformer_identity() and transformer_empty(). The functions for
 * "while" loops have been rewritten to check the genericity.
 *
 * Slightly different equations are used for the repeat loop, which
 * executes once or at least twice instead of zero or at least
 * once. These equations are given in the corresponding module,
 * complete_repeatloop_transformer() and
 * repeatloop_to_postcondition().
 *
 * The transformer associated to any kind of loop, but the
 * repeat_loop, is defined by equation:
 *
 * t_loop = t_init ; t_skip +
 *          t_init ; t_enter ; (t_body ; t_next)* ; t_body ; t_inc; t_exit
 *
 * where ";" is the combine operator and "+" the union. Note already
 * that t_init is not factored out because the union usually loses
 * information and its use must be postponed as much as possible.
 *
 * Transformers t_init, t_skip, t_inc, t_next and t_exit are dependent
 * on the kind of loops. For instance, t_skip is transformer_empty for
 * a repeat loop and t_next may include the incrementation (t_inc) and
 * the looping condition (for loop) or just the looping condition
 * (while loop). When no incrementation occurs, t_inc is t_identity
 * and
 *
 * t_next = t_continue. Elsewhere:
 *
 * t_next = t_inc ; t_continue
 *
 * Because transformers are used to compute preconditions, we need to
 * preserve:
 *
 * t_body_star =  t_init ; t_enter ;(t_body ; t_next)*
 *
 * to compute the body preconditions as t_body_star(pre). But it might
 * be better to use equation
 *
 * .t_init ; t_enter(pre) + (t_init ; t_enter ;(t_body ; t_next)* ; t_body ; t_next)(pre)
 *
 * to reduce the approximation in the * operator (see below).
 *
 * Since we store only one transformer per statement (no such thing as
 * proper and cumulated effects), we store t_body_star instead of
 * t_loop. Note that the range of t_body_star can be restricted by the
 * union of the ranges of t_enter and t_continue (or t_next) which are
 * the last transitions to occur in t_body_star.
 *
 * But to obtain a correct transformer for the bottom-up composition
 * of transformers, we fix the resulting transformer in
 * statement_to_transformer with the equation:
 *
 * t_loop = t_init ; t_skip + t_body_star ; t_body ; t_inc; t_exit
 *
 * When computing the loop postcondition, we do not use t_loop either
 * because we want to postpone the use of "+":
 *
 * post = (t_init ; t_skip)(pre) + (t_body_star ; t_body ; t_inc ; t_exit) (pre)
 *
 * When computing the body precondition, we use the following equation
 *
 * p_body = (t_init ; t_enter)(pre) + (t_body_star ; t_body ; t_next) (pre)
 *
 * in order to avoid t_body_star imprecision due to a union with
 * t_identity and to delay it in the precondition domain whose
 * dimension is half of the transformer domain dimension.
 *
 * Note also that the different transformers are computed relatively
 * to a current precondition. So t_init, t_skip etc... may have
 * different values at differente stages. This happens more when
 * transformers are computed in context (setproprety
 * SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT true) and even more when
 * they are recomputed (apply REFINE_TRANSFORMERS). For this reason, a
 * first crude precondition is computed with effects information to be
 * used to compute the body transformer. And since t_enter cannot be
 * computed without its precondition, the later is passed as post_init
 * instead of the loop precondition pre:
 *
 * post_init = t_init(pre)
 *
 * A new transformer is allocated. None of the arguments are modified.
 *
 * The source code has been restructured with a lot of the processing
 * moved into any_transformer_to_k_closure
 */
/* Would be nicer with a proper graph. The transformer associated to a
   loop in a PIPS print out is the transformer leading from the loop
   precondition to the loop invariant, t_????. But the transformer
   recursively returned is t_loop, leading from the loop precondition
   to the loop postcondition. Designed in West Palm Beach, Dec. 2009.
   Notations may have sliglhty changed during the coding.

      x loop precondition-------------------------------------
      | t_init                               | t_?????       | t_loop
      v                                      |               |
      x post_init                            |               |
      |                                      |               |
      --------------------------             |               |
      |                        | t_enter     |               |
      | t_skip                 v             |               |
      |                        x post_enter  |               |
      |                        | identity    |               |
      |                        v             |               |
      |                     -> x <------------loop invariant |
      |        tf_body_star/__/| t_body      |               |
      |                        v             |               |
      |                        x             |               |
      |                        | t_inc       |               |
      |                        v             |               |
      |                        x             |               !
      |                        |             |               |
      |       ------------------             |               |
      |       | t_exit         | t_continue  |               |
      v       v                v             |               |
      ---------                x--------------               |
      |                                                      |
      v                                                      |
      x loop postcondition <----------------------------------

  The same scheme is used for all kinds of loops. Of course, t_inc,
  t_exit, t_skip and t_continue have to be adapted to the loop kind.
 */

/* k is the periodicity sought. The normal standard default value is
   1. If k == 1, the body transformer is computed first and the loop
   transformer is derived from it. If k>1, the body transformer is
   retrieved and the loop transformer is based on the assumption that
   the number of iterations executed is always a multiple of k.

   This is obsolete and k should always be equal to 1. When a
   different value of k is required, call directly
   any_transformer_to_k_closure().
 */
transformer any_loop_to_k_transformer(transformer t_init,
				      transformer t_enter,
				      transformer t_next,
				      statement body,
				      list __attribute__ ((unused)) lel, // loop effect list
				      transformer post_init,
				      int k)
{
  transformer t_body = transformer_undefined; // Body transformer
  transformer t_body_star = transformer_undefined; // result: t_init ; t_enter ; tfbodystar
  transformer pre_body = transformer_undefined;
  transformer post_body = transformer_undefined;
  // This does not include the loop effects, e.g. the index incrementation
  //list bel = load_cumulated_rw_effects_list(body);
  //transformer t_effects = effects_to_transformer(bel);
  transformer t_effects = effects_to_transformer(lel);
  transformer post_enter = transformer_apply(t_enter, post_init);
  //post_enter = transformer_normalize(post_enter, 0);
  transformer post_next = transformer_undefined;
  transformer t_next_star = transformer_undefined;

  // They are displayed by any_transformer_to_k_closure()
  ifdebug(9) {
    fprintf(stderr, "t_init:\n");
    print_transformer(t_init);
    fprintf(stderr, "t_enter:\n");
    print_transformer(t_enter);
    fprintf(stderr, "t_next:\n");
    print_transformer(t_next);
    fprintf(stderr, "post_init:\n");
    print_transformer(post_init);
  }

  /* Compute a first rough body precondition, pre_body, using
   * different heuristics:
   *
   * - a very approximate fix-point and the conditions met for the
   * body to execute, pre_iteration.
   *
   * -
   *
   * - the convex hull of the precondition holding for the first
   *   iteration and the range of the extended body transformer,
   *   t_next.
   */
  //
  // First heuristics:
  //pre_body = invariant_wrt_transformer(post_enter, t_effects);
  // memory leak for pre_body? for transformer_range(pre_iteration)?
  //pre_body = invariant_wrt_transformer(pre_body, t_next);
  /* Add the loop body execution condition */
  //pre_body = transformer_combine(pre_body, pre_iteration);
  //
  // Second heuristics: either we enter directly or we loop
  post_body = transformer_apply(t_effects, post_enter);
  // temporary memory leaks
  t_next_star = (* transformer_fix_point_operator)(t_next);
  post_next = transformer_range(transformer_apply(transformer_combine(t_next_star, t_next), post_body));
  //post_next = transformer_normalize(post_next, 0);
  transformer pre_body_2 = transformer_range(transformer_convex_hull(post_next, post_enter));
  //
  // Third heuristics: either it is the first iteration, or we are in
  // the range of t_next (added for Semantics-New/flip-flop01.c, but
  // turns out to be useless for flip-flop02)
  // transformer pre_body_3 = transformer_convex_hull(post_enter, transformer_range(t_next));

  // pre_body = transformer_intersection(pre_body_2, pre_body_3);
  //free_transformers(pre_body_2, pre_body_3, NULL);
  pre_body = pre_body_2;

  /* Compute the body transformer */
  // THIS PART SHOULD BE CLEANED-UP and k not used to avoid
  // recomputing the transformers inside the loop body
  // There is no longer any reason to call this function with k !=1
  // For k == 2, call directly any_transformer_to_k_closure()
  // statement_to_transformer() allocated a new transformer which is not the stored transformer
  //t_body = transformer_dup(statement_to_transformer(body, pre_body));
  if(k==1) {
    t_body = statement_to_transformer(body, pre_body);
    // Experimental: produces wrong results
    // t_body = transformer_intersect_range_with_domain(t_body);
  }
  else // assume k==2
    t_body = copy_transformer(load_statement_transformer(body));


  // Insert a call to any_transformer_to_k_closure()
  t_body_star = any_transformer_to_k_closure(t_init,
					     t_enter,
					     t_next,
					     t_body,
					     post_init,
					     k, // unrollling degree
					     false); // assume a first
						     // execution of t_body

  /* Any transformer or other data structure to free? */
  //free_transformer(t_body); FI: no idea if it should be freed or not...
  free_transformer(pre_body);
  free_transformer(post_body);
  free_transformer(post_next);
  free_transformer(t_effects);
  free_transformer(post_enter);
  free_transformer(t_next_star);

  ifdebug(8) {
    fprintf(stderr, "t_body_star:\n");
    print_transformer(t_body_star);
  }

  return t_body_star;
}

transformer any_loop_to_transformer(transformer t_init,
				    transformer t_enter,
				    transformer t_next,
				    statement body,
				    list __attribute__ ((unused)) lel, // loop effect list
				    transformer post_init)
{
  return any_loop_to_k_transformer(t_init, t_enter, t_next, body, lel, post_init, 1);
}

transformer forloop_to_transformer(forloop fl,
				   transformer pre,
				   list flel) /* effects of forloop fl */
{
  /* t_body_star =  t_init ; t_enter ;(t_body ; t_next)* */
  transformer t_body_star = transformer_undefined;
  statement body_s = forloop_body(fl);

  /* Deal with initialization expression */
  expression init_e = forloop_initialization(fl);
  transformer t_init = safe_expression_to_transformer(init_e, pre);
  transformer post_init = transformer_apply(t_init, pre);

  /* Deal with condition expression */
  expression cond_e = forloop_condition(fl);
  transformer t_enter = condition_to_transformer(cond_e, post_init, true);
  /* An effort could be made to compute the precondition for t_continue,
     especially if the precondition to t_inc is available. */
  transformer p_continue = transformer_identity();
  transformer t_continue = condition_to_transformer(cond_e, p_continue, true);

  /* Deal with increment expression */
  expression inc_e = forloop_increment(fl);
  /* An effort could be made to compute the precondition for t_inc */
  transformer t_inc = safe_expression_to_transformer(inc_e, transformer_undefined);
  transformer t_next = transformer_combine(t_inc, t_continue);

  t_body_star = any_loop_to_transformer(t_init, t_enter, t_next, body_s, flel, post_init);

  /* Let's clean up the memory */

  free_transformer(p_continue);

  free_transformer(t_init);
  free_transformer(post_init);

  free_transformer(t_enter);
  free_transformer(t_continue);

  // free_transformer(t_inc); it is absorbed by t_next
  free_transformer(t_next);

  return t_body_star;
}

list forloop_to_transformer_list(forloop l __attribute__ ((unused)),
				 transformer pre __attribute__ ((unused)),
				 list e __attribute__ ((unused)))
{
  list tfl = NIL;
  pips_internal_error("Not implemented yet.");
  return tfl;
}

transformer new_whileloop_to_transformer(whileloop wl,
					 transformer pre,
					 list wlel) /* effects of whileloop wl */
{
  /* Equation:
   *
   * t_body_star =  t_init ; t_enter ;(t_body ; t_continue)*
   */
  transformer t_body_star = transformer_undefined;
  statement body_s = whileloop_body(wl);

  /* Deal with initialization expression, which may be included in
     the condition as in while(i++, j=0, i<m)? No because the
     expression is going to be evaluated at each cycle. The ised
     effects must be part of the condition transformer, tcond */
  transformer t_init = transformer_identity();

  /* Deal with condition expression */
  expression cond_e = whileloop_condition(wl);
  transformer t_enter = condition_to_transformer(cond_e, pre, true);
  /* An effort could be made to compute the precondition for t_continue. */
  transformer p_continue = transformer_identity();
  transformer t_continue = condition_to_transformer(cond_e, p_continue, true);

  t_body_star = any_loop_to_transformer(t_init, t_enter, t_continue, body_s, wlel, pre);

  /* Let's clean up the memory */

  free_transformer(p_continue);

  free_transformer(t_enter);
  free_transformer(t_continue);

  return t_body_star;
}

transformer new_whileloop_to_k_transformer(whileloop wl,
					   transformer pre,
					   list wlel, /* effects of
							 whileloop wl
					   */
					   int k) // unrolling
{
  /* t_body_star =  t_init ; t_enter ;(t_body ; t_next)* */
  transformer t_body_star = transformer_undefined;
  statement body_s = whileloop_body(wl);

  /* Deal with initialization expression */
  transformer t_init = transformer_identity();

  /* Deal with condition expression */
  expression cond_e = whileloop_condition(wl);
  transformer t_enter = condition_to_transformer(cond_e, pre, true);
  /* An effort could be made to compute the precondition for t_continue. */
  transformer p_continue = transformer_identity();
  transformer t_continue = condition_to_transformer(cond_e, p_continue, true);

  t_body_star = any_loop_to_k_transformer(t_init, t_enter, t_continue, body_s, wlel, pre, k);

  /* Let's clean up the memory */

  free_transformer(p_continue);

  free_transformer(t_enter);
  free_transformer(t_continue);

  return t_body_star;
}

transformer repeatloop_to_transformer(whileloop wl,
				      transformer pre,
				      list wlel) /* effects of whileloop wl */
{
  /* t_body_star =  t_init ; t_enter ; (t_body ; t_next)+
   *
   * t_once = t_body; t_exit
   *
   * t_repeat = t_body_star + t_once
   */
  transformer t_body_star = transformer_undefined;
  statement body_s = whileloop_body(wl);
  transformer t_body = transformer_undefined; // load_statement_transformer(body_s);
  transformer t_init = transformer_identity();
  expression cond_e = whileloop_condition(wl);
  transformer t_enter = transformer_identity();
  /* An effort could be made to compute the precondition for t_continue,
     especially if the precondition to t_inc is available. */
  transformer t_continue = condition_to_transformer(cond_e, transformer_undefined, true);
  /* FI: it should be computed with the postcondition of the body */
  transformer t_exit = condition_to_transformer(cond_e, transformer_undefined, false);
  //transformer t_inc = transformer_identity();
  transformer t_next = t_continue;

  /* The loop is executed at least twice; FI: I'm note sure the twice is captured */
  /* FI: example dowhile02 seems to show this is wrong with t_next
     empty, in spite of the star */
  t_body_star = any_loop_to_transformer(t_init, t_enter, t_next, body_s, wlel, pre);

  /* The loop is executed only once */
  // any_loop_to_transformer() has computed the body transformer
  t_body = load_statement_transformer(body_s);
  transformer t_once = transformer_combine(copy_transformer(t_body), t_exit);

  /* global transformer */
  transformer t_repeat = transformer_convex_hull(t_once, t_body_star);

  // free_transformer(t_once), free_transformer(t_body_star), free_transformer(t_next), free_transformer(t_exit), free_transformer(t_enter);

  return t_repeat;
}

#define IS_LOWER_BOUND 0
#define IS_UPPER_BOUND 1

/* tf and pre are a unique data structure when preconditions are computed */
transformer add_loop_skip_condition(transformer tf, loop l, transformer pre)
{
  /* It is assumed that loop l is not entered */
  range r = loop_range(l);
  expression e_lb = range_lower(r);
  expression e_ub = range_upper(r);
  expression e_incr = range_increment(r);
  normalized n_lb = NORMALIZE_EXPRESSION(e_lb);
  normalized n_ub = NORMALIZE_EXPRESSION(e_ub);
  int incr = 0;
  int incr_lb = 0;
  int incr_ub = 0;

  /* EXPRESSION_TO_TRANSFORMER() should be used */

  pips_debug(8,"begin with transformer tf=%p\n", tf);
  ifdebug(8) {
    (void) print_transformer(tf);
    pips_debug(8,"and precondition pre=%p\n", pre);
    (void) print_transformer(pre);
  }

  /* is the loop increment numerically known? Is its sign known? */
  expression_and_precondition_to_integer_interval(e_incr, pre, &incr_lb, &incr_ub);

  if(incr_lb==incr_ub) {
    if(incr_lb==0) {
      user_error("add_loop_skip_condition", "Illegal null loop increment\n");
    }
    else
      incr = incr_lb;
  }
  else if(incr_lb>=1) {
    incr = 1;
  }
  else if(incr_ub<=-1) {
    incr = -1;
  }
  else
    incr = 0;

  /* incr == 0 is used below as a give-up condition */

  /* find the real upper and lower bounds */
  if(incr<0) {
    /* exchange bounds */
    n_lb = NORMALIZE_EXPRESSION(e_ub);
    n_ub = NORMALIZE_EXPRESSION(e_lb);
  }

  if(incr!=0 && normalized_linear_p(n_lb) && normalized_linear_p(n_ub)) {
    /* ub < lb, i.e. ub + lb + 1 <= 0 */
    Pvecteur v_ub = (Pvecteur) normalized_linear(n_ub);
    Pvecteur v_lb = (Pvecteur) normalized_linear(n_lb);

    if(value_mappings_compatible_vector_p(v_lb)
       && value_mappings_compatible_vector_p(v_ub)) {
      Pvecteur v = vect_substract(v_ub, v_lb);

      vect_add_elem(&v, TCST, (Value) 1);
      tf = transformer_inequality_add(tf, v);

      ifdebug(8) {
	debug(8,"add_loop_skip_condition","Skip condition:\n");
	vect_fprint(stderr, v, (char * (*)(Variable)) external_value_name);
      }
    }
    else {
      pips_debug(8,"Non-analyzed variable in loop bound(s)\n");
    }
  }
  else {
    pips_debug(8,"increment sign unknown or non-affine bound\n");
  }

  ifdebug(8) {
    pips_debug(8,"end with new tf=%p\n", tf);
    (void) print_transformer(tf);
  }

  return tf;
}

/* FI: could be moved somewhere else, e.g. in transformer library. */
/* lower_or_upper: 0 for lower, 1 for upper, i.e. upper_p */
static transformer add_affine_bound_conditions(transformer pre,
					       entity index,
					       Pvecteur v_bound,
					       bool lower_or_upper,
					       transformer tfb)
{
  Pvecteur v = vect_dup(v_bound);

  /* check that v is not affected by tfb:
   * N = 10
   * DO I = 1, N
   *   N = 1
   *   {1<=I<=N} !wrong!
   *   T(I) = 0.
   * ENDDO
   * and make sure that aliasings (I,J) and (I,X) are correctly handled
   */

  /* Achtung: value_mappings_compatible_vector_p() has a side effect
   * on its argument; it has to be evaluated before the second half of
   * the test else effects would be wrongly interpreted in case of
   * equivalences
   */

  if(value_mappings_compatible_vector_p(v) &&
     !transformer_affect_linear_p(tfb,v)) {
    if (lower_or_upper == IS_LOWER_BOUND)
      vect_add_elem(&v,
		    (Variable) entity_to_new_value(index), VALUE_MONE);
    else{
      vect_chg_sgn(v);
      vect_add_elem(&v, (Variable) entity_to_new_value(index), VALUE_ONE);
    }
    pre = transformer_inequality_add(pre, v);
  }
  else{
    vect_rm(v);
    v = VECTEUR_UNDEFINED;
  }
  return pre;
}

/* Side effect on pre */
/* lower_or_upper: 0 for lower, 1 for upper, i.e. upper_p */
static transformer add_index_bound_conditions(transformer pre,
					      entity index,
					      expression bound,
					      int lower_or_upper,
					      transformer tfb)
{
  normalized n = NORMALIZE_EXPRESSION(bound);

  /* It is assumed on entry that index has values recognized
   * by the semantics analysis
   */
  /* pips_assert("add_index_bound_conditions", entity_has_values_p(index)); */

  if(normalized_linear_p(n)) {
    /* Old implementation, careful about the impact of the loop body
       but not about side effets in bound */
    Pvecteur v_bound = (Pvecteur) normalized_linear(n);
    /* tfb does not take into account the index incrementation */
    transformer t_iter = transformer_dup(tfb);

    transformer_arguments(t_iter) =
      arguments_add_entity(transformer_arguments(t_iter), index);

    add_affine_bound_conditions(pre, index, v_bound, lower_or_upper, t_iter);
    free_transformer(t_iter);
  }
  else { /* Why not use loop_bound_evaluation_to_transformer()? */
    type it = ultimate_type(entity_type(index));
    entity bv = make_local_temporary_value_entity(it);
    transformer pre_r = transformer_range(pre);
    transformer bt = safe_any_expression_to_transformer(bv, bound, pre_r, true);
    transformer br = transformer_range(bt);
    transformer npre = transformer_undefined;

    /* An inequation between index and bv should be added */
    if(lower_or_upper)
      br = transformer_add_inequality(br, index, bv, false);
    else
      br = transformer_add_inequality(br, bv, index, false);

    br = transformer_temporary_value_projection(br);
    reset_temporary_value_counter();
    br = safe_transformer_projection(br, transformer_arguments(tfb));

    /* FI: Fixt the result of the intersection in case of side
       effects in loop range*/
    npre = transformer_range_intersection(pre, br);
    //transformer_arguments(npre) = arguments_union(transformer_arguments(pre),
    //					 transformer_arguments(npre));
    /* Make sure the loop body does not modify the loop bounds */
    // FI: ipre is not a range, invariant_wrt_transformer() cannot be used
    //npre = invariant_wrt_transformer(ipre, tfb);
    // FI: removes to many variables of ipre or npre; induction
    // variables are lost when computing the preconditions
    //npre = safe_transformer_projection(npre, transformer_arguments(tfb));
    /* FI: we need a side effect on pre... */
    //gen_free_list(transformer_arguments(pre));
    free_predicate(transformer_relation(pre));
    /* Likely memory leak here: arguments_union may allocate a new list*/
    /* This might or not make sense, depending on the caller
    transformer_arguments(pre) =
      arguments_difference(arguments_union(transformer_arguments(pre),
					   transformer_arguments(npre)),
			   transformer_arguments(tfb));
    */
    /* When dealing with the loop body precondition, there is no
       reasons to remove the arguments of tfb, as tfb is part of the
       definition of pre */
    transformer_arguments(pre)
      = arguments_union(transformer_arguments(pre),
			transformer_arguments(npre));

    transformer_relation(pre) = transformer_relation(npre);
    free_transformer(bt);
    free_transformer(br);
    transformer_arguments(npre) = NIL;
    transformer_relation(npre) = predicate_undefined;
    free_transformer(npre);
    free_transformer(pre_r);
  }

  pips_assert("The resulting transformer is consistent",
	      transformer_consistent_p(pre));

  return(pre);
}

transformer add_index_range_conditions(transformer pre,
				       entity i,
				       range r,
				       transformer tfb)
{
  /* if tfb is not undefined, then it is a loop;
     loop bounds can be kept as preconditions for the loop body
     if the loop increment is numerically known and if they
     are linear and if they are loop body invariant, i.e.
     indices are accepted */

  expression lb = range_lower(r);
  expression ub = range_upper(r);
  expression e_incr = range_increment(r);
  int incr = 0;
  int incr_lb = 0;
  int incr_ub = 0;

  pips_debug(8, "begin\n");

  if(entity_has_values_p(i)) {

    /* is the loop increment numerically known? Is its sign known? */
    expression_and_precondition_to_integer_interval(e_incr, pre, &incr_lb, &incr_ub);

    if(incr_lb==incr_ub) {
      if(incr_lb==0) {
	pips_user_error("Illegal null increment\n");
      }
      else
	incr = incr_lb;
    }
    else if(incr_lb>=1) {
      incr = 1;
    }
    else if(incr_ub<=-1) {
      incr = -1;
    }
    else {
      /* incr == 0 is used below as a give-up condition */
      incr = 0;
    }

    /* When lost, try to exploit type information thanks to unsigned variables */
    if(incr==0) {
      if(positive_expression_p(e_incr))
        incr = 1;
      else if(negative_expression_p(e_incr))
        incr = -1;
    }

    /* find the real upper and lower bounds */
    if(incr<0) {
      ub = range_lower(r);
      lb = range_upper(r);
    }

    if(incr!=0) {
      if(simple_dead_loop_p(lb, ub)) {
	transformer new_pre = transformer_empty();

	free_transformer(pre);
	pre = new_pre;
      }
      else {
	/* try to add the lower bound */
	add_index_bound_conditions(pre, i, lb, IS_LOWER_BOUND, tfb);

	/* try to add the upper bound */
	add_index_bound_conditions(pre, i, ub, IS_UPPER_BOUND, tfb);
      }
    }

  }

  pips_debug(8, "end\n");
  return pre;
}

static transformer add_good_loop_conditions(transformer pre,
					    loop l)
{
  /* loop bounds can be kept as preconditions for the loop body
     if the loop increment is numerically known and if they
     are linear and if they are loop body invariant, i.e.
     indices are accepted */
  /* arg. tf is unused, it was replaced by tfb to correct a bug */
  statement b = loop_body(l);
  entity i = loop_index(l);
  range r = loop_range(l);
  /* Only OK if transformers already have been computed */
  /* transformer tfb = load_statement_transformer(b); */
  list eb = load_cumulated_rw_effects_list(b);
  transformer tfb = effects_to_transformer(eb);

  pips_debug(8, "begin\n");

  if(false) { /* New version to deal with complex do loop bounds? */
    transformer lbt = loop_bound_evaluation_to_transformer(l, pre);
    transformer opre = pre;
    pre = transformer_apply(pre, lbt);
    free_transformer(opre);
  }
  else {
    /* Old version */
    pre = add_index_range_conditions(pre, i, r, tfb);
  }

  pips_debug(8, "end\n");

  return(pre);
}

/* Always returns newly allocated memory */
static transformer add_loop_index_initialization(transformer tf,
					  loop l,
					  transformer pre)
{
    entity i = loop_index(l);
    expression init = range_lower(loop_range(l));
    transformer post = transformer_undefined;
    transformer t_init = transformer_undefined;
    //list lef = expression_to_proper_effects(init);
    list lef = expression_to_proper_constant_path_effects(init);
    transformer pre_r = transformer_range(pre);

    t_init = any_scalar_assign_to_transformer(i, init, lef, pre_r);
    free_transformer(pre_r);

    if(t_init==transformer_undefined)
	t_init = effects_to_transformer(lef);
    post = transformer_apply(t_init, tf);

    transformer_free(t_init);

    return post;
}

/* The exit value is known if
 *  - the loop index is an analyzed integer scalar variable
 *    real index are standard-compliant, integer index can be equivalenced to
 *    a real variable,...
 *  - the increment is affine (non-necessary assumption made: it is affine if the
 *    increment sign is known)
 *  - the increment sign is known
 *  - the body and loop initialization execution does not modify
 *    the value of the upper bound
 *  - the upper bound is affine (?)
 *
 * Affine increments can be handled when their signs only are known.
 *
 * For instance, for increment==k:
 *  i >= ub
 *  i-k <= ub-1
 *
 * Note the simplification when k==1.
 *
 * But, first of all, the value of the loop index in post must be
 * incremented. This changes its relationship with induction variables.
 *
 * Most tests here are redundant because this function is only called if it has
 * been proved that the loop was executed which implies that the upper and
 * lower bounds are affine, that the increment is affine and that the
 * increment sign is known.
 *
 * Always returns a newly allocated value
 */
transformer add_loop_index_exit_value(
    transformer post, /* postcondition of the last iteration */
    loop l,           /* loop to process */
    transformer pre)  /* precondition on loop entrance */
{
  entity i = loop_index(l);
  expression e_incr = range_increment(loop_range(l));
  normalized n_incr = NORMALIZE_EXPRESSION(e_incr);
  expression e_ub = range_upper(loop_range(l));
  normalized n_ub = NORMALIZE_EXPRESSION(e_ub);
  transformer t_incr = transformer_undefined;
  transformer t_body = load_statement_transformer(loop_body(l));
  list li = CONS(ENTITY, i, NIL);
  int ub_inc = 0;
  int lb_inc = 0;

  ifdebug(8) {
    pips_debug(8, "begin with post:\n");
    (void) print_transformer(post);
  }

  if(!entity_has_values_p(i)) {
    ifdebug(8) {
      pips_debug(8, "give up because %s has no values:\n",
	    entity_local_name(i));
      pips_debug(8, "end with post:\n");
      (void) print_transformer(post);
    }
    return transformer_dup(post);
  }

  expression_and_precondition_to_integer_interval(e_incr, pre, &lb_inc, &ub_inc);

  /* This part should be useless because post really is a loop postcondition.
   * There should be no need for an index incrementation or anything.
   * It is used to recover some of the fix-point failures:-(
   *
   * In fact, it may be useful. The loop transformer and its invariant are
   * computed without knowledge about the number of iteration, which may be
   * zero. By adding one execution of the loop body and the index incrementation,
   * we change b* into b+ which should add effective information.
   */
  if(normalized_linear_p(n_incr)) {
    Pvecteur v_incr = (Pvecteur) normalized_linear(n_incr);
    if(value_mappings_compatible_vector_p(v_incr)) {
      entity i_new = entity_to_new_value(i);
      entity i_rep = value_to_variable(i_new);
      t_incr = affine_increment_to_transformer(i_rep, v_incr);
    }
    else {
      t_incr = transformer_undefined;
    }
  }
  else {
    t_incr = transformer_undefined;
  }
  if(t_incr==transformer_undefined)
    t_incr = args_to_transformer(li);
  /* Do not apply an extra iteration! */
  /* post = transformer_apply(t_body, post); */
  post = transformer_apply(t_incr, post);
  transformer_free(t_incr);

  ifdebug(8) {
    pips_debug(8, "post after index incrementation:\n");
    (void) print_transformer(post);
  }

  if(normalized_linear_p(n_ub)
     && !transformer_affect_linear_p(t_body, (Pvecteur) normalized_linear(n_ub))) {
    if(lb_inc >= 1 || ub_inc <= -1) {
      Pvecteur v_ub = (Pvecteur) normalized_linear(n_ub);
      Pvecteur v_incr = (Pvecteur) normalized_linear(n_incr);

      if(value_mappings_compatible_vector_p(v_ub)
	 && value_mappings_compatible_vector_p(v_incr)) {
	entity i_new = entity_to_new_value(i);
	entity i_rep = value_to_variable(i_new);
	Pvecteur v_i = vect_new((Variable) i_rep, (Value) 1);
	Pvecteur c1 = VECTEUR_UNDEFINED;
	Pvecteur c2 = VECTEUR_UNDEFINED;

	pips_assert("The increment is an affine function", normalized_linear_p(n_incr));
	if(lb_inc==ub_inc && ABS(lb_inc)==1) {
	  /* v_i - v_incr == v_ub, regardless of the sign */
	  c1 = vect_substract(v_i, v_incr);
	  c1 = vect_cl(c1, (Value) -1, v_ub);
	  transformer_equality_add(post, c1);
	}
	else {
	  if(lb_inc>=1) {
	    /* v_i - v_incr <= v_ub < v_i
	     * or:
	     * v_i - v_incr - v_ub <= 0, v_ub - v_i + 1 <= 0
	     */
	    c1 = vect_substract(v_i, v_incr);
	    c2 = vect_substract(v_ub, v_i);

	    c1 = vect_cl(c1, (Value) -1, v_ub);
	    vect_add_elem(&c2, (Variable) TCST, (Value) 1);
	  }
	  else if(ub_inc<=-1) {
	    /* v_i - v_incr >= v_ub > v_i
	     *
	     * or:
	     * - v_i + v_incr + v_ub <= 0, - v_ub + v_i + 1 <= 0
	     */
	    c1 = vect_substract(v_incr, v_i);
	    c2 = vect_substract(v_i, v_ub);

	    c1 = vect_cl(c1, (Value) 1, v_ub);
	    vect_add_elem(&c2, (Variable) TCST, (Value) 1);
	  }
	  else {
	    /* should never happen! */
	    pips_assert("add_loop_index_exit_value", false);
	  }
	  transformer_inequality_add(post, c1);
	  transformer_inequality_add(post, c2);
	}

	ifdebug(8) {
	  pips_debug(8, "post with exit conditions:\n");
	  (void) print_transformer(post);
	}
      }
      else {
	pips_debug(8,
	      "post is unchanged because the increment or the upper bound"
	      " reference unanalyzed variables\n");
      }
    }
    else {
      pips_debug(8,
	    "post is unchanged because the increment sign is unknown\n");
    }
  }
  else {
    pips_debug(8,
	  "post is unchanged because the upper bound is not affine\n");
  }

  ifdebug(8) {
    pips_debug(8, "end: post:\n");
    (void) print_transformer(post);
  }

  return post;
}

bool simple_dead_loop_p(expression lower, expression upper)
{
  bool dead_loop_p = false;
  normalized n_lower = NORMALIZE_EXPRESSION(lower);
  normalized n_upper = NORMALIZE_EXPRESSION(upper);

  if(normalized_linear_p(n_upper) && normalized_linear_p(n_lower)) {
    Pvecteur v_lower = normalized_linear(n_lower);
    Pvecteur v_upper = normalized_linear(n_upper);

    if(VECTEUR_NUL_P(v_lower)) {
      if (!VECTEUR_NUL_P(v_upper)) {
	if(term_cst(v_upper)
	   && VECTEUR_NUL_P(vecteur_succ(v_upper))) {
	  dead_loop_p =  value_neg_p(vecteur_val(v_upper));
	}
      }
    }
    else if(VECTEUR_NUL_P(v_upper)) {
      if (!VECTEUR_NUL_P(v_lower)) {
	if(term_cst(v_lower)
	   && VECTEUR_NUL_P(vecteur_succ(v_lower))) {
	  dead_loop_p =  value_pos_p(vecteur_val(v_lower));
	}
      }
    }
    else if(term_cst(v_upper) && term_cst(v_lower)
	    && VECTEUR_NUL_P(vecteur_succ(v_upper))
	    && VECTEUR_NUL_P(vecteur_succ(v_lower))) {
      dead_loop_p =
	value_gt(vecteur_val(v_lower),vecteur_val(v_upper));
    }
  }

  return dead_loop_p;
}

transformer precondition_filter_old_values(transformer pre)
{
    Psysteme sc = predicate_system(transformer_relation(pre));
    Pbase b = BASE_UNDEFINED;

    for(b = sc_base(sc); !BASE_NULLE_P(b); b = vecteur_succ(b)) {
      entity old_v = (entity) vecteur_var(b);

      if(old_value_entity_p(old_v)) {
	pips_assert("No old values are left", false);
      }
    }

    return pre;
}

/* The loop initialization is performed before tf */
transformer transformer_add_loop_index_initialization(transformer tf,
							     loop l,
							     transformer pre)
{
  entity i = loop_index(l);
  range r = loop_range(l);
  normalized nlb = NORMALIZE_EXPRESSION(range_lower(r));

  /* EXPRESSION_TO_TRANSFORMER SHOULD BE USED */
  pips_assert("To please the compiler", pre==pre);

  if(entity_has_values_p(i) && normalized_linear_p(nlb)) {
    Psysteme sc = (Psysteme) predicate_system(transformer_relation(tf));
    Pcontrainte eq = CONTRAINTE_UNDEFINED;
    Pvecteur v_lb = vect_dup(normalized_linear(nlb));
    Pbase b_tmp, b_lb = make_base_from_vect(v_lb);
    entity i_init = entity_to_old_value(i);

    vect_add_elem(&v_lb, (Variable) i_init, VALUE_MONE);
    eq = contrainte_make(v_lb);
    /* The new variables in eq must be added to sc; otherwise,
     * further consistency checks core dump. bc.
     */
    /* sc_add_egalite(sc, eq); */
    /* The call to sc_projection_with_eq frees eq */
    sc = sc_projection_by_eq(sc, eq, (Variable) i_init);
    b_tmp = sc_base(sc);
    sc_base(sc) = base_union(b_tmp, b_lb);
    sc_dimension(sc) = base_dimension(sc_base(sc));
    base_rm(b_tmp);
    base_rm(b_lb);
    if(SC_RN_P(sc)) {
      /* FI: a NULL is not acceptable; I assume that we cannot
       * end up with a SC_EMPTY...
       */
      predicate_system_(transformer_relation(tf)) =
	newgen_Psysteme
	(sc_make(CONTRAINTE_UNDEFINED, CONTRAINTE_UNDEFINED));
    }
    else
      predicate_system_(transformer_relation(tf)) =
	newgen_Psysteme(sc);
  }
  else if(entity_has_values_p(i)) {
    /* Get rid of the initial value since it is unknowable */
    entity i_init = entity_to_old_value(i);
    list l_i_init = CONS(ENTITY, i_init, NIL);

    tf = transformer_projection(tf, l_i_init);
  }

return tf;
}

transformer transformer_add_loop_index_incrementation(transformer tf,
							     loop l,
							     transformer pre)
{
  entity i = loop_index(l);
  range r = loop_range(l);
  expression incr = range_increment(r);
  Pvecteur v_incr = VECTEUR_UNDEFINED;

  /* SHOULD BE REWRITTEN WITH EXPRESSION_TO_TRANSFORMER */

  pips_assert("To please the compiler", pre==pre);

  pips_assert("Transformer tf is consistent before update",
	      transformer_consistency_p(tf));

  /* it does not contain the loop index update the loop increment
     expression must be linear to find inductive variables related to
     the loop index */
  if(!VECTEUR_UNDEFINED_P(v_incr = expression_to_affine(incr))) {
    if(entity_has_values_p(i)) {
      if(value_mappings_compatible_vector_p(v_incr)) {
	tf = transformer_add_variable_incrementation(tf, i, v_incr);
      }
      else {
	entity i_old = entity_to_old_value(i);
	entity i_new = entity_to_new_value(i);
	Psysteme sc = predicate_system(transformer_relation(tf));
	Pbase b = sc_base(sc);

	transformer_arguments(tf) = arguments_add_entity(transformer_arguments(tf), i);
	b = base_add_variable(b, (Variable) i_old);
	b = base_add_variable(b, (Variable) i_new);
	sc_base(sc) = b;
	sc_dimension(sc) = base_dimension(sc_base(sc));
      }
    }
    else {
      pips_user_warning("non-integer or equivalenced loop index %s?\n",
			entity_local_name(i));
    }
  }
  else {
    if(entity_has_values_p(i)) {
      entity i_old = entity_to_old_value(i);
      entity i_new = entity_to_new_value(i);
      Psysteme sc = predicate_system(transformer_relation(tf));
      Pbase b = sc_base(sc);

      transformer_arguments(tf) = arguments_add_entity(transformer_arguments(tf), i);
      b = base_add_variable(b, (Variable) i_old);
      b = base_add_variable(b, (Variable) i_new);
      sc_base(sc) = b;
      sc_dimension(sc) = base_dimension(sc_base(sc));
    }
  }

  pips_assert("Transformer tf is consistent after update",
	      transformer_consistency_p(tf));

  return tf;
}

/* Side effects in loop bounds and increment are taken into account.  The
 * conditions on the loop index are given by the range of this
 * transformer.
 */
transformer loop_bound_evaluation_to_transformer(loop l, transformer pre)
{
  transformer r = transformer_undefined;
  entity i = loop_index(l);

  //pips_assert("No temporary variables are allocated",
  //      number_of_temporary_values()==0);

  if(entity_has_values_p(i)) {
    expression lbe = range_lower(loop_range(l));
    transformer iit = loop_initialization_to_transformer(l, pre);
    entity lbv = make_local_temporary_value_entity(entity_type(i));
    transformer lbt = any_expression_to_transformer(lbv, lbe, pre, true);

    transformer preub = transformer_safe_apply(iit, pre);
    expression ube = range_upper(loop_range(l));
    entity ubv = make_local_temporary_value_entity(entity_type(i));
    transformer ubt = any_expression_to_transformer(ubv, ube, preub, true);

    transformer prei = transformer_safe_apply(ubt, preub);
    expression ie = range_increment(loop_range(l));
    entity iv = make_local_temporary_value_entity(entity_type(i));
    transformer it = any_expression_to_transformer(iv, ie, prei, true);
    transformer pre_inc = transformer_safe_intersection(prei, it);
    entity ni = entity_to_new_value(i);
    int inc_lb = 0;
    int inc_ub = 0;
    expression eiv = entity_to_expression(iv);

    expression_and_precondition_to_integer_interval(eiv, pre_inc, &inc_lb, &inc_ub);
    free_expression(eiv);

    if(inc_lb>0 || inc_ub<0) {
      Pvecteur lb_ineq = vect_new((Variable) ni, VALUE_MONE);
      Pvecteur ub_ineq = vect_new((Variable) ni, VALUE_ONE);

      vect_add_elem(&lb_ineq, (Variable) lbv, VALUE_ONE);
      vect_add_elem(&ub_ineq, (Variable) ubv, VALUE_MONE);

      if(inc_ub<0) {
	lb_ineq = vect_multiply(lb_ineq, VALUE_MONE);
	ub_ineq = vect_multiply(ub_ineq, VALUE_MONE);
      }
      r = transformer_safe_apply(it, prei);
      r = transformer_safe_image_intersection(r, lbt);
      r = transformer_safe_image_intersection(r, ubt);
      r = transformer_inequality_add(r, lb_ineq);
      r = transformer_inequality_add(r, ub_ineq);
      r = transformer_temporary_value_projection(r);
    }
    reset_temporary_value_counter();
  }
  return r;
}

/* Note: It used to be implemented by computing the effect list of the
   lower bound expression and and new allocated effect for the loop index
   definition. It turns out to be very heavy, because cells must be of
   kind preference to be usable by several functions because macro
   effect_reference() expects so without testing it.

   However, it is also difficult to add a variable to the transformer t_init
   because of aliasing. So let's stick to the initial implementation. */
transformer loop_initialization_to_transformer(loop l, transformer pre)
{
  effect init_e = make_effect(make_cell(is_cell_preference,
					make_preference(make_reference(loop_index(l), NIL))),
			      make_action_write_memory(),
			      make_approximation_exact(),
			      make_descriptor_none());
  list l_init_e = CONS(EFFECT, init_e, NIL);
  //list l_expr_e = expression_to_proper_effects(range_lower(loop_range(l)));
  expression lbe = range_lower(loop_range(l));
  list l_expr_e = expression_to_proper_constant_path_effects(lbe);
  list el = list_undefined;

  transformer r_pre = transformer_safe_range(pre);
  transformer t_init = transformer_undefined;

  ifdebug(9) {
    print_effects(l_init_e);
    print_effects(l_expr_e);
  }

  el = gen_nconc(l_init_e, l_expr_e);

  ifdebug(9) {
    print_effects(el);
  }

  t_init =
    any_scalar_assign_to_transformer(loop_index(l),
				     range_lower(loop_range(l)),
				     el, /* over-approximation of effects */
				     r_pre);

  ifdebug(9) {
    print_effects(el);
  }

  gen_free_list(el);
  /* free_effects() is not enough, because it is a persistant reference */
  free_reference(preference_reference(cell_preference(effect_cell(init_e))));
  free_effect(init_e);
  free_transformer(r_pre);

  return t_init;
}

/* The transformer associated to a DO loop does not include the exit
 * condition because it is used to compute the precondition for any
 * loop iteration.
 *
 * There is only one attachment for the unbounded transformer and
 * for the bounded one.
 */
//transformer loop_to_transformer(loop l, transformer pre, list e)
transformer old_loop_to_transformer(loop l, transformer pre, list e)
{
  /* loop transformer tf = tfb* or tf = tfb+ or ... */
  transformer tf = transformer_undefined;
  /* loop body transformer */
  transformer tfb = transformer_undefined;
  /* approximate loop transformer, including loop index updates */
  transformer abtf = effects_to_transformer(e);
  /* loop body precondition */
  transformer preb = invariant_wrt_transformer(pre, abtf);
  /* Information about loop local variables is lost */
  list lv = loop_locals(l);
  /* range r = loop_range(l); */
  statement b = loop_body(l);
  transformer t_init = transformer_undefined;
  transformer old_tf = transformer_undefined;

  pips_debug(8,"begin with precondition pre=%p\n", pre);
  ifdebug(8) {
    (void) print_transformer(pre);
  }

  /* eliminate all information about local variables
   *
   * Mostly useful for Fortran code.
   *
   * In C, variables declared in the loop body do not exist before the
   * loop is entered and so do not have to be projected. But local
   * variables may have been introduced by a privatization phase.
   */
  preb = safe_transformer_projection(preb, lv);

  /* compute the loop body transformer under loop body precondition preb */
  if(!transformer_undefined_p(preb))
    preb = add_good_loop_conditions(preb, l);

  pips_debug(8,"body precondition preb=%p\n", preb);
  ifdebug(8) {
    (void) print_transformer(preb);
  }

  tfb = statement_to_transformer(b, preb);
  /* add indexation step under loop precondition pre */
  tfb = transformer_add_loop_index_incrementation(tfb, l, pre);

  pips_debug(8,"body transformer tfb=%p\n", tfb);
  ifdebug(8) {
    (void) print_transformer(tfb);
  }

  /* compute tfb's fix point according to pips flags */
  tf = (* transformer_fix_point_operator)(tfb);

  free_transformer(tfb);

  ifdebug(8) {
    pips_debug(8, "intermediate fix-point tf=\n");
    fprint_transformer(stderr, tf, (get_variable_name_t) external_value_name);
  }

  /* restrict the domain of tf by the range of pre, except for the loop
     index which is assigned in between */
  if(!transformer_undefined_p(pre)
     && get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT")) {
    list li = CONS(ENTITY, loop_index(l), NIL);
    transformer preb = transformer_filter(transformer_dup(pre), li);
    transformer r_pre = transformer_range(preb);

    tf = transformer_domain_intersection(tf, r_pre);
    free_transformer(r_pre);
    free_transformer(preb);
    gen_free_list(li);
  }

  /* add initialization for the unconditional initialization of the loop
     index variable */
  t_init = loop_initialization_to_transformer(l, pre);
  old_tf = tf;
  tf = transformer_combine(t_init, tf);
  tf = add_index_range_conditions(tf, loop_index(l), loop_range(l),
				  load_statement_transformer(b));
  free_transformer(old_tf);

  /* we have a problem here: to compute preconditions within the
     loop body we need a tf using private variables; to return
     the loop transformer, we need a filtered out tf; only
     one hook is available in the ri..; let'a assume there
     are no private variables and that if they are privatizable
     they are not going to get in our way */

  ifdebug(8) {
    (void) fprintf(stderr,"%s: %s\n","loop_to_transformer",
		   "resultat tf =");
    (void) (void) print_transformer(tf);
    pips_debug(8,"end\n");
  }

  return tf;
}

list loop_to_transformer_list(loop l __attribute__ ((unused)),
			      transformer pre __attribute__ ((unused)),
			      list e __attribute__ ((unused)))
{
  list tfl = NIL;
  pips_internal_error("Not implemented yet.");
  return tfl;
}

/* Transformer expression the loop initialization */
transformer loop_to_initialization_transformer(loop l, transformer pre)
{
  entity i = loop_index(l);
  range r = loop_range(l);
  expression low = range_lower(r);
  transformer t = assigned_expression_to_transformer(i, low, pre);
  return t;
}

/* Transformer expressiong the conditions between the index and the
 * loop bounds according to the sign of the increment.
 */
transformer loop_to_enter_transformer(loop l, transformer pre)
{
  entity i = loop_index(l);
  range r = loop_range(l);
  expression low = range_lower(r);
  expression up = range_upper(r);
  expression inc = range_increment(r);
  int il, iu;
  transformer t = transformer_identity();

  if(scalar_integer_type_p(entity_type(i))) {

    integer_expression_and_precondition_to_integer_interval(inc, pre, &il, &iu);

    if(il==iu && il==0) {
      /* A zero increment is not legal because the Fortran standard
	 imposes a division by the increment. */
      /* Only the loop bopy statement number is available... */
      pips_user_error("Null increment for DO loop detected.\n");
    }
    else if(il>0||iu<0) {
      /* Assumption: no side effect in loop bound expressions */
      entity lv = make_local_temporary_value_entity(entity_type(i));
      transformer lt = safe_integer_expression_to_transformer(lv, low, pre, true);
      entity uv = make_local_temporary_value_entity(entity_type(i));
      transformer ut = safe_integer_expression_to_transformer(uv, up, pre, true);

      if(il>0) {
	/* The increment is strictly positive */
	t = transformer_add_inequality_with_linear_term(t, lv, i, 1, true);
	t = transformer_add_inequality_with_linear_term(t, i, uv, 1, true);
      }
      else if(iu<0) {
	/* The increment is strictly negative */
	t = transformer_add_inequality_with_linear_term(t, uv, i, 1, true);
	t = transformer_add_inequality_with_linear_term(t, i, lv, 1, true);
      }
      t = transformer_intersection(t, lt);
      t = transformer_intersection(t, ut);
      t = transformer_temporary_value_projection(t);
      //t = transformer_normalize(t, 0);
      t = transformer_normalize(t, 1);
      free_transformers(lt, ut, NULL);
    }
    else {
      /* The sign of the increment is unknown, no information is available */
      ;
    }
  }
  else {
    /* No information is available for loop indices that are not integer */
    /* This could be improved, but floating point indices should never
       be used. */
    ;
  }

  return t;
}

transformer loop_to_continue_transformer(loop l, transformer pre)
{
  transformer t = transformer_identity();
  /* FI: the function transformer_add_loop_index_incrementation should
     be rewritten to exploit preconditions better. */
  t = transformer_add_loop_index_incrementation(t, l, pre);
  transformer et = loop_to_enter_transformer(l, pre);
  t = transformer_combine(t, et);
  free_transformer(et);
  return t;
}

/* loop_to_transformer() was developed first but is not as powerful as
 * the new algorithm used for all kinds of loops. See flip-flop01.c.
 */
//transformer new_loop_to_transformer(loop l, transformer pre, list lel)
transformer loop_to_transformer(loop l, transformer pre, list lel)
{
  /* Equation:
   *
   * t_body_star =  t_init ; t_enter ;(t_body ; t_inc; t_continue)*
   *
   * t_init, t_enter and t_continue, which includes t_inc, must be
   * computed from the loop and its precondition, pre, and extended
   * body effects, lel. In case the loop index appears in the bound or
   * increment expressions, the Fortran standard should be used to
   * determine the precondition to use. The same is true in case of
   * side-effects in the loop expressions.
   *
   * t_body must be computed from the loop body and a preliminary loop
   * invariant.
   */
  transformer t_body_star = transformer_undefined;
  statement body_s = loop_body(l);

  /* Deal with initialization expression, which may be included in
     the condition as in while(i++, j=0, i<m)? No because the
     expression is going to be evaluated at each cycle. The ised
     effects must be part of the condition transformer, tcond */
  transformer t_init = loop_to_initialization_transformer(l, pre);
  // FI Memory leak
  transformer post_i = transformer_range(transformer_apply(t_init, pre));

  /* Deal with enter transformers */
  transformer t_enter = loop_to_enter_transformer(l, post_i);
  transformer pre_b = transformer_apply(t_enter, post_i);

  /* An effort could be made to compute the precondition for t_continue. 
   * Precondition pre and the loop effect list lel could be used.
   */
  //transformer p_continue = transformer_identity();
  /* approximate loop transformer, including loop index updates */
  transformer abtf = effects_to_transformer(lel);
  /* first approximation of the loop body precondition */
  // invariant_wrt_transformer() seems to add arguments to the
  // filtered precondition...
  // FI: memory leak
  transformer p_continue = transformer_range(invariant_wrt_transformer(pre_b, abtf));
  /* FI: could do better with body_transformer */
  transformer t_continue = loop_to_continue_transformer(l, p_continue);

  t_body_star = any_loop_to_transformer(t_init, t_enter, t_continue, body_s, lel, pre_b);

  /* Let's clean up the memory */

  free_transformer(p_continue);
  free_transformer(t_enter);
  free_transformer(t_continue);
  free_transformer(post_i);
  free_transformer(pre_b);

  return t_body_star;
}

/* FI: used to be complete_any_loop_transformer() with a direct
   reduction by convex hull.
*/
list complete_any_loop_transformer_list(transformer t_init,
					transformer t_skip,
					transformer t_body_star,
					transformer t_body,
					transformer t_inc,
					transformer t_exit)
{
  /* t_loop = t_init ; t_skip + t_bodystar ; t_body ; t_inc; t_exit
   *
   * Before entering t_body, t_enter or t_next have been applied which
   * let us restrict the set of state modified by t_body. But this
   * information should already been included in t_body_star.
   */

  /* Several cases must be tested:
   *
   * 1. The loop is always entered and exited (usual case for
   * scientific code: t_skip = transformer_empty).
   *
   * 2. The loop is always entered and never exited (real-time code
   * situation, t_exit = transformer_empty)
   *
   * 3. The loop is never entered (in its current context, t_bodystar
   * = transformer_empty because t_enter = transformer_empty).
   *
   * 4. The loop may be entered and exited (usual case for general
   * purpose computing, think of list handling).
   *
   * 5. The loop may be entered but is then never exited (real-time
   * code again).
   *
   * These cases are checked with complete01.c,
   * complete02.c,... complete05.c
   */
  list ltl = NIL;
  // transformer ct = transformer_undefined;
  /* loop skipped transformer */
  transformer lst = transformer_combine(copy_transformer(t_init), t_skip);
  /* add loop entered transformer/condition: should already be in t_body_star */
  //transformer post_enter = transformer_range(t_enter);
  //transformer post_continue = transformer_range(t_continue);
  //transformer pre_body = transformer_convex_hull(post_enter, post_continue);
  //transformer let = transformer_combine(copy_transformer(t_body_star), pre_body);
  transformer let = copy_transformer(t_body_star);

  ifdebug(8) {
    fprintf(stderr, "t_init:\n");
    print_transformer(t_init);
    fprintf(stderr, "t_skip:\n");
    print_transformer(t_skip);
    fprintf(stderr, "t_body_star:\n");
    print_transformer(t_body_star);
    fprintf(stderr, "t_body:inc\n");
    print_transformer(t_body);
    fprintf(stderr, "t_inc:\n");
    print_transformer(t_inc);
    fprintf(stderr, "t_exit:\n");
    print_transformer(t_exit);
  }

  let = transformer_combine(let, t_body);
  let = transformer_combine(let, t_inc);
  let = transformer_combine(let, t_exit);
  let = transformer_normalize(let, 2);

  /* combine both cases in a transformer list */
  //ct = transformer_convex_hull(lst, let);
  ltl = two_transformers_to_list(lst, let);

  //free_transformer(post_enter);
  //free_transformer(post_continue);
  //free_transformer(pre_body);

  ifdebug(8) {
    fprintf(stderr, "ltl:\n");
    print_transformers(ltl);
  }

  return ltl;
}

/* Reduce the transformer list associated to a loop to a unique
   transformer using a convex hull. An empty list is converted into
   an empty transformer. The input list is freed. */
transformer complete_any_loop_transformer(transformer t_init,
					  transformer __attribute__ ((unused)) t_enter,
					  transformer t_skip,
					  transformer t_body_star,
					  transformer t_body,
					  transformer __attribute__ ((unused)) t_continue,
					  transformer t_inc,
					  transformer t_exit)
{
  transformer ltf = transformer_undefined;
  list ltl = complete_any_loop_transformer_list(t_init,
						t_skip,
						t_body_star,
						t_body,
						t_inc,
						t_exit);

  ltf = transformer_list_to_transformer(ltl);

  gen_free_list(ltl);

  return ltf;
}

transformer complete_forloop_transformer(transformer t_body_star,
					 transformer pre,
					 forloop fl)
{
  transformer ct = transformer_undefined;
  statement body_s = forloop_body(fl);
  transformer t_body = load_statement_transformer(body_s);
  transformer ct_body = transformer_undefined;
  transformer pre_body = transformer_undefined;
  expression init_e = forloop_initialization(fl);
  /* This function does not seem to exist because we only used
     statement effects in the past and because those are stored in the
     database. */
  transformer t_init = safe_expression_to_transformer(init_e, pre);
  transformer post_init = transformer_apply(t_init, pre);
  expression cond_e = forloop_condition(fl);
  expression inc_e = forloop_increment(fl);
  transformer t_skip = condition_to_transformer(cond_e, post_init, false);
  transformer t_enter = condition_to_transformer(cond_e, post_init, true);
/* An effort could be made to compute the precondition for t_exit,
     especially if the precondition to t_inc is available. */
  transformer t_continue = condition_to_transformer(cond_e, transformer_undefined, true);
  transformer t_exit = condition_to_transformer(cond_e, transformer_undefined, false);
  /* An effort could be made to compute the precondition for t_inc */
  transformer t_inc = safe_expression_to_transformer(inc_e, transformer_undefined);

  /* Make sure you have the effective statement transformer: it is not
     stored for loops and this is key for nested loops. */
  pre_body = transformer_apply(t_body_star, pre);
  ct_body = complete_statement_transformer(t_body, pre_body, body_s);

  ct = complete_any_loop_transformer(t_init, t_enter, t_skip, t_body_star, ct_body, t_continue, t_inc, t_exit);

  free_transformer(ct_body);
  free_transformer(pre_body);
  free_transformer(t_init);
  free_transformer(post_init);
  free_transformer(t_skip);
  free_transformer(t_enter);
  free_transformer(t_continue);
  free_transformer(t_exit);
  free_transformer(t_inc);

  return ct;
}

list complete_forloop_transformer_list(transformer t_body_star,
				       transformer pre,
				       forloop fl)
{
  list tfl = list_undefined;
  statement body_s = forloop_body(fl);
  transformer t_body = load_statement_transformer(body_s);
  transformer ct_body = transformer_undefined;
  transformer pre_body = transformer_undefined;
  expression init_e = forloop_initialization(fl);
  /* This function does not seem to exist because we only used
     statement effects in the past and because those are stored in the
     database. */
  transformer t_init = safe_expression_to_transformer(init_e, pre);
  transformer post_init = transformer_apply(t_init, pre);
  expression cond_e = forloop_condition(fl);
  expression inc_e = forloop_increment(fl);
  transformer t_skip = condition_to_transformer(cond_e, post_init, false);
  transformer t_enter = condition_to_transformer(cond_e, post_init, true);
/* An effort could be made to compute the precondition for t_exit,
     especially if the precondition to t_inc is available. */
  transformer t_continue = condition_to_transformer(cond_e, transformer_undefined, true);
  transformer t_exit = condition_to_transformer(cond_e, transformer_undefined, false);
  /* An effort could be made to compute the precondition for t_inc */
  transformer t_inc = safe_expression_to_transformer(inc_e, transformer_undefined);

  pips_internal_error("function is not implemented");

  /* Make sure you have the effective statement transformer: it is not
     stored for loops and this is key for nested loops. */
  pre_body = transformer_apply(t_body_star, pre);
  ct_body = complete_statement_transformer(t_body, pre_body, body_s);

  (void)complete_any_loop_transformer(t_init, t_enter, t_skip, t_body_star, ct_body, t_continue, t_inc, t_exit);

  free_transformer(ct_body);
  free_transformer(pre_body);
  free_transformer(t_init);
  free_transformer(post_init);
  free_transformer(t_skip);
  free_transformer(t_enter);
  free_transformer(t_continue);
  free_transformer(t_exit);
  free_transformer(t_inc);

  return tfl;
}

/* entered_p is used to for the execution of at least one iteration */
list new_complete_whileloop_transformer_list(transformer t_body_star,
					     transformer pre,
					     whileloop wl,
					     bool entered_p __attribute__ ((__unused__)))
{
  list tfl = NIL;
  //transformer ct = transformer_undefined;
  statement body_s = whileloop_body(wl);
  transformer t_body = load_statement_transformer(body_s);
  transformer ct_body = transformer_undefined;
  transformer pre_body = transformer_undefined;
  transformer post_body = transformer_undefined;
  transformer t_init = transformer_identity();
  expression cond_e = whileloop_condition(wl);
  //transformer t_skip = entered_p?
  //  transformer_empty() : condition_to_transformer(cond_e, pre, false);
  transformer t_skip = condition_to_transformer(cond_e, pre, false);
  transformer t_enter = condition_to_transformer(cond_e, pre, true);
  /* An effort could be made to compute the preconditions for t_continue and t_exit: see while04.c. */
  //transformer t_continue = condition_to_transformer(cond_e, transformer_undefined, true);
  //transformer t_exit = condition_to_transformer(cond_e, transformer_undefined, false);
  transformer t_continue = transformer_undefined;
  transformer t_exit = transformer_undefined;
  transformer t_inc = transformer_identity();

  /* Make sure you have the effective statement transformer: it is not
     stored for loops and this is key for nested loops. */
  pre_body = transformer_apply(t_body_star, pre);
  /* Should we instead compute recursively a transformer list? */
  ct_body = complete_statement_transformer(t_body, pre_body, body_s);

  post_body = transformer_apply(ct_body, pre_body);
  t_continue = condition_to_transformer(cond_e, post_body, true);
  t_exit = condition_to_transformer(cond_e, post_body, false);

  tfl = complete_any_loop_transformer_list(t_init, t_skip, t_body_star,
					   ct_body, t_inc, t_exit);

  free_transformer(ct_body);
  free_transformer(pre_body);
  free_transformer(post_body);
  free_transformer(t_init);
  free_transformer(t_skip);
  free_transformer(t_enter);
  free_transformer(t_continue);
  free_transformer(t_exit);
  free_transformer(t_inc);

  return tfl;
}

/* entered_p is used to for the execution of at least one iteration */
transformer new_complete_whileloop_transformer(transformer t_body_star,
					       transformer pre,
					       whileloop wl,
					       bool entered_p)
{
  list tfl =
    new_complete_whileloop_transformer_list(t_body_star, pre, wl, entered_p);
  transformer ct = transformer_list_to_transformer(tfl);
  // tfl is destroyed by transformer_list_to_transformer()
  return ct;
}

list complete_repeatloop_transformer_list(transformer t_body_star,
					    transformer __attribute__ ((unused)) pre,
					    whileloop wl)
{
  list tfl = NIL;
  statement body_s = whileloop_body(wl);
  transformer pt_body = load_statement_transformer(body_s);
  transformer t_body = transformer_undefined;
  transformer pre_body = transformer_undefined;
  transformer t_init = transformer_identity();
  expression cond_e = whileloop_condition(wl);
  transformer t_continue = condition_to_transformer(cond_e, transformer_undefined, true);
  transformer t_enter = transformer_identity();
  transformer t_skip = transformer_empty();
  transformer t_exit = condition_to_transformer(cond_e, transformer_undefined, false);
  /* An effort could be made to compute the precondition for t_exit */
  transformer t_inc = transformer_identity();
  transformer t_loop_1 = transformer_undefined;
  transformer t_loop_2 = transformer_undefined;;

  /* Make sure you have the effective statement transformer: it is not
     stored for loops and this is key for nested loops. */
  pre_body = transformer_apply(t_body_star, pre);
  t_body = complete_statement_transformer(pt_body, pre_body, body_s);

  t_loop_1 = copy_transformer(t_body);
  t_loop_2 = copy_transformer(t_body);

  /* The generic equation is not adpated to repeat loops, which may
   * execute once or at least twice instead of zero or twice
   *
   * The equation is:
   *.
   * t_loop = (t_body ; t_exit) +
   *          (t_body ; t_continue ; t_body_star ; t_body ; t_exit)
   *
   * where we assume that t_body_star includes the continuation condition.
   */

  //  ct = complete_any_loop_transformer(t_init, t_enter, t_skip, t_body_star, t_body, t_continue, t_inc, t_exit);

  t_loop_1 = transformer_combine(t_loop_1, t_exit);

  t_loop_2 = transformer_combine(t_loop_2, t_continue);
  t_loop_2 = transformer_combine(t_loop_2, t_body_star);
  t_loop_2 = transformer_combine(t_loop_2, t_body);
  t_loop_2 = transformer_combine(t_loop_2, t_exit);

  //ct = transformer_convex_hull(t_loop_1, t_loop_2);
  tfl = two_transformers_to_list(t_loop_1, t_loop_2);

  free_transformer(t_body);
  free_transformer(pre_body);
  free_transformer(t_init);
  free_transformer(t_enter);
  free_transformer(t_skip);
  free_transformer(t_exit);
  free_transformer(t_inc);

  //free_transformer(t_loop_1);
  //free_transformer(t_loop_2);

  return tfl;
}

transformer complete_repeatloop_transformer(transformer t_body_star,
					    transformer pre,
					    whileloop wl)
{
  list tl = complete_repeatloop_transformer_list(t_body_star, pre, wl);
  transformer ct = transformer_list_to_transformer(tl);
  return ct;
}

/* The transformer computed and stored for a loop is useful to compute the
   loop body precondition, but it is not useful to analyze a higher level
   construct, which need the loop transformer with the exit
   condition. Only Fortran DO loops are handled here. The need for nested
   WHILE loops has not been identified yet. */

/* The index variable is always initialized and then the loop is either
   entered and exited or not entered */
transformer complete_loop_transformer(transformer ltf, transformer pre, loop l)
{
  transformer tf = transformer_undefined;
  transformer t_enter = transformer_undefined;
  transformer t_skip = transformer_undefined;
  /* loop body transformer */
  transformer btf = transformer_undefined;
  range r = loop_range(l);
  statement s = loop_body(l);

  pips_debug(8,"begin with loop precondition\n");
  ifdebug(8) {
    (void) print_transformer(pre);
    pips_debug(8,"and loop transformer:\n");
    (void) print_transformer(ltf);
  }

  /* compute the loop body transformer */
  if(statement_loop_p(s)) {
    /* Since it is not stored, we need to go down recursively. A way to
       avoid this would be to always have sequences as loop
       bodies... Let's hope that perfectly nested loops are neither
       frequent nor deep! */
    loop il = instruction_loop(statement_instruction(s));
    /* compute the loop body precondition */
    transformer raw_ipre = transformer_apply(ltf, pre);
    transformer ipre = transformer_range(raw_ipre);
    ipre = transformer_normalize(ipre, 2);
    /* You do not need a range to recurse. A full precondition with
       arguments is expected by complete_loop_transformer(). Maybe,
       but it's much easier to debug with ranges as they only carry
       the useful information. And the result is correct for
       Semantics-New/for03.c and 04.c */
    transformer st = load_statement_transformer(s);
    btf = complete_loop_transformer(st, ipre, il);
    btf = transformer_normalize(btf, 2);
    free_transformer(ipre);
    free_transformer(raw_ipre);
  }
  else {
    btf = transformer_dup(load_statement_transformer(s));
  }
  /* The final index incrementation is performed later by add_loop_index_exit_value() */
  /* btf = transformer_add_loop_index_incrementation(btf, l, pre); */

  /* compute the transformer when the loop is entered: T o T* */
  /* Semantics-New/for04.c: this leads to t_enter ==
     empty_transformer because the same conditions on the initial
     values of cpi and cpj are preserved in both ltf, which is OK, and
     btf, which is not OK. */
  t_enter = transformer_combine(transformer_dup(ltf), btf);

  ifdebug(8) {
    pips_debug(8, "entered loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, (get_variable_name_t) external_value_name);
  }

  /* add the entry condition */
  /* but it seems to be in t already */
  /* t_enter = transformer_add_loop_index_initialization(t_enter, l); */

  /* It would make sense to apply the incrementation, but this is
     performed as a side effect by add_loop_index_exit_value(), which
     avoids unfeasability wrt the final loop bound. */
  /*
  transformer t_inc = transformer_identity();
  t_inc = transformer_add_loop_index_incrementation(t_inc, l, pre);
  t_enter = transformer_combine(t_enter, t_inc);
  */

  /* add the exit condition, without any information pre to estimate the
     increment */
  transformer tmp = t_enter;
  /* FI: oops, pre is used in spite of previous comment */
  t_enter = add_loop_index_exit_value(t_enter, l, pre);
  transformer_free(tmp);

  ifdebug(8) {
    pips_debug(8, "entered and exited loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, (get_variable_name_t) external_value_name);
  }

  /* add initialization for the unconditional initialization of the loop
     index variable */
  tmp = transformer_undefined_p(pre)?
    transformer_identity() :
    transformer_dup(pre);
  t_skip = add_loop_index_initialization(tmp, l, pre);
  //transformer ipre = transformer_apply(t_skip, pre);
  //transformer ripre = transformer_range(ipre);
  transformer_free(tmp);
  t_skip = add_loop_skip_condition(t_skip, l, pre);
  t_skip = transformer_normalize(t_skip, 2);

  ifdebug(8) {
    pips_debug(8, "skipped loop transformer t_skip=\n");
    fprint_transformer(stderr, t_skip, (get_variable_name_t) external_value_name);
  }

  /* It might be better not to compute useless transformer, but it's more
     readable that way. Since pre is information free, only loops with
     constant lower and upper bound and constant increment can benefit
     from this. */
  /* pre cannot be used as such. the loop initialization must be
     applied first: the previous comment seems to be correct. */
  // if(empty_range_wrt_precondition_p(r, ripre)) {
  if(empty_range_wrt_precondition_p(r, pre)) {
    tf = t_skip;
    free_transformer(t_enter);
  }
  // else if(non_empty_range_wrt_precondition_p(r, ripre)) {
  else if(non_empty_range_wrt_precondition_p(r, pre)) {
    tf = t_enter;
    free_transformer(t_skip);
  }
  else {
    tf = transformer_convex_hull(t_enter, t_skip);
    free_transformer(t_enter);
    free_transformer(t_skip);
  }
  //free_transformer(ipre);
  //free_transformer(ripre);

  tf = transformer_normalize(tf, 2);

  ifdebug(8) {
    pips_debug(8, "full refined loop transformer tf=\n");
    fprint_transformer(stderr, tf, (get_variable_name_t) external_value_name);
    pips_debug(8, "end\n");
  }

  return tf;
}

list complete_loop_transformer_list(transformer ltf, transformer pre, loop l)
{
  list tfl = list_undefined;
  transformer tf = transformer_undefined;
  transformer t_enter = transformer_undefined;
  transformer t_skip = transformer_undefined;
  /* loop body transformer */
  transformer btf = transformer_undefined;
  range r = loop_range(l);
  statement s = loop_body(l);

  pips_internal_error("Function not implemented.");

  pips_debug(8,"begin with loop precondition\n");
  ifdebug(8) {
    (void) print_transformer(pre);
    pips_debug(8,"and loop transformer:\n");
    (void) print_transformer(ltf);
  }

  /* compute the loop body transformer */
  if(statement_loop_p(s)) {
    /* Since it is not stored, we need to go down recursively. A way to
       avoid this would be to always have sequences as loop
       bodies... Let's hope that perfectly nested loops are neither
       frequent nor deep! */
    loop il = instruction_loop(statement_instruction(s));
    /* compute the loop body precondition */
    transformer raw_ipre = transformer_apply(ltf, pre);
    //transformer ipre = transformer_range(raw_ipre);
    //ipre = transformer_normalize(ipre, 2);
    /* You do not need a range to recurse. A full precondition with
       arguments is expected by complete_loop_transformer(). */
    btf = complete_loop_transformer(load_statement_transformer(s), raw_ipre, il);
    btf = transformer_normalize(btf, 2);
    //free_transformer(ipre);
    free_transformer(raw_ipre);
  }
  else {
    btf = transformer_dup(load_statement_transformer(s));
  }
  /* The final index incrementation is performed later by add_loop_index_exit_value() */
  /* btf = transformer_add_loop_index_incrementation(btf, l, pre); */

  /* compute the transformer when the loop is entered: T o T* */
  t_enter = transformer_combine(transformer_dup(ltf), btf);

  ifdebug(8) {
    pips_debug(8, "entered loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, (get_variable_name_t) external_value_name);
  }

  /* add the entry condition */
  /* but it seems to be in t already */
  /* t_enter = transformer_add_loop_index_initialization(t_enter, l); */

  /* It would make sense to apply the incrementation, but this is
     performed as a side effect by add_loop_index_exit_value(), which
     avoids unfeasability wrt the final loop bound. */
  /*
  transformer t_inc = transformer_identity();
  t_inc = transformer_add_loop_index_incrementation(t_inc, l, pre);
  t_enter = transformer_combine(t_enter, t_inc);
  */

  /* add the exit condition, without any information pre to estimate the
     increment */
  transformer tmp = t_enter;
  /* FI: oops, pre is used in spite of previous comment */
  t_enter = add_loop_index_exit_value(t_enter, l, pre);
  transformer_free(tmp);

  ifdebug(8) {
    pips_debug(8, "entered and exited loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, (get_variable_name_t) external_value_name);
  }

  /* add initialization for the unconditional initialization of the loop
     index variable */
  tmp = transformer_undefined_p(pre)?
    transformer_identity() :
    transformer_dup(pre);
  t_skip = add_loop_index_initialization(tmp, l, pre);
  transformer_free(tmp);
  t_skip = add_loop_skip_condition(t_skip, l, pre);
  t_skip = transformer_normalize(t_skip, 2);

  ifdebug(8) {
    pips_debug(8, "skipped loop transformer t_skip=\n");
    fprint_transformer(stderr, t_skip, (get_variable_name_t) external_value_name);
  }

  /* It might be better not to compute useless transformer, but it's more
     readable that way. Since pre is information free, only loops with
     constant lower and upper bound and constant increment can benefit
     from this. */
  if(empty_range_wrt_precondition_p(r, pre)) {
    tf = t_skip;
    free_transformer(t_enter);
  }
  else if(non_empty_range_wrt_precondition_p(r, pre)) {
    tf = t_enter;
    free_transformer(t_skip);
  }
  else {
    tf = transformer_convex_hull(t_enter, t_skip);
    free_transformer(t_enter);
    free_transformer(t_skip);
  }

  tf = transformer_normalize(tf, 2);

  ifdebug(8) {
    pips_debug(8, "full refined loop transformer tf=\n");
    fprint_transformer(stderr, tf, (get_variable_name_t) external_value_name);
    pips_debug(8, "end\n");
  }

  return tfl;
}

/* FI: I'm not sure this function is useful */
transformer complete_whileloop_transformer(transformer ltf,
					   transformer pre,
					   whileloop wl)
{
  transformer t = transformer_undefined;
  evaluation lt = whileloop_evaluation(wl);

  if(evaluation_before_p(lt))
    t = new_complete_whileloop_transformer(ltf, pre, wl, false);
  else
    t = complete_repeatloop_transformer(ltf, pre, wl);

  return t;
}

list complete_whileloop_transformer_list(transformer ltf, // loop transformer
					 transformer pre,
					 whileloop wl)
{
  list tfl = NIL;
  evaluation lt = whileloop_evaluation(wl);

  if(evaluation_before_p(lt))
    tfl = new_complete_whileloop_transformer_list(ltf, pre, wl, false);
  else
    tfl = complete_repeatloop_transformer_list(ltf, pre, wl);

  return tfl;
}

transformer old_complete_whileloop_transformer(transformer ltf,
					   transformer pre,
					   whileloop l)
{
  transformer tf = transformer_undefined;
  transformer t_enter = transformer_undefined;
  transformer t_skip = transformer_undefined;
  /* loop body transformer */
  transformer btf = transformer_undefined;
  expression cond = whileloop_condition(l);
  statement s = whileloop_body(l);
  transformer preb = invariant_wrt_transformer(pre, ltf);

  pips_debug(8,"begin with whileloop precondition\n");
  ifdebug(8) {
    (void) print_transformer(pre);
    pips_debug(8,"and whileloop transformer:\n");
    (void) print_transformer(ltf);
  }

  /* Recompute the exact loop body transformer. This is weird: it
     should have already been done by statement_to_transformer and
     propagated bask. However, we need to recompute it because it has
     not been stored and cannot be retrieved. It might be better to
     use complete_statement_transformer(retrieved t, preb, s). */
  if(statement_loop_p(s)) {
    /* Since it is not stored, we need to go down recursively. A way to
       avoid this would be to always have sequences as loop
       bodies... Let's hope that perfectly nested loops are neither
       frequent nor deep! */
    loop il = instruction_loop(statement_instruction(s));
    btf = complete_loop_transformer(load_statement_transformer(s), pre, il);
  }
  else if(statement_whileloop_p(s)) {
    whileloop il = instruction_whileloop(statement_instruction(s));
    btf = complete_whileloop_transformer(load_statement_transformer(s), pre, il);
  }
  else { /* The proper transformer has been stored. */
    btf = transformer_dup(load_statement_transformer(s));
  }

  /* btf = transformer_add_condition_information(btf, cond, preb, true); */
  t_enter = transformer_combine(transformer_dup(ltf), btf);

  ifdebug(8) {
    pips_debug(8, "entered loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, (get_variable_name_t) external_value_name);
  }

  /* add the exit condition */
  t_enter =  transformer_add_condition_information(t_enter, cond, preb, false);

  ifdebug(8) {
    pips_debug(8, "entered and exited loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, (get_variable_name_t) external_value_name);
  }

  /* add initialization for the unconditional initialization of the loop
     index variable */
  t_skip = transformer_undefined_p(pre)?
    transformer_identity() :
    transformer_dup(pre);
  t_skip = transformer_add_condition_information(t_skip, cond, pre, false);

  ifdebug(8) {
    pips_debug(8, "skipped loop transformer t_skip=\n");
    fprint_transformer(stderr, t_skip, (get_variable_name_t) external_value_name);
  }

  /* It might be better not to compute useless transformer, but it's more
     readable that way. Since pre is information free, only loops with
     constant lower and upper bound and constant increment can benefit
     from this. */
  if(!transformer_undefined_p(pre) && condition_false_wrt_precondition_p(cond, pre)) {
    tf = t_skip;
    free_transformer(t_enter);
  }
  else if(!transformer_undefined_p(pre) && condition_true_wrt_precondition_p(cond, pre)) {
    tf = t_enter;
    free_transformer(t_skip);
  }
  else {
    tf = transformer_convex_hull(t_enter, t_skip);
    free_transformer(t_enter);
    free_transformer(t_skip);
  }

  ifdebug(8) {
    pips_debug(8, "full refined loop transformer tf=\n");
    fprint_transformer(stderr, tf, (get_variable_name_t) external_value_name);
    pips_debug(8, "end\n");
  }

  return tf;
}

/* Recompute a fixpoint conditionnally to a valid precondition for all iterations */
static transformer recompute_loop_transformer(loop l, transformer pre)
{
  statement s = loop_body(l);
  entity i = loop_index(l);
  list list_mod = list_undefined;
  list list_old = list_undefined;
  list list_projectable = list_undefined;
  list list_val = list_undefined;
  transformer tf = load_statement_transformer(s);
  transformer tfb = transformer_add_loop_index_incrementation(transformer_dup(tf), l, pre);
  transformer new_tf = transformer_undefined;
  transformer preb = transformer_dup(pre);
  Psysteme sc_pre = SC_UNDEFINED;
  Psysteme sc = predicate_system(transformer_relation(tfb));
  
  ifdebug(5) {
    pips_debug(5, "Begin with precondition:\n");
    print_transformer(pre);
    pips_assert("Precondition pre is consistent", 
		transformer_internal_consistency_p(pre));
    pips_debug(5, "and transformer:\n");
    print_transformer(tf);
  }

  /* get rid of information modified by the loop body or the loop header */
  list_mod = dup_arguments(transformer_arguments(tf));
  arguments_add_entity(list_mod, i);
  gen_list_and(&list_mod, transformer_arguments(preb));
  ifdebug(5) {
    pips_debug(5, "Variables eliminated from the preconditions"
	       " because they are modified by tf:");
    print_arguments(list_mod);
  }
  list_val = variables_to_values(list_mod);
  /* get rid of old values in preb which are meaning less in tf */
  list_old = arguments_difference(transformer_arguments(preb),
				  transformer_arguments(tf));
  ifdebug(5) {
    pips_debug(5, "Variables eliminated from the preconditions"
	       " because they are linked to the module entry point:");
    print_arguments(list_old);
  }
  /* do not try to eliminate the same variable twice */
  gen_list_and_not(&list_old, list_mod);

  list_val = gen_nconc(list_val, variables_to_old_values(list_old));
  ifdebug(5) {
    pips_debug(5, "Values which should be eliminated from the preconditions:");
    print_arguments(list_val);
  }
  /* Not all values can be projected in preb because some may appear in tf
     and not in preb */
  list_projectable = transformer_projectable_values(preb);
  gen_list_and(&list_val, list_projectable);
  ifdebug(5) {
    pips_debug(5, "Values which can be eliminated from the preconditions:");
    print_arguments(list_val);
  }
  preb = transformer_projection(preb, list_val);

  /* add the remaining conditional information to the loop transformer */
  sc_pre = predicate_system(transformer_relation(preb));
  sc = sc_append(sc, sc_pre);
  predicate_system(transformer_relation(tfb)) = sc;
  ifdebug(1) {
    if(!transformer_internal_consistency_p(tfb)) {
      print_transformer(tfb);
      sc_fprint(stderr, sc, (char * (*)(Variable)) external_value_name);
    }
    pips_assert("Conditional loop body transformer is consistent",
		transformer_internal_consistency_p(tfb));
  }

  tfb = transformer_add_loop_index_incrementation(tfb, l, pre);

  new_tf = (* transformer_fix_point_operator)(tfb);
  new_tf = transformer_add_loop_index_initialization(new_tf, l, pre);

  free_transformer(preb);
  gen_free_list(list_mod);
  gen_free_list(list_val);

  ifdebug(5) {
    pips_debug(5, "End with fixpoint:\n");
    print_transformer(new_tf);
  }

  return new_tf;
}

/* NOT USED. NOT FULLY IMPLEMENTED YET. SHOULD BE REDUNDANT WITH whileloop_to_tramsformer() */
/* Recompute a fixpoint conditionnally to a valid precondition for all iterations */
/* Could/Should be later called from whileloop_to_postcondition() */
static transformer __attribute__ ((unused)) recompute_whileloop_transformer(whileloop wl, transformer pre)
{
  transformer new_tf = transformer_undefined;
  pips_assert("To shut up gcc", wl==wl && pre==pre);
  return new_tf;
}

/* FI: I do not have one test case to show that this function is of some use. */

static transformer loop_body_transformer_add_entry_and_iteration_information(transformer tfb, transformer pre)
{
  /* add information about the old value of tfb as convex hull of the
     entry precondition and of tfb's own image. For instance, if variable
     I is equal to 0 on entry and set ot either 0 or 1, the loop is always
     started with in [0, 1] */
  transformer initial = transformer_range(pre);
  transformer next = transformer_range(tfb);
  transformer pre_body = transformer_convex_hull(initial, next);


  ifdebug(8) {
    pips_debug(5, "Begin with body transformer:\n");
    print_transformer(tfb);
    pips_debug(5, "initial precondition:\n");
    print_transformer(initial);
    pips_debug(5, "next precondition:\n");
    print_transformer(next);
    pips_debug(5, "body precondition:\n");
    print_transformer(pre_body);
  }


  tfb = transformer_domain_intersection(tfb, pre_body);

  free_transformer(initial);
  free_transformer(next);
  free_transformer(pre_body);

  ifdebug(8) {
    pips_debug(5, "End with new body transformer:\n");
    print_transformer(tfb);
  }

  return tfb;
}

/* This function computes the effect of K loop iteration, with K positive.
 * This function does not take the loop exit into account because its result
 * is used to compute the precondition of the loop body.
 * Hence the loop exit condition only is added when preconditions are computed.
 * This is confusing when transformers are prettyprinted with the source code.
 */

transformer standard_whileloop_to_transformer(whileloop l,
					      transformer pre,
					      list e) // effects of whileloop l
{
  /* loop transformer tf = tfb* or tf = tfb+ or ... */
  transformer tf;
  /* loop body transformer */
  transformer tfb;
  expression cond = whileloop_condition(l);
  statement s = whileloop_body(l);

  pips_debug(8,"begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT)) {
    transformer pre_n = transformer_undefined;

    if(transformer_undefined_p(pre)) {
      pre_n = transformer_identity();
    }
    else {
      /* Make sure not to leave too much information in pre. Perform a very
	 simplistic fix point based on effects. */
      transformer tf_star = effects_to_transformer(e);

      pre_n = invariant_wrt_transformer(pre, tf_star);
      free_transformer(tf_star);
    }
    /* If the while entry condition is usable, it must be added
     * on the old values
     */
    /* I'd like to use pre_n as context to evaluate the condition cond,
       but I'm not sure it's safe (Francois Irigoin) */
    /* Side effects in cond are ignored! */
    pre_n = precondition_add_condition_information(pre_n, cond,
						   transformer_undefined, true);

    ifdebug(8) {
      pips_debug(8, "Precondition for loop body pre_n=\n");
      fprint_transformer(stderr, pre_n, (get_variable_name_t) external_value_name);
    }

    /* compute the whileloop body transformer, including the initial conditions */
    tfb = transformer_dup(statement_to_transformer(s, pre_n));

    /* The convex hull of the image of pre and of the image of tfb can be
       added as conditions on tfb's domain, if some information is available in pre */
    if(!transformer_undefined_p(pre))
      tfb = loop_body_transformer_add_entry_and_iteration_information(tfb, pre);

    /* compute tfb's fix point according to pips flags */
    if(pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
      tf = transformer_halbwachs_fix_point(tfb);
    }
    else if ((!transformer_undefined_p(pre)
	     && condition_false_wrt_precondition_p(cond, pre))
	     /* The second clause is probably stronger than the first one */
	     || transformer_empty_p(pre_n)) {
      /* The loop is never entered */
      /* Side effects of the condition evaluation should be taken into account */
      tf = transformer_identity();
      /* A temporary variable is expected as first argument. There is something to fix. */
      /* tf = any_expression_to_transformer(); */
    }
    else if(transformer_empty_p(tfb)) {
      /* The loop is never exited, e.g. because there is a STOP or an infinite loop inside */
      tf = transformer_empty();
    }
    else {
      transformer ftf = (* transformer_fix_point_operator)(tfb);

      if(*transformer_fix_point_operator==transformer_equality_fix_point) {
	Psysteme fsc = predicate_system(transformer_relation(ftf));
	Psysteme sc = SC_UNDEFINED;

	/* Dirty looking fix for a fix point computation error:
	 * sometimes, the basis is restricted to a subset of
	 * the integer scalar variables. Should be useless with proper
	 * fixpoint opertors.
	 */
	tf = effects_to_transformer(e);
	sc = (Psysteme) predicate_system(transformer_relation(tf));

	sc = sc_append(sc, fsc);

	free_transformer(ftf);
      }
      else {
	tf = ftf;
      }

      ifdebug(8) {
	pips_debug(8, "intermediate fix-point tf=\n");
	fprint_transformer(stderr, tf, (get_variable_name_t) external_value_name);
      }

    }

    free_transformer(pre_n);

    /* we have a problem here: to compute preconditions within the
       whileloop body we need a tf using private variables; to return
       the loop transformer, we need a filtered out tf; only
       one hook is available in the ri..; let'a assume there
       are no private variables and that if they are privatizable
       they are not going to get in our way */
  }
  else {
    /* basic cheap version: do not use the whileloop body transformer and
       avoid fix-points; local variables do not have to be filtered out
       because this was already done while computing effects */

    /* The loop body transformers could benefit from pre_n instead of
       transformer_undefined, but who would think of combining these
       two options? */
    (void) statement_to_transformer(s, transformer_undefined);
    tf = effects_to_transformer(e);
  }

  ifdebug(8) {
    (void) fprintf(stderr,"%s: %s\n","standard_whileloop_to_transformer",
		   "resultat tf =");
    (void) print_transformer(tf);
  }
  pips_debug(8,"end\n");
  return tf;
}

transformer whileloop_to_transformer(whileloop l,
				     transformer pre,
				     list e) /* effects of whileloop l */
{
  transformer t = transformer_undefined;
  evaluation lt = whileloop_evaluation(l);

  if(evaluation_before_p(lt))
    t = new_whileloop_to_transformer(l, pre, e);
  else
    t = repeatloop_to_transformer(l, pre, e);
  return t;
}

#if 0
list whileloop_to_transformer_list(whileloop l __attribute__ ((unused)),
				   transformer pre __attribute__ ((unused)),
				   list e __attribute__ ((unused)))
{
  list tfl = NIL;
  //evaluation lt = whileloop_evaluation(l);

  pips_internal_error("This function should never be called.");

  /*
  if(evaluation_before_p(lt))
    tfl = new_whileloop_to_transformer_list(l, pre, e);
  else
    tfl = repeatloop_to_transformer_list(l, pre, e);
  */
  return tfl;
}
#endif

transformer whileloop_to_k_transformer(whileloop l,
				       transformer pre,
				       list e, /* effects of whileloop
						  l */
				       int k) // number of iterations
{
  transformer t = transformer_undefined;
  evaluation lt = whileloop_evaluation(l);

  if(evaluation_before_p(lt))
    t = new_whileloop_to_k_transformer(l, pre, e, k);
  else
    pips_internal_error("repeatloop_to_k_transformer() not implemented.");
    //t = repeatloop_to_k_transformer(l, pre, e);
  return t;
}

transformer any_loop_to_postcondition(statement body,
				      transformer t_init,
				      transformer t_enter,
				      transformer t_skip,
				      transformer t_body_star,
				      transformer t_body,
				      transformer t_next,
				      transformer t_inc,
				      transformer t_exit,
				      transformer pre)
{
  /* The precondition to propagate in the body is:
   *
   * p_body = (t_init ; t_enter)(pre) + (t_body_star ; t_body ; t_next) (pre)
   *
   * The loop postcondition to return is:
   *
   * post = (t_init ; t_skip)(pre) + (t_body_star ; t_body ; t_inc ; t_exit) (pre)
   */
  transformer p_body = transformer_undefined;
  transformer post = transformer_undefined;
  /* To restrict the state at the beginning of the last iteration:
     should be part of t_body_star */
  //transformer post_enter = transformer_range(t_enter);
  //transformer post_continue = transformer_range(t_next);
  //transformer pre_iteration = transformer_convex_hull(post_enter, post_continue);
  /* Decompose the computation of p_body */
  transformer p_body_1 = transformer_apply(t_enter, transformer_apply(t_init, pre));
  transformer p_body_2 = transformer_apply(t_body_star, pre);
  //transformer p_body_3 = transformer_apply(pre_iteration, p_body_2);
  transformer p_body_4 = transformer_apply(t_body, p_body_2);
  transformer p_body_5 = transformer_apply(t_next, p_body_4);
  transformer a_post = transformer_undefined;
  /* Decompose the computation of post */
  transformer post_1 = transformer_apply(t_skip, transformer_apply(t_init, pre));
  transformer post_2 = transformer_apply(t_body_star, pre);
  //transformer post_3 = transformer_apply(pre_iteration, post_2);
  transformer post_4 = transformer_apply(t_body, post_2);
  transformer post_5 = transformer_apply(t_inc, post_4);
  transformer post_6 = transformer_apply(t_exit, post_5);

  p_body = transformer_convex_hull(p_body_1, p_body_5);
  a_post = statement_to_postcondition(p_body, body);

  /* a_post should be used because it is more accurate than direct
     recomputation, but we chose for the time being to recompute post
     entirely */
  post = transformer_convex_hull(post_1, post_6);

  /* Get rid of now useless transformers */

  free_transformer(a_post);

  //free_transformer(post_enter);
  //free_transformer(post_continue);
  //free_transformer(pre_iteration);

  free_transformer(p_body_1);
  free_transformer(p_body_2);
  //free_transformer(p_body_3);
  free_transformer(p_body_4);
  free_transformer(p_body_5);

  free_transformer(post_1);
  free_transformer(post_2);
  //free_transformer(post_3);
  free_transformer(post_4);
  free_transformer(post_5);
  free_transformer(post_6);

  return post;
}

transformer forloop_to_postcondition(transformer pre, forloop fl, transformer t_body_star)
{
  transformer post = transformer_undefined;

  statement body_s = forloop_body(fl);
  transformer t_body = load_statement_transformer(body_s);

  expression init_e = forloop_initialization(fl);
  /* This function does not seem to exist because we only used
     statement effects in the past and because those are stored in the
     database. */
  transformer t_init = safe_expression_to_transformer(init_e, pre);
  transformer post_init = transformer_apply(t_init, pre);

  expression cond_e = forloop_condition(fl);
  transformer t_skip = condition_to_transformer(cond_e, post_init, false);
  transformer t_enter = condition_to_transformer(cond_e, post_init, true);
/* An effort could be made to compute the precondition for t_exit,
     especially if the precondition to t_inc is available. */
  transformer t_continue = condition_to_transformer(cond_e, transformer_undefined, true);
  transformer t_exit = condition_to_transformer(cond_e, transformer_undefined, false);

  expression inc_e = forloop_increment(fl);
  /* An effort could be made to compute the precondition for t_inc */
  transformer t_inc = safe_expression_to_transformer(inc_e, transformer_undefined);

  transformer t_next = transformer_combine(transformer_dup(t_inc), t_continue);

  post = any_loop_to_postcondition(body_s,
				   t_init,
				   t_enter,
				   t_skip,
				   t_body_star,
				   t_body,
				   t_next,
				   t_inc,
				   t_exit,
				   pre);

  /* Clean up memory */

  free_transformer(t_init);
  free_transformer(post_init);
  free_transformer(t_skip);
  free_transformer(t_enter);
  free_transformer(t_continue);
  free_transformer(t_exit);
  free_transformer(t_inc);
  free_transformer(t_next);

    return post;
}

transformer repeatloop_to_postcondition(transformer pre, whileloop wl, transformer t_body_star)
{
  transformer post = transformer_undefined;

  statement body_s = whileloop_body(wl);
  transformer t_body = load_statement_transformer(body_s);
  transformer t_body_c = complete_statement_transformer(t_body, pre, body_s);
  transformer t_init = transformer_identity();
  transformer post_init = copy_transformer(pre);
  expression cond_e = whileloop_condition(wl);
  transformer t_continue = condition_to_transformer(cond_e, transformer_undefined, true);
  transformer t_exit = condition_to_transformer(cond_e, transformer_undefined, false);
  transformer t_skip = transformer_empty();
  //transformer t_skip = transformer_combine(copy_transformer(t_body), t_exit);
  transformer t_enter = transformer_identity();
  //transformer t_enter = transformer_combine(copy_transformer(t_body), t_continue);
  /* An effort could be made to compute the precondition for t_exit,
     especially if the precondition to t_inc is available. */
  transformer t_inc = transformer_identity();
  transformer post_1 = transformer_undefined;
  transformer post_2 = transformer_undefined;
  transformer post_3 = transformer_undefined;
  transformer post_4 = transformer_undefined;
  transformer post_5 = transformer_undefined;
  transformer post_6 = transformer_undefined;

  post = any_loop_to_postcondition(body_s,
				   t_init,
				   t_enter,
				   t_skip,
				   t_body_star,
				   t_body_c,
				   t_continue, //since t_inc is ineffective
				   t_inc,
				   t_exit,
				   pre);

  /* The repeat loop does not fit well in the current generic
   * approach. Most of the basic transformers such as t_init, t_skip,
   * t_enter,... are meaningless. Instead of dealing with zero or at
   * least one iteration, we have to deal with one or at least two.
   *
   * post = (t_body_c ; t_exit)(pre) +
   *        (t_body_c ; t_continue ; t_body_star ; t_body_c ; t_exit)(pre)
   *
   * where we assume that t_body_star includes the continuation condition. 
   */

  free_transformer(post);

  post_1 = transformer_apply(t_body_c, pre);
  post_2 = transformer_apply(t_exit, post_1);

  post_3 = transformer_apply(t_continue, post_1);
  post_4 = transformer_apply(t_body_star, post_3);
  post_5 = transformer_apply(t_body_c, post_4);
  post_6 = transformer_apply(t_exit, post_5);

  post = transformer_convex_hull(post_2, post_6);

  /* Clean up memory */

  free_transformer(t_init);
  free_transformer(post_init);
  free_transformer(t_skip);
  free_transformer(t_enter);
  free_transformer(t_continue);
  free_transformer(t_exit);
  free_transformer(t_inc);
  free_transformer(t_body_c);

  free_transformer(post_1);
  free_transformer(post_2);
  free_transformer(post_3);
  free_transformer(post_4);
  free_transformer(post_5);
  free_transformer(post_6);

    return post;
}

transformer loop_to_postcondition(transformer pre,
				  loop l,
				  transformer tf)
{
  transformer post = transformer_undefined;
  statement s = loop_body(l);
  range r = loop_range(l);

  pips_debug(8,"begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT) && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
    pips_internal_error("Halbwachs not implemented");
  }
  else {
    /* pips_internal_error("Equality option not implemented"); */
    /* the standard version should be OK for SEMANTICS_EQUALITY_INVARIANTS... */

    /* basic cheap version: add information on loop index in pre
       and propagate preb downwards in the loop body; compute the
       loop postcondition independently using the loop transformer */
    /* preb = precondition for loop body; includes a lousy fix-point */
    transformer preb = transformer_dup(pre);

    /* Get rid of information related to variables modified in
     * iterations of the loop body (including loop indices).
     *
     * Apparently, the loop index incrementation is not in tf...
     * according to a test with DO I = I, N
     * although it should be... according to the caller of this function
     * and the display of loop transformers!
     */
    preb = transformer_combine(preb, tf);

    /* Triolet's good loop algorithm */
    preb = add_good_loop_conditions(preb, l);

    /* It might be useful to normalize preb and to detect unfeasibility.
     * I choose not to do it because:
     *  - it almost never happens in user code
     *  - it's time consuming
     *  - when it happens in automatically generated code,
     *    dead code elimination is performed.
     *
     * So basically, I shift the burden from precondition computation
     * to dead code elimination. Dead loop testing must use a strong
     * feasibility test.
     *
     * If this decision is reversed, dead code elimination can be
     * speeded-up.
     *
     * Note that add_good_loop_conditions() can test trivial cases
     * automatically generated.
     *
     * Decision reverted: Francois Irigoin, 4 June 1997
     */

    /* preb can now be used to obtain a refined tf, for instance to show that a
       variable is monotonically increasing. */
    if(get_bool_property("SEMANTICS_RECOMPUTE_FIX_POINTS_WITH_PRECONDITIONS")) {
      transformer new_tf = recompute_loop_transformer(l, preb);

      free_transformer(preb);
      preb = transformer_dup(pre);
      preb = transformer_combine(preb, new_tf);
      preb = add_good_loop_conditions(preb, l);
    }


    /* FI: this is not correct when an invariant is found; we should add one
     * more incrementation of I (to modify the output of the invariant and
     * express the fact that the loop bound is no more true, at least when
     * the increment is one. 6 July 1993
     *
     * Note 1: this comments was wrong! See Validation/Semantics/induc1.f
     * Note 2: but the result was wrong anyway because it assumed that at
     * least one iteration always was performed.
     * Note 3: This is fixed by reverting the above decision.
     */
    if(empty_range_wrt_precondition_p(r, pre)) {
      pips_debug(8, "The loop is never executed\n");

      /* propagate an impossible precondition in the loop body */
      (void) statement_to_postcondition(transformer_empty(), s);
      /* The loop body precondition is not useful any longer */
      free_transformer(preb);

      transformer tmp = transformer_dup(pre);
      post = add_loop_index_initialization(tmp, l, pre);
      transformer_free(tmp);
    }
    else if(non_empty_range_wrt_precondition_p(r, pre)) {
      debug(8, "loop_to_postcondition", "The loop certainly is executed\n");

      /* propagate preconditions in the loop body and compute its postcondition... */
      post = statement_to_postcondition(preb, s);

      /* ... or compute directly the postcondition using the loop body transformer.
       * This second approach is slightly less precise because the transformer
       * cannot use preconditions to avoid some convex approximations. */
      /* post = transformer_apply(tf, pre); */

      /* The loop body effects should be passed as fourth argument
       * to check that the value of the upper bound expression is
       * not modified when the body is executed. But it is not available
       * and it is not yet used by ad_loop_index_exit_value()!
       */
      transformer tmp = post;
      post = add_loop_index_exit_value(post, l, pre);
      transformer_free(tmp);
    }
    else {
      /* First version: OK, but could be better! */
      /* transformer postloop = transformer_apply(tf, pre); */
      /* pre must be copied because sc_convex_hull updates is arguments
       * and may make them inconsistent when the two bases are merged
       * to perform the convex hull in a common vector space
       */
      /*
	transformer preloop = transformer_dup(pre);
	debug(8, "loop_to_postcondition", "The loop may be executed or not\n");

	post = transformer_convex_hull(postloop, preloop);
	transformer_free(postloop);
	transformer_free(preloop);
      */

      /* Second version: assume it is empty or non-empty and perform
       * the convex hull of both
       * (should be checked on Validation/Semantics/induc1.f)
       */
      transformer t_init = loop_initialization_to_transformer(l, pre);
      transformer post_ne = transformer_apply(t_init, pre);
      transformer post_al = transformer_undefined;
      transformer lpre = transformer_range(post_ne);

      pips_debug(8, "The loop may be executed or not\n");

      /* propagate preconditions in the loop body */
      post_al =  statement_to_postcondition(preb, s);
      /* post_al = transformer_apply(tf, pre); */

      /* We should add (when possible) the non-entry condition in post_ne!
       * For instance, DO I = 1, N leads to N <= 0
       */
      post_ne = add_loop_skip_condition(post_ne, l, lpre);
      // FI: already done with t_init
      //post_ne = add_loop_index_initialization(post_ne, l, lpre);
      free_transformer(lpre);

      transformer tmp = post_al;
      post_al = add_loop_index_exit_value(post_al, l, post_al);
      transformer_free(tmp);

      ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		       "Never executed: post_ne %p =");
	(void) print_transformer(post_ne);
	(void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		       "Always executed: post_al %p = ");
	(void) print_transformer(post_al);
      }
      post = transformer_convex_hull(post_ne, post_al);
      transformer_free(post_ne);
      transformer_free(post_al);
    }

    /* FI: Cannot be freed because it was stored for statement s:
       transformer_free(preb);
       did I mean to free pre, which is useless since it was changed into
       preb?
    */
  }

  ifdebug(8) {
    (void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		   "resultat post =");
    (void) print_transformer(post);
  }
  pips_debug(8,"end\n");
  return post;
}

transformer loop_to_total_precondition(
    transformer t_post,
    loop l,
    transformer tf,
    transformer context)
{
  transformer t_pre = transformer_undefined;
  statement b = loop_body(l);
  range r = loop_range(l);

  pips_debug(8,"begin\n");

  if(get_bool_property("SEMANTICS_RECOMPUTE_FIX_POINTS_WITH_PRECONDITIONS")) {
    transformer new_tf = transformer_undefined;
    transformer preb  = transformer_dup(context);

    preb = transformer_combine(preb, tf);

    /* Triolet's good loop algorithm */
    preb = add_good_loop_conditions(preb, l);

    new_tf = recompute_loop_transformer(l, preb);

    free_transformer(preb);
    /* preb = transformer_dup(pre); */
    preb = transformer_combine(preb, new_tf);
    preb = add_good_loop_conditions(preb, l);
  }

  if(empty_range_wrt_precondition_p(r, context)) { /* The loop is never executed */
    transformer tf_init = loop_initialization_to_transformer(l, context);
    transformer tf_empty = transformer_empty();
    transformer b_t_pre = transformer_undefined;

    pips_debug(8, "The loop is never executed\n");

    b_t_pre = statement_to_total_precondition(tf_empty, b);
    pips_assert("The body total precondition is consistent",
		transformer_consistency_p(b_t_pre));

    /* impact of loop index initialization, i.e. I = IT(J),.. or I = K/L,.. */
    t_pre = transformer_inverse_apply(tf_init, t_post);
    t_pre = transformer_to_domain(t_pre);

    free_transformer(tf_empty);
    free_transformer(tf_init);
  }
  else /* the loop may be entered for sure, or entered or not */ {

    transformer t_pre_ne = transformer_undefined;
    transformer t_pre_al = transformer_undefined;
    /* We need a fix_point without the initial condition but with the exit
       condition */
    transformer btf = transformer_dup(load_statement_transformer(b));
    transformer ltf = transformer_undefined;
    transformer b_t_post = transformer_undefined;
    transformer b_t_pre = transformer_undefined;
    transformer tf_init = loop_initialization_to_transformer(l, context);

    btf = transformer_add_loop_index_incrementation(btf, l, context);
    ltf =  (* transformer_fix_point_operator)(btf);
    ltf = add_loop_index_exit_value(ltf, l, context); /* Also performs
                                                         last
                                                         incrementation */
    b_t_post = transformer_inverse_apply(ltf, t_post);
    b_t_post = transformer_to_domain(b_t_post);
    b_t_pre = statement_to_total_precondition(b_t_post, b);
    pips_assert("The body total precondition is consistent",
		transformer_consistency_p(b_t_pre));
    t_pre_al = transformer_inverse_apply(tf_init, b_t_pre);
    t_pre_al = transformer_to_domain(t_pre_al);

    /* free_transformer(b_t_pre); it is associated to the loop body!*/
    free_transformer(b_t_post);
    free_transformer(btf);
    free_transformer(ltf);

    if(non_empty_range_wrt_precondition_p(r, context)) {
      /* The loop is always entered */
      t_pre = t_pre_al;

      pips_debug(8, "The loop certainly is executed\n");
    }
    else /* The loop may be skipped or entered */ {

      pips_debug(8, "The loop may be executed or not\n");

      /* skipped case computed here too */
      t_pre_ne = transformer_inverse_apply(tf_init, t_post);
      t_pre_ne = transformer_to_domain(t_pre_ne);
      t_pre_ne = add_loop_skip_condition(t_pre_ne, l, context);


      ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		       "Never executed: t_pre_ne =");
	(void) print_transformer(t_pre_ne);
	(void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		       "Always executed: post_al =");
	(void) print_transformer(t_pre_al);
      }

      t_pre = transformer_convex_hull(t_pre_ne, t_pre_al);
      transformer_free(t_pre_ne);
      transformer_free(t_pre_al);
    }
    free_transformer(tf_init);
  }

  ifdebug(8) {
    pips_debug(8, "resultat t_pre =%p", t_pre);
    (void) print_transformer(t_pre);
    pips_debug(8,"end\n");
  }
  return t_pre;
}

/**/
transformer whileloop_to_postcondition(
    transformer pre,
    whileloop l,
    transformer tf)
{
  transformer post = transformer_undefined;
  statement s = whileloop_body(l);
  expression c = whileloop_condition(l);

  pips_debug(8, "begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT)
     && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
    pips_internal_error("Halbwachs not implemented");
  }

  if(false_condition_wrt_precondition_p(c, pre)) {
    transformer c_t = condition_to_transformer(c, pre, false);
    pips_debug(8, "The loop is never executed\n");

    /* propagate an impossible precondition in the loop body */
    (void) statement_to_postcondition(transformer_empty(), s);
    /* do not add the exit condition since it is redundant with pre,
       but take care of side effects in the condition c */
    /* transformer_apply() generates a lot of warnings */
    post = transformer_combine(copy_transformer(pre), c_t);
    //post = transformer_apply(c_t, pre);
    free_transformer(c_t);
  }
  else { /* The loop may be entered at least once. */
    transformer pre_next = transformer_dup(pre);
    transformer pre_init =
      precondition_add_condition_information(transformer_dup(pre),
					     c, pre, true);
    // FI: this should work but is not compatible with the following
    //code; by definition, tf also include the side effect of pre
    //transformer pre_init = transformer_apply(c_t, pre);
    transformer preb = transformer_undefined; // body precondition
    transformer postb = transformer_undefined; // body postcondition
    transformer tb = load_statement_transformer(s); // body transformer
    int k = get_int_property("SEMANTICS_K_FIX_POINT");

    /* The standard transformer tb is not enough, especially if the
       loop body s is a loop since then it is not even the statement
       transformer, but more generally we do not want the identity to
       be taken into account in tb since it is already added with
       P0. So we would like to guarantee that at least one state change
       occurs: we are not interested in identity iterations. For
       instance, if s is a loop, this means that the loop is entered,
       except if the loop condition has side effects.

       To recompute this transformer, we use a pre_fuzzy=T*(P0)
       because we have nothing better.

       complete_statement_transformer() is not really useful here
       because we usually do not have tighly nested while loops.
    */
    transformer pre_fuzzy = transformer_apply(tf, pre);
    //tb = complete_non_identity_statement_transformer(tb, pre_fuzzy, s);
    list btl = list_undefined;
    int fbtll = 0;

    if(get_bool_property("SEMANTICS_USE_TRANSFORMER_LISTS")) {
      list fbtl = statement_to_transformer_list(s, pre_fuzzy);
      fbtll = (int) gen_length(fbtl);
      // filter out transformers that do not modify the state
      // FI: this is not a general approach since it depends on the
      // framework used and on the other variables, but it helps!
      btl = transformer_list_to_active_transformer_list(fbtl);
      gen_full_free_list(fbtl);

      if(gen_length(btl)==0) {
	tb = empty_transformer(transformer_identity());
      }
      else if(gen_length(btl)==1) {
	/* Nothing special in the loop body: no tests, no while,... */
	if(fbtll==1)
	  tb = complete_statement_transformer(tb, pre_fuzzy, s);
	else
	  // FI: not to sure about reuse and memory leaks...
	  tb = copy_transformer(TRANSFORMER(CAR(btl)));
      }
      else {
	/* Recompute the body transformer without taking identity
	   transformers into account. This is not enough because the
	   decision about "activity" should be made dimension by
	   dimension. We cannot get good result in general with a
	   convex hull performed here: only specific cases are
	   handled. We need instead a complex formulae to compute the
	   loop precondition as a function of p_0 and all t_i*/
	/* btl is copied because the function below frees at least
	   its components */
	tb = active_transformer_list_to_transformer(gen_full_copy_list(btl));
      }
    }
    else {
      tb = complete_statement_transformer(tb, pre_fuzzy, s);
    }

    pips_debug(8, "The loop may be executed and preconditions must"
	       " be propagated in the loop body\n");

    if(k==1) {
      if(!get_bool_property("SEMANTICS_USE_TRANSFORMER_LISTS")
	 || gen_length(btl)==1) {
	/* The loop fix point transformer T* could be used to obtain the
	 * set of stores for any number of iterations, including
	 * 0. Instead, use T+ and a convex hull with the precondition for
	 * the first iteration, which preserves more information when the
	 * fixpoint is not precise:
	 *
	 * P^* = P_0 U cond(c)(tb(cond(c)(tb^*(P_0))))
	 *
	 * Bertrand Jeannet suggests that we compute P0 U T(P0) U
	 * T^2(P0) U T_3(P0) where T_3 is the transitive closure
	 * obtained for iterations 3 to infinity by setting the initial
	 * iteration number k to 3 before projection. NSAD 2010. No test
	 * case has been forwarded to show that this would be useful.
	 */
	// FI: I do not know why pre_next==pre is used instead of
	// pre_init==P_0 in the statement just below
	pre_next = transformer_combine(pre_next, tf);
	pre_next = precondition_add_condition_information(pre_next, c,
							  pre_next, true);
	pre_next = transformer_combine(pre_next, tb);
	pre_next = precondition_add_condition_information(pre_next, c,
							  pre_next, true);
	preb = transformer_convex_hull(pre_init, pre_next);
      }
      else { // transformer lists are used and at least two
	     // transformers have been found
	transformer c_t = condition_to_transformer(c, pre_fuzzy, true);
	//transformer preb1 = transformer_list_closure_to_precondition(btl, c_t, pre_init);
	transformer preb1 = transformer_list_multiple_closure_to_precondition(btl, c_t, pre_init);
	//pre_next = transformer_combine(pre_next, tf);
	//pre_next = precondition_add_condition_information(pre_next, c,
	//						  pre_next, true);
      //pre_next = transformer_combine(pre_next, tb);
      //pre_next = precondition_add_condition_information(pre_next, c,
	//					  pre_next, true);
    //transformer preb2 = transformer_convex_hull(pre_init, pre_next);
    //pips_assert("The two preconditions have the same arguments",
    //		    arguments_equal_p(transformer_arguments(preb1),
    //			      transformer_arguments(preb2)));
	// FI: the intersection generates overflows
	//preb = transformer_intersection(preb1, preb2);
	preb = preb1;
      }
    }
    else if (k==2) {
      /* We need the loop effects to recompute the unrolled
	 transformer. Let's use NIL to start with... disaster.
	 Let's use the body effects and hope for no side effects in
	 loop condition.
      */
      list bel = load_cumulated_rw_effects_list(s); // Should be lel
      transformer tf2 = whileloop_to_k_transformer(l, pre, bel, 2);
      transformer pre_next2 = transformer_undefined;
      pre_next = transformer_combine(pre_next, tf2);
      pre_next = precondition_add_condition_information(pre_next, c,
							pre_next, true);
      pre_next = transformer_combine(pre_next, tb);
      pre_next = precondition_add_condition_information(pre_next, c,
							pre_next, true);
      preb = transformer_convex_hull(pre_init, pre_next);

      /* FI: since pre_next is no longer useful, pre_next2 could be
	 avoided. It just makes debugging easier. */
      pre_next2 = copy_transformer(pre_next);
      pre_next2 = precondition_add_condition_information(pre_next2, c,
							 pre_next2, true);
      pre_next2 = transformer_combine(pre_next2, tb);
      pre_next2 = precondition_add_condition_information(pre_next2, c,
							 pre_next2, true);
      preb = transformer_convex_hull(preb, pre_next2);
      free_transformer(tf2);
      free_transformer(pre_next2);
    }
    else
      pips_user_error("Unexpected value %d for k.\n", k);

    free_transformer(pre_fuzzy);

    /* propagate preconditions in the loop body and get its postcondition */

    postb = statement_to_postcondition(preb, s);

    if(true_condition_wrt_precondition_p(c, pre)) {
      /* At least one iteration is executed. The postcondition can be
       * computed into three different ways:
       *
       *  - use the loop body postcondition and apply the loop exit
       * condition transformer or precondition_add_condition_information;
       *
       *  - or use the loop precondition, the loop transformer, the loop
       * entry condition, the loop body transformer and the loop exit
       * transformer;
       *
       * - or use both and use their intersection as unique
       * postcondition (the added redundant information seems to
       * result in *less* information after a projection for w09.f,
       * Halbwachs car example).
       *
       * The second way is more likely to suffer from non-convexity as
       * it uses many more steps.
       *
       * Also, note that precondition_add_condition_information() is
       * more geared towards Fortran as it assumes no side effects in
       * the condition evaluation. However, it is better at handling
       * non-convex condition than condition_to_transformer(), but
       * condition_to_transformer(), which is built on top of
       * precondition_add_condition_information() could be
       * improved/might be improvable... In case the condition is not
       * convex, there is no single transformer which fits it. But the
       * postcondition can be updated with different convex components
       * and then different results united in a unique postcondition
       * by a convex hull.
       */

      pips_debug(8, "The loop certainly is executed.\n");

      if(false) {
	transformer ntl = transformer_undefined;
	transformer cpost = transformer_undefined; // combined postcondition
	ntl = transformer_apply(tf, pre);
	/* Let's execute the last iteration since it certainly exists */
	ntl = precondition_add_condition_information(ntl, c, ntl, true);
	post = transformer_apply(tb, ntl);
	free_transformer(ntl);
	post = precondition_add_condition_information(post, c, post, false);

	postb = precondition_add_condition_information(postb, c, postb, false);

	cpost = transformer_intersection(post, postb);

	free_transformer(post);
	free_transformer(postb);

	post = cpost;
      }
      else {
	// FI: does not work with side effects
	//post = precondition_add_condition_information(postb, c, postb, false);
	transformer pre_c = transformer_range(postb);
	transformer ctf = condition_to_transformer(c, pre_c, false);
	post = transformer_apply(ctf, postb);
	free_transformer(pre_c);
	free_transformer(ctf);
      }
    }
    else {
      /* Assume the loop is entered, post_al, or not, post_ne, and perform
       * the convex hull of both
       */
      transformer post_ne = transformer_dup(pre);
      transformer post_al = transformer_undefined;
      //transformer tb = load_statement_transformer(s);

      pips_debug(8, "The loop may be executed or not\n");

      /* The loop is executed at least once: let's execute the last iteration */
      //post_al = transformer_apply(tb, preb);

	// FI: does not work with side effects
	//post = precondition_add_condition_information(postb, c, postb, false);
	transformer pre_c = transformer_range(postb);
	transformer ctf = condition_to_transformer(c, pre_c, false);
	// Mmeory leak? Do we still need postb?
	post_al = transformer_apply(ctf, postb);
	free_transformer(pre_c);
	free_transformer(ctf);
	// post_al = precondition_add_condition_information(postb, c, postb, false);

      /* The loop is never executed */
	// FI: does not work with side effects
	// post_ne = precondition_add_condition_information(post_ne, c, post_ne, false);
	pre_c = transformer_range(post_ne);
	ctf = condition_to_transformer(c, pre_c, false);
	// Mmeory leak
	post_ne = transformer_apply(ctf, post_ne);
	free_transformer(pre_c);
	free_transformer(ctf);

      post = transformer_convex_hull(post_ne, post_al);
      // free for postb too? hidden within post_al?
      transformer_free(post_ne);
      transformer_free(post_al);
    }
  }

  ifdebug(8) {
    pips_debug(8, "resultat post =");
    (void) print_transformer(post);
  }
  pips_debug(8, "end\n");
  return post;
}

transformer whileloop_to_total_precondition(
    transformer t_post,
    whileloop l,
    transformer tf,
    transformer context)
{
  transformer t_pre = transformer_undefined;
  statement s = whileloop_body(l);
  expression c = whileloop_condition(l);

  pips_assert("not implemented yet", false && t_post==t_post);

  pips_debug(8,"begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT) && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
    pips_internal_error("Halbwachs not implemented");
  }
  else {
    transformer preb /*= transformer_dup(pre)*/ ;

    /* Apply the loop fix point transformer T* to obtain the set of stores
     * for any number of iteration, including 0.
     */
    preb = transformer_combine(preb, tf);

    if(false_condition_wrt_precondition_p(c, context)) {
      pips_debug(8, "The loop is never executed\n");

      /* propagate an impossible precondition in the loop body */
      (void) statement_to_postcondition(transformer_empty(), s);
      /* The loop body precondition is not useful any longer */
      free_transformer(preb);
      /* do not add the exit condition since it is redundant with pre */
      /* post = transformer_dup(pre); */
    }
    else if(true_condition_wrt_precondition_p(c, context)) {
      /* At least one iteration is executed. The transformer of
       * the loop body is not useful!
       */
      /* transformer tb = load_statement_transformer(s); */
      transformer ntl = transformer_undefined;

      pips_debug(8, "The loop certainly is executed\n");

      /* propagate preconditions in the loop body */
      preb = precondition_add_condition_information(preb, c, preb, true);
      (void) statement_to_postcondition(preb, s);

      /* ntl = transformer_apply(tf, pre); */
      /* Let's execute the last iteration since it certainly exists */
      ntl = precondition_add_condition_information(ntl, c, ntl, true);
      /* post = transformer_apply(tb, ntl); */
      free_transformer(ntl);
      /* post = precondition_add_condition_information(post, c, post, false); */
    }
    else {
      /* Assume the loop is entered, post_al, or not, post_ne, and perform
       * the convex hull of both
       */
      transformer post_ne /* = transformer_dup(pre) */ ;
      transformer post_al = transformer_undefined;
      transformer tb = load_statement_transformer(s);

      pips_debug(8, "The loop may be executed or not\n");

      /* propagate preconditions in the loop body */
      precondition_add_condition_information(preb, c, preb, true);
      (void) statement_to_postcondition(preb, s);

      /* The loop is executed at least once: let's execute the last iteration */
      post_al = transformer_apply(tb, preb);
      post_al = precondition_add_condition_information(post_al, c, post_al, false);

      /* The loop is never executed */
      post_ne = precondition_add_condition_information(post_ne, c, post_ne, false);

      /* post = transformer_convex_hull(post_ne, post_al); */
      transformer_free(post_ne);
      transformer_free(post_al);
    }
  }

  ifdebug(8) {
    pips_debug(8, "resultat t_pre=%p\n", t_pre);
    (void) print_transformer(t_pre);
    pips_debug(8,"end\n");
  }

  return t_pre;
}
