 /* semantic analysis: processing of loops
  *
  * $Id$
  *
  * $Log: loop.c,v $
  * Revision 1.2  2001/07/13 15:01:44  irigoin
  * First multitype version
  *
  * Revision 1.1  2001/06/21 09:03:46  irigoin
  * Initial revision
  *
  *
  *
  */
#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
/* #include "database.h" */
#include "linear.h"
#include "ri.h"
/*
#include "text.h"
#include "text-util.h"
*/
#include "ri-util.h"
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

#define IS_LOWER_BOUND 0
#define IS_UPPER_BOUND 1

transformer add_loop_skip_condition(transformer pre, loop l)
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

    debug(8,"add_loop_skip_condition","begin with pre\n");
    ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[add_loop_skip_condition]",
		       "input pre =");
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
	    pre = transformer_inequality_add(pre, v);

	    ifdebug(8) {
		debug(8,"add_loop_skip_condition","Skip condition:\n");
		vect_fprint(stderr, v, (char * (*)(Variable)) external_value_name);
	    }
	}
	else {
	    debug(8,"add_loop_skip_condition","Non-analyzed variable in loop bound(s)\n");
	}
    }
    else {
	debug(8,"add_loop_skip_condition","increment sign unknown or non-affine bound\n");
    }

    debug(8,"add_loop_skip_condition","end with pre\n");
    ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[add_loop_skip_condition]",
		       "new pre =");
	(void) print_transformer(pre);
    }

    return pre;
}

static transformer 
add_affine_bound_conditions(transformer pre, 
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

    /* Achtung: value_mappings_compatible_vector_p() has a side
     * effect on its argument; it has to be evaluated before
     * the second half of the test else effects would be wrongly
     * interpreted in case of equivalences 
     */

    if(value_mappings_compatible_vector_p(v) &&
       !transformer_affect_linear_p(tfb,v)) {
	if (lower_or_upper == IS_LOWER_BOUND)
	    vect_add_elem(&v,
			  (Variable) entity_to_new_value(index), VALUE_MONE);
	else{
	    vect_chg_sgn(v);
	    vect_add_elem(&v, 
			  (Variable) entity_to_new_value(index), VALUE_ONE);
	}
	pre = transformer_inequality_add(pre, v);
    }
    else{
	vect_rm(v);
	v = VECTEUR_UNDEFINED;
    }	
    return pre;
}

static transformer 
add_index_bound_conditions(
    transformer pre,
    entity index,
    expression bound,
    int lower_or_upper,
    transformer tfb)
{
    normalized n = NORMALIZE_EXPRESSION(bound);
    /* tfb does not take into account the index incrementation */
    transformer t_iter = transformer_dup(tfb);

    /* It is assumed on entry that index has values recognized 
     * by the semantics analysis
     */
    /* pips_assert("add_index_bound_conditions", entity_has_values_p(index)); */

    transformer_arguments(t_iter) = 
	arguments_add_entity(transformer_arguments(t_iter), index);

    if(normalized_linear_p(n)) {
	Pvecteur v_bound = (Pvecteur) normalized_linear(n);
	add_affine_bound_conditions(pre, index, v_bound, lower_or_upper, t_iter);
    }

    free_transformer(t_iter);
    return(pre);
}

transformer 
add_index_range_conditions(
    transformer pre,
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

    debug(8,"add_index_range_conditions","begin\n");

    if(entity_has_values_p(i)) {

	/* is the loop increment numerically known? Is its sign known? */
	expression_and_precondition_to_integer_interval(e_incr, pre, &incr_lb, &incr_ub);

	if(incr_lb==incr_ub) {
	    if(incr_lb==0) {
		user_error("add_index_range_conditions", "Illegal null increment\n");
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

    debug(8,"add_index_range_conditions","end\n");
    return pre;
}

static transformer 
add_good_loop_conditions(
    transformer pre,
    loop l,
    transformer tf)
{
    /* loop bounds can be kept as preconditions for the loop body
       if the loop increment is numerically known and if they
       are linear and if they are loop body invariant, i.e.
       indices are accepted */
    /* arg. tf is unused, it was replaced by tfb to correct a bug */
    statement b = loop_body(l);
    entity i = loop_index(l);
    range r = loop_range(l);
    transformer tfb = load_statement_transformer(b);

    debug(8,"add_good_loop_conditions","begin\n");

    pre = add_index_range_conditions(pre, i, r, tfb);

    debug(8,"add_good_loop_conditions","end\n");
    return(pre);
    
}


transformer add_loop_index_initialization(transformer pre, loop l)
{
    entity i = loop_index(l);
    expression init = range_lower(loop_range(l));
    transformer post = transformer_undefined;
    transformer t_init = transformer_undefined;
    list lef = expression_to_proper_effects(init);

    t_init = any_scalar_assign_to_transformer(i, init, lef, pre);
    if(t_init==transformer_undefined)
	t_init = effects_to_transformer(lef);
    post = transformer_apply(t_init, pre);

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
 */
transformer
add_loop_index_exit_value(
    transformer post, /* postcondition of the last iteration */
    loop l,           /* loop to process */
    transformer pre,  /* precondition on loop entrance */
    list lbe          /* list of loop body effects */ )
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
    return post;
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
	    pips_assert("add_loop_index_exit_value", TRUE);
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

bool 
simple_dead_loop_p(expression lower, expression upper)
{
    bool dead_loop_p = FALSE;
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
	pips_assert("No old values are left", FALSE);
      }
    }

    return pre;
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
  transformer tfb = transformer_add_loop_index_incrementation(transformer_dup(tf), l);
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

  tfb = transformer_add_loop_index_incrementation(tfb, l);

  new_tf = (* transformer_fix_point_operator)(tfb);
  new_tf = transformer_add_loop_index_initialization(new_tf, l);

  free_transformer(preb);
  gen_free_list(list_mod);
  gen_free_list(list_val);

  ifdebug(5) {
    pips_debug(5, "End with fixpoint:\n");
    print_transformer(new_tf);
  }

  return new_tf;
}

transformer loop_to_postcondition(
    transformer pre,
    loop l,
    transformer tf)
{
  transformer post = transformer_undefined;
  statement s = loop_body(l);
  range r = loop_range(l);

  debug(8,"loop_to_postcondition","begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT) && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
    pips_error("loop_to_postcondition","Halbwachs not implemented\n");
  }
  else {
    /* pips_error("loop_to_postcondition",
       "Equality option not implemented\n"); */
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
    preb = add_good_loop_conditions(preb, l, tf);

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
      preb = add_good_loop_conditions(preb, l, new_tf);
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
      debug(8, "loop_to_postcondition", "The loop is never executed\n");

      /* propagate an impossible precondition in the loop body */
      (void) statement_to_postcondition(transformer_empty(), s);
      /* The loop body precondition is not useful any longer */
      free_transformer(preb);
      post = transformer_dup(pre);
      post = add_loop_index_initialization(post, l);
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
      post = add_loop_index_exit_value(post, l, pre, NIL);
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
      transformer post_ne = transformer_dup(pre);
      transformer post_al = transformer_undefined;

      debug(8, "loop_to_postcondition", "The loop may be executed or not\n");

      /* propagate preconditions in the loop body */
      post_al =  statement_to_postcondition(preb, s);
      /* post_al = transformer_apply(tf, pre); */

      /* We should add (when possible) the non-entry condition in post_ne!
       * For instance, DO I = 1, N leads to N <= 0
       */
      post_ne = add_loop_skip_condition(post_ne, l);

      post_ne = add_loop_index_initialization(post_ne, l);
      post_al = add_loop_index_exit_value(post_al, l, pre, NIL);
      ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		       "Never executed: post_ne =");
	(void) print_transformer(post_ne);
	(void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		       "Always executed: post_al =");
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
  debug(8,"loop_to_postcondition","end\n");
  return post;
}

transformer whileloop_to_postcondition(
    transformer pre,
    whileloop l,
    transformer tf)
{
  transformer post = transformer_undefined;
  statement s = whileloop_body(l);
  expression c = whileloop_condition(l);

  debug(8,"whileloop_to_postcondition","begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT) && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
    pips_error("whileloop_to_postcondition","Halbwachs not implemented\n");
  }
  else {
    transformer preb = transformer_dup(pre);

    /* Apply the loop fix point transformer T* to obtain the set of stores
     * for any number of iteration, including 0.
     */
    preb = transformer_combine(preb, tf);

    if(false_condition_wrt_precondition_p(c, pre)) {
      debug(8, "whileloop_to_postcondition", "The loop is never executed\n");

      /* propagate an impossible precondition in the loop body */
      (void) statement_to_postcondition(transformer_empty(), s);
      /* The loop body precondition is not useful any longer */
      free_transformer(preb);
      /* do not add the exit condition since it is redundant with pre */
      post = transformer_dup(pre);
    }
    else if(true_condition_wrt_precondition_p(c, pre)) {
      /* At least one iteration is executed. The transformer of
       * the loop body is useful.
       */
      transformer tb = load_statement_transformer(s);
      transformer ntl = transformer_undefined;

      debug(8, "whileloop_to_postcondition", "The loop certainly is executed\n");

      /* propagate preconditions in the loop body */
      preb = precondition_add_condition_information(preb, c, TRUE);
      (void) statement_to_postcondition(preb, s);

      ntl = transformer_apply(tf, pre);
      /* Let's execute the last iteration since it certainly exists */
      ntl = precondition_add_condition_information(ntl, c, TRUE);
      post = transformer_apply(tb, ntl);
      free_transformer(ntl);
      post = precondition_add_condition_information(post, c, FALSE);
    }
    else {
      /* Assume the loop is entered, post_al, or not, post_ne, and perform
       * the convex hull of both
       */
      transformer post_ne = transformer_dup(pre);
      transformer post_al = transformer_undefined;
      transformer tb = load_statement_transformer(s);

      debug(8, "whileloop_to_postcondition", "The loop may be executed or not\n");

      /* propagate preconditions in the loop body */
      precondition_add_condition_information(preb, c, TRUE);
      (void) statement_to_postcondition(preb, s);

      /* The loop is executed at least once: let's execute the last iteration */
      post_al = transformer_apply(tb, preb);
      post_al = precondition_add_condition_information(post_al, c, FALSE);

      /* The loop is never executed */
      post_ne = precondition_add_condition_information(post_ne, c, FALSE);

      post = transformer_convex_hull(post_ne, post_al);
      transformer_free(post_ne);
      transformer_free(post_al);
    }
  }

  ifdebug(8) {
    (void) fprintf(stderr,"%s: %s\n","[whileloop_to_postcondition]",
		   "resultat post =");
    (void) print_transformer(post);
  }
  debug(8,"whileloop_to_postcondition","end\n");
  return post;
}
