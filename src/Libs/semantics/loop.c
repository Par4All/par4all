 /* semantic analysis: processing of loops
  *
  * $Id$
  *
  * $Log: loop.c,v $
  * Revision 1.6  2003/07/12 16:46:07  irigoin
  * Temporary version, with unsatisfactory implementation of loop_initialization_to_transformer()
  *
  * Revision 1.5  2002/03/21 09:36:32  irigoin
  * debug() replaced by pips_debug() and similar cosmetic modifications
  *
  * Revision 1.4  2001/12/05 17:16:24  irigoin
  * Reformatting + additions to compute total preconditions
  *
  * Revision 1.3  2001/10/22 15:44:38  irigoin
  * Code reformatting. Exploits expression transformers.
  *
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
      debug(8,"add_loop_skip_condition","Non-analyzed variable in loop bound(s)\n");
    }
  }
  else {
    debug(8,"add_loop_skip_condition","increment sign unknown or non-affine bound\n");
  }

  pips_debug(8,"end with new tf=%p\n", tf);
  ifdebug(8) {
    (void) print_transformer(tf);
  }

  return tf;
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

static transformer 
add_good_loop_conditions(
    transformer pre,
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

  pre = add_index_range_conditions(pre, i, r, tfb);

  pips_debug(8, "end\n");
  return(pre);
}


transformer add_loop_index_initialization(transformer tf, loop l, transformer pre)
{
    entity i = loop_index(l);
    expression init = range_lower(loop_range(l));
    transformer post = transformer_undefined;
    transformer t_init = transformer_undefined;
    list lef = expression_to_proper_effects(init);

    t_init = any_scalar_assign_to_transformer(i, init, lef, pre);
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
 */
transformer
add_loop_index_exit_value(
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
	    pips_assert("add_loop_index_exit_value", FALSE);
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

/* The loop initialization is performed before tf */
static transformer 
transformer_add_loop_index_initialization(transformer tf, loop l, transformer pre)
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

static transformer 
transformer_add_loop_index_incrementation(
					  transformer tf,
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

  /* it does not contain the loop index update
     the loop increment expression must be linear to find inductive 
     variables related to the loop index */
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

  pips_assert("No temporary variables are allocated",
	      number_of_temporary_values()==0);

  if(entity_has_values_p(i)) {
    expression lbe = range_lower(loop_range(l));
    transformer iit = loop_initialization_to_transformer(l, pre);
    entity lbv = make_local_temporary_value_entity(entity_type(i));
    transformer lbt = any_expression_to_transformer(lbv, lbe, pre, TRUE);

    transformer preub = transformer_safe_apply(iit, pre);
    expression ube = range_upper(loop_range(l));
    entity ubv = make_local_temporary_value_entity(entity_type(i));
    transformer ubt = any_expression_to_transformer(ubv, ube, preub, TRUE);

    transformer prei = transformer_safe_apply(ubt, preub);
    expression ie = range_increment(loop_range(l));
    entity iv = make_local_temporary_value_entity(entity_type(i));
    transformer it = any_expression_to_transformer(iv, ie, prei, TRUE);
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
   effect_reference() expects so without testing it. */
transformer loop_initialization_to_transformer(loop l, transformer pre)
{
  effect init_e = make_effect(make_cell(is_cell_preference,
					make_preference(make_reference(loop_index(l), NIL))),
			      make_action(is_action_write, UU),
			      make_approximation(is_approximation_must, UU),
			      make_descriptor(is_descriptor_none,UU));
  list l_init_e = CONS(EFFECT, init_e, NIL);
  list l_expr_e = expression_to_proper_effects(range_lower(loop_range(l)));
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
  free_reference();
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

transformer 
loop_to_transformer(loop l, transformer pre, list e)
{
  /* loop transformer tf = tfb* or tf = tfb+ or ... */
  transformer tf = transformer_undefined;
  /* loop body transformer */
  transformer tfb = transformer_undefined;
  /* approximate loop transformer, including loop index updates */
  transformer abtf = effects_to_transformer(e);
  /* loop body precondition */
  transformer preb = invariant_wrt_transformer(pre, abtf);
  /* range r = loop_range(l); */
  statement b = loop_body(l);
  transformer t_init = transformer_undefined;
  transformer old_tf = transformer_undefined;

  pips_debug(8,"begin with precondition pre=%p\n", pre);
  ifdebug(8) {
    (void) print_transformer(pre);
  }

  /* compute the loop body transformer under loop body precondition preb */
  if(!transformer_undefined_p(preb))
    preb = add_good_loop_conditions(preb, l);

  pips_debug(8,"body precondition preb=%p\n", preb);
  ifdebug(8) {
    (void) print_transformer(preb);
  }

  tfb = transformer_dup(statement_to_transformer(b, preb));
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
    fprint_transformer(stderr, tf, external_value_name);
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
    debug(8,"loop_to_transformer","end\n");
  }

  return tf;
}

/* The index variable is always initialized and then the loop is either
   entered and exited or not entered */
transformer 
refine_loop_transformer(transformer ltf, transformer pre, loop l)
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
    btf = refine_loop_transformer(load_statement_transformer(s), pre, il);
  }
  else {
    btf = transformer_dup(load_statement_transformer(s));
  }
  /* The final index incrementation is performed later by add_loop_index_exit_value() */
  /* btf = transformer_add_loop_index_incrementation(btf, l, pre); */

  /* compute the transformer when the loop is entered */
  t_enter = transformer_combine(transformer_dup(ltf), btf);

  ifdebug(8) {
    pips_debug(8, "entered loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, external_value_name);
  }

  /* add the entry condition */
  /* but it seems to be in t already */
  /* t_enter = transformer_add_loop_index_initialization(t_enter, l); */

  /* add the exit condition, without any information pre to estimate the
     increment */
  t_enter = add_loop_index_exit_value(t_enter, l, pre);

  ifdebug(8) {
    pips_debug(8, "entered and exited loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, external_value_name);
  }

  /* add initialization for the unconditional initialization of the loop
     index variable */
  t_skip = transformer_undefined_p(pre)?
    transformer_identity() :
    transformer_dup(pre);
  t_skip = add_loop_index_initialization(t_skip, l, pre);
  t_skip = add_loop_skip_condition(t_skip, l, pre);

  ifdebug(8) {
    pips_debug(8, "skipped loop transformer t_skip=\n");
    fprint_transformer(stderr, t_skip, external_value_name);
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

  ifdebug(8) {
    pips_debug(8, "full refined loop transformer tf=\n");
    fprint_transformer(stderr, tf, external_value_name);
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

/* This function computes the effect of K loop iteration, with K positive.
 * This function does not take the loop exit into account because its result
 * is used to compute the precondition of the loop body.
 * Hence the loop exit condition only is added when preconditions are computed.
 * This is confusing when transformers are prettyprinted with the source code.
 */

transformer 
whileloop_to_transformer(whileloop l, transformer pre, list e) /* effects of whileloop l */
{
  /* loop transformer tf = tfb* or tf = tfb+ or ... */
  transformer tf;
  /* loop body transformer */
  transformer tfb;
  expression cond = whileloop_condition(l);
  statement s = whileloop_body(l);

  debug(8,"whileloop_to_transformer","begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT)) {
    /* compute the whileloop body transformer */
    tfb = transformer_dup(statement_to_transformer(s, pre));

    /* If the while entry condition is usable, it must be added
     * on the old values
     */
    tfb = transformer_add_condition_information(tfb, cond, pre, TRUE);

    /* compute tfb's fix point according to pips flags */
    if(pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
      tf = transformer_halbwachs_fix_point(tfb);
    }
    else if (transformer_empty_p(tfb)) {
      /* The loop is never entered */
      tf = transformer_identity();
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
	fprint_transformer(stderr, tf, external_value_name);
      }

    }
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

    (void) statement_to_transformer(s, pre);
    tf = effects_to_transformer(e);
  }

  ifdebug(8) {
    (void) fprintf(stderr,"%s: %s\n","whileloop_to_transformer",
		   "resultat tf =");
    (void) print_transformer(tf);
  }
  pips_debug(8,"end\n");
  return tf;
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
    pips_internal_error("Halbwachs not implemented\n");
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
      debug(8, "loop_to_postcondition", "The loop is never executed\n");

      /* propagate an impossible precondition in the loop body */
      (void) statement_to_postcondition(transformer_empty(), s);
      /* The loop body precondition is not useful any longer */
      free_transformer(preb);
      post = transformer_dup(pre);
      post = add_loop_index_initialization(post, l, pre);
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
      post = add_loop_index_exit_value(post, l, pre);
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

      debug(8, "loop_to_postcondition", "The loop may be executed or not\n");

      /* propagate preconditions in the loop body */
      post_al =  statement_to_postcondition(preb, s);
      /* post_al = transformer_apply(tf, pre); */

      /* We should add (when possible) the non-entry condition in post_ne!
       * For instance, DO I = 1, N leads to N <= 0
       */
      post_ne = add_loop_skip_condition(post_ne, l, lpre);
      post_ne = add_loop_index_initialization(post_ne, l, lpre);
      free_transformer(lpre);

      post_al = add_loop_index_exit_value(post_al, l, post_al);

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

transformer whileloop_to_postcondition(
    transformer pre,
    whileloop l,
    transformer tf)
{
  transformer post = transformer_undefined;
  statement s = whileloop_body(l);
  expression c = whileloop_condition(l);

  pips_debug(8, "begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT) && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
    pips_internal_error("Halbwachs not implemented\n");
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
      preb = precondition_add_condition_information(preb, c, preb, TRUE);
      (void) statement_to_postcondition(preb, s);

      ntl = transformer_apply(tf, pre);
      /* Let's execute the last iteration since it certainly exists */
      ntl = precondition_add_condition_information(ntl, c, ntl, TRUE);
      post = transformer_apply(tb, ntl);
      free_transformer(ntl);
      post = precondition_add_condition_information(post, c, post, FALSE);
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
      preb = precondition_add_condition_information(preb, c, preb, TRUE);
      (void) statement_to_postcondition(preb, s);

      /* The loop is executed at least once: let's execute the last iteration */
      post_al = transformer_apply(tb, preb);
      post_al = precondition_add_condition_information(post_al, c, post_al, FALSE);

      /* The loop is never executed */
      post_ne = precondition_add_condition_information(post_ne, c, post_ne, FALSE);

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

  pips_assert("not implemented yet", FALSE);

  pips_debug(8,"begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT) && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
    pips_internal_error("Halbwachs not implemented\n");
  }
  else {
    transformer preb /*= transformer_dup(pre)*/ ;

    /* Apply the loop fix point transformer T* to obtain the set of stores
     * for any number of iteration, including 0.
     */
    preb = transformer_combine(preb, tf);

    if(false_condition_wrt_precondition_p(c, context)) {
      debug(8, "whileloop_to_postcondition", "The loop is never executed\n");

      /* propagate an impossible precondition in the loop body */
      (void) statement_to_postcondition(transformer_empty(), s);
      /* The loop body precondition is not useful any longer */
      free_transformer(preb);
      /* do not add the exit condition since it is redundant with pre */
      /* post = transformer_dup(pre); */
    }
    else if(true_condition_wrt_precondition_p(c, context)) {
      /* At least one iteration is executed. The transformer of
       * the loop body is useful.
       */
      transformer tb = load_statement_transformer(s);
      transformer ntl = transformer_undefined;

      pips_debug(8, "The loop certainly is executed\n");

      /* propagate preconditions in the loop body */
      preb = precondition_add_condition_information(preb, c, preb, TRUE);
      (void) statement_to_postcondition(preb, s);

      /* ntl = transformer_apply(tf, pre); */
      /* Let's execute the last iteration since it certainly exists */
      ntl = precondition_add_condition_information(ntl, c, ntl, TRUE);
      /* post = transformer_apply(tb, ntl); */
      free_transformer(ntl);
      /* post = precondition_add_condition_information(post, c, post, FALSE); */
    }
    else {
      /* Assume the loop is entered, post_al, or not, post_ne, and perform
       * the convex hull of both
       */
      transformer post_ne /* = transformer_dup(pre) */ ;
      transformer post_al = transformer_undefined;
      transformer tb = load_statement_transformer(s);

      debug(8, "whileloop_to_postcondition", "The loop may be executed or not\n");

      /* propagate preconditions in the loop body */
      precondition_add_condition_information(preb, c, preb, TRUE);
      (void) statement_to_postcondition(preb, s);

      /* The loop is executed at least once: let's execute the last iteration */
      post_al = transformer_apply(tb, preb);
      post_al = precondition_add_condition_information(post_al, c, post_al, FALSE);

      /* The loop is never executed */
      post_ne = precondition_add_condition_information(post_ne, c, post_ne, FALSE);

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
