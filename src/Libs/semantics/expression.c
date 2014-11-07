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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
 /* semantic analysis: processing of expressions, with or without
  * preconditions.
  *
  * Expressions in Fortran were assumed to be side effect free most of
  * the time and transformers were assumed to be linked to statements
  * only. This is not true in C, where many constructs have side
  * effects: Consider for instance the following three statements
  *
  * n++; j = k = 2; i = j, k = i;
  *
  * which may also appear as conditions, initializations or
  * incrementations in compound statements (while, for, if).
  *
  * Most functions here may return transformer_undefined when no
  * interesting polyhedral approximation exists because the problem
  * was fixed later at the statement level by using effects. For C, we
  * need new functions which always succeed. So we need effects at the
  * expression level although they are not stored in the PIPS
  * database.
  *
  * Moreover, temporary values are used to combine
  * subexpressions. These temporary values must be projected when a
  * significant transformer has been obtained, but not earlier. The
  * temporary value counter cannot be reset before all temporary
  * variables have been projected (eliminated). So functions which are
  * called recursively cannot project temporary values and reset the
  * temporary value counter. We can use wrapping functions and/or an
  * additional argument, is_internal, to know whether temporary
  * variables should be eliminated or not when returning the transformer.
  *
  * Note that since expressions have side effects, preconditions must
  * be updated along the left-to-right expression abstraction
  * process. For example:
  *
  * j = 2; i = j + (j = 0) + j;
  *
  * but such a case is forbidden by the C99 standard. The result of
  * the evaluation of such an expression is undefined.
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
/* #include "database.h" */
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "alias-classes.h"

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


/* TYPE INDEPENDENT OPERATIONS */

/* May return an  undefined transformer */
// static: no longer because used i ri_to_transformers.c because
// assignments are handled there
transformer generic_reference_to_transformer(entity v,
						    reference r,
						    transformer pre,
						    bool is_internal)
{
  transformer tf = transformer_undefined;
  entity rv = reference_variable(r);

  if((!is_internal && analyzable_scalar_entity_p(rv))
     || entity_has_values_p(rv)) {
    entity rv_new = is_internal?
    entity_to_new_value(rv) : external_entity_to_new_value(rv);
    transformer tfr =  simple_equality_to_transformer(v, rv_new, false);
    if(!transformer_undefined_p(pre)) {
      // FI: assume pre is a range
      tf = transformer_intersection(tfr, pre);
    }
    free_transformer(tfr);
  }
  else { /* If you are dealing with a C array reference, find
	    transformers for each subscript expression in case of
	    side effects. */
    entity m = get_current_module_entity();
    if(c_language_module_p(m)) {
      tf = expressions_to_transformer(reference_indices(r), pre);
    }
  }
  return tf;
}

static transformer
generic_minmax_to_transformer(entity e,
			      list args,
			      transformer pre,
			      bool minmax,
			      bool is_internal)
{
  transformer tf = transformer_undefined;
  transformer tf_acc = transformer_undefined;
  list cexpr;
  Pcontrainte cl = CONTRAINTE_UNDEFINED;

  pips_debug(8, "begin\n");

  pips_assert("At least one argument", gen_length(args)>=1);

  /* pips_assert("Precondition is unused", transformer_undefined_p(pre)); */

  for(cexpr = args; !ENDP(cexpr); POP(cexpr)) {
    expression arg = EXPRESSION(CAR(cexpr));
    type t = entity_type(e);
    entity tmp = make_local_temporary_value_entity(t);
    transformer tfe = any_expression_to_transformer(tmp, arg, pre, is_internal);

    Pvecteur v = vect_new((Variable) tmp, VALUE_ONE);
    Pcontrainte cv = CONTRAINTE_UNDEFINED;

    vect_add_elem(&v, (Variable) e, VALUE_MONE);

    if(minmax) {
      v = vect_multiply(v, VALUE_MONE);
    }

    cv = contrainte_make(v);
    cv->succ = cl;
    cl = cv;

    /* memory leak! */
    tf_acc = transformer_safe_image_intersection(tf_acc, tfe);
  }


  if(CONTRAINTE_UNDEFINED_P(cl) || CONTRAINTE_NULLE_P(cl)) {
    Psysteme sc = sc_make(CONTRAINTE_UNDEFINED, cl);

    sc_base(sc) = base_add_variable(BASE_NULLE, (Variable) e);
    sc_dimension(sc) = 1;
    tf = make_transformer(NIL,
			  make_predicate(sc));
  }
  else {
    /* A miracle occurs and the proper basis is derived from the
       constraints ( I do not understand why the new and the old value
       of e both appear... so it may not be necessary for the
       consistency check... I'm lost, FI, 6 Jan. 1999) */
    if(!transformer_undefined_p(tf_acc)) {
      tf_acc = transformer_inequalities_add(tf_acc, cl);
    }
    else {
      /* cl could be kept but would be always projected later */
      contraintes_free(cl);
    }
    tf = tf_acc;
  }

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

/* Type independent. Most of it should be factored out for unary operations. */
static transformer unary_minus_operation_to_transformer(entity v,
							expression e1,
							transformer pre,
							bool is_internal)
{
  transformer tf = transformer_undefined;
  entity tmp1 = make_local_temporary_value_entity(entity_type(v));
  /* sub-expressions are not necessarily integer? Then the expression
     should not be integer? */
  transformer tf1 = any_expression_to_transformer(tmp1, e1, pre, is_internal);
  transformer tf_op = simple_unary_minus_to_transformer(v, tmp1);

  pips_debug(9, "Begin for %s\n", entity_local_name(v));

  tf = transformer_safe_intersection(tf1, tf_op);

  /* Much too early: you are in between recursive calls! */
  /* tf = transformer_temporary_value_projection(tf); */

  free_transformer(tf1);
  free_transformer(tf_op);

  pips_debug(9, "End with %p\n", tf);
  ifdebug(9) dump_transformer(tf);

  return tf;
}

/* See also any_basic_update_to_transformer(), which should be based on this function */
transformer any_basic_update_operation_to_transformer(entity tmp, entity v, entity op)
{
  transformer tf = transformer_undefined;
  expression ve = entity_to_expression(v);
  expression inc = expression_undefined;
  expression n_rhs = expression_undefined;
  entity plus = entity_intrinsic(PLUS_C_OPERATOR_NAME);
  list args = list_undefined;

  if(ENTITY_POST_INCREMENT_P(op) || ENTITY_PRE_INCREMENT_P(op))
    inc = int_to_expression(1);
  else
    inc = int_to_expression(-1);

  n_rhs = MakeBinaryCall(plus, inc, entity_to_expression(v));

  /* FI: why*/
  args = CONS(EXPRESSION, ve, CONS(EXPRESSION, n_rhs, NIL));
  tf = any_assign_operation_to_transformer(v,
					   args,
					   transformer_undefined,
					   false); 
  if(ENTITY_POST_INCREMENT_P(op) || ENTITY_POST_DECREMENT_P(op))
    tf = transformer_add_equality(tf, entity_to_old_value(v), tmp);
  else
    tf = transformer_add_equality(tf, v, tmp);
  gen_full_free_list(args);

  pips_debug(6,"return tf=%p\n", tf);
  ifdebug(6) (void) dump_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

/* forward declaration */
static transformer generic_abs_to_transformer(
       entity, expression, transformer, bool);

static transformer 
generic_unary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  if(ENTITY_UNARY_MINUS_P(op)) {
    tf = unary_minus_operation_to_transformer(e, e1, pre, is_internal);
  }
  else if(ENTITY_IABS_P(op) || ENTITY_ABS_P(op) || ENTITY_DABS_P(op)
      || ENTITY_CABS_P(op)
      || ENTITY_C_ABS_P(op) || ENTITY_LABS_P(op) || ENTITY_LLABS_P(op)
      || ENTITY_IMAXABS_P(op)
      || ENTITY_FABS_P(op) || ENTITY_FABSF_P(op) || ENTITY_FABSL_P(op)
      || ENTITY_C_CABS_P(op) || ENTITY_CABSF_P(op) || ENTITY_CABSL_P(op)) {
    tf = generic_abs_to_transformer(e, e1, pre, is_internal);
  }
  else if(ENTITY_POST_INCREMENT_P(op) || ENTITY_POST_DECREMENT_P(op)
	  || ENTITY_PRE_INCREMENT_P(op) || ENTITY_PRE_DECREMENT_P(op)) {
    syntax s1 = expression_syntax(e1);
    entity v1 = reference_variable(syntax_reference(s1));

    if(entity_has_values_p(v1)) {
      transformer tfu = any_basic_update_operation_to_transformer(e, v1, op);
      if(transformer_undefined_p(pre)) {
	tf = tfu;
      }
      else {
	/* FI: Oops, the precondition is lost here! See any_basic_update_to_transformer() */
	tf = transformer_combine(transformer_range(pre), tfu);
	free_transformer(tfu);
      }
    }
  }

  return tf;
}

/* Type independent. Most of it should be factored out for binary
   operations. v = e1 + e2 or v = e1 - e2 */
static transformer addition_operation_to_transformer(entity v,
						     expression e1,
						     expression e2,
						     transformer pre,
						     bool addition_p,
						     bool is_internal)
{
  transformer tf = transformer_undefined;
  entity tmp1 = make_local_temporary_value_entity(entity_type(v));
  entity tmp2 = make_local_temporary_value_entity(entity_type(v));
  /* sub-expressions are not necessarily integer? Then the expression
     should not be integer? */
  transformer tf1 = safe_any_expression_to_transformer(tmp1, e1, pre, is_internal);
  transformer post = transformer_undefined_p(pre)? transformer_identity()
    : transformer_apply(tf1, pre);
  transformer npre = transformer_range(post);
  transformer tf2 = safe_any_expression_to_transformer(tmp2, e2, npre, is_internal);
  transformer tf12 = transformer_undefined;
  transformer tf_op = simple_addition_to_transformer(v, tmp1, tmp2, addition_p);

  free_transformer(post);
  free_transformer(npre);

  pips_debug(9, "Begin officialy... but a lot has been done already!\n");

  tf12 = transformer_safe_combine_with_warnings(tf1, tf2);
  tf = transformer_safe_image_intersection(tf12, tf_op);

  /*
    ifdebug(9) {
    pips_debug(9, "before projection, with temporaries v=%s, tmp1=%s, tmp2=%s,"
    " transformer rf=%p\n",
    entity_local_name(v), entity_local_name(tmp1),
    entity_local_name(tmp2), tf);
    dump_transformer(tf);
    }
  */

  /* Too early: you are projecting v and loosing all useful information
     within an expression! */
  /* tf = transformer_temporary_value_projection(tf); */

  free_transformer(tf2);
  free_transformer(tf12);
  free_transformer(tf_op);

  pips_debug(9, "End with transformer tf=%p\n", tf);
  ifdebug(9) dump_transformer(tf);

  return tf;
}

/* Similar to any_assign_to_transformer(), which is a simpler case. */
/* Here, we cope with expression such as the lhs of j = (i=1); */
transformer any_assign_operation_to_transformer(entity tmp,
						list args, /* arguments for assign */
						transformer pre, /* precondition */
						bool is_internal __attribute__ ((unused)))
{
  transformer tf = transformer_undefined;
  //transformer tfe = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  expression rhs = EXPRESSION(CAR(CDR(args)));
  syntax slhs = expression_syntax(lhs);

  pips_assert("2 args to assign", CDR(CDR(args))==NIL);

  /* The lhs must be a scalar reference to perform an interesting analysis */
  if(syntax_reference_p(slhs)) {
    reference rlhs = syntax_reference(slhs);
    if(ENDP(reference_indices(rlhs))) {
      entity v = reference_variable(rlhs);

      if(entity_has_values_p(v)) {
	entity v_new = entity_to_new_value(v);
	entity v_old = entity_to_old_value(v);
	entity tmp2 = make_local_temporary_value_entity(entity_type(v));

	tf = any_expression_to_transformer(tmp2, rhs, pre, true);

	if(!transformer_undefined_p(tf)) {

	  pips_debug(9, "A transformer has been obtained:\n");
	  ifdebug(9) dump_transformer(tf);

	  if(entity_is_argument_p(v, transformer_arguments(tf))) {
	    /* Is it standard compliant? The assigned variable is
	       modified by the rhs. */
	    transformer teq = simple_equality_to_transformer(v, tmp, true);
	    string s =
	      words_to_string(words_syntax(expression_syntax(rhs),NIL));

	    pips_user_warning("Variable %s in lhs is uselessly updated by the rhs '%s'\n",
			      entity_local_name(v), s);

	    free(s);

	    tf = transformer_combine(tf, teq);
	    free_transformer(teq);
	  }
	  else {
	    /* Take care of aliasing */
	    entity v_repr = value_to_variable(v_new);

	    /* tf = transformer_value_substitute(tf, v_new, v_old); */
	    tf = transformer_value_substitute(tf, v_new, v_old);

	    pips_debug(9,"After substitution v_new=%s -> v_old=%s\n",
		       entity_local_name(v_new), entity_local_name(v_old));

	    tf = transformer_value_substitute(tf, tmp2, v_new);

	    pips_debug(9,"After substitution tmp=%s -> v_new=%s\n",
		       entity_local_name(tmp2), entity_local_name(v_new));

	    transformer_add_modified_variable(tf, v_repr);
	  }
	  //tf = generic_equality_to_transformer(tmp, tmp2, false, false);
	  //tf = transformer_combine(tf, tfe);
	}
      }
    }
  }

  pips_debug(6,"return tf=%p\n", tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

transformer safe_any_assign_operation_to_transformer(entity tmp,
						list args, /* arguments for assign */
						transformer pre, /* precondition */
						bool is_internal)
{
  transformer tf = any_assign_operation_to_transformer(tmp, args, pre, is_internal);
  if(transformer_undefined_p(tf)) {
    // FI: I'd like tmp to appear in the basis...
    tf = transformer_identity();
  }
  return tf;
}

/* v = (e1 = e1 op e2) ; e1 must be a reference; does not compute
   information for dereferenced pointers such as in "*p += 2;" */
static transformer update_operation_to_transformer(entity v,
						   entity op,
						   expression e1,
						   expression e2,
						   transformer pre,
						   bool is_internal)
{
  transformer tf = transformer_undefined;

  if(expression_reference_p(e1)) {
    //transformer utf = transformer_undefined;
    entity tmp1 = make_local_temporary_value_entity(entity_type(v));
    entity tmp2 = make_local_temporary_value_entity(entity_type(v));
    /* sub-expressions are not necessarily integer? Then the expression
       should not be integer? */
    /* We assume that the update operation is syntactically correct */
    entity v1 = reference_variable(syntax_reference(expression_syntax(e1)));
    entity sop = update_operator_to_regular_operator(op);
    expression n_exp = MakeBinaryCall(sop, copy_expression(e1), copy_expression(e2));
    list args = CONS(EXPRESSION, e1, CONS(EXPRESSION, n_exp, NIL));

    pips_debug(9, "Begin officialy... but a lot has been done already!\n");

    /* Effects are used at a higher level */
    tf = any_assign_operation_to_transformer(v, args, /* ef,*/ pre, is_internal);
    gen_free_list(args);
    free_expression(n_exp);

    if(!transformer_undefined_p(tf))
      tf = transformer_add_equality(tf, v, v1);

    ifdebug(8) {
      pips_debug(8, "before projection, with temporaries v=%s, tmp1=%s, tmp2=%s,"
		 " transformer rf=%p\n",
		 entity_local_name(v), entity_local_name(tmp1),
		 entity_local_name(tmp2), tf);
      dump_transformer(tf);
    }

    /* Too early: you are projecting v and loosing all useful information
       within an expression! */
    /* tf = transformer_temporary_value_projection(tf); */

    //free_transformer(tf2);
    //free_transformer(tf12);
    //free_transformer(tf12u);
    //free_transformer(tf_op);
  }

  pips_debug(8, "End with transformer tf=%p\n", tf);
  ifdebug(8) dump_transformer(tf);

  return tf;
}

static transformer constant_to_transformer(entity v,
					   expression rhs)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

  pips_debug(9, "Begin for entity %s",
	     entity_local_name(v));

  /* the rhs is a call to a constant function */
  if(syntax_call_p(srhs)) {
    entity f = call_function(syntax_call(srhs));

    pips_debug(9, " and constant %s\n", module_local_name(f));

    if(entity_constant_p(f)) {
      basic fb = entity_basic(f);

      if(basic_string_p(fb)) {
	basic vb = entity_basic(v);
	if(basic_string_p(vb)) {
	  int llhs = string_type_size(vb);
	  int lrhs = string_type_size(fb);

	  if(llhs==-1 || llhs >= lrhs) {
	    /* Implicitly sufficient length or implicit padding with SPACEs */
	    tf = simple_equality_to_transformer(v, f, false);
	  }
	  else {
	    /* Take the prefix of the constant. Expect problems with PIPS quotes... */
	    string n = malloc(llhs+3);
	    entity f1 = entity_undefined;

	    n = strncpy(n, entity_local_name(f), llhs+1);
	    *(n+llhs+1) = *n;
	    pips_assert("A simple or a double quote is used", *n=='"' || *n=='\'');
	    *(n+llhs+2) = 0;
	    f1 = make_constant_entity(n, is_basic_string, llhs);
	    tf = simple_equality_to_transformer(v, f1, false);
	  }
	}
	else {
	  pips_user_error("CHARACTER variable assigned a non CHARACTER constant, \"%s\"\n",
			  module_local_name(f));
	}
      }
      else {
	/* This is not a constant string */
	int i;
	if(integer_constant_p(f, &i) && i==0) {
	  tf = transformer_identity();
	  tf = transformer_add_equality_with_integer_constant(tf, v, 0);
	}
	else if(float_constant_p(f) && float_constant_to_double(f)==0.) {
	  tf = transformer_identity();
	  tf = transformer_add_equality_with_integer_constant(tf, v, 0);
	}
	else
	  tf = simple_equality_to_transformer(v, f, false);
      }
    }
  }

  pips_debug(9, "End for entity %s\n",
	     entity_local_name(v));

  return tf;
}

/* HANDLING OF CONDITIONS */

#   define DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN 7

/* call transformer_add_condition_information_updown() recursively
 * on both sub-expressions if veracity == true; else
 * try your best...
 *
 */
static transformer transformer_add_condition_information_updown
(transformer, expression, transformer, bool, bool);

static transformer 
transformer_add_anded_conditions_updown(
    transformer pre,
    expression c1,
    expression c2,
    transformer context,
    bool veracity,
    bool upwards)
{
  transformer newpre = transformer_undefined;

  pips_debug(9,"Begin with pre=%p, veracity=%s, upwards=%s\n,",
	     pre, bool_to_string(veracity),
	     bool_to_string(upwards));

  if(veracity) {
    /* FI: this is a bit confusing because of aliasing conditions.
     * The second condition is added after the first one was analyzed
     * and the "and" effect is enforced by side effect
     *
     * This is equivalent to allocating copies of pre and then computing
     * their intersection.
     */
    transformer pre1 = transformer_undefined;
    ifdebug(9) {
      pips_debug(9, "pre=%p:\n", pre);
      dump_transformer(pre);
    }

    pre1 = transformer_add_condition_information_updown
      (pre, c1, context, veracity, upwards);

    ifdebug(9) {
      pips_debug(9, "pre1=%p:\n", pre1);
      dump_transformer(pre1);
    }

    newpre = transformer_add_condition_information_updown
      (pre1, c2, context, veracity, upwards);

    ifdebug(9) {
      pips_debug(9, "newpre=%p:\n", newpre);
      dump_transformer(newpre);
    }
  }
  else if(!upwards  || upwards) { /* This is useful when computing transformers in context */
    /* use precondition pre and hope than the convex hull won't destroy
       all information */
    /* compute !(c1&&c2) as !c1 || !c2 */
    transformer pre1 = transformer_dup(pre);
    transformer pre2 = pre;

    pre1 = transformer_add_condition_information_updown
      (pre1, c1, context, false, upwards);
    pre2 = transformer_add_condition_information_updown
      (pre2, c2, context, false, upwards);
    newpre = transformer_convex_hull(pre1, pre2);

    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
		 "pre1 =\n");
      dump_transformer(pre1);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
		 "pre2 =\n");
      dump_transformer(pre2);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
		 "newpre =\n");
      dump_transformer(newpre);
    }
    free_transformer(pre1);
    free_transformer(pre2);
  }
  else {
    /* might be possible to add a convex hull on the initial values
     * as in !upwards
     */
    newpre = pre;
  }

  ifdebug(9) {
    pips_debug(9, "end with newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

/* call transformer_add_condition_information_updown() recursively on both
 * sub-expressions if veracity == false; else try to do your best...
 *
 * Should be fused with the .AND. case?!? Careful with veracity...
 *
 */

static transformer transformer_add_ored_conditions_updown(
    transformer pre,
    expression c1,
    expression c2,
    transformer context,
    bool veracity,
    bool upwards)
{
  transformer newpre = transformer_undefined;

  pips_debug(9,"Begin with pre=%p, context=%p, veracity=%s, upwards=%s\n,",
	     pre, context, bool_to_string(veracity),
	     bool_to_string(upwards));

  if(!veracity) {
    /* compute !(c1||c2) as !c1 && !c2 */
    newpre = transformer_add_condition_information_updown
      (pre, c1, context, false, upwards);
    newpre = transformer_add_condition_information_updown
      (newpre, c2, context, false, upwards);
  }
  else if(!upwards) {
    /* compute (c1||c2) as such */
    transformer pre1 = transformer_dup(pre);
    transformer pre2 = pre;

    pre1 = transformer_add_condition_information_updown
      (pre1, c1, context, true, upwards);
    pre2 = transformer_add_condition_information_updown
      (pre2, c2, context, true, upwards);
    newpre = transformer_convex_hull(pre1, pre2);

    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	    "pre1 =\n");
      (void) (void) dump_transformer(pre1);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	    "pre2 =\n");
      (void) dump_transformer(pre2);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	    "newpre =\n");
      (void) (void) dump_transformer(newpre);
    }

    free_transformer(pre1);
    free_transformer(pre2);
  }
  else {
    /* might be possible to add a convex hull on the initial values
     * as in !upwards
     */
    newpre = pre;
  }

  ifdebug(9) {
    pips_debug(9, "end with newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

static transformer transformer_add_call_condition_information_updown(
    transformer pre,
    entity op,
    list args,
    transformer context,
    bool veracity,
    bool upwards)
{
  transformer newpre = transformer_undefined;
  expression c1 = expression_undefined;
  expression c2 = expression_undefined;

  pips_debug(9,"Begin with pre=%p, op=%s, veracity=%s, upwards=%s\n,",
	     pre, module_local_name(op), bool_to_string(veracity),
	     bool_to_string(upwards));

  if(!ENDP(args)) {
    c1 = EXPRESSION(CAR(args));
    if(!ENDP(CDR(args))) {
      c2 = EXPRESSION(CAR(CDR(args)));
    }
  }

  if(ENTITY_RELATIONAL_OPERATOR_P(op)) {
    newpre = transformer_add_any_relation_information(pre, op, c1, c2, context,
						      veracity, upwards);
  }
  else if(ENTITY_AND_P(op)) {
    newpre = transformer_add_anded_conditions_updown
      (pre, c1, c2, context, veracity, upwards);
  }
  else if(ENTITY_OR_P(op)) {
    newpre = transformer_add_ored_conditions_updown
      (pre, c1, c2, context, veracity, upwards);
  }
  else if(ENTITY_NOT_P(op)) {
    newpre = transformer_add_condition_information_updown
      (pre, c1, context, !veracity, upwards);
  }
  else if(((ENTITY_TRUE_P(op) || ENTITY_ONE_P(op)) && !veracity) ||
	  ((ENTITY_FALSE_P(op) || ENTITY_ZERO_P(op)) && veracity)) {
    free_transformer(pre);
    newpre = transformer_empty();
  }
  else {
    /* do not know what to do with other logical operators, for the
     * time being! keep pre unmodified
     *
     * FI: two problems, you can bump into arithmetic operators and
     * you can have side effects as in i++
     */
    /* Minimal service: use effects */
    newpre = pre;
    FOREACH(EXPRESSION, a, args) {
      transformer tf = expression_effects_to_transformer(a);
      newpre = transformer_combine(tf, newpre);
    }
  }

  ifdebug(9) {
    pips_debug(9, "end with newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

/* Non-convex information can be made convex when moving postcondition downwards
 * because some of the parts introducing non-convexity may be made empty by
 * the pre-existing precondition.
 *
 * This is not true for transformers computed upwards, although conditions on
 * initial values could also be added. Let's wait for a case which would prove
 * this useful.
 *
 * Argument pre is modified and may or not be returned as newpre. It may also be
 * freed and newpre allocated. In other words, argument "pre" should not be used
 * after a call to this function and the returned value alswayd used.
 */
static transformer transformer_add_condition_information_updown(
    transformer pre,
    expression c,
    transformer context,
    bool veracity,
    bool upwards)
{
  /* default option: no condition can be added */
  transformer newpre = pre;
  syntax s = expression_syntax(c);

  ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
    pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	  "begin upwards=%s veracity=%s c=",
	  bool_to_string(upwards), bool_to_string(veracity));
    print_expression(c);
    (void) fprintf(stderr,"pre=%p\n", pre);
    dump_transformer(pre);
    (void) fprintf(stderr,"and context=%p\n", context);
    dump_transformer(context);
  }

  switch(syntax_tag(s)){
  case is_syntax_call:
    {
      entity f = call_function(syntax_call(s));
      // FI: for the time being, I do not use ultimate_type()
      type rt = functional_result(type_functional(entity_type(f)));
      basic cb = copy_basic(variable_basic(type_variable(rt)));
      if(basic_overloaded_p(cb)) {
	type ert = expression_to_type(c);
	cb = copy_basic(variable_basic(type_variable(ert)));
      }

      if(ENTITY_LOGICAL_OPERATOR_P(f) || ENTITY_TRUE_P(f) || ENTITY_FALSE_P(f) ) {
	list args = call_arguments(syntax_call(s));

	newpre = transformer_add_call_condition_information_updown
	  (pre, f, args, context, veracity, upwards);
      }
      else if(basic_int_p(cb)) {
	if(( ENTITY_ONE_P(f) && !veracity) ||
	   (ENTITY_ZERO_P(f) && veracity)) {
	  free_transformer(pre);
	  newpre = transformer_empty();
	}
	else {
	  entity tmp_v = make_local_temporary_integer_value_entity();
	  transformer ct =
	    safe_integer_expression_to_transformer(tmp_v, c, context, true);
	  if(!veracity)
	    ct = transformer_add_equality_with_integer_constant(ct, tmp_v, 0);
	  newpre = transformer_apply(ct, pre);
	  free_transformer(ct);
	  free_transformer(pre);
	}
      }
      else {
	transformer tf = expression_effects_to_transformer(c);
	newpre = transformer_apply(tf, pre);
	free_transformer(tf);
	free_transformer(pre);
      }
      free_basic(cb);
      break;
    }
  case is_syntax_reference:
    {
      /* A logical variable must be referenced in Fortran. Any
	 variable can be referenced in C. */
      entity l = reference_variable(syntax_reference(s));
      if(entity_has_values_p(l)) {
	entity l_new = entity_to_new_value(l);
	type lt = ultimate_type(entity_type(l));
	basic lb = variable_basic(type_variable(lt));

	if(basic_logical_p(lb)) {
	  Pvecteur eq = vect_new((Variable) l_new, VALUE_ONE);

	  if(veracity) {
	    vect_add_elem(&eq, TCST, VALUE_MONE);
	  }

	  newpre = transformer_equality_add(pre, eq);
	}
	else {
	  Pvecteur eq = vect_new((Variable) l_new, VALUE_ONE);

	  if(veracity) {
	    eq = vect_multiply(eq, VALUE_MONE);
	    vect_add_elem(&eq, TCST, VALUE_ONE);
	  }
	  newpre = transformer_inequality_add(pre, eq);
	}
      }
      break;
    }
  case is_syntax_range:
    {
      pips_internal_error("range used as test condition!");
      break;
    }
  case is_syntax_subscript:
    {
        /* information may be lost */
        newpre = pre;
        break;
    }
  default:
      pips_internal_error("ill. expr. as test condition");
  }

  ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
    pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	  "end newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

transformer transformer_add_condition_information(
    transformer pre,
    expression c,
    transformer context,
    bool veracity)
{
    transformer post =
	transformer_add_condition_information_updown(pre, c, context,
						     veracity, true);

    post = transformer_temporary_value_projection(post);
    reset_temporary_value_counter();

    return post;
}

/* context might be derivable from pre as transformer_range(pre) but this
   is sometimes very computationally intensive, e.g. in ocean. */
transformer
precondition_add_condition_information(
    transformer pre,
    expression c,
    transformer context,
    bool veracity)
{
  transformer post = transformer_undefined;

  if(transformer_undefined_p(context)
     || ENDP(transformer_arguments(context))) {
    post = transformer_add_condition_information_updown
      (pre, c, context, veracity, false);
  }
  else {
    transformer new_context = transformer_range(context);

    post = transformer_add_condition_information_updown
      (pre, c, new_context, veracity, false);
    free_transformer(new_context);
  }

  post = transformer_temporary_value_projection(post);
  reset_temporary_value_counter();

  return post;
}

transformer transformer_add_domain_condition(transformer tf,
					     expression c,
					     transformer context,
					     bool veracity)
{
  tf = transformer_add_condition_information(tf, c, context, veracity);
  return tf;
}

transformer transformer_add_range_condition(transformer tf,
					    expression c,
					    transformer context,
					    bool veracity)
{
  tf = precondition_add_condition_information(tf, c, context, veracity);
  return tf;
}

/* INTEGER EXPRESSIONS */

/* FI: I do no longer understand the semantics of
   "is_internal"... although I designed it. The intent was probably to
   manage temporary values: they should be preserved as long as the
   analysis is internal and else projected.

   Furthermore, this function is no longer very useful as
   normalization can be performed dynamically again and again at a low
   cost.
 */
transformer simple_affine_to_transformer(entity e, Pvecteur a, bool is_internal)
{
  transformer tf = transformer_undefined;
  Pvecteur ve = vect_new((Variable) e, VALUE_ONE);
  Pvecteur vexpr = vect_dup(a);
  Pcontrainte c;
  Pvecteur eq = VECTEUR_NUL;

  pips_debug(8, "begin with is_internal=%s\n", bool_to_string(is_internal));

  ifdebug(9) {
    pips_debug(9, "\nLinearized expression:\n");
    vect_dump(vexpr);
  }

  if(!is_internal
     || (value_mappings_compatible_vector_p(ve) &&
	 value_mappings_compatible_vector_p(vexpr))) {
    /* The renaming used to eliminate equivalenced variables cannot be
       performed because e may be a temporary value. It is better to
       perform it at a higher level, for instance at the assignement
       level. */
    eq = vect_substract(ve, vexpr);
    vect_rm(ve);
    vect_rm(vexpr);
    c = contrainte_make(eq);
    tf = make_transformer(NIL,
			  make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
  }
  else {
    vect_rm(eq);
    vect_rm(ve);
    vect_rm(vexpr);
    tf = transformer_undefined;
  }

  pips_debug(8, "end with tf=%p\n", tf);
  ifdebug(8) dump_transformer(tf);

  return tf;
}

transformer affine_to_transformer(entity e, Pvecteur a, bool assignment)
{
  transformer tf = transformer_undefined;
  Pvecteur ve = vect_new((Variable) e, VALUE_ONE);
  entity e_new = entity_to_new_value(e);
  entity e_old = entity_to_old_value(e);
  cons * tf_args = CONS(ENTITY, e_new, NIL);
  /* must be duplicated right now  because it will be
     renamed and checked at the same time by
     value_mappings_compatible_vector_p() */
  Pvecteur vexpr = vect_dup(a);
  Pcontrainte c;
  Pvecteur eq = VECTEUR_NUL;

  pips_debug(8, "begin\n");

  ifdebug(9) {
    pips_debug(9, "\nLinearized expression:\n");
    vect_dump(vexpr);
  }

  if(!assignment) {
    vect_add_elem(&vexpr, (Variable) e, (Value) 1);

    ifdebug(8) {
      pips_debug(8, "\nLinearized expression for incrementation:\n");
      vect_dump(vexpr);
    }
  }

  if(value_mappings_compatible_vector_p(ve) &&
     value_mappings_compatible_vector_p(vexpr)) {
    ve = vect_variable_rename(ve,
			      (Variable) e,
			      (Variable) e_new);
    (void) vect_variable_rename(vexpr,
				(Variable) e_new,
				(Variable) e_old);
    eq = vect_substract(ve, vexpr);
    vect_rm(ve);
    vect_rm(vexpr);
    c = contrainte_make(eq);
    tf = make_transformer(tf_args,
			  make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
  }
  else {
    vect_rm(eq);
    vect_rm(ve);
    vect_rm(vexpr);
    tf = transformer_undefined;
  }

  pips_debug(8, "end\n");

  return tf;
}

transformer affine_increment_to_transformer(entity e, Pvecteur a)
{
    transformer tf = transformer_undefined;

    tf = affine_to_transformer(e, a, false);

    return tf;
}

static transformer modulo_of_a_constant_to_transformer(entity e, int lb1, int lb2, int ub2)
{
  /* Since arg1 is a constant, ub1=lb1 */
  transformer tf3 = transformer_undefined;

     /* The first argument is numerically known */
      if(lb1==0) {
	/* r == 0 not matter what */
	tf3 = transformer_add_equality_with_integer_constant(transformer_identity(), e, 0);
      }
      else if(lb2==ub2) {
	if(lb2==0) {
	  /* A floating point exception is going to be raised */
	  pips_user_warning("Zero divide encountered\n");
	  tf3 = transformer_empty();
	}
	else {
	  /* The result is known: r = lb1 % lb2 if lb2!=0 */
	  int r = lb1 % lb2;
	  tf3 = transformer_add_equality_with_integer_constant(transformer_identity(), e, r);
	}
      }
      else if(ub2<0) {
	/* The second argument is negative:
	 *  lb1>0? r>=0, r<= ub1, r<=-lb2-1
	 * : r<=0, -ub1<=r, r>= lb2+1 */
	if(lb1>0) {
	  if(lb1<-ub2) {
	    /* the modulo has no impact r=ub1=lb1 */
	    tf3 = transformer_add_equality_with_integer_constant(transformer_identity(), e, lb1);
	  }
	  else {
	    /* r>=0, r<= ub1, r<=-lb2-1 */
	    tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
								     e, 0, false);
	    tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, lb1, true);
	    tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, -lb2-1, true);
	  }
	}
	else {
	  if(lb1>ub2) {
	    /* the modulo has no impact r=ub1=lb1 */
	    tf3 = transformer_add_equality_with_integer_constant(transformer_identity(), e, lb1);
	  }
	  else {
	  /* r<=0, -ub1<=r, r>= lb2+1 */
	  tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
								 e, 0, true);
	  tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, lb1, false);
	  tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, lb2+1, false);
	  }
	}
      }
      else if(lb2>0) {
	/* The second argument is positive:
	 * lb1>0? r>=0, r<= ub1, r<=ub2-1
	 * : r<=0, -ub1<=r, -ub2+1<=r
	 */
	if(lb1>0) {
	  if(lb1<lb2) {
	    /* the modulo has no impact r=ub1=lb1 */
	    tf3 = transformer_add_equality_with_integer_constant(transformer_identity(), e, lb1);
	  }
	  else {
	    /* r>=0, r<= ub1, r<=-lb2-1 */
	    /* r>=0, r<= ub1, r<=ub2-1 */
	    tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
								     e, 0, false);
	    tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, lb1, true);
	    tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, ub2-1, true);
	  }
	}
	else {
	  if(-lb1<ub2) {
	    /* the modulo has no impact r=ub1=lb1 */
	    tf3 = transformer_add_equality_with_integer_constant(transformer_identity(), e, lb1);
	  }
	  else {
	    /* r<=0, -ub1<=r, -ub2+1<=r */
	    tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
								     e, 0, true);
	    tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, lb1, false);
	    tf3 = transformer_add_inequality_with_integer_constraint(tf3, e, -ub2+1, false);
	  }
	}
      }
      else {
	/* no information is available on the second argument: the
	   sign of the first argument is preserved */
	  tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
								 e, 0, lb1<0);
      }
      return tf3;
}

static transformer modulo_of_a_negative_value_to_transformer(entity e,
						      int lb1, int ub1,
						      int lb2, int ub2)
{
  transformer tf3 = transformer_undefined;
  /* arg1 is negative, hence r is always negative: r<=0 */
  if(ub2<0) {
    if(ub2<lb1) {
      /* the moduloe is the identity: lb1<=r<=ub1*/
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, lb1, false);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, ub1, true);
    }
    else {
      /* arg2 is negative: r>=arg2, r>= lb2+1 */
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, 0, true);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, lb2+1, false);
    }
  }
  else if(lb2>0) {
    if(lb2>-ub1) {
      /* the modulo is the identity: lb1<=r<=ub1*/
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, lb1, false);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, ub1, true);
    }
    else {
      /* arg2 is positive: r<=-arg2+1, r<= -lb2+1 */
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, 0, true);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, -lb2+1, true);
    }
  }
  else {
    /* No information is available on arg2: r<=0 */
    tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							     e, 0, true);
  }

  return tf3;
}

static transformer modulo_of_a_positive_value_to_transformer(entity e,
							     int lb1, int ub1,
							     int lb2, int ub2)
{
  transformer tf3 = transformer_undefined;
  /* arg1 is positive, hence r is always positive: r>=0 */
  if(ub2<0) {
    if(-ub2>ub1) {
      /* the modulo is the identity: lb1<=r<=ub1 */
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, lb1, false);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, ub1, true);
    }
    else {
      /* arg2 is negative: 0<=r<=-arg2+1, 0<=r<= -ub2+1 */
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, 0, false);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, -ub2+1, true);
    }
  }
  else if(lb2>0) {
    if(lb2>ub1) {
      /* the modulo is the identity: lb1<=r<=ub1*/
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, lb1, false);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, ub1, true);
    }
    else {
      /* arg2 is positive: 0<=r<=arg2-1, 0<=r<= ub2-1 */
      tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							       e, 0, false);
      tf3 = transformer_add_inequality_with_integer_constraint(tf3,
							       e, ub2-1, true);
    }
  }
  else {
    /* No information is available on arg2: r>=0 */
    tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							     e, 0, false);
  }

  return tf3;
}

/* Analyze v % k, with v constrainted by tf, assuming tf is a precondition
 *
 * See if exist a set v_i such that v = sum_i a_i v_i + c. Then v can
 * be rewritten v = gcd_i(a_i) lambda + c.
 *
 * if k divides gcd(a_i) then v % k = c % k
 *
 */
transformer modulo_by_a_constant_to_transformer(entity v1, transformer tf, entity v2, int k)
{
  transformer mtf = transformer_identity();
  Value gcd, c, K = (Value) k;

  // exists lambda s.t. v2 = lambda gcd + c ^ 0<=c<gcd
  if(transformer_to_1D_lattice(v2, tf, &gcd, &c)) {

  // false because we are waiting for function by Corinne
  if(gcd!=0) {
    if(gcd%k==0)
      mtf = transformer_add_equality_with_integer_constant(mtf, v1, c%K);
  }
  else
      mtf = transformer_add_equality_with_integer_constant(mtf, v1, c%K);
  }
  else {
    // There is no solution...
    // This case should be detected earlier and this piece of code is
    // probably unreachable
    mtf = empty_transformer(mtf);
  }
  return mtf;
}

/* Modulo and integer division
 *
 * Apparently identical in both C and Fortran
 *
 *           C    Fortran
 *  a  b  mod div  mod div
 *  3  2   1   1    1
 * -3  2  -1  -1   -1
 *  3 -2   1  -1    1
 * -3 -2  -1   1   -1
 *
 * FI: only implemented for positive dividends. Same for integer
 * division, I believe. Side effects are probably ignored.
 *
 * FI: to be improved by using expression_to_transformer() and the
 * precondition pre.
 *
 * FI: to be improved by using the lattice information in the
 * equations of the precondition.
 *
 * matrice_hermite()
 *
 * sc_to_matrices()
 */
static transformer modulo_to_transformer(entity e, /* assumed to be an integer value */
					 expression arg1,
					 expression arg2,
					 transformer pre, /* not used yet */
					 bool is_internal __attribute__ ((unused)))
{
  transformer tf = transformer_undefined;
  entity v1 =  make_local_temporary_integer_value_entity();
  transformer tf1 = safe_any_expression_to_transformer(v1, arg1, pre, true);
  transformer npre =
    transformer_apply(tf1, transformer_undefined_p(pre)? transformer_identity():pre);
  transformer pre2 = transformer_range(npre);
  entity v2 =  make_local_temporary_integer_value_entity();
  transformer tf2 = safe_any_expression_to_transformer(v2, arg2, pre2, true);
  npre = transformer_apply(tf2, pre2);
  transformer pre3 = transformer_range(npre);
  // To be cleaned up later
  //transformer tf3 = transformer_undefined;
  transformer tf3 = transformer_identity();

  pips_debug(8, "begin with pre=%p\n", pre);

  /* cut-and-pasted and adapted from multiply_to_transformer(); also useful for
     divide; the side effects of the two arguments and the derivation
     of the preconditions and the computation of the numerical bounds
     should be factorized in one function. */
  if(!transformer_undefined_p(pre) && !transformer_undefined_p(npre)) {
    int lb1 = 0;
    int ub1 = 0;
    int lb2 = 0;
    int ub2 = 0;
    expression ev1 = entity_to_expression(v1);
    expression ev2 = entity_to_expression(v2);

    expression_and_precondition_to_integer_interval(ev1, pre2, &lb1, &ub1);
    expression_and_precondition_to_integer_interval(ev2, pre3, &lb2, &ub2);
    free_expression(ev1);
    free_expression(ev2);

    if(lb2==ub2) {
      if(ub2==0) {
      /* Let's get rid of the exception case right away */
      /* A floating point exception is going to be raised */
      pips_user_warning("Zero divide encountered\n");
      tf3 = transformer_empty();
      }
      else {
      // lb2==ub2!=0
      // FI: we might have to impose lb2>0
      // FI: let's assume that pre is carried by tf1
	tf3 = modulo_by_a_constant_to_transformer(e, tf1, v1, lb2);
      }
    }

    if(transformer_identity_p(tf3)) {
      free_transformer(tf3);
      if(lb1==ub1) {
	tf3 = modulo_of_a_constant_to_transformer(e, lb1, lb2, ub2);
      }
      else if(ub1<=0) {
	tf3 = modulo_of_a_negative_value_to_transformer(e, lb1, ub1, lb2, ub2);
      }
      else if(lb1>=0) {
	tf3 = modulo_of_a_positive_value_to_transformer(e, lb1, ub1, lb2, ub2);
      }
      else {
	/* No sign information available about arg1 */
	if(lb2>0) {
	  /* arg2 is positive: -ub2+1<=r<=ub2-1 */
	  tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
								   e, -ub2+1, false);
	  tf3 = transformer_add_inequality_with_integer_constraint(tf3,
								   e, ub2-1, true);
	}
	else if(ub2<0) {
	  /* arg2 is negative: lb2+1<=r<=-lb2-1 */
	  tf3 = transformer_add_inequality_with_integer_constraint(transformer_identity(),
								   e, lb2+1, false);
	  tf3 = transformer_add_inequality_with_integer_constraint(tf3,
								   e, -lb2-1, true);
	}
	else
	  tf3 = transformer_identity();
      }
    }
  }
  else {
    /* Might be as good to leave it undefined... for the time
       being. Information about either arg1 or arg2 could be used */
    tf3 = transformer_identity();
    ;
  }


  /* Take care of side effects: tf = tf1 o tf2 o tf3 */
  tf = transformer_combine(tf1, tf2);
  tf = transformer_combine(tf, tf3);
  free_transformer(tf2);
  free_transformer(tf3);

  free_transformer(pre2);
  free_transformer(pre3);
  free_transformer(npre);

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

static transformer iabs_to_transformer(entity v, /* assumed to be a value */
				       expression expr,
				       transformer pre,
				       bool is_internal)
{
  transformer tf = transformer_identity();
  entity tv = make_local_temporary_value_entity(entity_type(v));
  transformer etf = integer_expression_to_transformer(tv, expr, pre, is_internal);
  Pvecteur vlb1 = vect_new((Variable) tv, VALUE_ONE);
  Pvecteur vlb2 = vect_new((Variable) tv, VALUE_MONE);

  pips_debug(8, "begin\n");


  vect_add_elem(&vlb1, (Variable) v, VALUE_MONE);
  vect_add_elem(&vlb2, (Variable) v, VALUE_MONE);

  tf = transformer_inequality_add(tf, vlb1);
  tf = transformer_inequality_add(tf, vlb2);
  tf = transformer_safe_image_intersection(tf, etf);
  free_transformer(etf);

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

/* Copy of iabs_to_transformer(), which is probably never called anymore */
static transformer generic_abs_to_transformer(entity v, /* assumed to be a value */
					      expression expr,
					      transformer pre,
					      bool is_internal)
{
  transformer tf = transformer_identity();
  /* Expression expr may be of a type different from v's type because
     abs maybe generic and because type conversions may be implicit */
  entity tv = make_local_temporary_value_entity(entity_type(v));
  type et = expression_to_type(expr);
  transformer etf = any_expression_to_transformer(tv, expr, pre, is_internal);
  int lb=INT_MIN, ub=INT_MAX;

  pips_debug(8, "begin\n");

  if(integer_type_p(et))
    integer_expression_and_precondition_to_integer_interval(expr, pre, &lb, &ub);
  free_type(et);

  if(ub==lb) {
    tf = transformer_add_equality_with_integer_constant(tf, v, abs(lb));
  }
  else if(lb>=0) {
    tf = transformer_add_equality(tf, v, tv);
  }
  else if(ub<=0) {
    tf = transformer_add_equality_with_affine_term(tf, v, tv,
						   VALUE_MONE, VALUE_ZERO);
  }
  else {
    Pvecteur vlb1 = vect_new((Variable) tv, VALUE_ONE);
    Pvecteur vlb2 = vect_new((Variable) tv, VALUE_MONE);


    vect_add_elem(&vlb1, (Variable) v, VALUE_MONE);
    vect_add_elem(&vlb2, (Variable) v, VALUE_MONE);

    tf = transformer_inequality_add(tf, vlb1);
    tf = transformer_inequality_add(tf, vlb2);
  }

  tf = transformer_safe_image_intersection(tf, etf);
  free_transformer(etf);

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

/* More could be done along the line of
   integer_multiply_to_transformer()... when need arises.*/
static transformer integer_divide_to_transformer(entity e,
						 expression arg1,
						 expression arg2,
						 transformer pre,
						 bool is_internal)
{
  transformer tf = transformer_undefined;
  normalized n1 = NORMALIZE_EXPRESSION(arg1);

  pips_debug(8, "begin with is_internal=%s\n",
	     bool_to_string(is_internal));

  /* pips_assert("Precondition is unused", transformer_undefined_p(pre)); */
  pips_assert("Shut up the compiler", pre==pre);

  if(integer_constant_expression_p(arg2) && normalized_linear_p(n1)
     && value_mappings_compatible_vector_p(normalized_linear(n1))) {
    int d = integer_constant_expression_value(arg2);
    /* must be duplicated right now  because it will be
       renamed and checked at the same time by
       value_mappings_compatible_vector_p() */
    Pvecteur vlb =
      vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE);
    Pvecteur vub = vect_dup(normalized_linear(n1));
    Pcontrainte clb = CONTRAINTE_UNDEFINED;
    Pcontrainte cub = CONTRAINTE_UNDEFINED;

    vect_add_elem(&vlb, (Variable) e, int_to_value(d));
    vect_add_elem(&vub, (Variable) e, int_to_value(-d));
    vect_add_elem(&vub, TCST, int_to_value(1-d));
    clb = contrainte_make(vlb);
    cub = contrainte_make(vub);
    clb->succ = cub;
    tf = make_transformer(NIL,
			  make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
  }

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}
/* More could be done along the line of
   integer_left_shift_to_transformer()... when need arises.*/
static transformer integer_right_shift_to_transformer(entity e,
						      expression arg1,
						      expression arg2,
						      transformer pre,
						      bool is_internal)
{
  transformer tf = transformer_undefined;
  normalized n1 = NORMALIZE_EXPRESSION(arg1);

  pips_debug(8, "begin with is_internal=%s\n",
	     bool_to_string(is_internal));

  /* pips_assert("Precondition is unused", transformer_undefined_p(pre)); */
  pips_assert("Shut up the compiler", pre==pre);

  if(integer_constant_expression_p(arg2) && normalized_linear_p(n1)
     && value_mappings_compatible_vector_p(normalized_linear(n1))) {
    int d = integer_constant_expression_value(arg2);
    Value val = value_lshift((Value) 1, (Value) d);
    /* must be duplicated right now  because it will be
       renamed and checked at the same time by
       value_mappings_compatible_vector_p() */
    Pvecteur vlb =
      vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE);
    Pvecteur vub = vect_dup(normalized_linear(n1));
    Pcontrainte clb = CONTRAINTE_UNDEFINED;
    Pcontrainte cub = CONTRAINTE_UNDEFINED;

    vect_add_elem(&vlb, (Variable) e, val);
    vect_add_elem(&vub, (Variable) e, -val);
    vect_add_elem(&vub, TCST, VALUE_ONE-val);
    clb = contrainte_make(vlb);
    cub = contrainte_make(vub);
    clb->succ = cub;
    tf = make_transformer(NIL,
			  make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
  }

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

/* Assumes that e1 and e2 are integer expressions, i.e. explicit casting
   is supposed to be used */
/* Better results might be obtained when e1 is an affine function of e2 or
   vice-versa as can occur when convex hulls generate coupling between
   variables. See non_linear11.f, 22L1=9L2+40, 1<=L1<=10. The smallest
   possible value is -2 and not -20. */
static transformer integer_multiply_to_transformer(entity v,
    expression e1,
    expression e2,
    transformer prec,
    bool is_internal)
{
  transformer tf = transformer_undefined;
  entity v1 = make_local_temporary_value_entity(entity_type(v));
  transformer ipre = transformer_undefined_p(prec)? transformer_identity() : transformer_range(prec);
  transformer t1 = safe_integer_expression_to_transformer(v1, e1, ipre, is_internal);
  entity v2 = make_local_temporary_value_entity(entity_type(v));
  transformer npre = transformer_safe_apply(t1, ipre);
  transformer t2 = safe_integer_expression_to_transformer(v2, e2, npre, is_internal);
  //transformer pre = transformer_undefined_p(ipre)? transformer_identity() :
  //copy_transformer(ipre);
  transformer pre = transformer_undefined_p(ipre)? transformer_identity() :
    ipre;

  pips_debug(8, "Begin\n");

  if(!transformer_undefined_p(t1) && !transformer_undefined_p(t2)) {
    int lb1 = 0;
    int ub1 = 0;
    int lb2 = 0;
    int ub2 = 0;

    /*
     * v1 and v2 are temp value,
     * so don't have normalize
     * so expression_and_precondition_to_integer_interval can't say anything interesting
     */
    //expression ev1 = entity_to_expression(v1);
    //expression ev2 = entity_to_expression(v2);

    /* FI: I had to switch the arguments to satisfy a reasonnable
       assert in image_intersection(), but the switch may be
       detrimental to memory allocation. */
    //t1 = transformer_safe_image_intersection(pre, t1);
    //t2 = transformer_safe_image_intersection(pre, t2);
    t1 = transformer_range(transformer_apply(pre, t1));
    t2 = transformer_range(transformer_apply(pre, t2));

    // Nelson Lossing: in some cases, these calls are useless and
    // return [MIN,MAX] because ev1 and ev2 are built with temporary values
    // In his case, it works if ev1 and ev2 are replaced by e1 and e2
    //expression_and_precondition_to_integer_interval(ev1, t1, &lb1, &ub1);
    //expression_and_precondition_to_integer_interval(ev2, t2, &lb2, &ub2);
    integer_expression_and_precondition_to_integer_interval(e1, t1, &lb1, &ub1);
    integer_expression_and_precondition_to_integer_interval(e2, t2, &lb2, &ub2);
    //free_expression(ev1);
    //free_expression(ev2);

    if(lb1==ub1) {
      /* The numerical value of expression e1 is known: v = lb1*v2 */
      Pvecteur veq = vect_new((Variable) v, VALUE_MONE);

      vect_add_elem(&veq, (Variable) v2, (Value) lb1);
      transformer_equality_add(t2, veq);
      tf = t2;
      t2 = transformer_undefined;
      //free_transformer(t1);
    }
    else if(lb2==ub2) {
      /* The numerical value of expression e2 is known: v = lb2*v1 */
      Pvecteur veq = vect_new((Variable) v, VALUE_MONE);

      vect_add_elem(&veq, (Variable) v1, (Value) lb2);
      tf = transformer_equality_add(t1, veq);
      t1 = transformer_undefined;
      //free_transformer(t2);
    }
    else if(ub1<=0) { // e1 is negative
      if(ub2<=0) { // e2 is negative
          // v >= v1 * ub2, v >= ub1 * v2
          // v <= v1 * lb2, v <= lb1 * v2 (if lb1 and lb2 exist)
          tf = transformer_intersection(t1,t2); // FI: not good if side effects!
          tf = transformer_normalize(tf, 2);
          tf = transformer_add_inequality_with_linear_term(tf,v,v1,ub2,false);
          tf = transformer_add_inequality_with_linear_term(tf,v,v2,ub1,false);
          if(lb2>INT_MIN)
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,lb2,true);
          if(lb1>INT_MIN)
            tf = transformer_add_inequality_with_linear_term(tf,v,v2,lb1,true);
      }
      else if(lb2>=0) { // e2 is positive
          // v <= v1 * lb2, v <= ub1 * v2
          // v >= v1 * ub2, v >= lb1 * v2 (if lb1 and ub2 exist)
          tf = transformer_intersection(t1,t2); // FI: not good if side effects!
          tf = transformer_normalize(tf, 2);
          tf = transformer_add_inequality_with_linear_term(tf,v,v1,lb2,true);
          tf = transformer_add_inequality_with_linear_term(tf,v,v2,ub1,true);
          if(ub2<INT_MAX)
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,ub2,false);
          if(lb1>INT_MIN)
            tf = transformer_add_inequality_with_linear_term(tf,v,v2,lb1,false);
      }
      else { // if lb2 and ub2 are known and their signs are different
          if(lb2>INT_MIN && ub2<INT_MAX) {
            // v1 * ub2 <= v <= v1 * lb2
            tf = transformer_intersection(t1,t2); // FI: not good if side effects!
            tf = transformer_normalize(tf, 2);
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,ub2,false);
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,lb2,true);
            ;
          }
          else {
            tf = transformer_identity();
          }

      }
    }
    else if(lb1>=0) {
      if(lb2>=0) {
          // v >= lb1 * v2, v >= v1 * lb2
          // v <= ub1 * v2, v <= v1 * ub2 (if ub1 and ub2 exist)
          tf = transformer_intersection(t1,t2); // FI: not good if side effects!
          tf = transformer_normalize(tf, 2);
          tf = transformer_add_inequality_with_linear_term(tf,v,v2,lb1,false);
          tf = transformer_add_inequality_with_linear_term(tf,v,v1,lb2,false);
          if(ub1<INT_MAX)
            tf = transformer_add_inequality_with_linear_term(tf,v,v2,ub1,true);
          if(ub2<INT_MAX)
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,ub2,true);
      }
      else if(ub2<=0) {
          // v <= v1 * ub2, v <= lb1 * v2
          // v >= v1 * lb2, v >= ub1 * v2 (if ub1 and lb2 exist)
          tf = transformer_intersection(t1,t2); // FI: not good if side effects!
          tf = transformer_normalize(tf, 2);
          tf = transformer_add_inequality_with_linear_term(tf,v,v1,ub2,true);
          tf = transformer_add_inequality_with_linear_term(tf,v,v2,lb1,true);
          if(lb2>INT_MIN)
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,lb2,false);
          if(ub1>INT_MAX)
            tf = transformer_add_inequality_with_linear_term(tf,v,v2,ub1,false);
          ;
      }
      else {
          if(lb2>INT_MIN && ub2<INT_MAX) { // FI: separated test needed
            // v1 * lb2 <= v <= v1 * ub2
            tf = transformer_intersection(t1,t2); // FI: not good if side effects!
            tf = transformer_normalize(tf, 2);
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,lb2,false);
            tf = transformer_add_inequality_with_linear_term(tf,v,v1,ub2,true);
          }
          else {
            tf = transformer_identity();
          }
      }
    }
    else {
      /* FI: This should now be obsolete... but it is not when the
	 signs are unknown */

      /* Do we have range information? */
      long long p1 = ((long long) lb1 )*((long long) lb2 );
      long long p2 = ((long long) lb1 )*((long long) ub2 );
      long long p3 = ((long long) ub1 )*((long long) lb2 );
      long long p4 = ((long long) ub1 )*((long long) ub2 );
      long long lb = (p2<p1)? p2 : p1;
      long long ub = (p2>p1)? p2 : p1;

      lb = (p3<lb)? p3 : lb;
      lb = (p4<lb)? p4 : lb;

      ub = (p3>ub)? p3 : ub;
      ub = (p4>ub)? p4 : ub;

      //free_transformer(t1);
      //free_transformer(t2);

      if(lb > INT_MIN || ub < INT_MAX)
          tf = transformer_identity();

      if(lb > INT_MIN) {
          Pvecteur vineql = vect_new((Variable) v, VALUE_MONE);

          vect_add_elem(&vineql, TCST, lb);
          tf = transformer_inequality_add(transformer_identity(), vineql);
      }
      if(ub < INT_MAX) {
          Pvecteur vinequ = vect_new((Variable) v, VALUE_ONE);

          vect_add_elem(&vinequ, TCST, -ub);
          tf = transformer_inequality_add(tf, vinequ);
      }
    }
  }
  else if(!transformer_undefined_p(t1)) {
    // free_transformer(t1);
  }
  else if(!transformer_undefined_p(t2)) {
    //free_transformer(t2);
    //t2 = transformer_undefined;
  }

  free_transformer(pre);

  /* let's try to exploit some properties of squaring, but the
     detection is poor. For instance 2*i*i won't be identified as a
     square. */
  if(transformer_undefined_p(tf)) {
    /* let's assume no impact from side effects */
    if(expression_equal_p(e1, e2)) {
      // FI: we could do a much better job here
      // a different function such as integer_square_to_transformer()
      // should be called to also generate the first terms
      // v>=t, v>=-t, v>=3t-2, v>=-3t-2
      // This terms are useful when t is connected because e1 is
      // affine and when the programmer plays on the behavior of
      // square around 0
      // See cavern01.c

      /* v is greater than v1 and -v1. This makes v>=0 redundant */
      if(!transformer_undefined_p(t1)) {
        tf = transformer_add_inequality(t1, v1, v, false);
        tf = transformer_add_inequality_with_affine_term(tf, v,
            v1, VALUE_MONE, VALUE_ZERO,
            false);
        /* Memory leak with tf... */
        tf = transformer_intersection(tf, t1);
      }
      else {
        /* The result is always positive: v>=0 */
        Pvecteur vineq = vect_new((Variable) v, VALUE_MONE);
        tf = transformer_identity();
        tf = transformer_inequality_add(tf, vineq);
      }
    }
    else if(expression_opposite_p(e1, e2)) {
      /* v is less than v1 and -v1. This makes v<=0 redundant */
      if(!transformer_undefined_p(t1)) {
        tf = transformer_add_inequality(t1, v, v1, false);
        tf = transformer_add_inequality_with_affine_term(tf, v,
            v1, VALUE_MONE, VALUE_ZERO,
            true);
        /* Memory leak with tf... */
        tf = transformer_intersection(tf, t1);
      }
      else {
        /* The result is always negative: v<=0 */
        Pvecteur vineq = vect_new((Variable) v, VALUE_ONE);
        tf = transformer_identity();
        tf = transformer_inequality_add(tf, vineq);
      }
    }
  }
  else {
    if(expression_equal_p(e1, e2)) {
      /* The result is always positive: v>=0 */
      Pvecteur vineq = vect_new((Variable) v, VALUE_MONE);
      tf = transformer_inequality_add(tf, vineq);
    }
    else if(expression_opposite_p(e1, e2)) {
      /* The result is always negative: v<=0 */
      Pvecteur vineq = vect_new((Variable) v, VALUE_ONE);
      tf = transformer_inequality_add(tf, vineq);
    }
  }

  if(!transformer_undefined_p(t1)) {
    free_transformer(t1);
  }
  if(!transformer_undefined_p(t2)) {
    free_transformer(t2);
  }

  pips_debug(8, "End with tf=%p\n", tf);
  ifdebug(8) (void) dump_transformer(tf);

  return tf;
}

/* cut-and-pasted and adapted from multiply_to_transformer();  */
// only take 1 arg as expression which will multiply by the sizeof
// it's a version simplify of multiply_to_transformer because we know that sizeof>0.
static transformer expression_multiply_sizeof_to_transformer(entity v,
    expression e1,
    transformer prec,
    bool is_internal)
{
  type tv = entity_type(v);
  type t = type_undefined;
  if (type_variable_p(tv)) {
    basic b = variable_basic(type_variable(tv));
    if (basic_pointer_p(b))
      t = basic_pointer(b);
  }
  if (type_undefined_p(t)) {
    return transformer_empty();
  }


  // create an entity for sizeof(type)
  add_sizeof_value(t);
  // get the sizeof entity
  entity val2 = type_to_sizeof_value(t);
  // convert to expression for transformer
  expression e2 = entity_to_expression(val2);

  transformer tf = transformer_undefined;
  entity v1 = make_local_temporary_value_entity(entity_type(v));
  transformer ipre = transformer_undefined_p(prec)? transformer_identity() : transformer_range(prec);
  transformer t1 = safe_integer_expression_to_transformer(v1, e1, ipre, is_internal);
  entity v2 = make_local_temporary_value_entity(entity_type(v));
  transformer npre = transformer_safe_apply(t1, ipre);
  transformer t2 = safe_integer_expression_to_transformer(v2, e2, npre, is_internal);
  transformer pre = transformer_undefined_p(ipre)? transformer_identity() :
    ipre;

  pips_debug(8, "Begin\n");

  if(!transformer_undefined_p(t1) && !transformer_undefined_p(t2)) {
    int lb1 = 0;
    int ub1 = 0;
    //e2 sizeof then positive

    /* FI: I had to switch the arguments to satisfy a reasonnable
       assert in image_intersection(), but the switch may be
       detrimental to memory allocation. */
    //t1 = transformer_safe_image_intersection(pre, t1);
    t1 = transformer_normalize(t1, 2);
    t1 = transformer_range(transformer_apply(pre, t1));

    integer_expression_and_precondition_to_integer_interval(e1, t1, &lb1, &ub1);

    if(lb1==ub1) {
      /* The numerical value of expression e1 is known: v = lb1*v2 */
      Pvecteur veq = vect_new((Variable) v, VALUE_MONE);

      vect_add_elem(&veq, (Variable) v2, (Value) lb1);
      transformer_equality_add(t2, veq);
      tf = t2;
      free_transformer(t1);
    }
    else if(ub1<=0) { // e1 is negative
      // e2 is positive
      // v <= v1 * lb2, v <= ub1 * v2
      // v >= v1 * ub2, v >= lb1 * v2 (if lb1 and ub2 exist)
      tf = transformer_intersection(t1,t2); // FI: not good if side effects!
      tf = transformer_normalize(tf, 2);
      tf = transformer_add_inequality_with_linear_term(tf,v,v2,ub1,true);
      if(lb1>INT_MIN)
        tf = transformer_add_inequality_with_linear_term(tf,v,v2,lb1,false);
    }
    else if(lb1>=0) {
      // v >= lb1 * v2, v >= v1 * lb2
      // v <= ub1 * v2, v <= v1 * ub2 (if ub1 and ub2 exist)
      tf = transformer_intersection(t1,t2); // FI: not good if side effects!
      tf = transformer_normalize(tf, 2);
      tf = transformer_add_inequality_with_linear_term(tf,v,v2,lb1,false);
      if(ub1<INT_MAX)
        tf = transformer_add_inequality_with_linear_term(tf,v,v2,ub1,true);
    }
    else { // FI: This should now be obsolete...
      if(lb1>INT_MIN && ub1<INT_MAX) { // FI: separated test needed
        // v1 * lb2 <= v <= v1 * ub2
        tf = transformer_intersection(t1,t2); // FI: not good if side effects!
        tf = transformer_normalize(tf, 2);
        tf = transformer_add_inequality_with_linear_term(tf,v,v2,lb1,false);
        tf = transformer_add_inequality_with_linear_term(tf,v,v2,ub1,true);
      }
      else {
        long long lb = lb1;
        long long ub = ub1;

        free_transformer(t1);
        free_transformer(t2);

        if(lb > INT_MIN || ub < INT_MAX)
            tf = transformer_identity();

        if(lb > INT_MIN) {
            Pvecteur vineql = vect_new((Variable) v, VALUE_MONE);

            vect_add_elem(&vineql, TCST, lb);
            tf = transformer_inequality_add(transformer_identity(), vineql);
        }
        if(ub < INT_MAX) {
            Pvecteur vinequ = vect_new((Variable) v, VALUE_ONE);

            vect_add_elem(&vinequ, TCST, -ub);
            tf = transformer_inequality_add(tf, vinequ);
        }
      }
    }
  }
  else if(!transformer_undefined_p(t1)) {
    free_transformer(t1);
  }
  else if(!transformer_undefined_p(t2)) {
    free_transformer(t2);
  }

  free_transformer(pre);
  free_expression(e2);

  pips_debug(8, "End with tf=%p\n", tf);
  ifdebug(8) (void) dump_transformer(tf);

  return tf;
}

/* Assumes that e1 and e2 are integer expressions, i.e. explicit casting
 * is supposed to be used.
 *
 * This piece of code has been copied and pasted from
 * integer_multiply_to_transformer(). Common issues like evaluating
 * the intervals for the values of the arguments should be factored
 * out. */
static transformer integer_left_shift_to_transformer(entity v,
						expression e1,
						expression e2,
						transformer prec,
						bool is_internal)
{
  transformer tf = transformer_undefined;
  // FI: ultimate_type should be called? Or int assumed?
  entity v1 = make_local_temporary_value_entity(entity_type(v));
  transformer ipre = transformer_undefined_p(prec)?
    transformer_identity() : transformer_range(prec);
  transformer t1 = safe_integer_expression_to_transformer(v1, e1, ipre, is_internal);
  entity v2 = make_local_temporary_value_entity(entity_type(v));
  transformer npre = transformer_safe_apply(t1, ipre);
  transformer t2 = safe_integer_expression_to_transformer(v2, e2, npre, is_internal);
  //transformer pre = transformer_undefined_p(ipre)? transformer_identity() :
  //  copy_transformer(ipre);
  transformer pre = ipre;

  pips_debug(8, "Begin\n");

  if(!transformer_undefined_p(t1) && !transformer_undefined_p(t2)) {
    intptr_t lb1 = 0;
    intptr_t ub1 = 0;
    intptr_t lb2 = 0;
    intptr_t ub2 = 0;

    /* FI: I had to switch the arguments to satisfy a reasonnable
       assert in image_intersection(), but the switch may be
       detrimental to memory allocation. */
    //t1 = transformer_safe_image_intersection(pre, t1);
    //t2 = transformer_safe_image_intersection(pre, t2);
    t1 = transformer_range(transformer_apply(pre, t1));
    t2 = transformer_range(transformer_apply(pre, t2));

    (void) precondition_minmax_of_value(v1, t1, &lb1, &ub1);
    (void) precondition_minmax_of_value(v2, t2, &lb2, &ub2);

    if(lb2==ub2) {
      /* The numerical value of expression e2 is known: v = v1*2^lb2 */
      Pvecteur veq = vect_new((Variable) v, VALUE_MONE);

      if(lb2>=0) {
	vect_add_elem(&veq, (Variable) v1,
		      value_lshift((Value) 1, (Value) lb2));
      }
      else  /* lb2<0 */ {
	// FI: Should look at the norm; seems to return 0 with gdb
	;
      }

      tf = transformer_equality_add(t1, veq);
      free_transformer(t2);
    }
    else if(0<lb2 && ub2<INT_MAX) {
      /* The value of v belongs to a bounded interval */
      if(0<=lb1) {
	// v1 is positive: v1*2^lb2 <= v <= v1*2^ub2
	Pvecteur veq1 = vect_new((Variable) v, VALUE_MONE);
	Pvecteur veq2 = vect_new((Variable) v, VALUE_ONE);
	vect_add_elem(&veq1, (Variable) v1,
		      value_lshift((Value) 1, (Value) lb2));
	vect_add_elem(&veq2, (Variable) v1,
		      -value_lshift((Value) 1, (Value) ub2));
	tf = transformer_inequality_add(t1, veq1);
	tf = transformer_inequality_add(t1, veq2);
	free_transformer(t2);
      }
      else if(ub1<=0) {
	// v1 is negative: v1*2^ub2 <= v <= v1*2^lb2
	Pvecteur veq1 = vect_new((Variable) v, VALUE_MONE);
	Pvecteur veq2 = vect_new((Variable) v, VALUE_ONE);
	vect_add_elem(&veq1, (Variable) v1,
		      value_lshift((Value) 1, (Value) ub2));
	vect_add_elem(&veq2, (Variable) v1,
		      -value_lshift((Value) 1, (Value) lb2));
	tf = transformer_inequality_add(t1, veq1);
	tf = transformer_inequality_add(t1, veq2);
	free_transformer(t2);
      }
      else {
	// No information is available because the sign of v1 is
	// unknown
	;
      }
    }
    else {
      // Let's hope that the sign of v2 is known as well as the sign
      // of v1. It does not matter if v1 is known exactly or not
      // a new transformer must be built
      //free_transformer(t1);
      free_transformer(t2);
      //transformer tf = transformer_identity();

      if(lb2>=0 && lb1>=1) {
	// The value of v is greater than the value of v1 multiplied
	// by 2
	tf = transformer_add_inequality_with_affine_term(t1, v, v1, 2, 0, false);
      }
      else if(lb2>=0 && lb1==0) {
	// The value of v is 0
	tf = transformer_add_inequality(t1, v1, v, false);
	//tf = transformer_add_equality_with_integer_constant(tf, v, 0);
      }
      else if(lb2>0 && ub1<0) {
	// The value of v is lesser than the value of v1
	tf = transformer_add_inequality(t1, v, v1, true);
      }
      else if(lb2>0 && ub1==0) {
	// The value of v is lesser than the value of v1
	tf = transformer_add_inequality(t1, v, v1, false);
      }
      else if(ub2<0) {
	// The value of v is zero
	free_transformer(t1);
	tf = transformer_identity();
	tf = transformer_add_equality_with_integer_constant(tf, v, 0);
      }
      else // nothing is know about v2: the sign of v1 is preserved
	// To be checked in C standard...
	if(0<=lb1)
	  tf = transformer_add_inequality_with_integer_constraint(t1, v, 0, false);
	else if(ub1<=0)
	  tf = transformer_add_inequality_with_integer_constraint(t1, v, 0, true);
    }
  }
  else if(!transformer_undefined_p(t1)) {
    free_transformer(t1);
  }
  else if(!transformer_undefined_p(t2)) {
    free_transformer(t2);
  }

  free_transformer(pre);

  pips_debug(8, "End with tf=%p\n", tf);
  ifdebug(8) (void) dump_transformer(tf);

  return tf;
}

static transformer integer_power_to_transformer(entity e,
						expression arg1,
						expression arg2,
						transformer prec,
						bool is_internal)
{
  transformer tf = transformer_undefined;
  normalized n1 = NORMALIZE_EXPRESSION(arg1);
  normalized n2 = NORMALIZE_EXPRESSION(arg2);
  transformer pre = transformer_undefined_p(prec)?
    transformer_identity() : transformer_range(prec);

  pips_debug(8, "begin\n");

  /* Should be rewritten using expression_to_transformer and expression
     evaluation as in integer_multiply_to_transformer */
  /* pips_assert("Precondition is unused", transformer_undefined_p(pre)); */
  //pips_assert("Shut up the compiler", pre==pre);
  pips_assert("Shut up the compiler", is_internal==is_internal);

  /* Is arg1 bounded? constant? */
  bool arg1_is_constant_p = false;
  //bool arg1_is_bounded_p = false;
  intptr_t lb1, ub1;
  if(precondition_minmax_of_expression(arg1, pre, &lb1, &ub1)) {
    if(lb1==ub1) {
      arg1_is_constant_p = true;
    }
  }
  /* Is arg2 bounded? constant? */
  bool arg2_is_constant_p = false;
  //bool arg2_is_bounded_p = false;
  intptr_t lb2, ub2;
  if(precondition_minmax_of_expression(arg2, pre, &lb2, &ub2)) {
    if(lb2==ub2) {
      arg2_is_constant_p = true;
    }
  }

  if(arg1_is_constant_p && arg2_is_constant_p) {
    int v = (int) exponentiate((Value) lb1, (Value) lb2);
    tf = transformer_identity();
    tf = transformer_add_equality_with_integer_constant(tf, e, v);
  }
  else if(arg2_is_constant_p && normalized_linear_p(n1)) {
    //int d = signed_integer_constant_expression_value(arg2);
    int d = (int) lb2;

    if(d%2==0) {

      if(d==0) {
	/* 1 is assigned unless arg1 equals 0... which is neglected */
	Pvecteur v = vect_new((Variable) e, VALUE_ONE);

	vect_add_elem(&v, TCST, VALUE_MONE);
	tf = make_transformer(NIL,
			      make_predicate(sc_make(contrainte_make(v),
						     CONTRAINTE_UNDEFINED)));
      }
      else if(d>0) {
	/* Does not work because unary minus is not seen as part of a constant */
	/* The expression value must be greater or equal to arg2 and positive */
	/* must be duplicated right now  because it will be
	   renamed and checked at the same time by
	   value_mappings_compatible_vector_p() */
	Pvecteur vlb1 = vect_dup(normalized_linear(n1));
	Pvecteur vlb2 = vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE);
	Pcontrainte clb1 = CONTRAINTE_UNDEFINED;
	Pcontrainte clb2 = CONTRAINTE_UNDEFINED;

	vect_add_elem(&vlb1, (Variable) e, VALUE_MONE);
	vect_add_elem(&vlb2, (Variable) e, VALUE_MONE);
	clb1 = contrainte_make(vlb1);
	clb2 = contrainte_make(vlb2);
	clb1->succ = clb2;
	tf = make_transformer(NIL,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb1)));
      }
      else {
	/* d is negative and even */
	Pvecteur vub = vect_new((Variable) e, VALUE_ONE);
	Pvecteur vlb = vect_new((Variable) e, VALUE_MONE);
	Pcontrainte clb = CONTRAINTE_UNDEFINED;
	Pcontrainte cub = CONTRAINTE_UNDEFINED;

	vect_add_elem(&vub, TCST, VALUE_MONE);
	clb = contrainte_make(vlb);
	cub = contrainte_make(vub);
	clb->succ = cub;
	tf = make_transformer(NIL,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
      }
    }
    else if(d<0) {
      /* d is negative, arg1 cannot be 0, expression value is -1, 0
	 or 1 */
      Pvecteur vub = vect_new((Variable) e, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
    else if(d==1) {
      Pvecteur v = vect_dup(normalized_linear(n1));

      vect_add_elem(&v, (Variable) e, VALUE_MONE);
      tf = make_transformer(NIL,
			    make_predicate(sc_make(contrainte_make(v),
						   CONTRAINTE_UNDEFINED)));
    }
  }
  else if(arg1_is_constant_p) {
    //int d = signed_integer_constant_expression_value(arg1);
    int d = (int) lb1;

    if(d==0||d==1) {
      /* 0 or 1 is assigned unless arg2 equals 0... which is neglected */
      Pvecteur v = vect_new((Variable) e, VALUE_ONE);

      vect_add_elem(&v, TCST, int_to_value(-d));
      tf = make_transformer(NIL,
			    make_predicate(sc_make(contrainte_make(v),
						   CONTRAINTE_UNDEFINED)));
    }
    else if(d > 1) { // e>=0, always

      tf = transformer_identity();

      /* e >= d^lb2 */
      if(lb2>=1) {
	Value elb2 = exponentiate((Value) d, (Value) lb2);
	tf =
	  transformer_add_inequality_with_integer_constraint(tf,
							     (Variable) e,
							     (int) elb2,
							     false);
      }

      /* e <= d^lb2 */
      if(ub2<INT_MAX) {
	// FI: let's hope no overflow is generated...
	Value eub2 = exponentiate((Value) d, (Value) ub2);
	tf =
	  transformer_add_inequality_with_integer_constraint(tf,
							     (Variable) e,
							     (int) eub2,
							     true);
      }


      /* the assigned value is positive */
      Pvecteur v1 = vect_new((Variable) e, VALUE_MONE);
      //Pcontrainte c1 = contrainte_make(v1);
      tf = transformer_inequality_add(tf, v1);

      if(normalized_linear_p(n2)) {
	Pvecteur v2 = vect_dup(normalized_linear(n2));
	//Pcontrainte c2 = CONTRAINTE_UNDEFINED;
	Pvecteur v3 = vect_multiply(vect_dup(normalized_linear(n2)), (Value) d);
	//Pcontrainte c3 = CONTRAINTE_UNDEFINED;

	vect_add_elem(&v2, TCST, VALUE_ONE);
	vect_add_elem(&v2, (Variable) e, VALUE_MONE);
	tf = transformer_inequality_add(tf, v2);
	// c2 = contrainte_make(v2);
	//contrainte_succ(c1) = c2;
	vect_add_elem(&v3, (Variable) e, VALUE_MONE);
	//c3 = contrainte_make(v3);
	//contrainte_succ(c2) = c3;
	tf = transformer_inequality_add(tf, v3);
      }

      //tf = make_transformer(NIL,
      //		    make_predicate(sc_make(CONTRAINTE_UNDEFINED, c1)));

    }
    else if(d == -1) {
      /* The assigned value is 1 or -1 */
      Pvecteur vub = vect_new((Variable) e, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
  }

  free_transformer(pre);

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

static transformer integer_minmax_to_transformer(entity v, /*value for minmax*/
						 list args,
						 transformer pre,
						 bool is_min,
						 bool is_internal)
{
  transformer tf = transformer_identity();
  list cexpr;
  intptr_t bound = is_min? LONG_MAX : LONG_MIN;

  pips_debug(8, "begin for %s %s\n",
	     is_min? "minimum" : "maximum",
	     is_internal? "intraprocedural" : "interprocedural");

  for(cexpr = args; !ENDP(cexpr); POP(cexpr)) {
    expression arg = EXPRESSION(CAR(cexpr));
    intptr_t lb, ub;
    if(precondition_minmax_of_expression(arg, pre, &lb, &ub)) {
      // FI: useless precision loss, anticipating
      // transformer_add_inequality_with+integer_constraint()
      bound = is_min? (MIN(bound, lb)) : (MAX(bound, ub));
    }
    else {
      bound = is_min? LONG_MIN : LONG_MAX;
    }
    entity tv = make_local_temporary_value_entity(entity_type(v));
    transformer etf = integer_expression_to_transformer(tv, arg, pre,
							is_internal);
    Pvecteur vineq = vect_new((Variable) tv, (Value) VALUE_ONE);

    vect_add_elem(&vineq, (Variable) v, VALUE_MONE);

    if(is_min) {
      vineq = vect_multiply(vineq, VALUE_MONE);
    }

    tf = transformer_inequality_add(tf, vineq);
    tf = transformer_safe_image_intersection(tf, etf);
    free_transformer(etf);
  }

  if(bound!=LONG_MIN && bound!=LONG_MAX) { // FI: this test does not
					   // seem to work as expected...
    int b = (int) bound;
    if((intptr_t) b == bound) { // FI: for safety...
      //tf = transformer_add_inequality_with_integer_constant(tf, v, bound, is_min);
      tf = transformer_add_inequality_with_integer_constraint(tf, v, b, !is_min);
    }
  }

  // FI: not strong enough to remove stupid redundancy. See
  // Semantics/minmax3
  // I do not dare add a stronger normalization because it could
  // impact the execution time too much...
  //tf = transformer_normalize(tf, 2);

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

static transformer min0_to_transformer(entity e,
				       list args,
				       transformer pre,
				       bool is_internal)
{
  return integer_minmax_to_transformer(e, args, pre, true, is_internal);
}

static transformer max0_to_transformer(entity e,
				       list args,
				       transformer pre,
				       bool is_internal)
{
  return integer_minmax_to_transformer(e, args, pre, false, is_internal);
}

/* Returns an undefined transformer in case of failure */
transformer assign_operation_to_transformer(entity val, // assumed to be a value, not to have values
					    expression lhs,
					    expression rhs,
					    transformer pre)
{
  transformer tf = transformer_undefined;

  pips_debug(8,"begin\n");

  if(expression_reference_p(lhs)) {
    entity e = reference_variable(syntax_reference(expression_syntax(lhs)));

    if(entity_has_values_p(e) /* && integer_scalar_entity_p(e) */) {
      entity ev = entity_to_new_value(e);
      //transformer teq = simple_equality_to_transformer(val, ev, true);
      tf = assigned_expression_to_transformer(ev, rhs, pre);
      if(!transformer_undefined_p(tf))
	tf = transformer_add_equality(tf, val, ev);
      //tf = transformer_combine(tf, teq);
      //free_transformer(teq);
    }
  }

  pips_debug(6,"return tf=%lx\n", (unsigned long)tf);
  ifdebug(6) (void) dump_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

/* */

static transformer points_to_unary_operation_to_transformer(entity e,
    entity op,
    expression e1,
    transformer pre,
    bool is_internal,
    bool is_pointer)
{
  pips_debug(8, "begin \n");
  transformer tf = transformer_undefined;

  if (current_statement_semantic_context_defined_p()) {
    statement curstat =  get_current_statement_semantic_context();
    points_to_graph ptg = get_points_to_graph_from_statement(curstat);

    list l = expression_to_points_to_sinks(e1, ptg);
    // compute the assignment for each cell pointed with the convex hull
    FOREACH(CELL, cp, l) {
      reference rrhs = cell_preference_p(cp)? preference_reference(cell_preference(cp)) : cell_reference(cp) ;
      entity rhs = reference_variable(rrhs);

      if (!is_pointer && entity_null_locations_p(rhs)) {
        if (gen_length(l) == 1) {
          tf = transformer_empty();
          pips_user_error("The pointer %s points to NULL\n", expression_to_string(e1));
        }
        else {
          pips_user_warning("The pointer %s can points to NULL\n", expression_to_string(e1));
        }
      } else if (entity_typed_nowhere_locations_p(rhs)) {
        if (gen_length(l) == 1) {
          tf = transformer_empty();
          pips_user_error("The pointer %s points to undefined/indeterminate (%s)\n", expression_to_string(e1), reference_to_string(rrhs));
        }
        else {
          pips_user_warning("The pointer %s can points to undefined/indeterminate (%s)\n", expression_to_string(e1), reference_to_string(rrhs));
        }
      }
      else {
        // check if rrhs is a constant path and can be analyze or not
        // (not really exact, see detail in analyzed_reference_p)
        if (analyzed_reference_p(rrhs)) {
          transformer rt = generic_reference_to_transformer(e, rrhs, pre, is_internal);
          // NL : It must have be a better way to do that but I don't know how
          if (transformer_undefined_p(tf))
            tf = rt;
          else {
            tf = transformer_convex_hull(tf, rt);
            free_transformer(rt);
          }
        }
      }
    }
  }

  if (transformer_undefined_p(tf)) {
    if(ENTITY_DEREFERENCING_P(op) || ENTITY_FIELD_P(op) || ENTITY_POINT_TO_P(op)) {
      pips_user_warning("activate TRANSFORMERS_INTER_FULL_WITH_POINTS_TO and setproperty SEMANTICS_ANALYZE_CONSTANT_PATH TRUE can maybe make better transformer\n");
    }
  }

  ifdebug (8) dump_transformer(tf);
  pips_debug(8, "end \n");
  return tf;
}

/* FI: this function is no longer useful (11 August 2013) */
static transformer 
integer_nullary_operation_to_transformer(
					 entity e __attribute__ ((unused)),
					 entity f __attribute__ ((unused)),
					 transformer pre __attribute__ ((unused)),
					 bool is_internal __attribute__ ((unused)))
{
  // eg : getchar and sizeof are nullary_operation
  transformer tf = transformer_undefined;
  /* rand() returns an integer between 0 and RAND_MAX. See
     Semantics/rand01.c

     How do you make sure that RAND_MAX is the same for the analyzed
     code and for the analyzer?
 */
  /* NL already parse by the function which call him (integer_call_expression_to_transformer)
  if (ENTITY_RAND_P(f)) {
    tf = transformer_add_inequality_with_integer_constraint(transformer_identity(),
							    e, 0, true);
  }
  */

  return tf;
}

static transformer integer_unary_operation_to_transformer(entity e,
							  entity op,
							  expression e1,
							  transformer pre,
							  bool is_internal)
{
  transformer tf = transformer_undefined;

  tf = generic_unary_operation_to_transformer(e, op, e1, pre, is_internal);

  if(transformer_undefined_p(tf)) {
    if(ENTITY_IABS_P(op)
        || ENTITY_C_ABS_P(op) || ENTITY_LABS_P(op) || ENTITY_LLABS_P(op)) {
      tf = iabs_to_transformer(e, e1, pre, is_internal);
    }
  }

  if(transformer_undefined_p(tf)) {
    tf = points_to_unary_operation_to_transformer(e, op, e1, pre, is_internal, false);
  }

  return tf;
}

static transformer integer_binary_operation_to_transformer(entity e,
							   entity op,
							   expression e1,
							   expression e2,
							   transformer pre,
							   bool is_internal)
{
  transformer tf = transformer_undefined;

  pips_debug(8, "Begin\n");

  if(ENTITY_PLUS_P(op) || ENTITY_MINUS_P(op)
     || ENTITY_PLUS_C_P(op) || ENTITY_MINUS_C_P(op)) {
    tf = addition_operation_to_transformer(e, e1, e2, pre,
					   ENTITY_PLUS_P(op) || ENTITY_PLUS_C_P(op),
					   is_internal);
  }
  if(ENTITY_PLUS_UPDATE_P(op) || ENTITY_MINUS_UPDATE_P(op)) {
    tf = update_operation_to_transformer(e, op, e1, e2, pre, is_internal);
  }
  else if(ENTITY_MULTIPLY_P(op)) {
    tf = integer_multiply_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_MODULO_P(op) || ENTITY_C_MODULO_P(op)) {
    tf = modulo_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_DIVIDE_P(op)) {
    tf = integer_divide_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_POWER_P(op)) {
    tf = integer_power_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_ASSIGN_P(op)) {
    tf = assign_operation_to_transformer(e, e1, e2, pre);
  }
  else if(ENTITY_LEFT_SHIFT_P(op)) {
    tf = integer_left_shift_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_RIGHT_SHIFT_P(op)) {
    tf = integer_right_shift_to_transformer(e, e1, e2, pre, is_internal);
  }

  pips_debug(8, "End with tf=%p\n", tf);

  return tf;
}

static transformer integer_call_expression_to_transformer(
    entity e,
    expression expr, /* needed to compute effects for user calls */
    transformer pre, /* Predicate on current store assumed not modified by
                        expr's side effects */
    bool is_internal)
{
  syntax sexpr = expression_syntax(expr);
  call c = syntax_call(sexpr);
  entity f = call_function(c);
  list args = call_arguments(c);
  transformer tf = transformer_undefined;
  int arity = gen_length(args);

  pips_debug(8, "Begin with precondition %p\n", pre);

  /* tests are organized to trap 0-ary user-defined functions as well as
     binary min and max */

  if(ENTITY_MIN0_P(f) || ENTITY_MIN_P(f)) {
    tf = min0_to_transformer(e, args, pre, is_internal);
  }
  else if(ENTITY_C_MIN_P(f)) {
    args=CDR(args);
    --arity;
    tf = min0_to_transformer(e,args,pre,is_internal);
  }
  else if(ENTITY_MAX0_P(f) || ENTITY_MAX_P(f)) {
    tf = max0_to_transformer(e, args, pre, is_internal);
  }
  else if(ENTITY_COMMA_P(f)) {
    // is_internal is dropped...
    tf = any_expressions_to_transformer(e, args, pre);
  }
  else if(ENTITY_CONDITIONAL_P(f)) {
    tf = any_conditional_to_transformer(e, args, pre);
  }
  else if(ENTITY_NOT_P(f)) {
    /* FI: should only happen for C code because int are often used
       as bool since there was no other way before C99. Quite time
       consuming here because it is applied to an int and not a
       bool argument. */
    tf = logical_not_to_transformer(e, args, pre);
  }
  else if(ENTITY_BITWISE_XOR_P(f)) {
  /* FI: it might be useful to handle here the bit wise XOR as in
     logical_binary_operation_to_transformer(). */
    tf = bitwise_xor_to_transformer(e, args, pre);
  }
  else if(ENTITY_RAND_P(f)) {
    tf = transformer_identity();
    tf = transformer_add_inequality_with_integer_constraint(tf, e, VALUE_ZERO, false);
  }
  else if(ENTITY_MODULO_P(f) || ENTITY_C_MODULO_P(f)) {
    expression arg1 = EXPRESSION(CAR(args));
    expression arg2 = EXPRESSION(CAR(CDR(args)));
    tf = modulo_to_transformer(e, arg1, arg2, pre, false);
  }
  else if(value_code_p(entity_initial(f))) {
    if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      tf = user_function_call_to_transformer(e, expr, pre);
    }
    else {
      /* No need to use effects here. It will be done at a higher level */
      ;
    }
  }
  else if(arity==0) {
    /* integer constant must have been detected earlier as linear (affine)
       expressions but there might be nullary intrinsic such as rand() */
    tf = integer_nullary_operation_to_transformer(e, f, pre, is_internal);
  }
  else if(arity==1) {
    expression e1 = EXPRESSION(CAR(args));
    tf = integer_unary_operation_to_transformer(e, f, e1, pre, is_internal);
  }
  else if(arity==2) {
    expression e1 = EXPRESSION(CAR(args));
    expression e2 = EXPRESSION(CAR(CDR(args)));
    tf = integer_binary_operation_to_transformer(e, f, e1, e2, pre, is_internal);
  }

  pips_debug(8, "End with tf=%p\n", tf);

  return tf;
}

transformer expression_effects_to_transformer(expression expr)
{
  //list el = expression_to_proper_effects(expr);
  list el = expression_to_proper_constant_path_effects(expr);
  transformer tf = effects_to_transformer(el);

  gen_full_free_list(el);
  return tf;
}

transformer
integer_expression_to_transformer(
    entity v,
    expression expr,
    transformer pre,
    bool is_internal) /* Do check wrt to value mappings... if you are
			 not dealing with interprocedural issues */
{
  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(expr);
  syntax sexpr = expression_syntax(expr);

  pips_debug(8, "begin with precondition %p for expression: ", pre);
  ifdebug(8) print_expression(expr);

  /* Assume: e is a value */

  if(normalized_linear_p(n)) {
    /* Is it really useful to keep using this function which does not
       take the precondition into acount? */
    if(transformer_undefined_p(pre)) {
      tf = simple_affine_to_transformer(v, (Pvecteur) normalized_linear(n), is_internal);
    }
    else {
      transformer tfl = simple_affine_to_transformer(v, (Pvecteur) normalized_linear(n), is_internal);
      if(transformer_undefined_p(tfl)) {
	tfl = expression_effects_to_transformer(expr);
      }

      tf = transformer_combine(copy_transformer(pre), tfl);
      free_transformer(tfl);
    }
  }
  else if(syntax_call_p(sexpr)) {
    tf = integer_call_expression_to_transformer(v, expr, pre, is_internal);
  }
  else if(syntax_reference_p(sexpr)) {
    //entity e = reference_variable(syntax_reference(sexpr));
    reference r = syntax_reference(sexpr);
    tf = generic_reference_to_transformer(v, r, pre, is_internal);
  }
  else if(syntax_cast_p(sexpr)) {
    cast c = syntax_cast(sexpr);
    type ct = cast_type(c);
    expression cexp = cast_expression(c);
    type cexpt = expression_to_type(cexp);
    if (type_equal_p(ct, cexpt))
      tf = integer_expression_to_transformer(v, cexp, pre, is_internal);
    else
      tf = transformer_undefined;
    free_type(cexpt);

  }
  else if(syntax_sizeofexpression_p(sexpr)) {
    if(get_bool_property("EVAL_SIZEOF")) {
      sizeofexpression soe = syntax_sizeofexpression(sexpr);
      value val = EvalSizeofexpression(soe);
      if(value_constant_p(val) && constant_int_p(value_constant(val))) {
	long long int i = constant_int(value_constant(val));
	tf = transformer_identity();
	tf = transformer_add_equality_with_integer_constant(tf, v, i);
      }
    }
    else
      tf = transformer_undefined;
  }
  else {
    /* vect_rm(ve); */
    tf = transformer_undefined;
  }

  pips_debug(8, "end with tf=%p\n", tf);

  return tf;
}

/* Always return a defined transformer, using effects in case a more precise analysis fails */
transformer safe_integer_expression_to_transformer(
    entity v,
    expression expr,
    transformer pre,
    bool is_internal) /* Do check wrt to value mappings... if you are
			 not dealing with interprocedural issues */
{
  transformer tf = integer_expression_to_transformer(v, expr, pre, is_internal);

  if(transformer_undefined_p(tf))
    tf = expression_effects_to_transformer(expr);

  return tf;
}

/* PROCESSING OF LOGICAL EXPRESSIONS */

transformer transformer_logical_inequalities_add(transformer tf, entity v)
{
  /* the values of v are between 0 and 1 */
  Pvecteur ineq1 = vect_new((Variable) v, VALUE_ONE);
  Pvecteur ineq2 = vect_new((Variable) v, VALUE_MONE);

  vect_add_elem(&ineq1, TCST, VALUE_MONE);

  tf = transformer_inequality_add(tf, ineq1);
  tf = transformer_inequality_add(tf, ineq2);

  return tf;
}

static transformer logical_constant_to_transformer(entity v,
						   entity f)
{
  transformer tf = transformer_undefined;
  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);
  Pcontrainte c;

  if(ENTITY_TRUE_P(f) || ENTITY_ONE_P(f)) {
    vect_add_elem(&eq, TCST , VALUE_MONE);
  }
  else if(ENTITY_FALSE_P(f) || ENTITY_ZERO_P(f)) {
    ;
  }
  else {
    pips_internal_error("Unknown logical constant %s",
	       entity_name(f));
  }
  c = contrainte_make(eq);
  tf = make_transformer(NIL,
			make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));

  return tf;
}

static transformer logical_unary_operation_to_transformer(entity v,
							  call c,
							  transformer pre,
							  bool is_internal)
{
  transformer tf = transformer_undefined;
  entity op = call_function(c);
  expression arg = EXPRESSION(CAR(call_arguments(c)));
  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);

  /* pips_assert("A unique argument", ENDP(CDR(call_arguments(c)))); */

  if(ENTITY_NOT_P(op)) {
    entity tmp = make_local_temporary_value_entity(entity_type(v));
    tf = logical_expression_to_transformer(tmp, arg, pre, is_internal);
    vect_add_elem(&eq, (Variable) tmp , VALUE_ONE);
    vect_add_elem(&eq, TCST , VALUE_MONE);
  }
  else {
//    pips_internal_error("Unknown logical constant %s",
//	       entity_name(op));
  }
  tf = transformer_equality_add(tf, eq);
  tf = transformer_logical_inequalities_add(tf, v);

  if(transformer_undefined_p(tf)) {
    pips_user_warning("Not Tested for logical\n");
    tf = points_to_unary_operation_to_transformer(v, op, arg, pre, is_internal, false);
  }

  return tf;
}

static transformer logical_binary_operation_to_transformer(entity v,
							   call c,
							   transformer pre,
							   bool is_internal)
{
  entity op = call_function(c);
  Pvecteur eq1 = VECTEUR_NUL;
  Pvecteur eq2 = VECTEUR_NUL;
  Pvecteur eq3 = VECTEUR_NUL;
  Pvecteur eq4 = VECTEUR_NUL;
  expression arg1 = EXPRESSION(CAR(call_arguments(c)));
  expression arg2 = EXPRESSION(CAR(CDR(call_arguments(c))));
  entity tmp1 = make_local_temporary_value_entity(entity_type(v));
  entity tmp2 = make_local_temporary_value_entity(entity_type(v));
  transformer tf1 = logical_expression_to_transformer(tmp1, arg1, pre, is_internal);
  transformer tf2 = logical_expression_to_transformer(tmp2, arg2, pre, is_internal);
  transformer tf = transformer_safe_intersection(tf1, tf2);

  ifdebug(9) {
    pips_debug(9, "Begin for value %s with subtransformers:\n",
	       entity_name(v));
    dump_transformer(tf1);
    dump_transformer(tf2);
  }

  free_transformer(tf1);
  free_transformer(tf2);

  ifdebug(9) {
    pips_debug(9, "Union of subtransformers:\n");
    dump_transformer(tf);
  }

  if(!transformer_undefined_p(tf)) {
    if(ENTITY_AND_P(op)) {
      /* v <= tmp1, v <= tmp2, v >= tmp1+tmp2-1 */
      eq1 = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&eq1, (Variable) tmp1, VALUE_MONE);
      eq2 = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&eq2, (Variable) tmp2, VALUE_MONE);
      eq3 = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&eq3, (Variable) tmp1, VALUE_ONE);
      vect_add_elem(&eq3, (Variable) tmp2, VALUE_ONE);
      vect_add_elem(&eq3, TCST, VALUE_MONE);
    }
    else if(ENTITY_OR_P(op)) {
      /* v >= tmp1, v>= tmp2, v <= tmp1+tmp2 */
      eq1 = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&eq1, (Variable) tmp1, VALUE_ONE);
      eq2 = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&eq2, (Variable) tmp2, VALUE_ONE);
      eq3 = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&eq3, (Variable) tmp1, VALUE_MONE);
      vect_add_elem(&eq3, (Variable) tmp2, VALUE_MONE);
    }
    else if(ENTITY_EQUAL_P(op)) {
      /* v >= 1-tmp1-tmp2, v >= tmp1+tmp2-1, i.e.
       * -v -tmp1 -tmp2 + 1 <= 0, -v+tmp1+tmp2-1<=0
       * v <= 1-tmp1+tmp2, v <= 1+tmp1-tmp2, i.e.
       * v + tmp1 - tmp2 - 1 <= 0, v - tmp1 + tmp2 - 1 <=0
       */
      eq1 = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&eq1, (Variable) tmp1, VALUE_MONE);
      vect_add_elem(&eq1, (Variable) tmp2, VALUE_MONE);
      vect_add_elem(&eq1, TCST , VALUE_ONE);

      eq2 = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&eq2, (Variable) tmp1, VALUE_ONE);
      vect_add_elem(&eq2, (Variable) tmp2, VALUE_ONE);
      vect_add_elem(&eq2, TCST , VALUE_MONE);

      eq3 = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&eq3, (Variable) tmp1, VALUE_ONE);
      vect_add_elem(&eq3, (Variable) tmp2, VALUE_MONE);
      vect_add_elem(&eq3, TCST , VALUE_MONE);

      eq4 = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&eq4, (Variable) tmp1, VALUE_MONE);
      vect_add_elem(&eq4, (Variable) tmp2, VALUE_ONE);
      vect_add_elem(&eq4, TCST , VALUE_MONE);
    }
    else if(ENTITY_NON_EQUAL_P(op) || ENTITY_BITWISE_XOR_P(op)) {
      /* v <= tmp1+tmp2, v <= 2-tmp1-tmp2, i.e.
       * v - tmp 1 - tmp 2 <= 0, v -tmp1-tmp2 -2 <= 0
       * v >= tmp1 - tmp2, v >= -tmp1+tmp2, i.e.
       * -v + tmp1 - tmp2 <=0, -v - tmp1 + tmp2 <=0
       */
      eq1 = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&eq1, (Variable) tmp1, VALUE_MONE);
      vect_add_elem(&eq1, (Variable) tmp2, VALUE_MONE);

      eq2 = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&eq2, (Variable) tmp1, VALUE_ONE);
      vect_add_elem(&eq2, (Variable) tmp2, VALUE_ONE);
      vect_add_elem(&eq2, TCST , VALUE_MONE+VALUE_MONE);

      eq3 = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&eq3, (Variable) tmp1, VALUE_ONE);
      vect_add_elem(&eq3, (Variable) tmp2, VALUE_MONE);

      eq4 = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&eq4, (Variable) tmp1, VALUE_MONE);
      vect_add_elem(&eq4, (Variable) tmp2, VALUE_ONE);

    }
    else {
      pips_internal_error("Unexpected binary logical operator %s",
			  entity_name(op));
    }
    tf = transformer_inequality_add(tf, eq1);
    tf = transformer_inequality_add(tf, eq2);
    if(!VECTEUR_NUL_P(eq3)) {
      if(!VECTEUR_NUL_P(eq4)) {
	tf = transformer_inequality_add(tf, eq4);
      }
      tf = transformer_inequality_add(tf, eq3);
    }
    tf = transformer_logical_inequalities_add(tf, v);
  }

  ifdebug(9) {
    pips_debug(9, "End with transformer:\n");
    dump_transformer(tf);
  }

  return tf;
}

static transformer logical_binary_function_to_transformer(entity v,
							  call c,
							  transformer pre,
							  bool is_internal)
{
  transformer tf = transformer_undefined;
  entity op = call_function(c);

  if(ENTITY_AND_P(op)||ENTITY_OR_P(op)||ENTITY_EQUAL_P(op)
     ||ENTITY_NON_EQUAL_P(op)||ENTITY_BITWISE_XOR_P(op)
     ||ENTITY_BITWISE_AND_P(op)||ENTITY_BITWISE_OR_P(op)) {
    tf = logical_binary_operation_to_transformer(v, c, pre, is_internal);
  }
  else if(ENTITY_ASSIGN_P(op)) {
    /* FI: slight signature mismatch: no need to propagate the call
       once you know it is a binary operator... */
    list args = call_arguments(c);
    expression lhs = EXPRESSION(CAR(args));
    expression rhs = EXPRESSION(CAR(CDR(args)));
    tf = assign_operation_to_transformer(v, lhs, rhs, pre);
  }
  else if(ENTITY_RELATIONAL_OPERATOR_P(op)) {
    expression expr1 = EXPRESSION(CAR(call_arguments(c)));
    expression expr2 = EXPRESSION(CAR(CDR(call_arguments(c))));
    basic b1 = basic_of_expression(expr1);
    basic b2 = basic_of_expression(expr2);

    if(basic_logical_p(b1) && basic_logical_p(b2)) {
      tf = logical_binary_operation_to_transformer(v, c, pre, is_internal);
    }
    else if(!transformer_undefined_p(pre)) {
      /* Non internal overloaded functions such as EQ, NEQ, GE, GT, LE, LT,... */
      //entity tmp = make_local_temporary_value_entity(entity_type(v));
      transformer tfe = transformer_add_any_relation_information(copy_transformer(pre),
								 op,
								 expr1,
								 expr2,
								 pre,
								 true,
								 true);

      // The normalization of tfe and tfne below is key because
      // transformer_empty_p() is quite weak (but fast) [FI: transformer_is_empty_p]
      tfe = transformer_normalize(tfe, 2);
      /* if the transformer is not feasible, return false */
      if(transformer_empty_p(tfe)) {
	Pvecteur eq = vect_new((Variable) v, VALUE_ONE);
	Pcontrainte c;

	c = contrainte_make(eq);
	tf = make_transformer(NIL,
			      make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
      }
      else {
	transformer tfne = transformer_add_any_relation_information(copy_transformer(pre),
								   op,
								   expr1,
								   expr2,
								   pre,
								   false,
								   true);

	tfne = transformer_normalize(tfne, 2);
	/* if the transformer is not feasible, return true */
	if(transformer_empty_p(tfne)) {
	  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);
	  Pcontrainte c;

	  vect_add_elem(&eq, TCST, VALUE_MONE);
	  c = contrainte_make(eq);
	  tf = make_transformer(NIL,
				make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
	}
	free_transformer(tfne);
      }
      free_transformer(tfe);
    }
    else {
      ;
    }

    free_basic(b1);
    free_basic(b2);
  }
  else {
    /* general case due to C convention about integers equal to 0 or
     *  not.
     *
     * if v is in [0..1], v can be used as such.
     * else if v>0, return 1
     * else if v<0, return 1
     * else 0<=v<=1
     */
    expression e1 = EXPRESSION(CAR(call_arguments(c)));
    expression e2 = EXPRESSION(CAR(CDR(call_arguments(c))));
    tf = integer_binary_operation_to_transformer(v, op, e1, e2,
						 pre, is_internal);
    if(!transformer_undefined_p(tf)) {
      // FI: should we restrict us to the range of tf?
      intptr_t vmin, vmax;
      bool success = precondition_minmax_of_value(v, tf, &vmin, &vmax);
      if(success) {
	if(vmin>=0 && vmax<=1)
	  ; // tf can be returned as such
	else if(vmin>=1 || vmax<=-1) {
	  // Must return v==1
	  free_transformer(tf);
	  tf = transformer_identity();
	  tf = transformer_add_equality_with_integer_constant(tf, v, 1);
	}
      }
    }

    if(transformer_undefined_p(tf)) {
      // Must return 0<=v<=1
      tf = transformer_identity();
      tf = transformer_add_inequality_with_integer_constraint(tf, v, 1, true);
      tf = transformer_add_inequality_with_integer_constraint(tf, v, 0, false);
    }
  }

  return tf;
}

/* v is assumed to be a temporary value and r a logical program variable */
static transformer logical_reference_to_transformer(entity v,
						    entity r,
						    bool is_internal)
{
  transformer tf = transformer_undefined;

  if(!is_internal || entity_has_values_p(r)) {
    entity r_new = is_internal? entity_to_new_value(r) : external_entity_to_new_value(r);


    tf = simple_equality_to_transformer(v, r_new, false);
    tf = transformer_logical_inequalities_add(tf, r_new);
  }

  return tf;
}

transformer logical_intrinsic_to_transformer(entity v,
					     call c,
					     transformer pre,
					     bool is_internal)
{
  transformer tf = transformer_undefined;

  switch(gen_length(call_arguments(c))) {
  case 0:
    tf = logical_constant_to_transformer(v, call_function(c));
    break;
  case 1:
    tf = logical_unary_operation_to_transformer(v, c, pre, is_internal);
    break;
  case 2:
    tf = logical_binary_function_to_transformer(v, c, pre, is_internal);
    break;
  default:
    pips_internal_error("Too many logical arguments, %d, for operator %s",
			gen_length(call_arguments(c)),
			entity_name(call_function(c)));
  }
  return tf;
}

/* Could be used to compute preconditions too. v is assumed to be a new
   value or a temporary value. */
transformer logical_expression_to_transformer(entity v,
					      expression rhs,
					      transformer pre,
					      bool is_internal)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

  switch(syntax_tag(srhs)) {
  case is_syntax_call: {
    call c = syntax_call(srhs);
    entity f = call_function(c);
    if(intrinsic_entity_p(f))
      tf = logical_intrinsic_to_transformer(v, c, pre, is_internal);
    else if(call_constant_p(c)) {
      entity cf = call_function(c);
      int i;
      if(logical_constant_p(cf)) {
	  /* Is likely to be handled as a nullary intrinsics */
	  tf = logical_intrinsic_to_transformer(v, c, pre, is_internal);
	}
      else if(integer_constant_p(cf, &i)) {
	if(i!=0)
	  i = 1;
	tf = transformer_identity();
	tf = transformer_add_equality_with_integer_constant(tf, v, i);
      }
      else
	pips_internal_error("Unexpected case.\n");
    }
    else {
      // call to a C or Fortran bool function
      //list ef = expression_to_proper_effects(rhs);
      tf = user_function_call_to_transformer(v, rhs, pre);
    }
    break;
  }
  case is_syntax_reference: {
    // FI: information in precondition pre is lost here
    transformer rtf = logical_reference_to_transformer(v, reference_variable(syntax_reference(srhs)),
					  is_internal);
    if(!transformer_undefined_p(rtf) && !transformer_undefined_p(pre)) {
      // FI: let us assume that pre is a range, not a transformer
      tf = transformer_intersection(rtf, pre);
    }
    break;
  }
  case is_syntax_range:
    pips_internal_error("Unexpected tag %d", syntax_tag(srhs));
    break;
  default:
    pips_internal_error("Illegal tag %d", syntax_tag(srhs));
  }
  return tf;
}

transformer string_expression_to_transformer(entity v,
					     expression rhs)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

  switch(syntax_tag(srhs)) {
  case is_syntax_call:
    switch(gen_length(call_arguments(syntax_call(srhs)))) {
    case 0:
      /* tf = constant_to_transformer(v, call_function(syntax_call(srhs))); */
      tf = constant_to_transformer(v, rhs);
      break;
    case 1:
      break;
    case 2:
      break;
    case 3:
      break;
    default:
      pips_internal_error("Too many arguments, %d, for operator %s",
			  gen_length(call_arguments(syntax_call(srhs))),
			  entity_name(call_function(syntax_call(srhs))));
    }
    break;
  case is_syntax_reference:
    {
      entity r = reference_variable(syntax_reference(srhs));

      if(entity_has_values_p(r)) {
	basic bv = variable_basic(type_variable(entity_type(v)));
	basic br = variable_basic(type_variable(entity_type(r)));

	if(basic_string_p(bv) && basic_string_p(br)) {
	  int llhs = string_type_size(bv);
	  int lrhs = string_type_size(br);

	  if(llhs == -1 || llhs >= lrhs) {
	    entity r_new = entity_to_new_value(r);
	    tf =  simple_equality_to_transformer(v, r_new, false);
	  }
	}
	else {
	  pips_user_error("Incompatible type in CHARACTER assignment: %s\n",
			  basic_to_string(br));
	}
      }
    break;
    }
  case is_syntax_range:
    pips_internal_error("Unexpected tag %d", syntax_tag(srhs));
    break;
  default:
    pips_internal_error("Illegal tag %d", syntax_tag(srhs));
  }

  return tf;
}

static transformer 
float_unary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  tf = generic_unary_operation_to_transformer(e, op, e1, pre, is_internal);

  if(transformer_undefined_p(tf)) {
    tf = points_to_unary_operation_to_transformer(e, op, e1, pre, is_internal, false);
  }

  return tf;
}

static transformer 
float_binary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    expression e2,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  if(ENTITY_PLUS_P(op) || ENTITY_MINUS_P(op)) {
    tf = addition_operation_to_transformer(e, e1, e2, pre, ENTITY_PLUS_P(op), is_internal);
  }
  else if(ENTITY_ASSIGN_P(op)) {
    tf = assign_operation_to_transformer(e, e1, e2, pre);
  }

  return tf;
}

static transformer 
float_call_expression_to_transformer(
    entity v,
    expression expr, /* needed to compute effects for user calls */
    transformer pre,
    bool is_internal)
{
  syntax sexpr = expression_syntax(expr);
  call c = syntax_call(sexpr);
  entity op = call_function(c);
  list args = call_arguments(c);
  transformer tf = transformer_undefined;
  int arity = gen_length(args);

  if(ENTITY_AMIN1_P(op)||ENTITY_DMIN1_P(op)||ENTITY_MIN_P(op)) {
    tf = generic_minmax_to_transformer(v, args, pre, true, is_internal);
  }
  if(ENTITY_C_MIN_P(op)) {
    args=CDR(args);
    --arity;
    tf = generic_minmax_to_transformer(v, args, pre, true, is_internal);
  }
  else if(ENTITY_AMAX1_P(op)||ENTITY_DMAX1_P(op)||ENTITY_MAX_P(op)) {
    tf = generic_minmax_to_transformer(v, args, pre, false, is_internal);
  }
  else if(value_code_p(entity_initial(op))
	  && get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
    /* Just in case some integer variables were modified by the function
       call, e.g. the count of its dynamic calls. */
    tf = user_function_call_to_transformer(v, expr, pre);
  }
  else {
    switch(arity) {
    case 0:
      /* tf = constant_to_transformer(v, call_function(syntax_call(srhs))); */
      tf = constant_to_transformer(v, expr);
      break;
    case 1:
      {
	expression e1 = EXPRESSION(CAR(args));
	tf = float_unary_operation_to_transformer(v, op, e1, pre, is_internal);
	break;
      }
    case 2:
      {
	expression e1 = EXPRESSION(CAR(args));
	expression e2 = EXPRESSION(CAR(CDR(args)));
	tf = float_binary_operation_to_transformer(v, op, e1, e2, pre, is_internal);
	break;
      }
    default:
      /* min, max,... */
      pips_user_warning("Operator %s not analyzed\n", entity_name(op));
    }
  }

  return tf;
}

transformer float_expression_to_transformer(entity v,
					    expression rhs,
					    transformer pre,
					    bool is_internal)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

  /* pre can be used if some integer variables with constant values have
     to be promoted to float. */

  /* pips_assert("Precondition is undefined", transformer_undefined_p(pre)); */

  switch(syntax_tag(srhs)) {
  case is_syntax_call:
    tf = float_call_expression_to_transformer(v, rhs, pre, is_internal);
    break;
  case is_syntax_reference:
    {
      //entity e = reference_variable(syntax_reference(srhs));
      // FI: some filtering needed if integer expressions should not
      // be exploited...
      reference r = syntax_reference(srhs);
      tf = generic_reference_to_transformer(v, r, pre, is_internal);
      break;
    }
  case is_syntax_range:
    pips_internal_error("Unexpected tag %d", syntax_tag(srhs));
    break;
  default:
    pips_internal_error("Illegal tag %d", syntax_tag(srhs));
  }

  return tf;
}

/* POINTER EXPRESSION */

/**
 * Maybe move in ri-util/expression.c or in effect-util/pointer_values.c
 * \details         search in an arithmetic pointer expression
 *                  if the pointer NULL is present
 * \param expr      arithmetic expression to scan
 * \return          true if expr contain NULL else false
 */
static bool have_null_value_in_pointer_expression_p(expression expr) {
  syntax sexpr = expression_syntax(expr);

  switch (syntax_tag(sexpr)) {
    case is_syntax_reference:
    {
      reference rexpr = syntax_reference(sexpr);
      entity var = reference_variable(rexpr);

      type bctvar = compute_basic_concrete_type(entity_type(var));
      basic b = variable_basic(type_variable(bctvar));

      if (basic_pointer_p(b)) {
        pips_user_warning("TODO : Need to check if %s is indirectly NULL\n", expression_to_string(expr));
        // TODO : function that check if we have an expression/entity e=NULL in the transformer t
        //        have to make projection of e and NULL (because can have  : e=e1, e1=NULL)
        //        is_entity/expression_equal_null_in_transformer(var/expr, pre)
        // Maybe we can use the info from points-to, we can want to analyze pointer without graph points-to, why?
        if (null_pointer_value_entity_p(var) || expression_null_p(expr)) //this test can be not sufficient
          return true;
        else
          return false;
      }
      else
        return false;
      break;
    }
    case is_syntax_call:
    {
      call cexpr = syntax_call(sexpr);
      entity func = call_function(cexpr);

      if (ENTITY_PLUS_P(func) || ENTITY_PLUS_C_P(func) ||
          ENTITY_MINUS_P(func) || ENTITY_MINUS_C_P(func)) {
        list args = call_arguments(cexpr);
        expression e1 = EXPRESSION(CAR(args));
        expression e2 = EXPRESSION(CAR(CDR(args)));
        pips_assert("2 args for arithmetic", CDR(CDR(args))==NIL);

        return have_null_value_in_pointer_expression_p(e1) || have_null_value_in_pointer_expression_p(e2);
      }
      else if (expression_constant_p(expr)) {
        return false;
      }
      else if(ENTITY_MULTIPLY_P(func)) {
        return false;
      }
      else {
        //this case normally never happen
        pips_internal_error("unexpected function call : %s\n", entity_name(func));
      }
      break;
    }
    default:
      //this case normally never happen
      pips_internal_error("unexpected syntaxe tag : %i\n", syntax_tag(sexpr));
  }

  //this case normally never happen
  pips_internal_error("error with expression : %s, syntax_tag : %i \n", expression_to_string(expr), syntax_tag(sexpr));
  return false;
}

/**
 * Maybe move in ri-util/expression.c or in effect-util/pointer_values.c
 * \details         search in an arithmetic pointer expression
 *                  if the pointer NULL is present
 * \param expr      arithmetic expression to scan
 * \return          true if expr contain NULL else false
 */
static bool have_sizeof_value_in_multiply_pointer_expression_p(expression expr) {
  syntax sexpr = expression_syntax(expr);

  switch (syntax_tag(sexpr)) {
    case is_syntax_reference:
    {
      reference rexpr = syntax_reference(sexpr);
      entity var = reference_variable(rexpr);

      if (sizeof_value_entity_p(var)) {
        return true;
      }
      else
        return false;
      break;
    }
    case is_syntax_call:
    {
      call cexpr = syntax_call(sexpr);
      entity func = call_function(cexpr);

      if (ENTITY_MULTIPLY_P(func)) {
        list args = call_arguments(cexpr);
        expression e1 = EXPRESSION(CAR(args));
        expression e2 = EXPRESSION(CAR(CDR(args)));
        pips_assert("2 args for arithmetic", CDR(CDR(args))==NIL);

        return have_sizeof_value_in_multiply_pointer_expression_p(e1) || have_sizeof_value_in_multiply_pointer_expression_p(e2);
      }
      break;
    }
    default:
      return false;
  }

  //this case normally never happen
  pips_internal_error("error with expression : %s, syntax_tag : %i \n", expression_to_string(expr), syntax_tag(sexpr));
  return false;
}
/*
static expression pointer_expression_to_pointer_expression_with_sizeof(expression exp) {
  expression result;


  return result;
}
*/
static transformer pointer_unary_operation_to_transformer(
    entity e,       //entity to be affected (value not variable)
    entity op,      //"Name" of the call function
    expression e1,  //arg of the call function
    transformer pre,/* Predicate on current store assumed not modified by
                        expr's side effects */
    bool is_internal)
{
  transformer tf = transformer_undefined;

  //tf = generic_unary_operation_to_transformer(e, op, e1, pre, is_internal);
  // NL: can't call generic_unary_operation_to_transformer because of the case ABS
  //     it's a copy of generic_unary_operation_to_transformer without the case ABS
  //     add the case ADDRESS_OF
  if(ENTITY_UNARY_MINUS_P(op)) {
    tf = unary_minus_operation_to_transformer(e, e1, pre, is_internal);
  }
  else if(ENTITY_POST_INCREMENT_P(op) || ENTITY_POST_DECREMENT_P(op)
            || ENTITY_PRE_INCREMENT_P(op) || ENTITY_PRE_DECREMENT_P(op)) {
    syntax s1 = expression_syntax(e1);
    entity v1 = reference_variable(syntax_reference(s1));

    if(entity_has_values_p(v1)) {
      transformer tfu = any_basic_update_operation_to_transformer(e, v1, op);
      if(transformer_undefined_p(pre)) {
          tf = tfu;
      }
      else {
          /* FI: Oops, the precondition is lost here! See any_basic_update_to_transformer() */
          tf = transformer_combine(transformer_range(pre), tfu);
          free_transformer(tfu);
      }
    }
  }
  else if (ENTITY_ADDRESS_OF_P(op)) {
    syntax s1 = expression_syntax(e1);

    if (syntax_reference_p(s1)) {
      reference r1 = syntax_reference(s1);
      // entity v1 = reference_variable(r1);

      // create an entity for the address_of
      add_address_of_value(r1, entity_type(e));
      // get the address_of entity
      entity a1 = reference_to_address_of_value(r1);

      // make the transformer for the equality (take from generic_reference_to_transformer)
      transformer tfr =  simple_equality_to_transformer(e, a1, false);
      if(!transformer_undefined_p(pre)) {
        // FI: assume pre is a range
        tf = transformer_intersection(tfr, pre);
      }
      free_transformer(tfr);
    }
    else if (syntax_call_p(s1)) {
      call c1 = syntax_call(s1);
      entity f1 = call_function(c1);
      // list args = call_arguments(c1);

      if (ENTITY_FIELD_P(f1)||ENTITY_POINT_TO_P(f1)) {
        pips_internal_error("ADDRESS_OF for Struct : Not Implemented yet\n");
      }
      else {
      //Normally this case never appear
        pips_internal_error("illegal expression_syntax, syntax_call_function\n syntax_tag : %i\n call_function : %s\n",
            syntax_tag(s1), entity_name(f1));
      }
    }
    else {
    //Normally this case never appear
      pips_internal_error("illegal expression_syntax, syntax_tag : %i",
          syntax_tag(s1));
    }
  }

  if(transformer_undefined_p(tf)) {
    tf = points_to_unary_operation_to_transformer(e, op, e1, pre, is_internal, true);
  }

  return tf;
}

static transformer pointer_binary_operation_to_transformer(
    entity e,
    expression expr,
    transformer pre,
    bool is_internal)
{
  syntax sexpr = expression_syntax(expr);
  call c = syntax_call(sexpr);
  entity op = call_function(c);
  list args = call_arguments(c);
  expression e1 = EXPRESSION(CAR(args));
  expression e2 = EXPRESSION(CAR(CDR(args)));
  transformer tf = transformer_undefined;

  pips_debug(8, "Begin\n");

  if(ENTITY_PLUS_P(op) || ENTITY_MINUS_P(op)) {
    // can occur for arithmetic for pointer
    // e1+e2 or e1-e2, e1 and e2 not pointer
    tf = addition_operation_to_transformer(e, e1, e2, pre,
        ENTITY_PLUS_P(op),
        is_internal);
  }
  if(ENTITY_PLUS_C_P(op)) {
    // can occur for arithmetic for pointer
    // e1+e2, e1 or e2 pointer
    if (have_null_value_in_pointer_expression_p(expr)) {
      tf = transformer_empty();
      pips_user_error("Try to make arithmetic with NULL pointer : %s\n", expression_to_string(expr));
    }
    else {
      tf = addition_operation_to_transformer(e, e1, e2, pre,
          ENTITY_PLUS_C_P(op),
          is_internal);
    }
  }
  if(ENTITY_MINUS_C_P(op)) {
    // can occur for arithmetic for pointer
    // e1+e2, e1 or/and e2 pointer
    if (have_null_value_in_pointer_expression_p(expr)) {
      tf = transformer_empty();
      pips_user_error("Try to make arithmetic with NULL pointer : %s\n", expression_to_string(expr));
    }
    else {
      tf = addition_operation_to_transformer(e, e1, e2, pre,
          ENTITY_PLUS_C_P(op),
          is_internal);
    }
  }
  else if(ENTITY_MULTIPLY_P(op)) {
    // can occur for arithmetic for pointer
    // already multiply by the sizeof
    pips_debug(9, "ENTITY_MULTIPLY_P\n");
    if (have_sizeof_value_in_multiply_pointer_expression_p(expr)) {
      tf = integer_multiply_to_transformer(e, e1, e2, pre, is_internal);
      pips_user_error("can't compute when sizeof already present in arithmetic yet : %s\n", expression_to_string(expr));
    }
    else {
      // compute the multiply with sizeof
      tf = expression_multiply_sizeof_to_transformer(e, expr, pre, is_internal);
    }
  }
  else if(ENTITY_PLUS_UPDATE_P(op) || ENTITY_MINUS_UPDATE_P(op)) {
    pips_user_error("UPDATE for rhs pointer not done yet : %s\n", expression_to_string(expr));
    //tf = update_operation_to_transformer(e, op, e1, e2, pre, is_internal);
  }
  else if(ENTITY_ASSIGN_P(op)) {
    pips_user_warning("ASSIGN for rhs pointer not tested : %s\n", expression_to_string(expr));
    // Maybe need to be redesign like any_assign_to_transformer or assign_rhs_to_reflhs_to_transformer
    tf = assign_operation_to_transformer(e, e1, e2, pre);
  }
  else {
    pips_user_warning("Operator %s not analyzed\n", entity_name(op));
  }

  pips_debug(8, "End with tf=%p\n", tf);

  return tf;
}

static transformer pointer_call_expression_to_transformer(
    entity e,        // value of the expression
    expression expr, /* needed to compute effects for user calls */
    transformer pre, /* Predicate on current store assumed not modified by
                        expr's side effects */
    bool is_internal)
{
  syntax sexpr = expression_syntax(expr);
  call c = syntax_call(sexpr);
  entity f = call_function(c);
  list args = call_arguments(c);
  transformer tf = transformer_undefined;
  int arity = gen_length(args);

  pips_debug(8, "Begin with precondition %p\n", pre);
  /* tests are organized to trap 0-ary user-defined functions as well as
     binary min and max */

  if(ENTITY_MIN0_P(f) || ENTITY_MIN_P(f)) {
    tf = min0_to_transformer(e, args, pre, is_internal);
  }
  else if(ENTITY_C_MIN_P(f)) {
    args=CDR(args);
    --arity;
    tf = min0_to_transformer(e,args,pre,is_internal);
  }
  else if(ENTITY_MAX0_P(f) || ENTITY_MAX_P(f)) {
    tf = max0_to_transformer(e, args, pre, is_internal);
  }
  else if(ENTITY_COMMA_P(f)) {
    // is_internal is dropped...
    tf = any_expressions_to_transformer(e, args, pre);
  }
  else if(ENTITY_CONDITIONAL_P(f)) {
    tf = any_conditional_to_transformer(e, args, pre);
  }
  else if(value_code_p(entity_initial(f)) &&
          get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      tf = user_function_call_to_transformer(e, expr, pre);
  }
  else {
    switch(arity) {
    case 0:
    {
      // expr is a constant
      //tf = constant_to_transformer(e, expr);
      // occur for arithmetic for pointer
      // multiply the constant with he sizeof lhs
      // compute the multiply with sizeof
      tf = expression_multiply_sizeof_to_transformer(e, expr, pre, is_internal);

      break;
    }
    case 1:
    {
      expression e1 = EXPRESSION(CAR(args));
      tf = pointer_unary_operation_to_transformer(e, f, e1, pre, is_internal);
      break;
    }
    case 2:
    {
      //expression e1 = EXPRESSION(CAR(args));
      //expression e2 = EXPRESSION(CAR(CDR(args)));
      tf = pointer_binary_operation_to_transformer(e, expr, pre, is_internal);
      break;
    }
    default:
      pips_user_warning("Operator %s not analyzed\n", entity_name(f));
    }
  }

  pips_debug(8, "End with tf=%p\n", tf);

  return tf;
}

/**
 * \details         This function may return an undefined transformers if it fails to
 *                  capture the semantics of expr in the polyhedral framework.
 *                  cf any_expression_to_transformer which call it
 * \param v         value of the expression
 * \param expr      pointer expression
 * \param pre       transformer, no side effect
 * \param is_internal ?? use in generic_reference_to_transformer
 * \return          transformer with the pointer expression
 *                  or transformer_undefined if failed
 */
transformer pointer_expression_to_transformer(entity v,  // value of the expression
                                            expression expr,
                                            transformer pre,
                                            bool is_internal)
{
  transformer tf = transformer_undefined;
  syntax sexpr = expression_syntax(expr);

  ifdebug(8) {
    pips_debug(8, "begin with precondition %p for expression: ", pre);
    print_expression(expr);
  }

  /* Assume: v is a value */
//  normalized n = NORMALIZE_EXPRESSION(expr);
//  if(normalized_linear_p(n))
//  {
//    ifdebug(8) print_syntax(sexpr);
//    pips_debug(8, "\nsyntaxe type : %i \n", syntax_tag(sexpr));
//    //print_transformer(pre);
//    if (have_null_value_in_pointer_expression_p(expr)) {
//      pips_user_error("Try to make arithmetic with NULL pointer : %s", expression_to_string(expr));
//    }
//    // Is it really useful to keep using this function which does not
//    //   take the precondition into acount?
//    else if(transformer_undefined_p(pre)) {
//      tf = simple_affine_to_transformer(v, (Pvecteur) normalized_linear(n), is_internal);
//    }
//    else {
//      transformer tfl = simple_affine_to_transformer(v, (Pvecteur) normalized_linear(n), is_internal);
//      if(transformer_undefined_p(tfl)) {
//        tfl = expression_effects_to_transformer(expr);
//      }
//
//      tf = transformer_combine(copy_transformer(pre), tfl);
//      free_transformer(tfl);
//    }
//  }
//  else
  {
    switch (syntax_tag(sexpr)) {
    case is_syntax_call:
    {
      tf = transformer_undefined;
      tf = pointer_call_expression_to_transformer(v, expr, pre, is_internal);
      break;
    }
    case is_syntax_reference:
    {
      // pips_user_warning("reference for pointer analysis not be tested\n");
      // can occur for arithmetic for pointer
      reference rexpr = syntax_reference(sexpr);

      entity rvexpr = reference_variable(rexpr);
      type texpr = entity_type(rvexpr);
      type bctexpr = compute_basic_concrete_type(texpr);
      if(type_variable_p(bctexpr)) {
        variable vexpr = type_variable(bctexpr);
        if (!volatile_variable_p(vexpr)
            && ENDP(variable_dimensions(vexpr))
            && !basic_pointer_p(variable_basic(vexpr))) {
          // compute the multiply with sizeof
          tf = expression_multiply_sizeof_to_transformer(v, expr, pre, is_internal);
        }
        else
          tf = generic_reference_to_transformer(v, rexpr, pre, is_internal);
      }
      break;
    }
    case is_syntax_cast:
    {
      pips_user_warning("cast for pointer analysis not be tested (except for NULL pointer)\n");
      cast c = syntax_cast(sexpr);
      type ct = cast_type(c);
      expression cexp = cast_expression(c);
      type cexpt = expression_to_type(cexp);

      if (expression_null_p(expr)) {
        // make the transformer for the equality (take from generic_reference_to_transformer)
        transformer tfr =  simple_equality_to_transformer(v, null_pointer_value_entity(), false);
        if(!transformer_undefined_p(pre)) {
          // FI: assume pre is a range
          tf = transformer_intersection(tfr, pre);
        }
        free_transformer(tfr);
      } else if (type_equal_p(ct, cexpt))
        tf = pointer_expression_to_transformer(v, cexp, pre, is_internal);
      else
        tf = transformer_undefined;
        //pips_internal_error("Illegal tag %d", syntax_tag(sexpr));
      free_type(cexpt);
      break;
    }
    default:
      tf = transformer_undefined;
      //pips_internal_error("Illegal tag %d", syntax_tag(sexpr));
    }
  }

  pips_debug(8, "end with tf=%p\n", tf);

  return tf;
}



/* Assimilate enum and logical to int (used when they appear in logical
 * operations)
 *
 * FI: Might be useful quite often in semantics... but forces the
 * programmer to use tags of basic instead of basic variables.
 */
static int semantics_basic_tag(basic b)
{
  int t = basic_tag(b);

  if(t==is_basic_derived) {
    entity d = basic_derived(b);
    type dt = entity_type(d);
    if(type_enum_p(dt))
      t = is_basic_int;
  }
  else if(t==is_basic_logical)
    t = is_basic_int;

  return t;
}

transformer transformer_add_any_relation_information(
	transformer pre, /* precondition */
	entity op,
	expression e1,
	expression e2,
	transformer context,
	bool veracity,   /* the relation is true or not */
	bool upwards)    /* compute transformer or precondition */
{
  basic b1 = basic_of_expression(e1);
  basic b2 = basic_of_expression(e2);
  int t1 = semantics_basic_tag(b1);
  int t2 = semantics_basic_tag(b2);

  pips_debug(8, "begin %s with pre=%p\n",
	     upwards? "upwards" : "downwards", pre);

  /* This is not satisfactory: when going upwards you might nevertheless
     benefit from some precondition information. But you would then need
     to pass two transformers as argument: a context transformer and a
     to-be-modified transformer */

  /* context = upwards? transformer_undefined : pre; */
  /* context = transformer_range(pre); */

  if( (t1==t2)
      || (t1==is_basic_int && t2==is_basic_string)
      ||  (t2==is_basic_int && t1==is_basic_string) ) {

    switch(t1) {
    case is_basic_logical:
      /* Logical are represented by integer values*/
    case is_basic_float:
      /* PIPS does not represent negative constants: call to unary_minus */
    case is_basic_complex:
      /* PIPS does not represent complex constants: call to CMPLX */
    case is_basic_string:
      /* Only constant string are processed */
    case is_basic_pointer:
      /* Pointer analysis */
    case is_basic_int:
      {
	/* This is not correct if side effects occur in e1 or e2 */
	/* This is very complicated to add constraints from tf1, tf2 and
           rel in pre! */
	entity tmp1 = make_local_temporary_value_entity_with_basic(b1);
	entity tmp2 = make_local_temporary_value_entity_with_basic(b2);
	transformer tf1 = safe_any_expression_to_transformer(tmp1, e1, context, true);
	/* tf1 may modify the initial context. See w10.f */
	transformer pcontext = transformer_undefined_p(context)? transformer_identity()
	  : transformer_apply(tf1, context);
	transformer ncontext = transformer_range(pcontext);
	transformer tf2 = safe_any_expression_to_transformer(tmp2, e2, ncontext, true);
	// FI: not precise because the context is not taken into
	// account when non-convex information is available,
	// non-convex information that could be convex within the
	// context. But see below.
	transformer rel = relation_to_transformer(op, tmp1, tmp2, veracity);
	transformer cond = transformer_undefined;
	transformer newpre = transformer_undefined;

	free_transformer(pcontext);
	free_transformer(ncontext);

	/* tf1 disappears into cond */
	cond = transformer_safe_combine_with_warnings(tf1, tf2);

	if(transformer_undefined_p(rel) && !transformer_undefined_p(cond)) {
	  /* try to break rel it into two convex components */
	  if((ENTITY_NON_EQUAL_P(op) && veracity)
	     || (ENTITY_EQUAL_P(op) && !veracity)) {
	    entity lt = local_name_to_top_level_entity(LESS_THAN_OPERATOR_NAME);
	    entity gt = local_name_to_top_level_entity(GREATER_THAN_OPERATOR_NAME);
	    transformer rel1 = relation_to_transformer(lt, tmp1, tmp2, true);
	    transformer rel2 = relation_to_transformer(gt, tmp1, tmp2, true);
	    transformer full_cond1 = transformer_safe_image_intersection(cond, rel1);
	    transformer full_cond2 = transformer_safe_image_intersection(cond, rel2);
	    /* pre disappears into newpre2 */
	    transformer newpre1 = transformer_combine(transformer_dup(pre), full_cond1);
	    transformer newpre2 = transformer_combine(pre, full_cond2);

	    free_transformer(rel1);
	    free_transformer(rel2);

	    newpre = transformer_convex_hull(newpre1, newpre2);

	    free_transformer(full_cond1);
	    free_transformer(full_cond2);
	    free_transformer(newpre1);
	    free_transformer(newpre2);
	  }
	}
	else if (!transformer_undefined_p(cond)){
	  transformer full_cond = transformer_safe_image_intersection(cond, rel);
	  /* pre disappears into newpre */
	  newpre = transformer_combine(pre, full_cond);
	  free_transformer(full_cond);
	}
	else {
	  /* newpre stays undefined or newpre is pre*/
	  newpre = pre;
	}

	transformer_free(cond);
	pre = newpre;

	pips_assert("Pre is still consistent with tf1 and tf2 and rel",
		    transformer_argument_consistency_p(pre));

	/* free_transformer(tf1); already gone */
	free_transformer(tf2);
	free_transformer(rel);
	break;
      }
    case is_basic_overloaded:
      pips_internal_error("illegal overloaded type for operator %s",
		 entity_name(op));
      break;
    case is_basic_bit:
      // Do not emit the same message during the second analysis of
      // the condition
      if(veracity)
	pips_user_warning("bit type not analyzed for operator %s\n",
			  entity_name(op));
      break;
//    case is_basic_pointer:
//      if(veracity)
//	pips_user_warning("pointer type not analyzed for operator %s\n",
//			  entity_name(op));
//      break;
    case is_basic_derived:
      /* Nothing to be done with struct and union */
      break;
    case is_basic_typedef:
      pips_internal_error("typedef should ne converted to concrete types for operator %s",
			entity_name(op));
      break;
    default:
      pips_internal_error("unknown basic b=%d", basic_tag(b1));
    }
  }

  free_basic(b1);
  free_basic(b2);

  pre = transformer_normalize(pre, 2);

  /* pre may be unchanged when no information is derived */
  pips_debug(8, "end with pre=%p\n", pre);

  return pre;
}

/* A set of functions to compute the transformer associated to an
   expression evaluated in a given context.

   The choices are:

   1. Do I need the value of the expression? It seems strange but in
   some cases, the expression value is discarded. See for instance the C for construct.

   Keyword: "any" implies that the expression value is needed

   2. In case of failure, do I need an approximate transformer based
   on memory effects or an undefined transformer because I want to try
   something else?

   Keyword: "safe" implies that effects are used and that no undefined
   transformer is returned.

   3. In case of interprocedural call, am I ready to fail? This might
   be useful for the libraries "control" and "effects", but is not
   implemented yet.

   Suggested keyword: "light"

   4. Conditional expressions require specific treatment because the
   bool analysis is not well linked to the integer analysis and
   because C uses implicit conditions: if(n) means if(n!=0).

   5. Expression lists are a special case of expression.
 */

/* This function may return an undefined transformers if it fails to
   capture the semantics of expr in the polyhedral framework. */
transformer any_expression_to_transformer(
			      entity v, // value of the expression
			      expression expr,
			      transformer pre,
			      bool is_internal)
{
  transformer tf = transformer_undefined;
  basic be = basic_of_expression(expr);
  basic bv = variable_basic(type_variable(ultimate_type(entity_type(v))));

  if(basic_typedef_p(be)) {
    entity te = basic_typedef(be);
    type nte = ultimate_type(entity_type(te));

    pips_assert("nte is of type variable", type_variable_p(nte));

    /* Should basic of expression take care of this? */
    pips_debug(8, "Fix basic be\n");

    free_basic(be);
    be = copy_basic(variable_basic(type_variable(nte)));
  }

  ifdebug(8) {
    pips_debug(8, "begin for entity %s of type %s and %s expression ",
	       entity_local_name(v),
	       basic_to_string(bv), basic_to_string(be));
    print_expression(expr);
  }

  /* Assume v is a value */
  if( (basic_tag(bv)==basic_tag(be))
      || (basic_float_p(bv) && basic_int_p(be))
      || (basic_derived_p(bv) && basic_int_p(be))
      || (basic_derived_p(be) && basic_int_p(bv))
      || (basic_logical_p(bv) && basic_int_p(be))
      || (basic_logical_p(be) && basic_int_p(bv))
      || (basic_pointer_p(bv) && basic_int_p(be))) {
    switch(basic_tag(be)) {
    case is_basic_int:
      if(integer_analyzed_p()) {
	if(basic_int_p(bv)) {
	  tf = integer_expression_to_transformer(v, expr, pre, is_internal);
	}
	else if(basic_derived_p(bv)) {
	  /* If we are here, it should be an enum type... */
	  tf = integer_expression_to_transformer(v, expr, pre, is_internal);
	}
          else if(basic_logical_p(bv)) {
            tf = logical_expression_to_transformer(v, expr, pre, is_internal);
          }
          else if(basic_pointer_p(bv)) {
            //tf = integer_expression_to_transformer(v, expr, pre, is_internal);
            // arithmetic for pointer
            if(pointer_analyzed_p())
              tf = pointer_expression_to_transformer(v, expr, pre, is_internal);
          }
	else {
	  pips_user_warning("Integer expression assigned to float value\n"
			    "Apply PIPS type checker for better results\n");
	  /* Constants, at least, could be typed coerced */
	  /* Redundant with explicit type coercion also available in PIPS */
	  /* To be done later */
	}
      }
      break;
    case is_basic_logical:
      if(basic_logical_p(bv) && boolean_analyzed_p())
	tf = logical_expression_to_transformer(v, expr, pre, is_internal);
      else if(basic_int_p(bv)) { // We should check that integers are analyzed...
	tf = integer_expression_to_transformer(v, expr, pre, is_internal);
      }
      break;
    case is_basic_float:
      /* PIPS does not represent negative constants: call to unary_minus */
      if(float_analyzed_p())
	tf = float_expression_to_transformer(v, expr, pre, is_internal);
      break;
    case is_basic_pointer:
      if(pointer_analyzed_p())
        tf = pointer_expression_to_transformer(v, expr, pre, is_internal);
      break;
    case is_basic_complex:
      /* PIPS does not represent complex constants: call to CMPLX */
      break;
    case is_basic_string:
      /* Only constant string are processed */
      if(string_analyzed_p())
	tf = string_expression_to_transformer(v, expr);
      break;
    case is_basic_overloaded: {
      /* The overloading is supposed to have been lifted by
	 basic_of_expression() */
      if(expression_call_p(expr) && ENTITY_CONTINUE_P(call_function(expression_call(expr))))
	tf = transformer_identity();
      else
	pips_internal_error("illegal overloaded type for an expression");
      break;
    }
    case is_basic_derived: {
      entity de = basic_derived(be);
      type det = ultimate_type(entity_type(de));
      if(type_enum_p(det)) {
	/* enum type are analyzed like int */
	tf = integer_expression_to_transformer(v, expr, pre, is_internal);
      }
      else {
      pips_internal_error("entities of type \"derived\" that are not \"enum\" cannot be analyzed");
      }
      break;
    }
    case is_basic_typedef:
      pips_internal_error("entities of type \"typedef\" cannot be analyzed");
      break;
    default:
      pips_internal_error("unknown basic b=%d", basic_tag(be));
    }
  }
  else {
    pips_user_warning("Implicit type coercion between %s variable and %s "
		      "expression may reduce semantic analysis accuracy.\n"
		      "Apply 'type_checker' to explicit all type coercions.\n",
		      basic_to_string(bv), basic_to_string(be));
    /* It might be interesting to go further in case the comma operator is used */
    if(comma_expression_p(expr)) {
      call c = syntax_call(expression_syntax(expr));
      list expr_l = call_arguments(c);

      /* No need to link the returned value as it must be cast to the
	 proper type. */
      tf = expressions_to_transformer(expr_l, pre);
    }
  }

  /* tf may be transformer_undefined when no information is derived */
  ifdebug(1) {
    if(!transformer_undefined_p(pre))
      pips_assert("No obvious aliasing between tf and pre", tf!=pre);
  }
  pips_debug(8, "end with tf=%p\n", tf);

  return tf;
}

/* Always return a usable transformer. Use effects if no better
   transformer can be found */
transformer safe_any_expression_to_transformer(
					       entity v, // value of the expression
					       expression expr,
					       transformer pre,
					       bool is_internal)
{
  transformer npre = transformer_undefined_p(pre)?
    transformer_identity() : copy_transformer(pre);
  transformer tf = any_expression_to_transformer(v, expr, npre, is_internal);

  if(transformer_undefined_p(tf)) {
    //list el = expression_to_proper_effects(expr);
    list el = expression_to_proper_constant_path_effects(expr);
    tf = effects_to_transformer(el);

    /*
      pips_assert("effect references are protected by persistant",
      effect_list_can_be_safely_full_freed_p(el));*/

    (void) effect_list_can_be_safely_full_freed_p(el);

    gen_full_free_list(el);
  }
  free_transformer(npre);

  return tf;
}

/* Just to capture side effects as the returned value is
   ignored. Example: "(void) inc(&i);' */
transformer expression_to_transformer(
				      expression exp,
				      transformer pre,
				      list el)
{
  type et = expression_to_type(exp);
  //entity tmpv = entity_undefined;
  transformer tf = transformer_undefined;

  if(type_void_p(et)) {
    /* FI: I assume this implies a cast to (void) but I'm wrong for
       any call to a void function. */
    syntax s_exp = expression_syntax(exp);

    if(syntax_cast_p(s_exp)) {
      expression sub_exp = cast_expression(syntax_cast(s_exp));
      type cet = expression_to_type(sub_exp);

      if(analyzable_type_p(cet)) {
	entity tmpv = make_local_temporary_value_entity(cet);

	/* FI: I do not remember the meaning of the last parameter */
	// it's use in generic_reference_to_transformer
	tf = any_expression_to_transformer(tmpv, sub_exp, pre, false);
      }
      free_type(cet);
    }
    else if(syntax_call_p(s_exp)) {
      /* Must be a call to a void function */
      call c = syntax_call(s_exp);
      //list el = expression_to_proper_effects(exp);;
      list el = expression_to_proper_constant_path_effects(exp);
      tf = call_to_transformer(c, pre, el);
    }
    else {
      /* Wait till it happpens... */
      pips_internal_error("This case is not handled yet");
    }
  }
  else if(analyzable_type_p(et)) {
    entity tmpv = make_local_temporary_value_entity(et);
    /* FI: I do not remember the meaning of the last parameter */
    // it's use in generic_reference_to_transformer
    tf = any_expression_to_transformer(tmpv, exp, pre, false);
  }
  else {
    // FI: we should go down recursively anyway because of casts and
    // side effects in C...
    if(expression_reference_p(exp)) {
      reference r = expression_reference(exp);
      tf = generic_reference_to_transformer(entity_undefined, r, pre, false);
    }
    else if(expression_call_p(exp)) {
      call c = expression_call(exp);
      entity f = call_function(c);
      /* FI: there may be (many) other exceptions with intrinsics...
       *
       * Fortran intrinsics, such as IOs, are not taken into account
       * because these code should not be executed for Fortran code.
       */
      if(ENTITY_FIELD_P(f)||ENTITY_POINT_TO_P(f)) {
	tf = safe_expression_to_transformer(EXPRESSION(CAR(call_arguments(c))),
					    pre);
      }
      else
	tf = expressions_to_transformer(call_arguments(c), pre);
    }
    else if(expression_cast_p(exp)) {
      cast c = expression_cast(exp);
      expression sub_exp = cast_expression(c);
      tf = safe_expression_to_transformer(sub_exp, pre);
    }
    else if(expression_sizeofexpression_p(exp)) {
      sizeofexpression soe = expression_sizeofexpression(exp);
      if(sizeofexpression_expression_p(soe)) {
	expression sub_exp = sizeofexpression_expression(soe);
	tf = safe_expression_to_transformer(sub_exp, pre);
      }
    }
    else if(expression_subscript_p(exp)) {
      subscript sub = expression_subscript(exp);
      expression base = subscript_array(sub);
      transformer tf1 = safe_expression_to_transformer(base, pre);
      transformer tf2
	= expressions_to_transformer(subscript_indices(sub), pre);
      tf = transformer_combine(tf1, tf2);
      free_transformer(tf2); // tf1 is exported in tf
    }
    else if(expression_application_p(exp)) {
      application a = expression_application(exp);
      expression func = application_function(a);
      transformer tf1 = safe_expression_to_transformer(func, pre);
      transformer tf2
	= expressions_to_transformer(application_arguments(a), pre);
      tf = transformer_combine(tf1, tf2);
      free_transformer(tf2); // tf1 is exported in tf
    }
    // FI: Many other possible cases: range, sizeofexpression, subscript,
    // application, va_arg
  }

  /* If everything else has failed. */
  if(transformer_undefined_p(tf))
    tf = effects_to_transformer(el);
  else
    tf = transformer_temporary_value_projection(tf);

  free_type(et);

  return tf;
}

transformer safe_expression_to_transformer(expression exp, transformer pre)
{
  /* This simple function does not take points-to information into
     account. Furthermore, precise points-to information is not
     available when side effects occur as in comma expressions.

     FI: I do not see a simple way out. I can try to fix partly the
     problem in statement_to_transformer where the effects are
     known... Or I can play safer and use an effect function that
     replaces dereferencements by anywhere effects.

     See anywhere03.c as an example of the issue.
  */
  //list el = expression_to_proper_effects(exp);
  list el = expression_to_proper_constant_path_effects(exp);
  transformer tf = expression_to_transformer(exp, pre, el);

  gen_full_free_list(el);

  return tf;
}

/* To capture side effects and to add C twist for numerical
 * conditions. Argument pre may be undefined.
 *
 * Beware of tricky conditions using the comma or the conditional
 * operator, although they should be handled as the default case,
 * using effects.
 */
transformer condition_to_transformer(expression cond,
				     transformer pre,
				     bool veracity)
{
  //list el = expression_to_proper_effects(cond);
  list el = expression_to_proper_constant_path_effects(cond);
  transformer safe_pre = transformer_undefined_p(pre)?
    transformer_identity():
    transformer_range(pre);
  basic eb = basic_of_expression(cond);
  transformer tf = transformer_undefined;
  bool relation_p = false;
  /* upwards should be set to false when computing preconditions (but
     we have no way to know) or when computing tramsformers in
     context. And set to true in other cases. upwards is a way to gain
     execution time at the expense of precision. This speed/accuracy
     tradeoff has evolved with CPU technology. */
  bool upwards = false;
  expression tcond = cond; // By default; not true for the comma operations

  /* If a comma operator is used, the test is the last expression */
  /* FI: quid of a conditional operator? e.g. "x>0?1<m:1<n" */
  /* FI: quid of assignment operator? e.g. "i = j = k = 1" */
  /* FI: we probably need a function tcond = expression_to_condition(cond) */
  if(expression_call_p(cond)) {
    call c = syntax_call(expression_syntax(cond));
    entity op = call_function(c);
    if(ENTITY_COMMA_P(op)) {
      list args = call_arguments(c);
      tcond = EXPRESSION(CAR(gen_last(args)));
      tf = safe_expression_to_transformer(cond, pre);
    }
  }
  if(transformer_undefined_p(tf)) {
    tf= transformer_identity();
  }

  /* C comparison operators return an integer value */
  if(!basic_logical_p(eb) && expression_call_p(tcond)) {
    entity op = call_function(syntax_call(expression_syntax(tcond)));

    relation_p = ENTITY_LOGICAL_OPERATOR_P(op);
    //relation_p = ENTITY_RELATIONAL_OPERATOR_P(op);
  }

  if(basic_logical_p(eb)) {
    // entity tmpv = make_local_temporary_value_entity_with_basic(eb);
    //tf = logical_expression_to_transformer(tmpv, cond, safe_pre, true);
    //tf = transformer_add_condition_information_updown
    //  (transformer_identity(), cond, safe_pre, veracity, upwards);
    /* FI: not good when side effects in cond */
    //tf = transformer_add_condition_information_updown
    //  (copy_transformer(safe_pre), cond, safe_pre, veracity, upwards);
    transformer ctf = transformer_add_condition_information_updown
      (tf, tcond, safe_pre, veracity, upwards);
    tf = transformer_apply(ctf, safe_pre);
    free_transformer(ctf);
  }
  else if(relation_p) {
    //tf = transformer_add_condition_information_updown
    //  (transformer_identity(), cond, safe_pre, veracity, upwards);
    //tf = transformer_add_condition_information_updown
    //  (copy_transformer(safe_pre), cond, safe_pre, veracity, upwards);
    /* In case, there are side-effects in the condition. This is very
       unlikely for standard code and should be simplified with a
       test to transformer_identity_p(tf) */
    transformer new_pre = transformer_apply(tf, safe_pre);
    transformer new_pre_r = transformer_range(new_pre);
    new_pre_r = transformer_normalize(new_pre_r, 2);
    transformer ctf = transformer_add_condition_information_updown
      (transformer_identity(), tcond, new_pre_r, veracity, upwards);
    transformer_temporary_value_projection(ctf);
    ctf = transformer_normalize(ctf, 2);
    tf = transformer_combine(tf, ctf);
    free_transformer(ctf);
    free_transformer(new_pre);
    free_transformer(new_pre_r);
  }
  else {
    /* Make sure you can handle this kind of variable.

       This test is added for Semantics-New/transformer01.c which
       tests a pointer. The underlying bug may still be there when
       pointers are analyzed by PIPS.
    */
    if(analyzable_basic_p(eb)) {
      entity tmpv = make_local_temporary_value_entity_with_basic(eb);
      transformer ctf = safe_any_expression_to_transformer(tmpv, tcond, safe_pre, true);
      tf = transformer_combine(tf, ctf);
      if(veracity) {
	/* tmpv != 0 */
	transformer tf_plus = transformer_add_sign_information(copy_transformer(tf), tmpv, 1);
	transformer tf_minus = transformer_add_sign_information(copy_transformer(tf), tmpv, -1);

	ifdebug(8) {
	  fprintf(stderr, "tf_plus %p:\n", tf_plus);
	  dump_transformer(tf_plus);
	  fprintf(stderr, "tf_minus %p:\n", tf_minus);
	  dump_transformer(tf_minus);
	}

	free_transformer(tf);
	tf = transformer_convex_hull(tf_plus, tf_minus);
	free_transformer(tf_plus);
	free_transformer(tf_minus);
      }
      else {
	/* tmpv==0 */
	tf = transformer_add_sign_information(tf, tmpv, 0);
      }
    }
  }

  if(transformer_undefined_p(tf))
    tf = effects_to_transformer(el);
  else {
    /* Not yet? We may be in a = c? x : y; or in e1, e2,...; Does it matter? */
    tf = transformer_temporary_value_projection(tf);
    //reset_temporary_value_counter();
  }
  /* May be dangerous if this routine is called internally to another
     routine using temporary variables... */
  /* reset_temporary_value_counter(); */

  gen_full_free_list(el);
  free_basic(eb);
  free_transformer(safe_pre);

  return tf;
}

/* FI: not too smart to start with the special case with no value
   returned, just side-effects... */
transformer conditional_to_transformer(expression cond,
				       expression te,
				       expression fe,
				       transformer pre,
				       list ef)
{
  transformer tf = transformer_undefined;
  transformer ttf = condition_to_transformer(cond, pre, true);
  transformer t_pre = transformer_apply(ttf, pre);
  transformer t_pre_r = transformer_range(t_pre);
  transformer tet = safe_expression_to_transformer(te, t_pre_r);
  transformer ftf = condition_to_transformer(cond, pre, false);
  transformer f_pre = transformer_apply(ftf, pre);
  transformer f_pre_r = transformer_range(f_pre);
  transformer fet = safe_expression_to_transformer(fe, f_pre_r);

  ttf = transformer_combine(ttf, tet);
  ftf = transformer_combine(ftf, fet);
  tf = transformer_convex_hull(ttf, ftf);

  free_transformer(ttf);
  free_transformer(ftf);
  free_transformer(tet);
  free_transformer(fet);
  free_transformer(t_pre_r);
  free_transformer(f_pre_r);
  free_transformer(t_pre);
  free_transformer(f_pre);

  if(transformer_undefined_p(tf))
    tf = effects_to_transformer(ef);
  return tf;
}

/* Take care of the returned value. About a cut-and-paste of previous
   function, conditional_to_transformer(). */
transformer any_conditional_to_transformer(entity v,
					   list args,
					   transformer pre)
{
  expression cond = EXPRESSION(CAR(args));
  expression te = EXPRESSION(CAR(CDR(args)));
  expression fe = EXPRESSION(CAR(CDR(CDR(args))));
  transformer tf = transformer_undefined;
  transformer ttf = condition_to_transformer(cond, pre, true);
  transformer t_pre = transformer_apply(ttf, pre);
  transformer t_pre_r = transformer_range(t_pre);
  transformer tet = safe_any_expression_to_transformer(v, te, t_pre_r, true);
  transformer ftf = condition_to_transformer(cond, pre, false);
  transformer f_pre = transformer_apply(ftf, pre);
  transformer f_pre_r = transformer_range(f_pre);
  transformer fet = safe_any_expression_to_transformer(v, fe, f_pre_r, true);

  ttf = transformer_combine(ttf, tet);
  ftf = transformer_combine(ftf, fet);
  tf = transformer_convex_hull(ttf, ftf);

  free_transformer(ttf);
  free_transformer(ftf);
  free_transformer(tet);
  free_transformer(fet);
  free_transformer(t_pre_r);
  free_transformer(f_pre_r);
  free_transformer(t_pre);
  free_transformer(f_pre);

  //if(transformer_undefined_p(tf))
  //  tf = effects_to_transformer(ef);
  return tf;
}

/* returns the transformer associated to a C logical not, !, applied
 * to an integer argument, when meaningful, and transformer_undefined
 * otherwise. Effects must be used at a higher level to fall back on a
 * legal transformer.
 *
 * @param v is the value associated to the expression
 *
 * @param args is assumed to have only one expression element of type
 * int or enum negated by !
 *
 * @param pre is the current precondition
 */
transformer logical_not_to_transformer(entity v,
				       list args,
				       transformer pre)
{
  expression le = EXPRESSION(CAR(args));
  //basic be = basic_of_expression(le);
  transformer tf = transformer_undefined;

  //if(basic_int_p(be)) {
    entity tmp_v = make_local_temporary_integer_value_entity();
    tf = safe_any_expression_to_transformer(tmp_v, le, pre, true);
    transformer tfr = transformer_range(tf);
    intptr_t lb, ub;

    if(precondition_minmax_of_value(tmp_v, tfr, &lb, &ub)) {
      if(lb==ub && lb==0)
	tf = transformer_add_equality_with_integer_constant(tf, v, 1);
      else if(lb>0 || ub<0)
	tf = transformer_add_equality_with_integer_constant(tf, v, 0);
      else {
	tf = transformer_add_inequality_with_integer_constraint(tf, v, 1, true);
	tf = transformer_add_inequality_with_integer_constraint(tf, v, 0, false);
      }
    }
    else {
      tf = transformer_add_inequality_with_integer_constraint(tf, v, 1, true);
      tf = transformer_add_inequality_with_integer_constraint(tf, v, 0, false);
    }
    free_transformer(tfr);
    //  }

    //free_basic(be);

  return tf;
}

/* returns the transformer associated to a C bitwise xor, ^, applied
 * to two integer argument, when meaningful, and transformer_undefined
 * otherwise. Effects must be used at a higher level to fall back on a
 * legal transformer.
 *
 * @param v is the value associated to the expression
 *
 * @param args is assumed to have only one expression element of type
 * int negated by !
 *
 * @param pre is the current precondition
 */
transformer bitwise_xor_to_transformer(entity v,
				       list args,
				       transformer pre)
{
  /* Something can be done when both arguments have values in [0,1],
     when both arguments have known numerical values, when both
     arguments have the same value,... However, all this
     tests imply lots time consuming projections for a limited
     result... Unless somebody uses XOR to flip-flop between
     arrays... for instance to bother Sven Verdoolaege:-). A logical
     not might be more useful for this:-). */
  expression le1 = EXPRESSION(CAR(args));
  //basic be1 = basic_of_expression(le1);
  expression le2 = EXPRESSION(CAR(CDR(args)));
  //basic be2 = basic_of_expression(le2);
  transformer tf = transformer_undefined;

  //if(basic_int_p(be1) && basic_int_p(be2)) {
    entity tmp_v1 = make_local_temporary_integer_value_entity();
    // FI: Even with the "in context" property, the precondition is
    // not integrated into the transformer...
    transformer tf1 = safe_any_expression_to_transformer(tmp_v1, le1, pre, true);
    //transformer tfr1 = transformer_range(tf1);
    transformer tfr1 = transformer_apply(tf1, pre);
    intptr_t lb1, ub1;
    entity tmp_v2 = make_local_temporary_integer_value_entity();
    transformer tf2 = safe_any_expression_to_transformer(tmp_v2, le2, pre, true);
    // transformer tfr2 = transformer_range(tf2);
    transformer tfr2 = transformer_apply(tf2, pre);
    intptr_t lb2, ub2;

    if(precondition_minmax_of_value(tmp_v1, tfr1, &lb1, &ub1)
       && precondition_minmax_of_value(tmp_v2, tfr2, &lb2, &ub2)) {
      if(lb1==ub1 && lb2==ub2) {
	/* The two values are know at compile time */
	tf = transformer_identity();
	tf = transformer_add_equality_with_integer_constant(tf, v, lb1 ^ lb2);
      }
      else if(0<=lb1 && ub1 <= 1 && 0<=lb2 && ub2 <= 1) {
	tf = transformer_identity();
	/* The two values can be interpreted as booleans */
	/* Code about cut-and-pasted from
	   logical_binary_operation_to_transformer() */
	Pvecteur eq1 = vect_new((Variable) v, VALUE_ONE);
	vect_add_elem(&eq1, (Variable) tmp_v1, VALUE_MONE);
	vect_add_elem(&eq1, (Variable) tmp_v2, VALUE_MONE);

	Pvecteur eq2 = vect_new((Variable) v, VALUE_ONE);
	vect_add_elem(&eq2, (Variable) tmp_v1, VALUE_ONE);
	vect_add_elem(&eq2, (Variable) tmp_v2, VALUE_ONE);
	vect_add_elem(&eq2, TCST , VALUE_MONE+VALUE_MONE);

	Pvecteur eq3 = vect_new((Variable) v, VALUE_MONE);
	vect_add_elem(&eq3, (Variable) tmp_v1, VALUE_ONE);
	vect_add_elem(&eq3, (Variable) tmp_v2, VALUE_MONE);

	Pvecteur eq4 = vect_new((Variable) v, VALUE_MONE);
	vect_add_elem(&eq4, (Variable) tmp_v1, VALUE_MONE);
	vect_add_elem(&eq4, (Variable) tmp_v2, VALUE_ONE);

	tf = transformer_inequality_add(tf, eq1);
	tf = transformer_inequality_add(tf, eq2);
	tf = transformer_inequality_add(tf, eq3);
	tf = transformer_inequality_add(tf, eq4);
      }
    }
    if(transformer_undefined_p(tf)) {
      /* The two arguments may be equal because they are the same
	 expression and the xor is used to generate a zero or because
	 they have the same value. */
      if(expression_equal_p(le1, le2)) {
	tf = transformer_identity();
	tf = transformer_add_equality_with_integer_constant(tf, v, 0);
      }
      else {
	transformer tfr1 = transformer_range(tf1);
	transformer tfr2 = transformer_range(tf2);
	transformer tfi = transformer_intersection(tfr1, tfr2);
	entity tmp_d = make_local_temporary_integer_value_entity();
	Pvecteur eq = vect_new((Variable) tmp_d, VALUE_ONE);
	vect_add_elem(&eq, (Variable) tmp_v1, VALUE_MONE);
	vect_add_elem(&eq, (Variable) tmp_v2, VALUE_ONE);
	tfi = transformer_equality_add(tfi, eq);
	intptr_t lbd, ubd;

	if(precondition_minmax_of_value(tmp_d, tfi, &lbd, &ubd)) {
	  if(lbd==0 && ubd==0) {
	    /* Since the difference is zero the two values are equal */
	    tf = transformer_identity();
	    tf = transformer_add_equality_with_integer_constant(tf, v, 0);
	  }
	}
      }
    }
    //}
    //free_basic(be1), free_basic(be2);
    free_transformer(tfr1), free_transformer(tfr2);
  return tf;
}

/* Compute the transformer associated to a list of expressions such as
 * "i=0, j = 1;". The value returned is ignored.
 */
transformer expressions_to_transformer(list expl,
				       transformer pre)
{
  transformer tf = transformer_identity();
  transformer cpre = transformer_undefined_p(pre)?
    transformer_identity() : copy_transformer(pre);

  FOREACH(EXPRESSION, exp, expl) {
    /* el is an over-appoximation; should be replaced by a
       safe_expression_to_transformer() taking care of computing the
       precise effects of exp instead of using the effects of expl. */
    transformer cpre_r = transformer_range(cpre);
    transformer ctf = safe_expression_to_transformer(exp, cpre_r);
    transformer npre = transformer_undefined;

    tf = transformer_combine(tf, ctf);
    tf = transformer_normalize(tf, 2);
    npre = transformer_apply(ctf, cpre);
    npre = transformer_normalize(npre, 2);
    free_transformer(cpre);
    free_transformer(cpre_r);
    cpre = npre;
  }
  free_transformer(cpre);
  return tf;
}

/* Compute the transformer associated to a list of expressions such as
 * "i=0, j = 1;". The value returned is linked to v.
 *
 * To be merged with the previous function.
 */
transformer any_expressions_to_transformer(entity v,
					   list expl,
					   transformer pre)
{
  transformer tf = transformer_identity();
  transformer cpre = transformer_undefined_p(pre)?
    transformer_identity() : transformer_range(pre);
  expression l_exp = EXPRESSION(CAR(gen_last(expl)));

  FOREACH(EXPRESSION, exp, expl) {
    /* el is an over-appoximation; should be replaced by a
       safe_expression_to_transformer() taking care of computing the
       precise effects of exp instead of using the effects of expl. */
    transformer ctf = (exp==l_exp)?
      safe_any_expression_to_transformer(v, exp, cpre, false) :
      safe_expression_to_transformer(exp, cpre);
    transformer npre = transformer_undefined;
    transformer npre_r = transformer_undefined;

    tf = transformer_combine(tf, ctf);
    tf = transformer_normalize(tf, 2);
    npre = transformer_apply(ctf, cpre);
    npre_r = transformer_range(npre);
    npre_r = transformer_normalize(npre_r, 2);
    free_transformer(cpre);
    free_transformer(npre);
    free_transformer(ctf);
    cpre = npre_r;
  }
  free_transformer(cpre);
  return tf;
}

/* compute integer bounds @p pmax, @p pmin of value @p val under
 * preconditions @p tr require value mappings set !
 */
bool precondition_minmax_of_value(entity val,
				  transformer tr,
				  intptr_t* pmin, intptr_t* pmax)
{
    bool success;
    /* retrieve the associated psysteme*/
    Psysteme ps = sc_dup(predicate_system(transformer_relation(tr)));
    /* compute min / max bounds */
    Value vmin,vmax;
    if((success=sc_minmax_of_variable(ps,val,&vmin,&vmax)))
    {
        /* special case to handle VMIN and VMAX in 32 bits*/
        if(vmax != (Value)(intptr_t)(vmax) && vmax == VALUE_MAX) vmax= INT_MAX;
        if(vmin != (Value)(intptr_t)(vmin) && vmin == VALUE_MIN) vmin= INT_MIN;
        pips_assert("no data loss", vmin == (Value)(intptr_t)(vmin));
        pips_assert("no data loss", vmax == (Value)(intptr_t)(vmax));
        *pmin=(intptr_t)vmin;
        *pmax=(intptr_t)vmax;
    }
    // FI: ps is destroyed by sc_minmax_of_variable(). Should it
    // nevertheless be freed?
    return success;
}


/* compute integer bounds @p pmax, @p pmin of expression @p exp under preconditions @p tr
 * require value mappings set !
 */
bool precondition_minmax_of_expression(expression exp,
				       transformer tr, // precondition
				       intptr_t* pmin,
				       intptr_t* pmax)
{
    bool success;
    /* create a temporary value */
    basic bas = basic_of_expression(exp);
    entity var = make_local_temporary_value_entity_with_basic(bas);
    free_basic(bas);

    /* compute its preconditions */
    transformer var_tr = safe_any_expression_to_transformer(var,exp,tr,false);
    transformer pre = transformer_apply(var_tr, tr);

    //success = precondition_minmax_of_value(var, var_tr, pmin, pmax);
    success = precondition_minmax_of_value(var, pre, pmin, pmax);

    /* tidy & return */
    predicate_system(transformer_relation(var_tr))=SC_UNDEFINED;
    free_transformer(var_tr);
    free_entity(var);
    return success;
}

/* tries hard to simplify expression @p e if it is a min or a max
 * operator, by evaluating it under preconditions @p tr.
 * Two approaches are tested:
 * check bounds of lhs-rhs,
 * or compare bounds of lhs and rhs
 */
void simplify_minmax_expression(expression e,transformer tr)
{
    if(expression_minmax_p(e)) {
        call c =expression_call(e);
        bool is_max = ENTITY_MAX_P(call_function(c));

        expression lhs = binary_call_lhs(c);
        expression rhs = binary_call_rhs(c);
        expression diff = make_op_exp(
                MINUS_OPERATOR_NAME,
                copy_expression(lhs),
                copy_expression(rhs)
                );
        intptr_t lhs_lbound,lhs_ubound,rhs_lbound,rhs_ubound,diff_lbound,diff_ubound;
        if(precondition_minmax_of_expression(diff,tr,&diff_lbound,&diff_ubound) &&
            (diff_lbound>=0 || diff_ubound<=0)) {
            if(is_max) {
                if(diff_lbound>=0) local_assign_expression(e,lhs);
                else local_assign_expression(e,rhs);
            }
            else {
                if(diff_lbound>=0) local_assign_expression(e,rhs);
                else local_assign_expression(e,lhs);
            }
        }
        else if(precondition_minmax_of_expression(lhs,tr,&lhs_lbound,&lhs_ubound) &&
                precondition_minmax_of_expression(rhs,tr,&rhs_lbound,&rhs_ubound))
        {
            if(is_max)
            {
                if(lhs_lbound >=rhs_ubound) local_assign_expression(e,lhs);
                else if(rhs_lbound >= lhs_ubound) local_assign_expression(e,rhs);
            }
            else
            {
                if(lhs_lbound >=rhs_ubound) local_assign_expression(e,rhs);
                else if(rhs_lbound >= lhs_ubound) local_assign_expression(e,lhs);
            }
        }
    }
}

bool false_condition_wrt_precondition_p(expression c, transformer pre)
{
  bool result = false;

  result = eval_condition_wrt_precondition_p(c, pre, false);

  return result;
}

bool true_condition_wrt_precondition_p(expression c, transformer pre)
{
  bool result = false;

  result = eval_condition_wrt_precondition_p(c, pre, true);

  return result;
}

bool eval_condition_wrt_precondition_p(expression c, transformer pre, bool veracity)
{
  bool result = false;
  transformer f = transformer_dup(pre);

  if(veracity) {
    f = precondition_add_condition_information(f, c, pre, false);
  }
  else {
    f = precondition_add_condition_information(f, c, pre, true);
  }

  result = transformer_empty_p(f);

  return result;
}

/* It is supposed to be obsolete but is still called. Maybe, it's only
   partly obsolete... If upwards is false, it is worth performing more
   convex hulls because the precondition on entry may restrain the space. */
transformer transformer_add_integer_relation_information(
    transformer pre,
    entity relop,
    expression e1,
    expression e2,
    bool veracity,
    bool upwards) /* upwards = transformer, !upwards = precondition */
{
#   define DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION 7
  /* default: no change */
  transformer newpre = pre;
  /* both expressions e1 and e2 must be affine */
  normalized n1 = NORMALIZE_EXPRESSION(e1);
  normalized n2 = NORMALIZE_EXPRESSION(e2);

  ifdebug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION) {
    pips_debug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION,
	       "begin upwards=%s veracity=%s relop=%s e1=", 
	       bool_to_string(upwards), bool_to_string(veracity),
	       entity_local_name(relop));
    print_expression(e1);
    (void) fprintf(stderr,"e2=");
    print_expression(e2);
    (void) fprintf(stderr,"pre=");
    print_transformer(pre);
  }

  if(normalized_linear_p(n1) && normalized_linear_p(n2)
     && value_mappings_compatible_vector_p((Pvecteur) normalized_linear(n1))
     && value_mappings_compatible_vector_p((Pvecteur) normalized_linear(n2))) {
    Pvecteur v1 = vect_dup((Pvecteur) normalized_linear(n1));
    Pvecteur v2 = vect_dup((Pvecteur) normalized_linear(n2));

    /* Make sure that new values only are used in v1 and v2 */
    variables_to_new_values(v1);
    variables_to_new_values(v2);

    if((ENTITY_EQUAL_P(relop) && veracity) ||
       (ENTITY_NON_EQUAL_P(relop) && !veracity)) {
      /* v1 - v2 == 0 */
      Pvecteur v = vect_substract(v1, v2);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_equality_add(pre, v);
    }
    else if((ENTITY_EQUAL_P(relop) && !veracity) ||
	    (ENTITY_NON_EQUAL_P(relop) && veracity)) {
      /* v2 - v1 + 1 <= 0 ou v1 - v2 + 1 <= 0 */
      /* FI: I do not know if this is valid when your are moving upwards
       * variables in v2 and v1 may have to be renamed as #init values (i.e. old values)
       */
      transformer prea = transformer_dup(pre);
      transformer preb = pre;
      Pvecteur va = vect_substract(v2, v1);
      Pvecteur vb = vect_substract(v1, v2);
      vect_add_elem(&va, TCST, VALUE_ONE);
      vect_add_elem(&vb, TCST, VALUE_ONE);
      /* FI: I think that this should be programmed (see comment above)
       * but I'm waiting for a bug to occur... (6 July 1993)
       *
       * FI: Well, the bug was eventually seen:-) (8 November 1995)
       */
      if(upwards) {
	upwards_vect_rename(va, pre);
	upwards_vect_rename(vb, pre);
      }
      prea = transformer_inequality_add(prea, va);
      preb = transformer_inequality_add(preb, vb);
      newpre = transformer_convex_hull(prea, preb);
      /* free_transformer(prea); */
      /* free_transformer(preb); */
    }
    else if ((ENTITY_GREATER_THAN_P(relop) && veracity) ||
	     (ENTITY_LESS_OR_EQUAL_P(relop) && !veracity)) {
      /* v2 - v1 + 1 <= 0 */
      Pvecteur v = vect_substract(v2, v1);
      vect_add_elem(&v, TCST, VALUE_ONE);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else if ((ENTITY_GREATER_THAN_P(relop) && !veracity) ||
	     (ENTITY_LESS_OR_EQUAL_P(relop) && veracity)) {
      /* v1 - v2 <= 0 */
      Pvecteur v = vect_substract(v1, v2);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else if ((ENTITY_GREATER_OR_EQUAL_P(relop) && veracity) ||
	     (ENTITY_LESS_THAN_P(relop) && !veracity)) {
      /* v2 - v1 <= 0 */
      Pvecteur v = vect_substract(v2, v1);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else if ((ENTITY_GREATER_OR_EQUAL_P(relop) && !veracity) ||
	     (ENTITY_LESS_THAN_P(relop) && veracity)) {
      /* v1 - v2 + 1 <= 0 */
      Pvecteur v = vect_substract(v1, v2);
      vect_add_elem(&v, TCST, VALUE_ONE);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else {
      /* do nothing... although Malik may have tried harder! */
      newpre = pre;
    }
    vect_rm(v1);
    vect_rm(v2);
  }
  else
    /* do nothing, although MODULO and INTEGER DIVIDE could be handled */
    newpre = pre;

  ifdebug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION) {
    pips_debug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION,
	       "end newpre=\n");
    print_transformer(newpre);
    pips_assert("Transformer is internally consistent",
		transformer_internal_consistency_p(newpre));
  }

  return newpre;
}

/* Simplification of bool expressions with precondition
 *
 * Expression e is modified, if necessary, by side effect.
 *
 * The value returned is 1 if the expression e is always true (never
 * false) under condition p, 0 if is the expression is always false
 * (never true), and -1 otherwise.
 *
 * This function is not used within the semantics library. It is
 * exported for partial_eval and suppress_dead_code or similar passes.
 *
 * Because of C flexibility, all kinds of "boolean" expressions may
 * arise. Think of the conditional and comma operators, think of all
 * kinds of side effects, think of integer expressions,...
 *
 * FI: In the short term, I only need to deal with bool operators
 * and, or and not.
 */
int simplify_boolean_expression_with_precondition(expression e,
						  transformer p)
{
  int validity = -1; // unknown
  bool done = false;
  if(logical_operator_expression_p(e)) {
    // We could try the global simplification as in the other case...
    syntax s = expression_syntax(e);
    call c = syntax_call(s);
    entity op = call_function(c);
    if(ENTITY_NOT_P(op)) {
      expression a = EXPRESSION(CAR(call_arguments(c)));
      validity = simplify_boolean_expression_with_precondition(a,p);
      if(validity>=0)
	validity = 1 - validity;
      done = true;
    }
    else if(ENTITY_OR_P(op)) {
      expression a1 = EXPRESSION(CAR(call_arguments(c)));
      int v1 = simplify_boolean_expression_with_precondition(a1,p);
      if(v1==1)
	validity = 1;
      else {
	expression a2 = EXPRESSION(CAR(CDR(call_arguments(c))));
	int v2 = simplify_boolean_expression_with_precondition(a2,p);
	if(v2==1)
	  validity = 1;
	else if(v2==0) { // a2 has no impact
	  if(v1==0)
	    validity = 0;
	  else {
	    validity = -1;
	    replace_expression_content(e, copy_expression(a1));
	  }
	}
	else // v2==-1
	  if(v1==0) { // a1 has no impact
	    validity = -1;
	    replace_expression_content(e, copy_expression(a2));
	  }
	  else
	    validity = -1;
      }
      done =true;
    }
    else if(ENTITY_AND_P(op)) {
      expression a1 = EXPRESSION(CAR(call_arguments(c)));
      int v1 = simplify_boolean_expression_with_precondition(a1,p);
      if(v1==0)
	validity = 0;
      else { // v1 == -1 or 1, no conclusion yet
	expression a2 = EXPRESSION(CAR(CDR(call_arguments(c))));
	int v2 = simplify_boolean_expression_with_precondition(a2,p);
	if(v2==0)
	  validity = 0;
	else if(v2==1) {
	  if(v1==1)
	    validity = 1;
	  else { // get rid of a2
	    validity = -1;
	    replace_expression_content(e, copy_expression(a1));
	  }
	}
	else { // v2==-1
	  if(v1==1) {// a1 is useless
	    validity = -1;
	    replace_expression_content(e, copy_expression(a2));
	  }
	  else {
	    validity = -1;
	  }
	}
      }
      done =true;
    }
  }
  if (!done) {
    transformer tt = condition_to_transformer(e,p, true);
    transformer ft = condition_to_transformer(e,p, false);

    if(transformer_empty_p(tt)) {
      // Then the condition is always false
      validity = 0;
    }
    if(transformer_empty_p(ft)) {
      // Then the condition is always true
      validity = 1;
    }
    // Both can be true simultaneously when the precondition p is
    // empty... but it does not matter
  }

  if(validity==0||validity==1) {
    // e must be replaced by a call to true or false
    expression ne = validity?make_true_expression():make_false_expression();
    replace_expression_content(e, ne);
  }

  return validity;
}
