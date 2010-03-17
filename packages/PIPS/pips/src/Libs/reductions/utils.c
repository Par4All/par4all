/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/*
 * utilities for reductions.
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "semantics.h"

/******************************************************************** MISC */
/* ??? some arrangements with words_range to print a star in this case:-)
 */
static syntax make_star_syntax()
{
    return make_syntax(is_syntax_range,
		       make_range(expression_undefined,
				  expression_undefined,
				  expression_undefined));
}

/***************************************************** REFERENCED ENTITIES */
/* returns the list of referenced variables
 */
static list /* of entity */ referenced;
static void ref_rwt(reference r)
{ referenced = gen_once(reference_variable(r), referenced);}

static list /* of entity */
referenced_variables(reference r)
{
    referenced = NIL;
    gen_recurse(r, reference_domain, gen_true, ref_rwt);
    return referenced;
}

/************************************** REMOVE A VARIABLE FROM A REDUCTION */
/* must be able to remove a modified variable from a reduction:
 * . A(I) / I -> A(*)
 * . A(B(C(I))) / C -> A(B(*))
 * . A(I+J) / I -> A(*)
 * . A(B(I)+C(I)) / I -> A(*) or A(B(*)+C(*)) ? former looks better
 */

/* the variable that must be removed is stored here
 */
static entity variable_to_remove;
/* list of expressions to be deleted (at rwt)
 */
static list /* of expression */ dead_expressions;
/* the first expression encountered which is a function call, so as
 * to avoid "*+J" results
 */
static expression first_encountered_call;
/* stack of reference expressions (if there is no call)
 */
DEFINE_LOCAL_STACK(ref_exprs, expression)

static bool expr_flt(expression e)
{
    if (expression_reference_p(e)) /* REFERENCE */
	ref_exprs_push(e);
    else if (!first_encountered_call && expression_call_p(e)) /* CALL */
	first_encountered_call = e;
    return TRUE;
}

static void expr_rwt(expression e)
{
    if (expression_reference_p(e))
	ref_exprs_pop();
    else if (first_encountered_call==e)
	first_encountered_call = NULL;

    if (gen_in_list_p(e, dead_expressions))
    {
	free_syntax(expression_syntax(e));
	expression_syntax(e) = make_star_syntax();
    }
}

static bool ref_flt(reference r)
{
    if (reference_variable(r)==variable_to_remove)
    {
	dead_expressions =
	    gen_once(first_encountered_call? first_encountered_call:
		     ref_exprs_head(), dead_expressions);
	return FALSE;
    }
    return TRUE;
}

void
remove_variable_from_reduction(
    reduction red,
    entity var)
{
    variable_to_remove = var;
    dead_expressions = NIL;
    first_encountered_call = NULL;
    make_ref_exprs_stack();

    pips_debug(8, "removing %s from %s[%s]\n",
	       entity_name(var),
	       reduction_operator_tag_name
	          (reduction_operator_tag(reduction_op(red))),
	       entity_name(reduction_variable(red)));

    gen_multi_recurse(reduction_reference(red),
		      expression_domain, expr_flt, expr_rwt,
		      reference_domain, ref_flt, gen_null,
		      NULL);

    gen_free_list(dead_expressions);
    free_ref_exprs_stack();
}

bool
update_reduction_under_effect(
    reduction red,
    effect eff)
{
    entity var = effect_variable(eff);
    bool updated = FALSE;

    pips_debug(7, "reduction %s[%s] under effect %s on %s\n",
	       reduction_operator_tag_name
	          (reduction_operator_tag(reduction_op(red))),
	       entity_name(reduction_variable(red)),
	       effect_write_p(eff)? "W": "R",
	       entity_name(effect_variable(eff)));

    /* REDUCTION is dead if the reduction variable is affected
     */
    if (entity_conflict_p(reduction_variable(red),var))
    {
	reduction_operator_tag(reduction_op(red)) =
	    is_reduction_operator_none;
	return FALSE;
    }
    /* else */

    if (effect_read_p(eff)) return TRUE;

    /* now var is written */
    FOREACH (ENTITY, e, reduction_dependences(red)) {
      if (entity_conflict_p(var, e)) {
	updated = TRUE;
	remove_variable_from_reduction(red, e);
      }
    }

    if (updated)
    {
	gen_free_list(reduction_dependences(red));
	reduction_dependences(red) =
	     referenced_variables(reduction_reference(red));
    }

    return TRUE;
}

/* looks for a reduction about var in reds, and returns it.
 * tells whether it worths keeping on. It does not if there may be some
 * conflicts with other reduced variables...
 */
static bool
find_reduction_of_var(
    entity var,
    reductions reds,
    reduction *pr)
{
  FOREACH (REDUCTION, r, reductions_list(reds)) {
    entity red_var = reduction_variable(r);
    if (red_var==var) {
      *pr = copy_reduction(r);
      return TRUE;
    }
    else if (entity_conflict_p(red_var, var))
      return FALSE; /* I will not combine them... */
  }
  return TRUE;
}

/* merge two reductions into first so as to be compatible with both.
 * deletes the second. tells whether they where compatibles
 * quite basic at the time
 */
static bool
merge_two_reductions(reduction first, reduction second)
{
    pips_assert("same variable",
		reduction_variable(first)==reduction_variable(second));

    if (reduction_operator_tag(reduction_op(first))!=
	reduction_operator_tag(reduction_op(second)))
    {
	free_reduction(second);
	return FALSE;
    }

    if (!reference_equal_p(reduction_reference(first),
			   reduction_reference(second)))
    {
	/* actually merges, very simple at the time
	 */
	free_reference(reduction_reference(first));
	reduction_reference(first) =
	    make_reference(reduction_variable(second), NIL);
    }

    free_reduction(second);
    return TRUE;
}

/* update *pr according to r for variable var
 * r is not touched.
 */
bool
update_compatible_reduction_with(
    reduction *pr,
    entity var,
    reduction r)
{
    if (reduction_variable(r)!=var)
	return !entity_conflict_p(var, reduction_variable(r));

    /* else same var and no conflict */
    if (reduction_none_p(*pr))
    {
	free_reduction(*pr);
	*pr = copy_reduction(r);
	return TRUE;
    }
    /* else are they compatible?
     */
    if (reduction_tag(*pr)!=reduction_tag(r))
	return FALSE;
    /* ok, let us merge them
     */
    return merge_two_reductions(*pr, copy_reduction(r));
}

/* what to do with reduction *pr for variable var
 * under effects le and reductions reds.
 * returns whether worth to go on.
 * conditions:
 */  
bool
update_compatible_reduction(
    reduction *pr,
    entity var,
    list /* of effect */ le,
    reductions reds)
{
    reduction found = NULL;

    if (!find_reduction_of_var(var, reds, &found))
	return FALSE;

    if (found)
    {
      if (!reduction_none_p(*pr)) /* some reduction already available */
	return merge_two_reductions(*pr, found);
      else { /* must update the reduction with the encountered effects */
	FOREACH (ENTITY, e,reduction_dependences(*pr)) {
	  remove_variable_from_reduction(found, e);
	}
	free_reduction(*pr); *pr = found;
	return TRUE;
      }
    }
    /* else
     * now no new reduction waas found, must check *pr against effects
     */
    if (!reduction_none_p(*pr)) /* some reduction */
      {
	FOREACH (EFFECT, e, le) {
	  if (!update_reduction_under_effect(*pr, e)) {
	    DEBUG_REDUCTION(8, "kill of ", *pr);
	    pips_debug(8, "under effect to %s\n",
		       entity_name(effect_variable(e)));
	    return FALSE;
	  }
	}
    }
    else
    {
      FOREACH (EFFECT, e, le) {
	if (entity_conflict_p(effect_variable(e), var))
	  return FALSE;
	else if (effect_write_p(e)) /* stores for latter cleaning */
	  reduction_dependences(*pr) = gen_once(effect_variable(e),
						reduction_dependences(*pr));
      }
    }
    return TRUE;
}

/*************************************************** CALL PROPER REDUCTION */
/* extract the proper reduction of a call (instruction) if any.
 */

/* I trust intrinsics (operations?) and summary effects...
 */
bool pure_function_p(entity f)
{
    value v = entity_initial(f);

    if (value_symbolic_p(v) || value_constant_p(v) || value_intrinsic_p(v))
	return TRUE;
    /* else */

    if (entity_module_p(f))
    {
      FOREACH (EFFECT, e, load_summary_effects(f)) {
	if (effect_write_p(e)) /* a side effect!? */
	  return FALSE;
	if (io_effect_entity_p(effect_variable(e))) /* LUNS */
	  return FALSE;
      }
    }

    return TRUE;
}

/* tells whether r is a functional reference...
 * actually I would need to recompute somehow the proper effects of obj?
 */
static bool is_functional;
static bool call_flt(call c)
{
    if (pure_function_p(call_function(c)))
	return TRUE;
    /* else */
    is_functional = FALSE;
    gen_recurse_stop(NULL);
    return FALSE;
}
static bool
functional_object_p(gen_chunk* obj)
{
    is_functional = TRUE;
    gen_recurse(obj, call_domain, call_flt, gen_null);
    return is_functional;
}

/****************************************************** REDUCTION OPERATOR */

#define OKAY(op,com) do { *op_tag=op; *commutative=com; return TRUE;} while(FALSE)
#define OKAY_WO_COMM(op) do { *op_tag=op; return TRUE;} while(FALSE)
/* tells whether entity f is a reduction operator function
 * also returns the corresponding tag, and if commutative
 * @return TRUE if the entity f is a reduction operator function
 * @param f, the entity to look at
 * @param op_tag, use to return the reduction operator (+, * ...)
 * @param commutative, use to return the operator commutativity
 */
static bool
function_reduction_operator_p(
    entity f,
    tag *op_tag,
    bool *commutative)
{
    if (ENTITY_PLUS_P(f))
      OKAY(is_reduction_operator_sum, TRUE);
    else if (ENTITY_MINUS_P(f))
      OKAY(is_reduction_operator_sum, FALSE);
    else if (ENTITY_MULTIPLY_P(f))
      OKAY(is_reduction_operator_prod, TRUE);
    else if (ENTITY_DIVIDE_P(f))
      OKAY(is_reduction_operator_prod, FALSE);
    else if (ENTITY_MIN_P(f) || ENTITY_MIN0_P(f))
      OKAY(is_reduction_operator_min, TRUE);
    else if (ENTITY_MAX_P(f) || ENTITY_MAX0_P(f))
      OKAY(is_reduction_operator_max, TRUE);
    else if (ENTITY_AND_P(f))
      OKAY(is_reduction_operator_and, TRUE);
    else if (ENTITY_OR_P(f))
      OKAY(is_reduction_operator_or, TRUE);
    else if (ENTITY_BITWISE_AND_P(f))
      OKAY(is_reduction_operator_bitwise_and, TRUE);
    else if (ENTITY_BITWISE_OR_P(f))
      OKAY(is_reduction_operator_bitwise_or, TRUE);
    else if (ENTITY_BITWISE_XOR_P(f))
      OKAY(is_reduction_operator_bitwise_xor, TRUE);
    else if (ENTITY_EQUIV_P(f))
      OKAY(is_reduction_operator_eqv, TRUE);
    else if (ENTITY_NON_EQUIV_P(f))
      OKAY(is_reduction_operator_neqv, TRUE);
    else
      return FALSE;
}

/* returns the possible operator of expression e if it is a reduction,
 * and the operation commutative. (- and / are not)
 */
static bool
extract_reduction_operator(
    expression e,
    tag *op_tag,
    bool *commutative)
{
    syntax s = expression_syntax(e);
    call c;
    entity f;

    if (!syntax_call_p(s)) return FALSE;
    c = syntax_call(s);
    f = call_function(c);
    return function_reduction_operator_p(f, op_tag, commutative);
}

/* Test if the operator is an update operator compatible with reduction
 * This also returns the corresponding tag, and if commutative
 * @return TRUE if the operator is an update operator compatible with reduction
 * @param operator, the operator (as an entity) to look at
 * @param op_tag, used to return the reduction operator (+, * ...)
 * @param commutative, used to return the operator commutativity
 */
static bool extract_reduction_update_operator (entity operator,
					       tag*   op_tag,
					       bool*  commutative)
{
  if (ENTITY_PLUS_UPDATE_P (operator))
    OKAY(is_reduction_operator_sum, TRUE);
  else if (ENTITY_MINUS_UPDATE_P (operator))
    OKAY(is_reduction_operator_sum, TRUE);
  else if (ENTITY_MULTIPLY_UPDATE_P (operator))
    OKAY(is_reduction_operator_prod, TRUE);
  else if (ENTITY_DIVIDE_UPDATE_P (operator))
    OKAY(is_reduction_operator_prod, TRUE);
  else if (ENTITY_BITWISE_OR_UPDATE_P (operator))
    OKAY (is_reduction_operator_bitwise_or, TRUE);
  else if (ENTITY_BITWISE_AND_UPDATE_P (operator))
    OKAY(is_reduction_operator_bitwise_and, TRUE);
  else if (ENTITY_BITWISE_XOR_UPDATE_P (operator))
    OKAY(is_reduction_operator_bitwise_xor, TRUE);

  return FALSE;
}

/* Test if the operator is an unary update operator compatible with reduction
 * This also returns the corresponding tag
 * @return TRUE if the operator is an update operator compatible with reduction
 * @param operator, the operator (as an entity) to look at
 * @param op_tag, used to return the reduction operator (+, ...)
 */
static bool extract_reduction_unary_update_operator (entity operator,
						     tag*   op_tag)
{
  if (ENTITY_POST_INCREMENT_P(operator) ||
      ENTITY_POST_DECREMENT_P(operator) ||
      ENTITY_PRE_INCREMENT_P(operator) ||
      ENTITY_PRE_DECREMENT_P(operator))
    OKAY_WO_COMM (is_reduction_operator_sum);

  return FALSE;
}

/* @return TRUE if f is compatible with tag op.
 * @param f the entity to check.
 * @param op the tag to check against.
 * @param pcomm the commutative return value.
 */
static bool
reduction_function_compatible_p(
    entity f,
    tag op,
    bool *pcomm)
{
    tag nop;
    if (!function_reduction_operator_p(f, &nop, pcomm))
	return FALSE;
    return nop==op;
}

/***************************************************** FIND SAME REFERENCE */

/* looks for an equal reference in e, for reduction rop.
 * the reference found is also returned.
 * caution:
 * - for integers, / is *not* a valid reduction operator:-(
 *   this case is detected here...
 * static variables:
 * - refererence looked for fsr_ref,
 * - reduction operator tag fsr_op,
 * - returned reference fsr_found,
 */
static reference fsr_ref;
static reference fsr_found;
static tag fsr_op;

static bool fsr_reference_flt(reference r)
{
  pips_debug(7,"r: %s, fsr_ref: %s\n",
	     entity_name(reference_variable(r)),
	     entity_name(reference_variable(fsr_ref)));
   if (reference_equal_p(r, fsr_ref))
    {
	fsr_found = r;
	/* stop the recursion if does not need to check int div
	 */
	if (!basic_int_p(entity_basic(reference_variable(fsr_ref))) ||
	    (fsr_op!=is_reduction_operator_prod))
	    gen_recurse_stop(NULL);
    }

    return FALSE; /* no candidate refs within a ref! */
}

/* @return TRUE if the all operators are compatible with the detected reduction operator
 * @param c, the call to search for operators
 */
static bool fsr_call_flt(call c)
{
    bool comm;
    if (!reduction_function_compatible_p(call_function(c), fsr_op, &comm))
      return FALSE;
    /* else */
    if (!comm)
    {
      list /* of expression */ le = call_arguments(c);
      pips_assert("length is two", gen_length(le)==2);
      gen_recurse_stop(EXPRESSION(CAR(CDR(le))));
    }

    return TRUE;
}

static bool
equal_reference_in_expression_p(
    reference r,       /* looked for */
    expression e,      /* visited object */
    tag rop,           /* assumed reduction */
    bool red_up_op,
    reference *pfound) /* returned */
{
    fsr_ref = r;
    fsr_op  = rop;
    fsr_found = NULL;

    gen_multi_recurse(e,
		      call_domain, fsr_call_flt, gen_null,
		      reference_domain, fsr_reference_flt, gen_null,
		      NULL);

    *pfound = fsr_found;
    return (red_up_op || (fsr_found != NULL));
}

/******************************************************** NO OTHER EFFECTS */

/* checks that the references are the only touched within this statement.
 * I trust the proper effects to store all references...
 */
bool
no_other_effects_on_references (
    statement s,
    list /* of reference on the same variable */ lr)
{
    list /* of effect */ le;
    entity var;
    if (ENDP(lr)) return TRUE;

    le = effects_effects(load_proper_references(s));
    var = reference_variable(REFERENCE(CAR(lr)));

    pips_debug(7,"entity name: %s\n", entity_name(var));

    FOREACH (EFFECT, e, le) {
      reference r = effect_any_reference(e);
      if (!gen_in_list_p(r, lr) &&
	  entity_conflict_p(reference_variable(r), var))
	return FALSE;
      pips_debug(7,"refrence r: %p of entity: %s\n", r, entity_name (reference_variable(r)));
    }

    return TRUE;
}

/************************************************ EXTRACT PROPER REDUCTION */
/* This function look for a reduction and return it if found
 * mallocs are avoided if nothing is found...
 * looks for v = v OP y or v OP= y, where y is independent of v.
 * @return TRUE if the call in s a reduction
 * @param s,
 * @param c, the call to test for reduction
 * @param red, the reduction to return
 */
bool
call_proper_reduction_p (
    statement s,    /* needed to query about proper effects */
    call c,         /* the call of interest */
    reduction *red) /* the returned reduction (if any) */
{
  tag op; // The operator tag (sum, mul ...)
  list le = NIL; // list of expression
  list lr = NIL; // list of reference
  list lp = NIL; // list of Preference
  bool comm      = FALSE; // The commutatity operator flag
  bool assign_op = FALSE; // The assign operator flag
  bool update_op = FALSE; // The reduction update operator flag;
  bool unary_op  = FALSE; // The reduction unary update operator flag;
  entity fct = call_function(c); // the call function to test for reduction
  reference  lhs   = reference_undefined;
  reference  other = reference_undefined;
  expression elhs  = expression_undefined;
  expression erhs  = expression_undefined;

  pips_debug(7, "call to %s (%p)\n", entity_name(call_function(c)), s);

  // First init the operation flags
  // only check the operator type if the previous test failed
  if ((assign_op = ENTITY_ASSIGN_P (fct)) == FALSE)
    if ((update_op=extract_reduction_update_operator(fct, &op, &comm))==FALSE)
      unary_op = extract_reduction_unary_update_operator (fct, &op);

  // if no suitable operator have been found : return false
  if ((unary_op == FALSE) && (update_op == FALSE) && (assign_op == FALSE))
    return FALSE;

  // get the left and rigth operand
  le = call_arguments(c);
  elhs = EXPRESSION(CAR(le));
  //no right operand for unary operator
  if (unary_op == FALSE) erhs = EXPRESSION(CAR(CDR(le)));
  if (syntax_reference_p(expression_syntax(elhs)) == FALSE) {
    pips_user_warning ("not handeled case, no reduction will be detected\n");
    return FALSE;
  }
  lhs = syntax_reference(expression_syntax(elhs));

  // the lhs and rhs (if exits) must be functionnal
  // (same location on different evaluations)
  if (!functional_object_p((gen_chunk *) lhs)	||
      ((unary_op == FALSE) && !functional_object_p((gen_chunk *) erhs)))
    return FALSE;
  pips_debug(8, "lhs and rhs are functional\n");

  // Check that the operation performed is valid for a reduction,
  // The check is useless for reduction update and unary operator because
  // already done previously by "extract_reduction_update_operator" and
  // "extract_reduction_unary_update_operator"
  if ((unary_op == FALSE) && (update_op == FALSE) &&
      (extract_reduction_operator(erhs, &op, &comm) == FALSE))
    return FALSE;
  pips_debug(8, "reduction operator %s\n", reduction_operator_tag_name(op));

  // there should be another direct reference to lhs if not unary
  // !!! syntax is a call if extract_reduction_operator returned TRUE
  if (unary_op == FALSE) {
    if (!equal_reference_in_expression_p(lhs, erhs, op, update_op, &other))
      return FALSE;
    pips_debug(8, "matching reference found (%p)\n", other);
  }

  // build the list of found reference to the reduced variable
  if ((update_op == TRUE) || (unary_op == TRUE))
    lr = CONS(REFERENCE, lhs, NIL);
  else
    lr = CONS(REFERENCE, lhs, CONS(REFERENCE, other, NIL));
  pips_debug(7,"list lr is: %p and %p\n", other, lhs);
  // there should be no extra effects on the reduced variable
  if (!no_other_effects_on_references (s, lr)) {
    gen_free_list(lr);
    return FALSE;
  }
  pips_debug(8, "no other effects\n");

  FOREACH (REFERENCE, r, lr) {
    lp = CONS(PREFERENCE, make_preference(r), lp);
  }
  gen_free_list(lr), lr = NIL;

  // well, it is ok for a reduction now!
  *red = make_reduction(copy_reference(lhs),
			make_reduction_operator(op, UU),
			referenced_variables(lhs),
			lp);

  DEBUG_REDUCTION(7, "returning\n", *red);
  return TRUE;
}
