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
bool reduction_star_p(reduction r) {
    reference ref = reduction_reference(r);
    FOREACH(EXPRESSION,exp,reference_indices(ref))
        if( syntax_range_p(expression_syntax(exp)) ) {
            range ran = syntax_range(expression_syntax(exp));
            if(expression_undefined_p(range_upper(ran)))
                return true;
        }
    return false;
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
    return true;
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
	return false;
    }
    return true;
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
    bool updated = false;

    pips_debug(7, "reduction %s[%s] under effect %s on %s\n",
	       reduction_operator_tag_name
	          (reduction_operator_tag(reduction_op(red))),
	       entity_name(reduction_variable(red)),
	       effect_write_p(eff)? "W": "R",
	       entity_name(effect_variable(eff)));

    if(!store_effect_p(eff)) {
      return true;
    }


    /* REDUCTION is dead if the reduction variable is affected
     */
    if (entities_may_conflict_p(reduction_variable(red),var))
    {
	reduction_operator_tag(reduction_op(red)) =
	    is_reduction_operator_none;
	return false;
    }
    /* else */

    if (effect_read_p(eff)) return true;

    /* now var is written */
    FOREACH (ENTITY, e, reduction_dependences(red)) {
      if (entities_may_conflict_p(var, e)) {
	updated = true;
	remove_variable_from_reduction(red, e);
      }
    }

    if (updated)
    {
	gen_free_list(reduction_dependences(red));
	reduction_dependences(red) =
	     referenced_variables(reduction_reference(red));
    }

    return true;
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
      return true;
    }
    else if (entities_may_conflict_p(red_var, var))
      return false; /* I will not combine them... */
  }
  return true;
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
	return false;
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
    return true;
}

/* update *pr according to r for variable var
 * r is not touched.
 */
bool update_compatible_reduction_with(reduction *pr, entity var, reduction r) {
  if(reduction_variable(r) != var)
    return !entities_may_conflict_p(var, reduction_variable(r));

  /* else same var and no conflict */
  if(reduction_none_p(*pr)) {
    free_reduction(*pr);
    *pr = copy_reduction(r);
    return true;
  }
  /* else are they compatible?
   */
  if(reduction_tag(*pr) != reduction_tag(r))
    return false;
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
	return false;

    if (found)
    {
      if (!reduction_none_p(*pr)) /* some reduction already available */
	return merge_two_reductions(*pr, found);
      else { /* must update the reduction with the encountered effects */
	FOREACH (ENTITY, e,reduction_dependences(*pr)) {
	  remove_variable_from_reduction(found, e);
	}
	free_reduction(*pr); *pr = found;
	return true;
      }
    }
    /* else
     * now no new reduction waas found, must check *pr against effects
     */
    if (!reduction_none_p(*pr)) /* some reduction */
      {
	FOREACH (EFFECT, e, le) {
    if(!store_effect_p(e)) continue;
	  if (!update_reduction_under_effect(*pr, e)) {
	    DEBUG_REDUCTION(8, "kill of ", *pr);
	    pips_debug(8, "under effect to %s\n",
		       entity_name(effect_variable(e)));
	    return false;
	  }
	}
    }
    else
    {
      FOREACH (EFFECT, e, le) {
        if(!store_effect_p(e)) continue;
	if (entities_may_conflict_p(effect_variable(e), var))
	  return false;
	else if (effect_write_p(e)) /* stores for latter cleaning */
	  reduction_dependences(*pr) = gen_once(effect_variable(e),
						reduction_dependences(*pr));
      }
    }
    return true;
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
      return true;
    /* else */

    if (entity_module_p(f))
    {
      FOREACH (EFFECT, e, load_summary_effects(f)) {
        if(!store_effect_p(e)) continue;
        if (effect_write_p(e)) /* a side effect!? */
          return false;
        if (io_effect_entity_p(effect_variable(e))) /* LUNS */
          return false;
      }
    }

    return true;
}

/* tells whether r is a functional reference...
 * actually I would need to recompute somehow the proper effects of obj?
 */
static bool is_functional;
static bool call_flt(call c)
{
    if (pure_function_p(call_function(c)))
	return true;
    /* else */
    is_functional = false;
    gen_recurse_stop(NULL);
    return false;
}
static bool
functional_object_p(gen_chunk* obj)
{
    is_functional = true;
    gen_recurse(obj, call_domain, call_flt, gen_null);
    return is_functional;
}

/****************************************************** REDUCTION OPERATOR */

#define OKAY(op,com) do { *op_tag=op; *commutative=com; return true;} while(false)
#define OKAY_WO_COMM(op) do { *op_tag=op; return true;} while(false)
/* tells whether entity f is a reduction operator function
 * also returns the corresponding tag, and if commutative
 * @return true if the entity f is a reduction operator function
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
      OKAY(is_reduction_operator_sum, true);
    else if (ENTITY_MINUS_P(f))
      OKAY(is_reduction_operator_sum, false);
    else if (ENTITY_MULTIPLY_P(f))
      OKAY(is_reduction_operator_prod, true);
    else if (ENTITY_DIVIDE_P(f))
      OKAY(is_reduction_operator_prod, false);
    else if (ENTITY_MIN_P(f) || ENTITY_MIN0_P(f))
      OKAY(is_reduction_operator_min, true);
    else if (ENTITY_MAX_P(f) || ENTITY_MAX0_P(f))
      OKAY(is_reduction_operator_max, true);
    else if (ENTITY_AND_P(f))
      OKAY(is_reduction_operator_and, true);
    else if (ENTITY_OR_P(f))
      OKAY(is_reduction_operator_or, true);
    else if (ENTITY_BITWISE_AND_P(f))
      OKAY(is_reduction_operator_bitwise_and, true);
    else if (ENTITY_BITWISE_OR_P(f))
      OKAY(is_reduction_operator_bitwise_or, true);
    else if (ENTITY_BITWISE_XOR_P(f))
      OKAY(is_reduction_operator_bitwise_xor, true);
    else if (ENTITY_EQUIV_P(f))
      OKAY(is_reduction_operator_eqv, true);
    else if (ENTITY_NON_EQUIV_P(f))
      OKAY(is_reduction_operator_neqv, true);
    else
      return false;
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

    if (!syntax_call_p(s)) return false;
    c = syntax_call(s);
    f = call_function(c);
    return function_reduction_operator_p(f, op_tag, commutative);
}

/* Test if the operator is an update operator compatible with reduction
 * This also returns the corresponding tag, and if commutative
 * @return true if the operator is an update operator compatible with reduction
 * @param operator, the operator (as an entity) to look at
 * @param op_tag, used to return the reduction operator (+, * ...)
 * @param commutative, used to return the operator commutativity
 */
static bool extract_reduction_update_operator (entity operator,
					       tag*   op_tag,
					       bool*  commutative)
{
  if (ENTITY_PLUS_UPDATE_P (operator))
    OKAY(is_reduction_operator_sum, true);
  else if (ENTITY_MINUS_UPDATE_P (operator))
    OKAY(is_reduction_operator_sum, true);
  else if (ENTITY_MULTIPLY_UPDATE_P (operator))
    OKAY(is_reduction_operator_prod, true);
  else if (ENTITY_DIVIDE_UPDATE_P (operator))
    OKAY(is_reduction_operator_prod, true);
  else if (ENTITY_BITWISE_OR_UPDATE_P (operator))
    OKAY (is_reduction_operator_bitwise_or, true);
  else if (ENTITY_BITWISE_AND_UPDATE_P (operator))
    OKAY(is_reduction_operator_bitwise_and, true);
  else if (ENTITY_BITWISE_XOR_UPDATE_P (operator))
    OKAY(is_reduction_operator_bitwise_xor, true);

  return false;
}

/* Test if the operator is an unary update operator compatible with reduction
 * This also returns the corresponding tag
 * @return true if the operator is an update operator compatible with reduction
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

  return false;
}

/* @return true if f is compatible with tag op.
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
	return false;
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

    return false; /* no candidate refs within a ref! */
}

/* @return true if the all operators are compatible with the detected reduction operator
 * @param c, the call to search for operators
 */
static bool fsr_call_flt(call c)
{
    bool comm;
    if (!reduction_function_compatible_p(call_function(c), fsr_op, &comm))
      return false;
    /* else */
    if (!comm)
    {
      list /* of expression */ le = call_arguments(c);
      pips_assert("length is two", gen_length(le)==2);
      gen_recurse_stop(EXPRESSION(CAR(CDR(le))));
    }

    return true;
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
    if (ENDP(lr)) return true;

    le = effects_effects(load_proper_references(s));
    var = reference_variable(REFERENCE(CAR(lr)));

    pips_debug(7,"entity name: %s\n", entity_name(var));

    FOREACH (EFFECT, e, le) {
      if(!store_effect_p(e)) continue;
      reference r = effect_any_reference(e);
      if (!gen_in_list_p(r, lr) && store_effect_p(e) &&
          entities_may_conflict_p(reference_variable(r), var)) {
        pips_debug(7,"Effect may touch variable : ");print_effect(e);
        return false;
      }
      pips_debug(7,"refrence r: %p of entity: %s\n", r, entity_name (reference_variable(r)));
    }

    return true;
}

/************************************************ EXTRACT PROPER REDUCTION */
/* This function look for a reduction and return it if found
 * mallocs are avoided if nothing is found...
 * looks for v = v OP y or v OP= y, where y is independent of v.
 * @return true if the call in s a reduction
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
  bool comm      = false; // The commutatity operator flag
  bool assign_op = false; // The assign operator flag
  bool update_op = false; // The reduction update operator flag;
  bool unary_op  = false; // The reduction unary update operator flag;
  entity fct = call_function(c); // the call function to test for reduction
  reference  lhs   = reference_undefined;
  reference  other = reference_undefined;
  expression elhs  = expression_undefined;
  expression erhs  = expression_undefined;

  pips_debug(7, "call to %s (%p)\n", entity_name(call_function(c)), s);

  // First init the operation flags
  // only check the operator type if the previous test failed
  if ((assign_op = ENTITY_ASSIGN_P (fct)) == false)
    if ((update_op=extract_reduction_update_operator(fct, &op, &comm))==false)
      unary_op = extract_reduction_unary_update_operator (fct, &op);

  // if no suitable operator has been found : return false
  if ((unary_op == false) && (update_op == false) && (assign_op == false)) {
    pips_debug(5,"No unary, nor update, no assign !\n");
    return false;
  }

  // get the left and right operands
  le = call_arguments(c);
  elhs = EXPRESSION(CAR(le));
  //no right operand for unary operator
  if (unary_op == false) erhs = EXPRESSION(CAR(CDR(le)));
  if (syntax_reference_p(expression_syntax(elhs)) == false) {
    pips_user_warning ("The left hand side of assignment is not a reference, "
        "this is not handled and no reduction will be detected\n");
    return false;
  }
  lhs = syntax_reference(expression_syntax(elhs));

  // the lhs and rhs (if exits) must be functionnal
  // (same location on different evaluations)
  if (!functional_object_p((gen_chunk *) lhs)	||
      ((unary_op == false) && !functional_object_p((gen_chunk *) erhs))) {
    pips_debug(5,"Lhs or Rhs not functional !\n");
    return false;
  }
  pips_debug(8, "lhs and rhs are functional\n");

  // Check that the operation performed is valid for a reduction,
  // The check is useless for reduction update and unary operator because
  // already done previously by "extract_reduction_update_operator" and
  // "extract_reduction_unary_update_operator"
  if ((unary_op == false) && (update_op == false) &&
      (extract_reduction_operator(erhs, &op, &comm) == false)) {
    pips_debug(5,"extract_reduction_operator returned false !!\n");
    return false;
  }
  pips_debug(8, "reduction operator %s\n", reduction_operator_tag_name(op));

  // there should be another direct reference to lhs if not unary
  // !!! syntax is a call if extract_reduction_operator returned TRUE
  if (unary_op == false) {
    if (!equal_reference_in_expression_p(lhs, erhs, op, update_op, &other)) {
      pips_debug(5,"!equal_reference_in_expression_p !!\n");
      return false;
    }
    pips_debug(8, "matching reference found (%p)\n", other);
  }

  // build the list of found reference to the reduced variable
  if ((update_op == true) || (unary_op == true))
    lr = CONS(REFERENCE, lhs, NIL);
  else
    lr = CONS(REFERENCE, lhs, CONS(REFERENCE, other, NIL));
  pips_debug(7,"list lr is: %p and %p\n", other, lhs);
  // there should be no extra effects on the reduced variable
  if (!no_other_effects_on_references (s, lr)) {
    pips_debug(5,"Other effects on references !!\n");
    gen_free_list(lr);
    return false;
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
  return true;
}


/**
 * Return the "other part" of the reduction. If the statement is :
 * sum = sum + a[i];
 * then we'll return a[i]
 */
expression get_complement_expression(statement s, reference reduced) {
  expression complement = expression_undefined;
  // We handle only trivial cases
  if(statement_call_p(s)) {
    call c = statement_call(s);
    entity fct = call_function(c);
    list le = call_arguments(c);
    tag op;
    bool comm = false; // Commutativity
    // Differentiate unary && update && binary operators
    if (ENTITY_ASSIGN_P (fct)){
      // Normal case, binary expected
      expression rhs = EXPRESSION(CAR(CDR(le)));
      if(expression_call_p(rhs)) {
        list args = call_arguments(expression_call(rhs));
        pips_assert("Have a binary operator\n", gen_length(args)==2);
        /* try to find out which of the two expression corresponds to the
         * reduced reference. We'll return the other...
         */
        expression e1 = EXPRESSION(CAR(args));
        expression e2 = EXPRESSION(CAR(CDR(args)));
        if(expression_reference_p(e1) &&
            same_ref_name_p(reduced,expression_reference(e1))) {
          complement = e2;
        } else if(expression_reference_p(e2) &&
            same_ref_name_p(reduced,expression_reference(e2))) {
          complement = e2;
        } else {
          pips_user_warning("Atomic operation replacement seems less general"
              " than reduction detection. This merits a bug report !\n");
        }
      } else {
        pips_user_warning("Atomic operation replacement seems less general than"
            " reduction detection. This merits a bug report !\n");
      }
    } else if(extract_reduction_unary_update_operator(fct, &op)) {
      // ++ or --  : we return 1
      pips_debug(3,"We have an unary operator\n");
      complement = int_to_expression(1);
    } else if(extract_reduction_update_operator(fct, &op, &comm)) {
      // += or similar, return directly rhs
      pips_debug(3,"We have an update operator\n");
      complement = EXPRESSION(CAR(CDR(le)));
    } else {
      pips_internal_error("We have a reduction, but the statement is neither an"
          " unary nor a binary ? It's a bug, sorry no choice but abort !\n");
    }
  }
  return complement;
}

