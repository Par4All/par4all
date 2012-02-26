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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

/*
 * This file contains a set of functions to evaluate store independent
 * expressions, especially integer expressions, which are key to loop
 * parallelization.
 *
 * The key functions are:
 *
 * value EvalExpression(expression)
 *
 * bool expression_integer_value(expression, intptr_t *)
 *
 * bool expression_negative_integer_value_p(expression)
 *
 * bool positive_expression_p()
 *
 * bool negative_expression_p()
 *
 * expression range_to_expression()
 *
 * bool range_count()
 *
 * A few misplaced functions are here too: expression_linear_p(),
 * ipow(), vect_const_p(), vect_product()
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "misc.h"

#include "ri-util.h"
#include "properties.h"

#include "operator.h"

/* Evaluate statically an expression. If the expression can be
 * evaluated regardeless of the store and regardless of the target
 * machine, and if the expression type is integer or float, return a
 * constant value. Else, return a value unknown.
 *
 * To accept dependencies on target architecture, set the property
 * "EVAL_SIZEOF" to true.
 *
 * A new value is always allocated.
 *
 * If you are not interested in the object value, but simply to the
 * integer value of a constant expression, see
 * expression_integer_value().
 *
 * This function is not fully implemented. It does not take care of
 * logical (apparently) and string expressions. Some intrinsic
 * operators are not evaluated. Floating point expressions are not as
 * well covered as integer expression.
 *
 * It is not clear if the evaluation of floating point expressions
 * should be considered target dependent (e.g. for GPU code) or if the
 * IEEE floating point standard is considered strongly enforced.
 *
 * If information about the store used to evaluate the expression,
 * i.e. preconditions, use precondition_minmax_of_expression()
 *
 * The algorithm is built on a recursive analysis of the
 * expression structure. Lower level functions are called until basic atoms
 * are reached. The success of basic atom evaluation depend on the atom
 * type:
 *
 * reference: right now, the evaluation fails because we do not compute
 * predicates on variables, unless the variable type is qualified by const.
 *
 * call: a call to a user function is not evaluated. a call to an intrinsic
 * function is successfully evaluated if arguments can be evaluated. a call
 * to a constant function is evaluated if its basic type is integer.
 *
 * range: a range is not evaluated.
 */
value EvalExpression(expression e)
{
    return EvalSyntax(expression_syntax(e));
}

value EvalSyntax(syntax s)
{
  value v;

  switch (syntax_tag(s)) {
  case is_syntax_reference: {
    /* Is it a reference to a const variable? */
    reference r = syntax_reference(s);
    entity var = reference_variable(r);
    if(const_variable_p(var)) {
      value i = entity_initial(var);
      if(value_constant_p(i))
	v = copy_value(entity_initial(var));
      else if(value_expression_p(i)) {
	expression ie = value_expression(i);
	v = EvalExpression(ie);
      }
      else
	v = make_value_unknown();
    }
    else
      v = make_value_unknown();
    break;
  }
  case is_syntax_range:
    v = make_value_unknown();
    break;
  case is_syntax_call:
    v = EvalCall((syntax_call(s)));
    break;
  case is_syntax_cast:
    v = make_value_unknown();
    break;
  case is_syntax_sizeofexpression:
    /* SG: sizeof is architecture dependant, it is better not to
       evaluate it by default */
    if(get_bool_property("EVAL_SIZEOF"))
        v = EvalSizeofexpression((syntax_sizeofexpression(s)));
    else
        v = make_value_unknown();
    break;
  case is_syntax_subscript:
  case is_syntax_application:
  case is_syntax_va_arg:
    v = make_value_unknown();
    break;
  default:
    fprintf(stderr, "[EvalExpression] Unexpected default case %d\n",
	    syntax_tag(s));
    abort();
  }

  return v;
}

/* only calls to constant, symbolic or intrinsic functions might be
 * evaluated. recall that intrinsic functions are not known.
 */
value EvalCall(call c)
{
  value vout, vin;
  entity f;

  f = call_function(c);
  vin = entity_initial(f);

  if (value_undefined_p(vin))
    pips_internal_error("undefined value for %s", entity_name(f));

  switch (value_tag(vin)) {
  case is_value_intrinsic:
    vout = EvalIntrinsic(f, call_arguments(c));
    break;
  case is_value_constant:
    vout = EvalConstant((value_constant(vin)));
    break;
  case is_value_symbolic:
    /* SG: it may be better not to evaluate symbolic and keep their symbolic name */
    if(get_bool_property("EVAL_SYMBOLIC_CONSTANT"))
      vout = EvalConstant((symbolic_constant(value_symbolic(vin))));
    else
      vout = make_value_unknown();
    break;
  case is_value_unknown:
    /* it might be an intrinsic function */
    vout = EvalIntrinsic(f, call_arguments(c));
    break;
  case is_value_code:
    vout = make_value_unknown();
    break;
  default:
    pips_internal_error("Unexpected default case.");
  }

  return(vout);
}

value EvalSizeofexpression(sizeofexpression soe)
{
  type t = type_undefined;
  value v = value_undefined;
  _int i;

  if(sizeofexpression_expression_p(soe)) {
    expression e = sizeofexpression_expression(soe);

    t = expression_to_type(e);
  }
  else {
    t = sizeofexpression_type(soe);
  }

  i = type_memory_size(t);
  v = make_value_constant(make_constant_int(i));

  if(sizeofexpression_expression_p(soe))
    free_type(t);

  return v;
}

/* Constant c is returned as field of value v. */
value EvalConstant(constant c)
{
  value v = value_undefined;

  if(constant_int_p(c)) {
    v = make_value(is_value_constant,
		   make_constant(is_constant_int,
				 (void*) constant_int(c)));
  }
  else if(constant_float_p(c)) {
    constant nc = make_constant(is_constant_float,
				NULL);
    constant_float(nc) = constant_float(c);
    v = make_value(is_value_constant, nc);
  }
  else if(constant_call_p(c)) {
    entity e = constant_call(c);
    type t = entity_type(e);
    if(type_functional_p(t)) {
      functional f = type_functional(t);
      t = functional_result(f);
    }
    if(scalar_integer_type_p(t)) {
      long long int val;
      sscanf(entity_local_name(e), "%lld", &val);
      v = make_value(is_value_constant,
		     make_constant(is_constant_int,
				   (void*) val));
    }
    else if(float_type_p(t)) {
      double val;
      sscanf(entity_local_name(e), "%lg", &val);
      constant nc = make_constant(is_constant_float,
				 NULL);
      constant_float(nc) = val;
      v = make_value(is_value_constant, nc);
    }
    else
      v = make_value(is_value_constant,
		     make_constant(is_constant_litteral, NIL));
  }
  else
    v = make_value(is_value_constant,
		   make_constant(is_constant_litteral, NIL));
  return v;
}

/* This function tries to evaluate a call to an intrinsic function.
 * right now, we only try to evaluate unary and binary intrinsic
 * functions, ie. Fortran operators.
 *
 * e is the intrinsic function.
 *
 * la is the list of arguments.
 */
value EvalIntrinsic(entity e, list la)
{
  value v;
  int token;

  if ((token = IsUnaryOperator(e)) > 0)
    v = EvalUnaryOp(token, la);
  else if ((token = IsBinaryOperator(e)) > 0)
    v = EvalBinaryOp(token, la);
  else if ((token = IsNaryOperator(e)) > 0)
    v = EvalNaryOp(token, la);
  else if (ENTITY_CONDITIONAL_P(e))
    v = EvalConditionalOp(la);
  else
    v = make_value(is_value_unknown, NIL);

  return(v);
}

value EvalConditionalOp(list la)
{
  value vout, v1, v2, v3;
  _int arg1 = 0, arg2 = 0, arg3 = 0;
  bool failed = false;

  pips_assert("Three arguments", gen_length(la)==3);

  v1 = EvalExpression(EXPRESSION(CAR(la)));
  if (value_constant_p(v1) && constant_int_p(value_constant(v1)))
    arg1 = constant_int(value_constant(v1));
  else
    failed = true;

  v2 = EvalExpression(EXPRESSION(CAR(CDR(la))));
  if (value_constant_p(v2) && constant_int_p(value_constant(v2)))
    arg2 = constant_int(value_constant(v2));
  else
    failed = true;

  v3 = EvalExpression(EXPRESSION(CAR(CDR(CDR(la)))));
  if (value_constant_p(v3) && constant_int_p(value_constant(v3)))
    arg3 = constant_int(value_constant(v3));
  else
    failed = true;

  if(failed)
    vout = make_value(is_value_unknown, NIL);
  else
    vout = make_value(is_value_constant,
		      make_constant(is_constant_int, (void *) (arg1? arg2: arg3)));

  free_value(v1);
  free_value(v2);
  free_value(v3);

  return vout;
}


value EvalUnaryOp(int t, list la)
{
  value vout, v;
  int arg;

  assert(la != NIL);
  v = EvalExpression(EXPRESSION(CAR(la)));
  if (value_constant_p(v) && constant_int_p(value_constant(v)))
    arg = constant_int(value_constant(v));
  else
    return(v);

  if (t == MINUS) {
    constant_int(value_constant(v)) = -arg;
    vout = v;
  }
  else if (t == PLUS) {
    constant_int(value_constant(v)) = arg;
    vout = v;
  }
  else if (t == NOT) {
    constant_int(value_constant(v)) = arg!=0;
    vout = v;
  }
  else {
    free_value(v);
    vout = make_value(is_value_unknown, NIL);
  }

  return(vout);
}

/* t defines the operator and la is a list to two sub-expressions. 
 *
 * Integer and floatint point constants are evaluated.
*/
value EvalBinaryOp(int t, list la)
{
  value v;
  long long int i_arg_l, i_arg_r;
  double f_arg_l, f_arg_r;
  bool int_p = true;

  pips_assert("non empty list", la != NIL);

  v = EvalExpression(EXPRESSION(CAR(la)));
  if (value_constant_p(v) && constant_int_p(value_constant(v))) {
    i_arg_l = constant_int(value_constant(v));
    f_arg_l = i_arg_l;
    free_value(v);
  }
  else if (value_constant_p(v) && constant_float_p(value_constant(v))) {
    int_p = false;
    f_arg_l = constant_float(value_constant(v));
  }
  else
    return(v);

  la = CDR(la);

  pips_assert("non empty list", la != NIL);
  v = EvalExpression(EXPRESSION(CAR(la)));

  if (value_constant_p(v) && constant_int_p(value_constant(v))) {
    i_arg_r = constant_int(value_constant(v));
    f_arg_r = i_arg_r;
  }
  else if (value_constant_p(v) && constant_float_p(value_constant(v))) {
    f_arg_r = constant_float(value_constant(v));
    int_p = false;
  }
  else
    return(v);

  switch (t) {
  case MINUS:
    if(int_p)
      constant_int(value_constant(v)) = i_arg_l-i_arg_r;
    else
      constant_float(value_constant(v)) = i_arg_l-i_arg_r;
    break;
  case PLUS:
    if(int_p)
      constant_int(value_constant(v)) = i_arg_l+i_arg_r;
    else
      constant_float(value_constant(v)) = f_arg_l+f_arg_r;
    break;
  case STAR:
    if(int_p)
      constant_int(value_constant(v)) = i_arg_l*i_arg_r;
    else
      constant_float(value_constant(v)) = f_arg_l*f_arg_r;
    break;
  case SLASH:
    if(int_p) {
      if (i_arg_r != 0)
	constant_int(value_constant(v)) = i_arg_l/i_arg_r;
      else {
	pips_user_error("[EvalBinaryOp] zero divide\n");
      }
    }
    else {
      if (f_arg_r != 0)
	constant_float(value_constant(v)) = f_arg_l/f_arg_r;
      else {
	pips_user_error("[EvalBinaryOp] zero divide\n");
      }
    }
    break;
  case MOD:
    if(int_p) {
      if (i_arg_r != 0)
	constant_int(value_constant(v)) = i_arg_l%i_arg_r;
      else {
	pips_user_error("[EvalBinaryOp] zero divide\n");
      }
    }
    else {
      if (f_arg_r != 0) {
	free_value(v);
	v = make_value(is_value_unknown, NIL);
      }
      else {
	pips_user_error("[EvalBinaryOp] zero divide\n");
      }
    }
    break;
  case POWER:
    if(int_p) {
      if (i_arg_r >= 0)
	constant_int(value_constant(v)) = ipow(i_arg_l,i_arg_r);
      else {
	free_value(v);
	v = make_value(is_value_unknown, NIL);
      }
    }
    else {
      /* FI: lazy... */
      free_value(v);
      v = make_value(is_value_unknown, NIL);
    }
    break;
    /*
     * Logical operators should return logical values...
     */
  case EQ:
    if(int_p)
      constant_int(value_constant(v)) = i_arg_l==i_arg_r;
    else
      constant_int(value_constant(v)) = f_arg_l==f_arg_r;
    break;
  case NE:
    if(int_p)
      constant_int(value_constant(v)) = i_arg_l!=i_arg_r;
    else
      constant_int(value_constant(v)) = f_arg_l!=f_arg_r;
    break;
  case EQV:
    if(int_p)
      constant_int(value_constant(v)) = i_arg_l==i_arg_r;
    else
      constant_int(value_constant(v)) = f_arg_l==f_arg_r;
    break;
  case NEQV:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l!=i_arg_r;
    else
       constant_int(value_constant(v)) = f_arg_l!=f_arg_r;
   break;
  case GT:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l>i_arg_r;
    else
       constant_int(value_constant(v)) = f_arg_l>f_arg_r;
   break;
  case LT:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l<i_arg_r;
    else
       constant_int(value_constant(v)) = f_arg_l<f_arg_r;
   break;
  case GE:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l>=i_arg_r;
    else
       constant_int(value_constant(v)) = f_arg_l>=f_arg_r;
   break;
   /* OK for Fortran Logical? Int value or logical value? */
  case OR:
    if(int_p)
       constant_int(value_constant(v)) = (i_arg_l!=0)||(i_arg_r!=0);
    else
       constant_int(value_constant(v)) = (f_arg_l!=0)||(f_arg_r!=0);
   break;
  case AND:
    if(int_p)
       constant_int(value_constant(v)) = (i_arg_l!=0)&&(i_arg_r!=0);
    else
       constant_int(value_constant(v)) = (f_arg_l!=0)&&(f_arg_r!=0);
   break;
  case BITWISE_OR:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l|i_arg_r;
    else
      pips_user_error("Bitwise or cannot have floating point arguments\n");
   break;
  case BITWISE_AND:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l&i_arg_r;
    else
      pips_user_error("Bitwise and cannot have floating point arguments\n");
   break;
  case BITWISE_XOR:
    if(int_p)
      constant_int(value_constant(v)) = i_arg_l^i_arg_r;
    else
      pips_user_error("Bitwise xor cannot have floating point arguments\n");
   break;
  case LEFT_SHIFT:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l<<i_arg_r;
    else {
      free_value(v);
      v = make_value(is_value_unknown, NIL);
    }
   break;
   case RIGHT_SHIFT:
    if(int_p)
       constant_int(value_constant(v)) = i_arg_l>>i_arg_r;
    else {
      free_value(v);
      v = make_value(is_value_unknown, NIL);
    }
   break;
 default:
    free_value(v);
    v = make_value(is_value_unknown, NIL);
  }

  return(v);
}

value EvalNaryOp(int t, list la)
{
    value v = value_undefined;
    value w = value_undefined;
    int new_arg = 0;
    bool first_arg_p = true;

    /* 2 operands at least are needed */
    assert(la != NIL && CDR(la) != NIL);

    MAP(EXPRESSION, e, {
	v = EvalExpression(e);
	if (value_constant_p(v) && constant_int_p(value_constant(v))) {
	    new_arg = constant_int(value_constant(v));
	    if (first_arg_p) {
		first_arg_p = false;
		w = v;
	    }
	    else {
		switch(t) {
		case MAXIMUM:
		    constant_int(value_constant(w))= MAX(constant_int(value_constant(w)),
							 new_arg);
		    break;
		case MINIMUM:
		    constant_int(value_constant(w))= MIN(constant_int(value_constant(w)),
							 new_arg);
		    break;
		default:
		    return v;
		}
		free_value(v);
	    }
	}
	else
	    return(v);
    }, la);

    return(w);
}

int IsUnaryOperator(entity e)
{
  int token;
  const char* n = entity_local_name(e);

  if (same_string_p(n, UNARY_MINUS_OPERATOR_NAME))
    token = MINUS;
  else if (same_string_p(n, UNARY_PLUS_OPERATOR_NAME))
    token = PLUS;
  else if (same_string_p(n, NOT_OPERATOR_NAME)
	   || same_string_p(n, C_NOT_OPERATOR_NAME))
    token = NOT;
  else if (same_string_p(n, POST_INCREMENT_OPERATOR_NAME))
    token = POST_INCREMENT;
  else if (same_string_p(n, POST_DECREMENT_OPERATOR_NAME))
    token = POST_DECREMENT;
  else if (same_string_p(n, PRE_INCREMENT_OPERATOR_NAME))
    token = PRE_INCREMENT;
  else if (same_string_p(n, PRE_DECREMENT_OPERATOR_NAME))
    token = PRE_DECREMENT;
  else
    token = -1;

  return(token);
}

/* FI: These string constants are defined in ri-util.h and the tokens
   in ri-util/operator.h */
int IsBinaryOperator(entity e)
{
  int token;
  const char* n = entity_local_name(e);

  if      (same_string_p(n, MINUS_OPERATOR_NAME)
	   || same_string_p(n, MINUS_C_OPERATOR_NAME))
    token = MINUS;
  else if (same_string_p(n, PLUS_OPERATOR_NAME)
	   || same_string_p(n, PLUS_C_OPERATOR_NAME))
    token = PLUS;
  else if (same_string_p(n, MULTIPLY_OPERATOR_NAME))
    token = STAR;
  else if (same_string_p(n, DIVIDE_OPERATOR_NAME))
    token = SLASH;
  else if (same_string_p(n, POWER_OPERATOR_NAME))
    token = POWER;
  else if (same_string_p(n, MODULO_OPERATOR_NAME)
	   || same_string_p(n, C_MODULO_OPERATOR_NAME))
    token = MOD;
  else if (same_string_p(n, EQUAL_OPERATOR_NAME)
	   || same_string_p(n, C_EQUAL_OPERATOR_NAME))
    token = EQ;
  else if (same_string_p(n, NON_EQUAL_OPERATOR_NAME)
	   || same_string_p(n, C_NON_EQUAL_OPERATOR_NAME))
    token = NE;
  else if (same_string_p(n, MODULO_OPERATOR_NAME)
	   || same_string_p(n, C_MODULO_OPERATOR_NAME))
    token = EQV;
  else if (same_string_p(n, EQUIV_OPERATOR_NAME))
    token = NEQV;
  else if (same_string_p(n, GREATER_THAN_OPERATOR_NAME)
	   || same_string_p(n, C_MODULO_OPERATOR_NAME))
    token = GT;
  else if (same_string_p(n, LESS_THAN_OPERATOR_NAME)
	   || same_string_p(n, C_LESS_THAN_OPERATOR_NAME))
    token = LT;
  else if (same_string_p(n, GREATER_OR_EQUAL_OPERATOR_NAME)
	   || same_string_p(n, C_GREATER_OR_EQUAL_OPERATOR_NAME))
    token = GE;
  else if (same_string_p(n, LESS_OR_EQUAL_OPERATOR_NAME)
	   || same_string_p(n, C_LESS_OR_EQUAL_OPERATOR_NAME))
    token = LE;
  else if (same_string_p(n, OR_OPERATOR_NAME)
	   || same_string_p(n, C_OR_OPERATOR_NAME))
    token = OR;
  else if (same_string_p(n, AND_OPERATOR_NAME)
	   || same_string_p(n, C_AND_OPERATOR_NAME))
    token = AND;
  else if (same_string_p(n, BITWISE_AND_OPERATOR_NAME))
    token = BITWISE_AND;
  else if (same_string_p(n, BITWISE_OR_OPERATOR_NAME))
    token = BITWISE_OR;
  else if (same_string_p(n, BITWISE_XOR_OPERATOR_NAME))
    token = BITWISE_XOR;
  else if (same_string_p(n, LEFT_SHIFT_OPERATOR_NAME))
    token = LEFT_SHIFT;
  else if (same_string_p(n, RIGHT_SHIFT_OPERATOR_NAME))
    token = RIGHT_SHIFT;

  else if (same_string_p(n, ASSIGN_OPERATOR_NAME)) // C operators
    token = ASSIGN;
  else if (same_string_p(n, MULTIPLY_UPDATE_OPERATOR_NAME))
    token = MULTIPLY_UPDATE;
  else if (same_string_p(n, DIVIDE_UPDATE_OPERATOR_NAME))
    token = DIVIDE_UPDATE;
  else if (same_string_p(n, PLUS_UPDATE_OPERATOR_NAME))
    token = PLUS_UPDATE;
  else if (same_string_p(n, MINUS_UPDATE_OPERATOR_NAME))
    token = MINUS_UPDATE;
  else if (same_string_p(n, LEFT_SHIFT_UPDATE_OPERATOR_NAME))
    token = LEFT_SHIFT_UPDATE;
  else if (same_string_p(n, RIGHT_SHIFT_UPDATE_OPERATOR_NAME))
    token = RIGHT_SHIFT_UPDATE;
  else if (same_string_p(n, BITWISE_OR_UPDATE_OPERATOR_NAME))
    token = BITWISE_OR_UPDATE;

  else if (same_string_p(entity_local_name(e), IMPLIED_COMPLEX_NAME) ||
	   same_string_p(entity_local_name(e), IMPLIED_DCOMPLEX_NAME))
    token = CAST_OP;
  else
    token = -1;

  return(token);
}

int IsNaryOperator(entity e)
{
	int token;

	if (strcmp(entity_local_name(e), MIN0_OPERATOR_NAME) == 0)
		token = MINIMUM;
	else if (strcmp(entity_local_name(e), MAX0_OPERATOR_NAME) == 0)
		token = MAXIMUM;
	else if (strcmp(entity_local_name(e), MIN_OPERATOR_NAME) == 0)
		token = MINIMUM;
	else if (strcmp(entity_local_name(e), MAX_OPERATOR_NAME) == 0)
		token = MAXIMUM;
	else
		token = -1;

	return token;
}

/* FI: such a function should exist in Linear/arithmetique
 *
 * FI: should it return a long long int?
 */
int 
ipow(int vg, int vd)
{
	int i = 1;

	assert(vd >= 0);

	while (vd-- > 0)
		i *= vg;

	return(i);
}


/*
 * evaluates statically the value of an integer expression.
 *
 * returns true if an integer value could be computed and placed in pval.
 *
 * returns false otherwise.
 *
 * Based on EvalExpression()
*/
bool
expression_integer_value(expression e, intptr_t * pval)
{
    bool is_int = false;
    value v = EvalExpression(e);

    if (value_constant_p(v) && constant_int_p(value_constant(v))) {
        *pval = constant_int(value_constant(v));
        is_int = true;
    }

    free_value(v);
    return is_int;
}


/* Return true iff the expression has an integer value known
 * statically and this value is negative. All leaves of the expression
 * tree must be constant integer.
 *
 * If preconditions are available, a more general expression can be
 * evaluated using precondition_minmax_of_expression().
 */
bool expression_negative_integer_value_p(expression e) {
  intptr_t v;
  return expression_integer_value(e, &v) && (v < 0);
}

/* Use constants and type information to decide if the value of
 * sigma(e) is always positive, e.g. >=0
 *
 * See negative_expression_p()
 *
 * Should we define positive_integer_expression_p() and check the expression type?
 */
bool positive_expression_p(expression e)
{
  bool positive_p = false; // In general, false because no conclusion can be reached
  intptr_t v;
  if(expression_integer_value(e, &v)) {
    positive_p = v>=0;
  }
  else {
    syntax s = expression_syntax(e);

    if(syntax_reference_p(s)) {
      entity v = reference_variable(syntax_reference(s));
      type t = ultimate_type(entity_type(v));
      positive_p = unsigned_type_p(t);
    }
    else if(syntax_call_p(s)) {
      call c = syntax_call(s);
      entity f = call_function(c);
      type t = entity_type(f);
      pips_assert("t is a functional type", type_functional_p(t));
      type rt = functional_result(type_functional(t));
      if(unsigned_type_p(rt))
	positive_p = true;
      else if(overloaded_type_p(rt)) { /* We assume an operator is used */
	/* We can check a few cases recursively... */
	list args = call_arguments(c);
	intptr_t l = gen_length(args);
	if(l==1) {
	  expression arg = EXPRESSION(CAR(args));
	  if(ENTITY_UNARY_MINUS_P(f)) {
	    positive_p = negative_expression_p(arg); // Maybe zero, but no chance for sigma(x)>0
	  }
	  else if(ENTITY_IABS_P(f) || ENTITY_ABS_P(f) || ENTITY_DABS_P(f)
		  || ENTITY_CABS_P(f)) {
	    positive_p = true;
	  }
	}
	else if(l==2) {
	  expression arg1 = EXPRESSION(CAR(args));
	  expression arg2 = EXPRESSION(CAR(CDR(args)));
	  if(ENTITY_MINUS_P(f)) {
	    positive_p = positive_expression_p(arg1) && negative_expression_p(arg2);
	  }
	  else if(ENTITY_PLUS_P(f)) {
	    positive_p = positive_expression_p(arg1) && positive_expression_p(arg2);
	  }
	  else if(ENTITY_MULTIPLY_P(f)) {
	    positive_p = (positive_expression_p(arg1) && positive_expression_p(arg2))
	      || (negative_expression_p(arg1) && negative_expression_p(arg2));
	  }
	  else if(ENTITY_DIVIDE_P(f)) {
	    positive_p = (positive_expression_p(arg1) && positive_expression_p(arg2))
	      || (negative_expression_p(arg1) && negative_expression_p(arg2));
	  }
	}
      }
    }
  }
  return positive_p;
}

/* Use constants and type information to decide if the value of
 * sigma(e) is always negative, e.g. <=0. If this can be proven, true is returned.
 *
 * false is returned when no decision can be made,
 * i.e. negative_expression_p(e)==false does not imply
 * positive_expression_p(e)
 *
 * Similar and linked to previous function
 */
bool negative_expression_p(expression e)
{
  bool negative_p = false; // In general, false because no conclusion can be reached
  intptr_t v;
  if(expression_integer_value(e, &v)) {
    negative_p = v<=0;
  }
  else {
    syntax s = expression_syntax(e);

    if(syntax_call_p(s)) {
      call c = syntax_call(s);
      entity f = call_function(c);
      type t = entity_type(f);
      pips_assert("t is a functional type", type_functional_p(t));
      type rt = functional_result(type_functional(t));
      if(overloaded_type_p(rt)) { /* We assume an operator is used */
	/* We can check a few cases recursively... */
	list args = call_arguments(c);
	intptr_t l = gen_length(args);
	if(l==1) {
	  expression arg = EXPRESSION(CAR(args));
	  if(ENTITY_UNARY_MINUS_P(f)) {
	    negative_p = positive_expression_p(arg); // Maybe zero, but no chance for sigma(x)>0
	  }
	}
	else if(l==2) {
	  expression arg1 = EXPRESSION(CAR(args));
	  expression arg2 = EXPRESSION(CAR(CDR(args)));
	  if(ENTITY_MINUS_P(f)) {
	    negative_p = negative_expression_p(arg1) && positive_expression_p(arg2);
	  }
	  else if(ENTITY_PLUS_P(f)) {
	    negative_p = negative_expression_p(arg1) && negative_expression_p(arg2);
	  }
	  else if(ENTITY_MULTIPLY_P(f)) {
	    negative_p = (negative_expression_p(arg1) && positive_expression_p(arg2))
	      || (positive_expression_p(arg1) && negative_expression_p(arg2));
	  }
	  else if(ENTITY_DIVIDE_P(f)) {
	    negative_p = (negative_expression_p(arg1) && positive_expression_p(arg2))
	      || (positive_expression_p(arg1) && negative_expression_p(arg2));
	  }
	}
      }
    }
  }
  return negative_p;
}

/* returns if e is normalized and linear.
 *
 * FI: should be moved into expression.c
 */
bool expression_linear_p(expression e)
{
    normalized n = expression_normalized(e);
    return !normalized_undefined_p(n) && normalized_linear_p(n);
}

/**
 * computes the distance between the lower bound and the upper bound of the range
 * @param r range to analyse
 * @param mode wether we compute the distance or count the number of iterations
 * @return appropriate distance or count
 */
expression range_to_expression(range r,enum range_to_expression_mode mode)
{
    expression distance =  make_op_exp(PLUS_OPERATOR_NAME,
            copy_expression(range_upper(r)),
            make_op_exp(MINUS_OPERATOR_NAME,
                int_to_expression(1),
                copy_expression(range_lower(r))));
    if( range_to_nbiter_p(mode) ) distance = make_op_exp(DIVIDE_OPERATOR_NAME,distance,copy_expression(range_increment(r)));
    return distance;
}

/* The range count only can be evaluated if the three range expressions
 * are constant and if the increment is non zero. On failure, a zero
 * count is returned. See also SizeOfRange().
 */
bool
range_count(range r, intptr_t * pcount)
{
    bool success = false;
    intptr_t l, u, inc;

    if(expression_integer_value(range_lower(r), &l)
       && expression_integer_value(range_upper(r), &u)
       && expression_integer_value(range_increment(r), &inc)
       && inc != 0 ) {

	if(inc<0) {
	    * pcount = ((l-u)/(-inc))+1;
	}
	else /* inc>0 */ {
	    * pcount = ((u-l)/inc)+1;
	}

	if(* pcount < 0)
	    *pcount = 0;

	success = true;
    }
    else {
	* pcount = 0;
	success = false;
    }

    return success;
}


/* returns true if v is not NULL and is constant */
/* I make it "static" because it conflicts with a Linear library function.
 * Both functions have the same name but a slightly different behavior.
 * The Linear version returns 0 when a null vector is passed as argument.
 * Francois Irigoin, 16 April 1990
 */
static bool
vect_const_p(Pvecteur v)
{
    pips_assert("vect_const_p", v != NULL);
    return vect_size(v) == 1 && value_notzero_p(vect_coeff(TCST, v));
}

/* 
  returns a Pvecteur equal to (*pv1) * (*pv2) if this product is 
  linear or NULL otherwise. 
  the result is built from pv1 or pv2 and the other vector is removed.
*/
Pvecteur 
vect_product(Pvecteur * pv1, Pvecteur * pv2)
{
    Pvecteur vr;

    if (vect_const_p(*pv1)) {
        vr = vect_multiply(*pv2, vect_coeff(TCST, *pv1));
        vect_rm(*pv1);
    }
    else if (vect_const_p(*pv2)) {
        vr = vect_multiply(*pv1, vect_coeff(TCST, *pv2));
        vect_rm(*pv2);
    }
    else {
        vr = NULL;
        vect_rm(*pv1);
        vect_rm(*pv2);
    }

    *pv1 = NULL;
    *pv2 = NULL;

    return(vr);
}
