/* This file contains a set of functions to evaluate integer constant
expressions. The algorithm is built on a recursive analysis of the
expression structure. Lower level functions are called until basic atoms
are reached. The succes of basic atom evaluation depend on the atom
type:

reference: right now, the evaluation fails because we do not compute
predicates on variables.

call: a call to a user function is not evaluated. a call to an intrinsic
function is successfully evaluated if arguments can be evaluated. a call
to a constant function is evaluated if its basic type is integer.

range: a range is not evaluated. 
*/

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "ri.h"
#include "misc.h"

#include "ri-util.h"

#include "operator.h"

value 
EvalExpression(expression e)
{
    return(EvalSyntax(expression_syntax(e)));
}

value 
EvalSyntax(syntax s)
{
	value v;

	switch (syntax_tag(s)) {
	    case is_syntax_reference:
	    case is_syntax_range:
		v = make_value(is_value_unknown, NIL);
		break;
	    case is_syntax_call:
		v = EvalCall((syntax_call(s)));
		break;
	    default:
		fprintf(stderr, "[EvalExpression] case default\n");
		abort();
	}

	return(v);
}

/* only calls to constant, symbolic or intrinsic functions might be
 * evaluated. recall that intrinsic functions are not known.
 */
value 
EvalCall(call c)
{
    value vout, vin;
    entity f;

    f = call_function(c);
    vin = entity_initial(f);

    if (value_undefined_p(vin))
	pips_internal_error("undefined value for %s\n", entity_name(f));
	
    switch (value_tag(vin)) {
      case is_value_intrinsic:
	vout = EvalIntrinsic(f, call_arguments(c));
	break;
      case is_value_constant:
	vout = EvalConstant((value_constant(vin)));
	break;
      case is_value_symbolic:
	vout = EvalConstant((symbolic_constant(value_symbolic(vin))));
	break;
      case is_value_unknown:
	/* it might be an intrinsic function */
	vout = EvalIntrinsic(f, call_arguments(c));
	break;
      case is_value_code:
	vout = make_value(is_value_unknown, NIL);
	break;
      default:
	fprintf(stderr, "[EvalCall] case default\n");
	abort();
    }

    return(vout);
}

value 
EvalConstant(constant c) 
{
    return((constant_int_p(c)) ?
	   make_value(is_value_constant, make_constant(is_constant_int,
						       constant_int(c))) :
	   make_value(is_value_constant, 
		      make_constant(is_constant_litteral, NIL)));

}

/* this function tries to evaluate a call to an intrinsic function.
right now, we only try to evaluate unary and binary intrinsic functions,
ie. fortran operators.

e is the intrinsic function.

la is the list of arguments.
*/

value 
EvalIntrinsic(entity e, list la)
{
    value v;
    int token;

    if ((token = IsUnaryOperator(e)) > 0)
	    v = EvalUnaryOp(token, la);
    else if ((token = IsBinaryOperator(e)) > 0)
	    v = EvalBinaryOp(token, la);
    else if ((token = IsNaryOperator(e)) > 0)
	    v = EvalNaryOp(token, la);
    else
	    v = make_value(is_value_unknown, NIL);

    return(v);
}

value 
EvalUnaryOp(int t, list la)
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
	else {
		gen_free(v);
		vout = make_value(is_value_unknown, NIL);
	}

	return(vout);
}

value 
EvalBinaryOp(int t, list la)
{
    value v;
    int argl, argr;

    pips_assert("non empty list", la != NIL);

    v = EvalExpression(EXPRESSION(CAR(la)));
    if (value_constant_p(v) && constant_int_p(value_constant(v))) {
	argl = constant_int(value_constant(v));
	gen_free(v);
    }
    else
	return(v);

    la = CDR(la);

    pips_assert("non empty list", la != NIL);
    v = EvalExpression(EXPRESSION(CAR(la)));

    if (value_constant_p(v) && constant_int_p(value_constant(v))) {
	argr = constant_int(value_constant(v));
    }
    else
	return(v);

    switch (t) {
      case MINUS:
	constant_int(value_constant(v)) = argl-argr;
	break;
      case PLUS:
	constant_int(value_constant(v)) = argl+argr;
	break;
      case STAR:
	constant_int(value_constant(v)) = argl*argr;
	break;
      case SLASH:
	if (argr != 0)
	    constant_int(value_constant(v)) = argl/argr;
	else {
	    fprintf(stderr, "[EvalBinaryOp] zero divide\n");
	    abort();
	}	
	break;
      case MOD:
	if (argr != 0)
	    constant_int(value_constant(v)) = argl%argr;
	else {
	    fprintf(stderr, "[EvalBinaryOp] zero divide in modulo\n");
	    abort();
	}	
	break;
      case POWER:
	if (argr >= 0)
	    constant_int(value_constant(v)) = ipow(argl,argr);
	else {
	    gen_free(v);
	    v = make_value(is_value_unknown, NIL);
	}
	break;
      default:
	gen_free(v);
	v = make_value(is_value_unknown, NIL);
    }

    return(v);
}

value 
EvalNaryOp(int t, list la)
{
    value v = value_undefined;
    value w = value_undefined;
    int new_arg = 0;
    bool first_arg_p = TRUE;

    /* 2 operands at least are needed */
    assert(la != NIL && CDR(la) != NIL);

    MAP(EXPRESSION, e, {
	v = EvalExpression(e);
	if (value_constant_p(v) && constant_int_p(value_constant(v))) {
	    new_arg = constant_int(value_constant(v));
	    if (first_arg_p) {
		first_arg_p = FALSE;
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
		gen_free(v);
	    }
	}
	else
	    return(v);
    }, la);

    return(w);
}

/* FI: These string constants are defined in ri-util.h */
int 
IsUnaryOperator(entity e)
{
	int token;

	if (strcmp(entity_local_name(e), "--") == 0)
		token = MINUS;
	else if (strcmp(entity_local_name(e), ".NOT.") == 0)
		token = NOT;
	else
		token = -1;

	return(token);
}

/* FI: These string constants are defined in ri-util.h */
int 
IsBinaryOperator(entity e)
{
	int token;

	if      (strcmp(entity_local_name(e), "-") == 0)
		token = MINUS;
	else if (strcmp(entity_local_name(e), "+") == 0)
		token = PLUS;
	else if (strcmp(entity_local_name(e), "*") == 0)
		token = STAR;
	else if (strcmp(entity_local_name(e), "/") == 0)
		token = SLASH;
	else if (strcmp(entity_local_name(e), "**") == 0)
		token = POWER;
	else if (strcmp(entity_local_name(e), "MOD") == 0)
		token = MOD;
	else if (strcmp(entity_local_name(e), ".EQ.") == 0)
		token = EQ;
	else if (strcmp(entity_local_name(e), ".NE.") == 0)
		token = NE;
	else if (strcmp(entity_local_name(e), ".EQV") == 0)
		token = EQV;
	else if (strcmp(entity_local_name(e), ".NEQV") == 0)
		token = NEQV;
	else if (strcmp(entity_local_name(e), ".GT.") == 0)
		token = GT;
	else if (strcmp(entity_local_name(e), ".LT.") == 0)
		token = LT;
	else if (strcmp(entity_local_name(e), ".GE.") == 0)
		token = GE;
	else if (strcmp(entity_local_name(e), ".LE.") == 0)
		token = LE;
	else if (strcmp(entity_local_name(e), ".OR.") == 0)
		token = OR;
	else if (strcmp(entity_local_name(e), ".AND.") == 0)
		token = AND;
	else if (same_string_p(entity_local_name(e), IMPLIED_COMPLEX_NAME) ||
		 same_string_p(entity_local_name(e), IMPLIED_DCOMPLEX_NAME))
	        token = CAST;
	else
		token = -1;

	return(token);
}

int 
IsNaryOperator(entity e)
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

/* FI: such a function should exist in Linear/arithmetique */
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
  computes the value of an integer expression.
  returns TRUE if an integer value has been found and placed in pval.
  returns FALSE otherwise.
*/
bool 
expression_integer_value(expression e, int * pval)
{
    bool is_int = FALSE;
    value v = EvalExpression(e);

    if (value_constant_p(v) && constant_int_p(value_constant(v))) {
        *pval = constant_int(value_constant(v));
        is_int = TRUE;
    }

    gen_free(v);
    return is_int;
}

/* returns TRUE if v is not NULL and is constant */
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
