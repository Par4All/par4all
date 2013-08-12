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

#ifndef lint
char vcid_syntax_eval[] = "%A% ($Date: 1998/04/14 21:28:15 $, ) version $Revision: 16236 $, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */

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

range: a range is not evaluated. */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "parser_private.h"
#include "syntax.h"
#include "toklex.h"


value EvalExpression(e)
expression e;
{
    return(EvalSyntax(expression_syntax(e)));
}



value EvalSyntax(s)
syntax s;
{
  value v;

  switch (syntax_tag(s)) {
  case is_syntax_reference:
  case is_syntax_range:
    v = make_value_unknown();
    break;
  case is_syntax_call:
    v = EvalCall((syntax_call(s)));
    break;
  case is_syntax_cast:
  case is_syntax_sizeofexpression:
  case is_syntax_subscript:
  case is_syntax_application:
  case is_syntax_va_arg:
    v = make_value_unknown();
    break;
  default:
    ParserError("EvalExpression", "cas default\n");
  }

  return(v);
}



/* only calls to constant, symbolic or intrinsic functions might be
evaluated. recall that intrinsic functions are not known. */

value EvalCall(c)
call c;
{
	value vout, vin;
	entity f;

	f = call_function(c);
	vin = entity_initial(f);
	
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
		vout = make_value_unknown();
		break;
	    default:
		ParserError("EvalCall", "case default\n");
	}

	return(vout);
}



value EvalConstant(c) 
constant c;
{
    return((constant_int_p(c)) ?
	   make_value(is_value_constant, make_constant(is_constant_int,
						       constant_int(c))) :
	   MakeValueLitteral());
}



/* this function tries to evaluate a call to an intrinsic function.
right now, we only try to evaluate unary and binary intrinsic functions,
ie. fortran operators.

e is the intrinsic function.

la is the list of arguments.
*/

value EvalIntrinsic(e, la)
entity e;
cons *la;
{
    value v;
    int token;

    if ((token = IsUnaryOperator(e)) > 0)
	    v = EvalUnaryOp(token, la);
    else if ((token = IsBinaryOperator(e)) > 0)
	    v = EvalBinaryOp(token, la);
    else
	    v = make_value_unknown();

    return(v);
}



value EvalUnaryOp(t, la)
int t;
cons *la;
{
	value vout, v;
	int arg;

	pips_assert("EvalUnaryOpt", la != NIL);
	v = EvalExpression(EXPRESSION(CAR(la)));
	if (value_constant_p(v) && constant_int_p(value_constant(v)))
		arg = constant_int(value_constant(v));
	else
		return(v);

	if (t == TK_MINUS) {
		constant_int(value_constant(v)) = -arg;
		vout = v;
	}
	else {
		gen_free(v);
		vout = make_value_unknown();
	}

	return(vout);
}

value EvalBinaryOp(t, la)
int t;
cons *la;
{
    value v;
    int argl, argr;

    pips_assert("EvalBinaryOpt", la != NIL);
    v = EvalExpression(EXPRESSION(CAR(la)));
    if (value_constant_p(v) && constant_int_p(value_constant(v))) {
	argl = constant_int(value_constant(v));
	gen_free(v);
    }
    else
	    return(v);

    la = CDR(la);
    pips_assert("EvalBinaryOpt", la != NIL);
    v = EvalExpression(EXPRESSION(CAR(la)));
    if (value_constant_p(v) && constant_int_p(value_constant(v))) {
	argr = constant_int(value_constant(v));
    }
    else
	    return(v);

    switch (t) {
      case TK_MINUS:
	constant_int(value_constant(v)) = argl-argr;
	break;
      case TK_PLUS:
	constant_int(value_constant(v)) = argl+argr;
	break;
      case TK_STAR:
	constant_int(value_constant(v)) = argl*argr;
	break;
      case TK_SLASH:
	if (argr != 0)
		constant_int(value_constant(v)) = argl/argr;
	else
		FatalError("EvalBinaryOp", "zero divide\n");
	break;
      case TK_POWER:
	if (argr >= 0)
		constant_int(value_constant(v)) = ipow(argl,argr);
	else {
	    gen_free(v);
	    v = make_value_unknown();
	}
	break;
      default:
	debug(9, "EvalBinaryOp", "pas encore d'evaluation\n");
	gen_free(v);
	v = make_value_unknown();
    }

    return(v);
}



int IsUnaryOperator(e)
entity e;
{
	int token;

	if (strcmp(entity_local_name(e), "--") == 0)
		token = TK_MINUS;
	else if (strcmp(entity_local_name(e), ".NOT.") == 0)
		token = TK_NOT;
	else
		token = -1;

	return(token);
}



int IsBinaryOperator(e)
entity e;
{
	int token;

	if      (strcmp(entity_local_name(e), "-") == 0)
		token = TK_MINUS;
	else if (strcmp(entity_local_name(e), "+") == 0)
		token = TK_PLUS;
	else if (strcmp(entity_local_name(e), "*") == 0)
		token = TK_STAR;
	else if (strcmp(entity_local_name(e), "/") == 0)
		token = TK_SLASH;
	else if (strcmp(entity_local_name(e), "**") == 0)
		token = TK_POWER;
	else if (strcmp(entity_local_name(e), ".EQ.") == 0)
		token = TK_EQ;
	else if (strcmp(entity_local_name(e), ".NE.") == 0)
		token = TK_NE;
	else if (strcmp(entity_local_name(e), ".EQV") == 0)
		token = TK_EQV;
	else if (strcmp(entity_local_name(e), ".NEQV") == 0)
		token = TK_NEQV;
	else if (strcmp(entity_local_name(e), ".GT.") == 0)
		token = TK_GT;
	else if (strcmp(entity_local_name(e), ".LT.") == 0)
		token = TK_LT;
	else if (strcmp(entity_local_name(e), ".GE.") == 0)
		token = TK_GE;
	else if (strcmp(entity_local_name(e), ".LE.") == 0)
		token = TK_LE;
	else if (strcmp(entity_local_name(e), ".OR.") == 0)
		token = TK_OR;
	else if (strcmp(entity_local_name(e), ".AND.") == 0)
		token = TK_AND;		
	else
		token = -1;

	return(token);
}



int ipow(vg, vd)
int vg, vd;
{
    /* FI: see arithmetique library */
	int i = 1;

	pips_assert("ipow", vd >= 0);

	while (vd-- > 0)
		i *= vg;

	return(i);
}
