/* 	%A% ($Date: 1998/07/13 19:40:38 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_syntax_expression[] = "%A% ($Date: 1998/07/13 19:40:38 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */

#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "misc.h"

#include "syntax.h"

/* this function creates a PARAMETER, ie a symbolic constant.

  e is an entity.

  x is an expression that represents the value of e. */

entity 
MakeParameter(e, x)
entity e;
expression x;
{
    type tp;

    tp = (entity_type(e) != type_undefined) ? entity_type(e) : ImplicitType(e);
    entity_type(e) = make_type(is_type_functional, make_functional(NIL, tp));
    entity_storage(e) = MakeStorageRom();
    entity_initial(e) = MakeValueSymbolic(x);

    return(e);
}


/* expressions from input output lists might contain implied do loops.
with our internal representation, implied do loops are stored as calls
to a special intrinsic function whose name is IMPLIED_DO_NAME and whose
first argument is the range of the loop.

v is a reference to the do variable.

r is the range of the loop.

l is the list of expressions which are to be read or written according
to this implied do loop. */

expression 
MakeImpliedDo(v, r, l)
syntax v;
range r;
cons *l;
{
    call c;
    expression er;

    if (!syntax_reference_p(v))
	    FatalError("MakeImpliedDo", "function call as DO variable\n");

    if (reference_indices(syntax_reference(v)) != NULL)
	    FatalError("MakeImpliedDo", "variable reference as DO variable\n");

    /* the range is enclosed in an expression */
    er = make_expression(make_syntax(is_syntax_range, r), 
			 normalized_undefined);

    l = CONS(EXPRESSION, make_expression(v, normalized_undefined), 
	     CONS(EXPRESSION, er, l));

    c = make_call(CreateIntrinsic(IMPLIED_DO_NAME), l);
    return(make_expression(make_syntax(is_syntax_call, c), 
			   normalized_undefined));
}



/* MakeAtom: 
 * this function creates a syntax, ie. a reference, a call or a range.
 *
 * there are a few good cases: e is a variable and its dimensionality is
 * equal to the number of expressions in indices, e is a function, e is a
 * constant or a symbolic constant.
 *
 * there are a few bad cases: e is a zero dimension variable and indices is
 * not equal to NIL (see comments of MakeExternalFunction), e is not known
 * and the list indices is not empty (it is a call to a user function), e
 * is not known and the list indices is empty (it is an implicit
 * declaration).
 * 
 * in this function, we first try to transform bad cases into good ones,
 * and then to create a syntax.
 * 
 * e is a variable or a function.
 * 
 * indices is a list of expressions (arguments or indices).
 * 
 * fc and lc are substring bound expressions.
 * 
 * HasParenthesis is a boolean that tells if the reference to e was done
 * with () or not. this is mandatory to make the difference between a call
 * to a function and a reference to a function.
 * 
 *  - MakeAtom in expression.c fixed to generate the proper message
 *    when the substring operator is used (Francois Irigoin, 6 June 1995).
 *    See lcart2.f in Validation.
 */

syntax 
MakeAtom(e, indices, fc, lc, HasParenthesis)
entity e;
cons * indices;
expression fc, lc;
int HasParenthesis;
{
    syntax s = syntax_undefined;
    type te;

    te = entity_type(e);

    /* checking errors ... */
    if (te != type_undefined) {
	if (type_statement_p(te)) {
	    FatalError("MakeAtom", "label dans une expression\n");
	}
	else if (type_area_p(te)) {
	    FatalError("MakeAtom", "area dans une expression\n");
	}
	else if (type_void_p(te)) {
	    FatalError("MakeAtom", "void dans une expression\n");
	}
	else if (type_unknown_p(te)) {
	    FatalError("MakeAtom", "unknown dans une expression\n");
	}
    }

    /* fixing bad cases */
    if (te == type_undefined) {
	/* FI: to handle parameterless function calls like t= second() - 11 March 1993 */
	/* if (indices == NULL) { */
	if (indices == NULL && !HasParenthesis) {
	    debug(2, "MakeAtom", "implicit declaration: %s\n",
		  entity_name(e));
	    DeclareVariable(e, type_undefined, indices, 
			    storage_undefined, value_undefined);
	}
	else {
	    type tr = ImplicitType(e);

	    debug(2, "MakeAtom", "new user function: %s\n",
		  entity_name(e));
	    /* e = MakeExternalFunction(e, type_undefined); */
	    e = MakeExternalFunction(e, tr);

	    /* use expression list to compute argument types */
	    update_functional_type_with_actual_arguments(e, indices);
	}
    }
    else if (type_variable_p(te)) {
	/* FI: same as in previous paragraph */
	/* if (variable_dimensions(type_variable(te))==NULL && indices!=NULL) { */
	/* if (variable_dimensions(type_variable(te))==NULL
	    && (indices!=NULL || HasParenthesis)) { */
	if (variable_dimensions(type_variable(te))==NULL
	    && (indices!=NULL || HasParenthesis)) {
	  if(fc==expression_undefined && lc==expression_undefined)
	  /*
	  if( !basic_string_p(variable_basic(type_variable(te)))
	      || (fc==expression_undefined && lc==expression_undefined)) */ {
	    e = MakeExternalFunction(e, type_undefined);
	    /* FI: probleme here for character returning function! You have to know if
	     * you are dealing with a substring operator or a function call.
	     *
	     * Fortunately, according to SUN f77 compiler, you are not allowed to
	     * take the substring of a function call!
	     */
	  }
	}
    }

    /* here, bad cases have been transformed into good ones. */
    te = entity_type(e);

    if (type_variable_p(te)) {
      if((gen_length(indices)==0) ||
	 (gen_length(indices)==
	  gen_length(variable_dimensions(type_variable(te))))) {
	if (lc == expression_undefined && fc == expression_undefined) {
	  s = make_syntax(is_syntax_reference, 
			  make_reference(e, indices));
	}
	else {
	  /* substring */
	  expression ref = 
	    make_expression(make_syntax(is_syntax_reference, 
					make_reference(e, indices)),
			    normalized_undefined);
	  expression lce = expression_undefined;
	  expression fce = expression_undefined;
	  list lexpr = NIL;
	  entity substr = entity_intrinsic(SUBSTRING_FUNCTION_NAME);
	  basic bt = variable_basic(type_variable(te));

	  pips_assert("Substring can only be applied to a string", basic_string_p(bt));

	  if(fc == expression_undefined) 
	    fce = int_to_expression(1);
	  else
	    fce = fc;

	  if(lc == expression_undefined)
	    lce = int_to_expression(basic_type_size(bt));
	  else
	    lce = lc;

	  lexpr = CONS(EXPRESSION, ref, 
		      CONS(EXPRESSION, fce,
			   CONS(EXPRESSION, lce, NIL)));
	  s = make_syntax(is_syntax_call, make_call(substr, lexpr));
	  /* ParserError("MakeAtom", "Substrings are not implemented\n"); */
	}
      }
      else {
	user_warning("MakeAtom",
		     "Too many or too few subscript expressions"
		     " for reference to %s\n",
		     entity_local_name(e));
	ParserError("MakeAtom", "Illegal array reference\n");
      }

    }
    else if (type_functional_p(te)) {
	if (value_unknown_p(entity_initial(e))) {
	    /* e is either called or passed as argument to a function. */
	    if (indices == NIL && HasParenthesis == FALSE) {
		s = make_syntax(is_syntax_reference, make_reference(e, NIL));
	    }
	    else {
		update_called_modules(e);
		s = make_syntax(is_syntax_call, make_call(e, indices));
	    }
	}
	else {
	    if (value_code_p(entity_initial(e))) {
		update_called_modules(e);
	    }
    	    s = make_syntax(is_syntax_call, make_call(e, indices));
	}
    }
    else {
	ParserError("MakeAtom", "unexpected type\n");
    }
	
    return(s);
}



/* this function takes a list of io elements (i, j, t(i,j)), and returns
the same list, with a cons cell pointing to a character constant
expression 'IOLIST=' before each element of the original list 

(i , j , t(i,j)) becomes ('IOLIST=' , i , 'IOLIST=' , j , 'IOLIST=' , t(i,j))
*/

cons *
MakeIoList(l)
cons *l;
{
    cons *pc; /* to walk thru l */
    cons *lr = NIL; /* result list */

    expression e = MakeCharacterConstantExpression("IOLIST=");
		
    pc = l;
    while (pc != NULL) {
	cons *p = CONS(EXPRESSION, e, NIL);

	CDR(p) = pc;
	pc = CDR(pc);
	CDR(CDR(p)) = NIL;

	lr = gen_nconc(p, lr);
    }

    return(lr);
}

/* Make sure that no call to implied do is in l */

list
FortranExpressionList(list l)
{
    MAP(EXPRESSION, e, {
	if(expression_implied_do_p(e))
	    ParserError("FortranExpressionList", "Unexpected implied DO\n");
    }, l);
    return l;
}

expression
MakeFortranBinaryCall(
    entity op,
    expression e1,
    expression e2)
{
    expression e = expression_undefined;

    if(expression_implied_do_p(e1) || expression_implied_do_p(e2)) {
	    ParserError("MakeFortranBinaryCall", "Unexpected implied DO\n");
    }

    e = MakeBinaryCall(op, e1, e2);

    return e;
}

expression
MakeFortranUnaryCall(
    entity op,
    expression e1)
{
    expression e = expression_undefined;

    if(expression_implied_do_p(e1)) {
	    ParserError("MakeFortranUnaryCall", "Unexpected implied DO\n");
    }

    e = MakeUnaryCall(op, e1);

    return e;
}
