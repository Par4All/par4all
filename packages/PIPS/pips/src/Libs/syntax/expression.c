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

#ifndef lint
char vcid_syntax_expression[] = "$Id$";
#endif /* lint */

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "misc.h"
#include "properties.h"

#include "syntax.h"

/* this function creates a PARAMETER, ie a symbolic constant.

  e is an entity.

  x is an expression that represents the value of e. */

entity MakeParameter(entity e, expression x)
{
    type tp;

    tp = (entity_type(e) != type_undefined) ? entity_type(e) : ImplicitType(e);
    entity_type(e) = make_type(is_type_functional, make_functional(NIL, tp));
    if(storage_undefined_p(entity_storage(e))) {
	entity_storage(e) = make_storage_rom();
    }
    else {
	if(storage_ram_p(entity_storage(e))) {
	    user_warning("MakeParameter", "Variable %s redefined as parameter\n",
			 entity_local_name(e));
	    ParserError("MakeParameter", "A variable cannot be redefined as a parameter\n");
	}
	else {
	    user_warning("MakeParameter", "Symbol %s redefined as parameter\n",
			 entity_local_name(e));
	    ParserError("MakeParameter", "A symbol cannot be redefined as a parameter\n");
	}
    }
    if(value_undefined_p(entity_initial(e)) || value_unknown_p(entity_initial(e))) {
      value v =  MakeValueSymbolic(x);
      symbolic s = value_symbolic(v);
      constant c = symbolic_constant(s);
      if(constant_int_p(c) && scalar_integer_type_p(tp))
	entity_initial(e) = v;
      else if(constant_float_p(c) && float_type_p(tp))
	entity_initial(e) = v;
      else if(constant_float_p(c) && scalar_integer_type_p(tp)) {
	/* Take the integer part of the floating point constant */
	double fval = constant_float(c);
	long long int ival = (long long int) fval;
	constant_tag(c) = is_constant_int;
	constant_int(c) = ival;
	entity_initial(e) = v;
      }
      else
	entity_initial(e) = v;
    }
    else {
	user_warning("MakeParameter", "Initial value for variable %s redefined\n",
		     entity_local_name(e));
	FatalError("MakeParameter", "An initial value cannot be redefined by parameter\n");
    }

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


/* this used to be a nested function, 
 * but compilation on macos dislikes nested functions ...
 */
static list make_arg_from_stmt( statement stmt, list args ) {
    instruction i = statement_instruction(stmt);
    expression expr;
    if ( instruction_expression_p(i) ) {
        expr = instruction_expression(i);
    } else if ( instruction_loop_p(i) ) {
        expr = loop_to_implieddo( instruction_loop(i) );
    } else {
        pips_internal_error("We can't handle anything other than expression"
                "and loop for loop-to-implieddo conversion.\n");
    }
    args = CONS(EXPRESSION,expr, args );
    return args;
}

/*
 * @brief Convert a loop to an IMPLIED-DO
 */
expression loop_to_implieddo( loop l ) {
  syntax index = make_syntax_reference( make_reference( loop_index(l), NIL ) );
  range r = loop_range(l);


  /* Fix last parameter */
  statement body = loop_body(l);
  instruction ibody = statement_instruction(body);
  list args = NIL;
  if(instruction_sequence_p(ibody)) {
    sequence seq = instruction_sequence(ibody);
    FOREACH(statement,stmt,sequence_statements(seq)) {
      args = make_arg_from_stmt(stmt, args);
    }
    args = gen_nreverse(args);
  } else {
    args = make_arg_from_stmt(body, args);
  }
  return MakeImpliedDo(index, r, args);
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
 * HasParenthesis is a bool that tells if the reference to e was done
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
	else if (type_functional_p(te)) {
	    if(!HasParenthesis) {
		/* It can be a PARAMETER or a functional variable or... */
		value iv = entity_initial(e);
		if(!value_undefined_p(iv) && value_code_p(iv)) {
		    user_warning("MakeAtom", "reference to functional entity %s\n",
				 entity_name(e));
		    /* Not enough information to decide to stop or not. */
		    /* ParserError("MakeAtom",
		       "unsupported use of a functional entity\n"); */
		}
	    }
	}
    }

    /* fixing bad cases */
    if (te == type_undefined) {
	/* FI: to handle parameterless function calls like t= second() - 11 March 1993 */
	/* if (indices == NULL) { */
	if (indices == NULL && !HasParenthesis) {
	    if(storage_undefined_p(entity_storage(e))) {
		debug(2, "MakeAtom", "implicit declaration of scalar variable: %s\n",
		  entity_name(e));
		DeclareVariable(e, type_undefined, indices, 
				storage_undefined, value_undefined);
	    }
	    else if(storage_formal_p(entity_storage(e))) {
		pips_debug(2, "reference to a functional parameter: %s\n",
		      entity_name(e));
		/* It has already been declared and should not be
                   redeclared because it may be an entry formal parameter
                   which is not declared in the current module. If e
                   represents an entry formal parameter (although its
                   top-level name is the current module), it does not
                   belong to the declarations of the current
                   module. Hence, it is hard to assert something here.

		   However, e has to be typed and valued. */
		if(type_undefined_p(entity_type(e))) {
		    entity_type(e) = ImplicitType(e);
		}
		if (value_undefined_p(entity_initial(e))) {
		    entity_initial(e) = make_value_unknown();
		}
	    }
	    else {
		debug(2, "MakeAtom", "implicit type declaration of scalar variable: %s\n",
		      entity_name(e));
		DeclareVariable(e, type_undefined, indices, 
				storage_undefined, value_undefined);
	    }
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

		/* use expression list to compute argument types */
		update_functional_type_with_actual_arguments(e, indices);

		/* FI: probleme here for character returning function! You have to know if
		 * you are dealing with a substring operator or a function call.
		 *
		 * Fortunately, according to SUN f77 compiler, you are not allowed to
		 * take the substring of a function call!
		 */
	    }
	}
    }
    else if (type_functional_p(te) && HasParenthesis) {
      /* In fact, only check compatability... if requested! */
      update_functional_type_with_actual_arguments(e, indices);
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

		if(!basic_string_p(bt)) {
		  /* pips_assert("Substring can only be applied to a string",
				 basic_string_p(bt)); */
		  if(!get_bool_property("PARSER_ACCEPT_ARRAY_RANGE_EXTENSION"))
		    ParserError("MakeAtom",
				"Substring operations can only be applied to "
				"strings in Fortran 77\n");
		  else {
		    /* Probably an extension we would have liked to have
                       for the DREAM-UP project. */
		    pips_internal_error("Not implemented yet");
		  }
		}

		if(fc == expression_undefined) 
		    fce = int_to_expression(1);
		else
		    fce = fc;

		if(lc == expression_undefined) {
		    /* The upper bound may be unknown for formal
                       parameters and for allocatable arrays and cannot be
                       retrieved from the type declaration */
		    value ub = basic_string(bt);

		    if(value_unknown_p(ub)) {
			lce = MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME));
		    }
		    else {
			lce = int_to_expression(basic_type_size(bt));
		    }
		}
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
	    /* e is either called or passed as argument to a function.
	     It cannot be a PARAMETER or its value would be known. */
	    if (indices == NIL && HasParenthesis == false) {
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

	    /* e is either called or passed as argument to a function, or
               it is a PARAMETER, in which case, it must really be
               called. */
	    if (indices == NIL && !HasParenthesis
		&& !value_symbolic_p(entity_initial(e))) {
		s = make_syntax(is_syntax_reference, make_reference(e, NIL));
	    }
	    else {
		s = make_syntax(is_syntax_call, make_call(e, indices));
	    }
	}
    }
    else {
	ParserError("MakeAtom", "unexpected type\n");
    }
	
    return(s);
}



/* This function takes a list of io elements (i, j, t(i,j)), and returns
the same list, with a cons cell pointing to a character constant
expression 'IOLIST=' before each element of the original list.

(i , j , t(i,j)) becomes ('IOLIST=' , i , 'IOLIST=' , j , 'IOLIST=' , t(i,j))

This IO list is later concatenated to the IO control list to form the
argument of an IO function. The tagging is necessary because of this
concatenation.

The IOLIST call used to be shared within one IO list. Since sharing is 
avoided in the PIPS internal representation, they are now duplicated.
*/

cons *
MakeIoList(l)
cons *l;
{
    cons *pc; /* to walk thru l */
    cons *lr = NIL; /* result list */
		
    pc = l;
    while (pc != NULL) {
        expression e = MakeCharacterConstantExpression(IO_LIST_STRING_NAME);
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

/* If a left hand side is a call, it should be a substring operator or a macro.
   If it is a call to an intrinsic with no arguments,
   the intrinsic is in fact masqued by a local variable.

   If s is not OK, it is freed and a new_s is allocated.
 */

syntax
CheckLeftHandSide(syntax s)
{
    syntax new_s = syntax_undefined;

    if(syntax_reference_p(s)) {
      entity v = reference_variable(syntax_reference(s));
      type vt = entity_type(v);

      if(type_variable_p(vt))
	new_s = s;
      else 
	pips_user_error("Illegal assignment to variable %s with type %s\n",
			entity_local_name(v), type_to_string(vt));
    }
    else {
	call c = syntax_call(s);
	entity f = call_function(c);

	if(intrinsic_entity_p(f)) {
	    if(strcmp(entity_local_name(f), SUBSTRING_FUNCTION_NAME)==0) {
		/* OK for substrings: They are processed later by MakeAssignInst() */
		pips_debug(7, "Substring assignment detected\n");
		new_s = s;
	    }
	    else if(ENDP(call_arguments(c))) {
		/* Oupss... This must be a local variable */
		entity v = FindOrCreateEntity(get_current_module_name(), entity_local_name(f));

		user_warning("CheckLeftHandSide",
			     "Name conflict between local variable %s and intrinsics %s\n",
			     entity_local_name(f), entity_name(f));

		free_syntax(s);
		reify_ghost_variable_entity(v);
		new_s = make_syntax(is_syntax_reference, make_reference(v, NIL));
	    }
	    else {
		/* A call to an intrinsic cannot be a lhs: statement function? 
		   Let's hope it works... */
		user_warning("CheckLeftHandSide",
			     "Name conflict between statement function %s and intrinsics %s\n",
			     entity_local_name(f), entity_name(f));
		new_s = s;
	    }
	}
	else {
	    /* Must be a macro... */
	    pips_debug(2, "Statement function definition %s\n", entity_name(f));
	    new_s = s;
	}
    }

    return new_s;
}
