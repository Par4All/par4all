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
  * Functions for the expressions
  *
  * Yi-Qing YANG, Lei ZHOU, Francois IRIGOIN, Fabien COELHO
  *
  * 12, Sep, 1991
  *
  * Alexis PLATONOFF, Sep. 25, 1995 : I have added some usefull functions from
  * static_controlize/utils.c
  */

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "text.h"
#include "text-util.h"

#include "arithmetique.h"
#include "pipsmake.h"

#include "ri-util.h"

/**************************************************** FORTRAN STRING COMPARE */

/* quite lazy...
 */
static string actual_fortran_string_to_compare(string fs, int * plength)
{
  string s = fs;
  int len;

  /* skip TOP-LEVEL header */
  if (strncmp(s, TOP_LEVEL_MODULE_NAME, strlen(TOP_LEVEL_MODULE_NAME))==0)
    s += strlen(TOP_LEVEL_MODULE_NAME);

  /* skip : header */
  if (strncmp(s, MODULE_SEP_STRING, strlen(MODULE_SEP_STRING))==0)
    s += strlen(MODULE_SEP_STRING);

  len = strlen(s);

  /* skip surrounding quotes */
  if (len>=2 &&
      ((s[0]=='\'' && s[len-1]=='\'') || (s[0]=='"' && s[len-1]=='"')))
  {
    s++;
    len -= 2;
  }

  /* skip trailing *spaces* (are these blanks?) if any. */
  while (len>0 && s[len-1]==' ')
    len--;

  *plength = len;
  return s;
}

/* compare pips fortran string constants from the fortran point of view.
 *
 * as of 3.1 and 6.3.5 of the Fortran 77 standard, the character order
 * is not fully specified. It states:
 *  - A < B < C ... < Z
 *  - 0 < 1 < 2 ... < 9
 *  - blank < 0
 *  - blank < A
 *  - 9 < A  *OR* Z < 0
 * since these rules are ascii compatible, we'll take ascii.
 * in practice, this may be implementation dependent?
 *
 * @param fs1 constant fortran string (entity name is fine)
 * @param fs2 constant fortran string (entity name is fine)
 * @return -n 0 +n depending on < == >, n first differing char.
 */
int fortran_string_compare(string fs1, string fs2)
{
  int l1, l2, i, c = 0;
  string s1, s2;

  /* skip headers, trailers... */
  s1 = actual_fortran_string_to_compare(fs1, &l1);
  s2 = actual_fortran_string_to_compare(fs2, &l2);

  /* collating sequence comparison. */
  for (i=0; c==0 && i<l1 && i<l2; i++)
  {
    if (s1[i] < s2[i]) c = -i-1;
    if (s1[i] > s2[i]) c = i+1;
  }

  /* equal string header case. */
  if (c==0 && l1!=l2)
    c = (l1<l2)? -l1-1: l2+1;

  return c;
}

/********************************************************************* BASIC */

/*  a BASIC tag is returned for the expression
 *  this is a preliminary version. should be improved.
 *  was in HPFC.
 */
tag suggest_basic_for_expression(expression e)
{
  tag	result = basic_tag(expression_basic(e));

  if (result==is_basic_overloaded)
    {
      syntax s = expression_syntax(e);

      /*  must be a call
       */
      assert(syntax_call_p(s));

      if (ENTITY_RELATIONAL_OPERATOR_P(call_function(syntax_call(s))))
	result = is_basic_logical;
      else
	{
	  /* else some clever analysis could be done
	   */
	  pips_user_warning("an overloaded is turned into an int...\n");
	  result = is_basic_int;
	}
    }

  return result;
}

expression expression_mult(expression ex)
{
  pips_internal_error("not implemented");
  return ex;
}

/* if v is a constant, returns a constant call.
 * if v is a variable, returns a reference to v.
 */
expression entity_to_expression(entity e)
{
  if(entity_symbolic_p(e))
    return call_to_expression(make_call(e, NIL));
  else if (entity_constant_p(e))
    return call_to_expression(make_call(e, NIL));
  else /* should be a scalar variable! */
    return reference_to_expression(make_reference(e, NIL));
}


expression make_entity_expression(entity e, cons *inds)
{
  syntax s = syntax_undefined;
  if( entity_constant_p(e) )
    {
      s = make_syntax_call(make_call(e,NIL));
    }
  else
    {
      reference r = make_reference(e, inds);
      s = make_syntax_reference(r);
    }
  return syntax_to_expression(s);
}

/*
 * remarks: why is the default to normalized_complex~?
 * should be undefined, OR normalized if possible.
 * I put normalize_reference... FC 27/09/93 and june 94
 */
expression reference_to_expression(reference r)
{
  expression e;
  syntax s = make_syntax(is_syntax_reference, r);

  e = make_expression(s, normalize_reference(r));

  return e;
}


/*
 * Get a reference from an expression. The expression has to contain
 * directly a syntax containing directly the reference.
 */
reference expression_to_reference(expression e)
{
    syntax s = expression_syntax(e);
    message_assert("reference", syntax_reference_p(s));
    return syntax_reference(s);
}

/* Add a zero subscript to a reference "r" by side effect.
 *
 * Used when array names are used to convert to the first array element
 */
void generic_reference_add_fixed_subscripts(reference r, type t, bool zero_p)
{
  pips_assert("type is of kind variable", type_variable_p(t));
  variable v = type_variable(t);

  // FI: this assert makes sense within the ri-util framework but is
  // too strong for the kind of references used in effects-util
  // pips_assert("scalar type", ENDP(reference_indices(r)));

  list dl = variable_dimensions(v);
  list sl = NIL; // subscript list
  FOREACH(DIMENSION, d, dl) {
    expression s = zero_p? int_to_expression(0) : make_unbounded_expression();
    // reference_indices(r) = CONS(EXPRESSION, s, reference_indices(r));
    sl = CONS(EXPRESSION, s, sl);
  }
  reference_indices(r) = gen_nconc(reference_indices(r), sl);
}

void reference_add_zero_subscripts(reference r, type t)
{
  generic_reference_add_fixed_subscripts(r, t, true);
}

void reference_add_unbounded_subscripts(reference r, type t)
{
  generic_reference_add_fixed_subscripts(r, t, false);
}

/* Build an expression that call a function or procedure.

   @param c is the call
 */
expression call_to_expression(call c)
{
  return make_expression(make_syntax(is_syntax_call, c),
			 normalized_undefined);
}


/* Build an expression that call an function entity with an argument list.

   @param e is the function entity to call
   @param l is the list of argument expressions given to the function to call
 */
expression make_call_expression(entity e, list l)
{
  return call_to_expression(make_call(e, l));
}


/* Creates a call expression to a function with zero arguments.

  @param f is the function entity to call
  */
expression MakeNullaryCall(entity f)
{
  return make_call_expression(f, NIL);
}


/* Creates a call expression to a function with one argument.

   @param f is the function entity to call
   @param a is the argument expression given to the function to call
 */
expression MakeUnaryCall(entity f, expression a)
{
  return make_call_expression(f, CONS(EXPRESSION, a, NIL));
}


/* Creates a call expression to a function with 2 arguments.

   @param f is the function entity to call
   @param eg is the first argument expression given to the function to call
   @param ed is the second argument expression given to the function to call
 */
expression MakeBinaryCall(entity f, expression eg, expression ed) {
  return make_call_expression(f, CONS(EXPRESSION, eg,
				      CONS(EXPRESSION, ed, NIL)));
}


/* Creates a call expression to a function with 3 arguments.

   @param f is the function entity to call
   @param e1 is the first argument expression given to the function to call
   @param e2 is the second argument expression given to the function to call
   @param e3 is the second argument expression given to the function to call
 */
expression MakeTernaryCall(entity f,
			       expression e1,
			       expression e2,
			       expression e3) {
  return make_call_expression(f,
			      CONS(EXPRESSION, e1,
				   CONS(EXPRESSION, e2,
					CONS(EXPRESSION, e3,
					     NIL))));
}


/* Make an assign expression, since in C the assignment is a side effect
   operator.

   Useful in for-loops.

   @param lhs must be a reference
   @param rhs is the expression to assign

   @return the expression "lhs = rhs"
*/
expression
make_assign_expression(expression lhs,
		       expression rhs) {
  /* RK: this assert should be relaxed to deal with *p and so on.
     pips_assert("Need a reference as lhs", expression_reference_p(lhs)); */
  return MakeBinaryCall(CreateIntrinsic(ASSIGN_OPERATOR_NAME), lhs, rhs);
}

expression make_subscript_expression(expression a, list sl)
{
  subscript i = make_subscript(a, sl);
  syntax s = make_syntax_subscript(i);
  expression e = syntax_to_expression(s);
  return e;
}

/* predicates and short cut accessors on expressions */

bool expression_brace_p(expression e) {
    if(expression_call_p(e)) {
        call c = expression_call(e);
        return ENTITY_BRACE_INTRINSIC_P(call_function(c));
    }
    return false;
}

bool expression_call_p(expression e)
{
  return(syntax_call_p(expression_syntax(e)));
}
bool expression_address_of_p(expression e)
{
  if (expression_call_p(e))
    {
      call c = expression_call(e);
      return(ENTITY_ADDRESS_OF_P(call_function(c)));
    }
  else
    return false;
}

call expression_call(expression e)
{
  return(syntax_call(expression_syntax(e)));
}

bool expression_cast_p(expression e)
{
  return(syntax_cast_p(expression_syntax(e)));
}

cast expression_cast(expression e)
{
  return(syntax_cast(expression_syntax(e)));
}

bool expression_sizeofexpression_p(expression e)
{
  return(syntax_sizeofexpression_p(expression_syntax(e)));
}

sizeofexpression expression_sizeofexpression(expression e)
{
  return(syntax_sizeofexpression(expression_syntax(e)));
}
/* Duplicate
bool expression_subscript_p(expression e)
{
  return(syntax_subscript_p(expression_syntax(e)));
}

subscript expression_subscript(expression e)
{
  return(syntax_subscript(expression_syntax(e)));
}
*/
bool expression_application_p(expression e)
{
  return(syntax_application_p(expression_syntax(e)));
}

application expression_application(expression e)
{
  return(syntax_application(expression_syntax(e)));
}

bool expression_field_p(expression e)
{
    return expression_call_p(e) && ENTITY_FIELD_P(call_function(expression_call(e)));
}
/* we get the type of the expression by calling expression_to_type()
 * which allocates a new one. Then we call ultimate_type() to have
 * the final type. Finally we test if it's a pointer by using pointer_type_p().*/
bool expression_pointer_p(expression e) {
  type et = expression_to_type(e);
  type t = ultimate_type(et);
  return pointer_type_p(t);

}

bool array_argument_p(expression e)
{
  if (expression_reference_p(e))
    {
      reference ref = expression_reference(e);
      entity ent = reference_variable(ref);
      if (array_entity_p(ent)) return true;
    }
  return false;
}



/* Test if an expression is a reference.
 */
bool expression_reference_p(expression e) {
  return(syntax_reference_p(expression_syntax(e)));
}

entity expression_variable(expression e)
{
  /* Assume e is a reference expression:
     expression_reference_p(e)==true  */
  return reference_variable(syntax_reference(expression_syntax(e)));
}

/* Test if an expression is a reference to a given variable entity.
 */
bool is_expression_reference_to_entity_p(expression e, entity v)
{
  bool is_e_reference_to_v = false;

  if(expression_reference_p(e)) {
    reference r = syntax_reference(expression_syntax(e));

    is_e_reference_to_v = (reference_variable(r)==v);
  }
  return is_e_reference_to_v;
}


/* This function returns true, if there exists a same expression in the list
 *                       false, otherwise
*/
bool same_expression_in_list_p(expression e, list le)
{
  MAP(EXPRESSION, f, if (same_expression_p(e,f)) return true, le);
  return false;
}

/* This function returns true, if there exists an expression equal in the list
 *                       false, otherwise
*/
bool expression_equal_in_list_p(expression e, list le)
{
  MAP(EXPRESSION, f, if (expression_equal_p(e,f)) return true, le);
  return false;
}

/* C xor is missing */
bool logical_operator_expression_p(expression e)
{
  /* Logical operators are : .NOT.,.AND.,.OR.,.EQV.,.NEQV.*/
  if (expression_call_p(e))
    {
      entity op = call_function(syntax_call(expression_syntax(e)));
      if (ENTITY_AND_P(op) || ENTITY_OR_P(op) || ENTITY_NOT_P(op) ||
	  ENTITY_EQUIV_P(op) ||ENTITY_NON_EQUIV_P(op))
	return true;
      return false;
    }
  return false;
}

bool relational_expression_p(expression e)
{
  /* A relational expression is a call whose function is either one of the following :
   * .LT.,.LE.,.EQ.,.NE.,.GT.,.GE. */
  if (expression_call_p(e))
    {
      entity op = call_function(syntax_call(expression_syntax(e)));
      if (ENTITY_RELATIONAL_OPERATOR_P(op))
	return true;
      return false;
    }
  return false;
}

bool integer_expression_p(expression e)
{
  basic b = basic_of_expression(e);
  bool integer_p = basic_int_p(b);

  free_basic(b);
  return integer_p;
}

bool logical_expression_p(expression e)
{
  /* A logical expression is either one of the following:
   * - a logical constant
   * - the symbolic name of a logical constant
   * - a logical variable name
   * - a logical array element name
   * - a logical function reference
   * - a relational expression (.LT.,.LE.,.EQ.,.NE.,.GT.,.GE.)
   * - is formed by combining together one or more of the above
   *   entities using parentheses and the logical operators
   *   .NOT.,.AND.,.OR.,.EQV.,.NEQV. */

  /* NN:  In fact, I didn't use the PIPS function : basic_of_expression because of 2 reasons :
   * - the function basic_of_intrinsic use the macro : ENTITY_LOGICAL_OPERATOR_P
   *   which is not like the Fortran Standard definition (the above comments)
   * - the case where an expression is a range is not considered here for a
   *   logical expression */

  syntax s = expression_syntax(e);
  basic b;
  entity func;

  pips_debug(2, "\n");

  switch(syntax_tag(s)) {
  case is_syntax_reference:
    {
      b = variable_basic(type_variable(entity_type(reference_variable(syntax_reference(s)))));
      if (basic_logical_p(b))
	return true;
      return false;
    }
  case is_syntax_call:
    {
      if (operator_expression_p(e,TRUE_OPERATOR_NAME) ||
	  operator_expression_p(e,FALSE_OPERATOR_NAME) ||
	  relational_expression_p(e)||
	  logical_operator_expression_p(e) )
	return true;
      func = call_function(syntax_call(expression_syntax(e)));
      b = variable_basic(type_variable(functional_result(type_functional(entity_type(func)))));
      if (basic_logical_p(b)) return true;

      /* The case of symbolic name of a logical constant is not treated here */

      return false;
    }
  case is_syntax_range:
    return false;
  default: pips_internal_error("Bad syntax tag");
    return false;
  }

  debug(2, "logical expression", " ends\n");
}


/* This function returns:
 *
 *  1, if e is a relational expression that is always true
 *
 * -1, if e is a relational expression that is always false
 *
 *  0, otherwise.
 *
 * It should be called trivial_condition_p().
 */

int trivial_expression_p(expression e)
{
  if (relational_expression_p(e))
    {
      /* If e is a relational expression*/
      list args = call_arguments(syntax_call(expression_syntax(e)));
      expression e1 =  EXPRESSION(CAR(args));
      expression e2 = EXPRESSION(CAR(CDR(args)));
      normalized n1,n2;
      entity op;
      if (expression_undefined_p(e1) ||expression_undefined_p(e2) ) return 0;
      n1 = NORMALIZE_EXPRESSION(e1);
      n2 = NORMALIZE_EXPRESSION(e2);
      op = call_function(syntax_call(expression_syntax(e)));

      ifdebug(3) {
	fprintf(stderr, "Normalizes of  expression:");
	print_expression(e);
	print_normalized(n1);
	print_normalized(n2);
      }

      if (normalized_linear_p(n1) && normalized_linear_p(n2))
	{
	  Pvecteur v1 = normalized_linear(n1);
	  Pvecteur v2 = normalized_linear(n2);
	  Pvecteur v = vect_substract(v1,v2);

	  /* The test if an expression is trivial (always true or false) or not
	   * depends on the operator of the expression :
	   * (op= {<=,<,>=,>,==,!=}) so we have to treat each different case */

	  if (vect_constant_p(v))
	    {
	      if (ENTITY_NON_EQUAL_P(op))
		{
		  /* Expression :  v != 0 */
		  if (VECTEUR_NUL_P(v)) return -1;
		  if (value_zero_p(val_of(v))) return -1;
		  if (value_notzero_p(val_of(v))) return 1;
		}
	      if (ENTITY_EQUAL_P(op))
		{
		  /* Expression :  v == 0 */
		  if (VECTEUR_NUL_P(v)) return 1;
		  if (value_zero_p(val_of(v))) return 1;
		  if (value_notzero_p(val_of(v))) return -1;
		}
	      if (ENTITY_GREATER_OR_EQUAL_P(op))
		{
		  /* Expression :  v >= 0 */
		  if (VECTEUR_NUL_P(v)) return 1;
		  if (value_posz_p(val_of(v))) return 1;
		  if (value_neg_p(val_of(v))) return -1;
		}
	      if (ENTITY_LESS_OR_EQUAL_P(op))
		{
		  /* Expression :  v <= 0 */
		  if (VECTEUR_NUL_P(v)) return 1;
		  if (value_negz_p(val_of(v))) return 1;
		  if (value_pos_p(val_of(v))) return -1;
		}
	      if (ENTITY_LESS_THAN_P(op))
		{
		  /* Expression :  v < 0 */
		  if (VECTEUR_NUL_P(v)) return -1;
		  if (value_neg_p(val_of(v))) return 1;
		  if (value_posz_p(val_of(v))) return -1;
		}
	      if (ENTITY_GREATER_THAN_P(op))
		{
		  /* Expression :  v > 0 */
		  if (VECTEUR_NUL_P(v)) return -1;
		  if (value_pos_p(val_of(v))) return 1;
		  if (value_negz_p(val_of(v))) return -1;
		}
	    }
	  return 0;
	}
      return 0;
    }
  return 0;
}


/* Test if an expression is a verbose reduction of the form :
   "i = i op v" or "i = v op i"

  @param e is the expression to analyse

  @param filter is a function that take an expression and return true iff
  expression is of the form "op(a,b,c...)". The filter may not be specific
  to binary operators, even if this function only deal with binary
  patterns. Use for example add_expression_p() to detect "i = i + v" or "i
  = v + i".

  @return the v expression if e is of the requested form or
  expression_undefined if not
 */
expression
expression_verbose_reduction_p_and_return_increment(expression incr,
						    bool filter(expression)) {
  if (assignment_expression_p(incr)) {
    /* The expression is an assignment, it is a good start. */
    list assign_params = call_arguments(syntax_call(expression_syntax(incr)));
    expression lhs = EXPRESSION(CAR(assign_params));

    /* Only deal with simple references right now: */
    if (expression_reference_p(lhs)) {
      expression rhs = EXPRESSION(CAR(CDR(assign_params)));
      if (filter(rhs)) {
	/* Operation found. */
	list op_params = call_arguments(syntax_call(expression_syntax(rhs)));
	/* Only deal with binary operators */
	if (gen_length(op_params) == 2) {
	  expression arg1 = EXPRESSION(CAR(op_params));
	  expression arg2 = EXPRESSION(CAR(CDR(op_params)));
	  if (expression_reference_p(arg1)
	      && reference_equal_p(expression_reference(lhs),
				   expression_reference(arg1)))
	    /* If arg1 is the same reference as lhs,
	       we are in the "i = i op v" case: */
	    return arg2;
	  else if (expression_reference_p(arg2)
		   && reference_equal_p(expression_reference(lhs),
					expression_reference(arg2)))
	    /* If arg2 is the same reference as lhs,
	       we are in the "i = v op i" case: */
	    return arg1;
	}
      }
    }
  }
  return expression_undefined;
}


bool expression_implied_do_p(e)
expression e ;
{
    if (expression_call_p(e)) {
	call c = syntax_call(expression_syntax(e));
	entity e = call_function(c);

	return(strcmp(entity_local_name(e), IMPLIED_DO_NAME) == 0);
    }

    return(false);
}

bool comma_expression_p(expression e)
{
  bool result = false;

  if (expression_call_p(e)) {
    call c = syntax_call(expression_syntax(e));
    entity f = call_function(c);

    result = ENTITY_COMMA_P(f);
  }

  return result;
}

bool expression_list_directed_p(e)
expression e ;
{
    if (expression_call_p(e)) {
	call c = syntax_call(expression_syntax(e));
	entity e = call_function(c);

	return(strcmp(entity_local_name(e), LIST_DIRECTED_FORMAT_NAME) == 0);
    }

    return(false);
}

/* More extensive than next function */
bool extended_integer_constant_expression_p(expression e)
{
  value v = EvalExpression(e);
  bool ice_p = false;

  if(value_constant_p(v)) {
    constant c = value_constant(v);

    ice_p = constant_int_p(c);
  }
  free_value(v);
  return ice_p;
}

// same as previous but also returns the integer constant in the integer
// pointed by parameter result if the expression is an integer constant
bool extended_integer_constant_expression_p_to_int(expression e, int * result)
{
  value v = EvalExpression(e);
  bool ice_p = false;

  if(value_constant_p(v)) {
    constant c = value_constant(v);

    ice_p = constant_int_p(c);
    if (ice_p) *result = constant_int(c);
  }
  free_value(v);
  return ice_p;
}


/* positive integer constant expression: call to a positive constant
   or to a sum of positive integer constant expressions (much too
   restrictive, but enough for the source codes submitted to PIPS up
   to now).

   Likely to fail and need further extension if subtraction and
   multiplication are used as probably allowed by C standard.

   NormalizeExpression() could be used instead, as it is in fact to compute
   the value of the expression.

   Use previous function instead of this one, and -1 will be a constant...
*/
bool integer_constant_expression_p(e)
expression e;
{
  syntax s = expression_syntax(e);
  bool ice = false;

  if(syntax_call_p(s)) {
    call c = syntax_call(s);
    entity cst = call_function(c);
    list args = call_arguments(c);
    int i;

    if(integer_constant_p(cst, &i)) {
      ice = true;
    }
    else if(integer_symbolic_constant_p(cst, &i)) {
      ice = true;
    }
    else if(ENTITY_PLUS_P(cst)||ENTITY_PLUS_C_P(cst)) {
      expression e1 = EXPRESSION(CAR(args));
      expression e2 = EXPRESSION(CAR(CDR(args)));

      ice = integer_constant_expression_p(e1) && integer_constant_expression_p(e2);
    }
  }

  return ice;
}

bool signed_integer_constant_expression_p(expression e)
{
  if(!integer_constant_expression_p(e)) {
    syntax s = expression_syntax(e);

    if(syntax_call_p(s)) {
	call c = syntax_call(s);
	entity um = call_function(c);

	if(um == gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
							 UNARY_MINUS_OPERATOR_NAME),
				    entity_domain)) {
	  expression e2 = binary_call_lhs(c);

	  return integer_constant_expression_p(e2);
	}
    }
    return false;
  }
  else {
    return true;
  }
}

/* The expression may be complicated but all its leaves are constants or
   parameters. It evaluates to a signed integer constant. I am too lazy to
   fully implement it as I should and I only take care of affine
   expressions (Francois). */
bool expression_with_constant_signed_integer_value_p(e)
expression e;
{
  normalized ne = NORMALIZE_EXPRESSION(e);
  bool constant_p = false;

  if(normalized_linear_p(ne)) {
    Pvecteur ve = normalized_linear(ne);
    /* No vecteur_constant_p() in linear library? */
    constant_p = VECTEUR_NUL_P(ve)
      || (term_cst(ve) && VECTEUR_UNDEFINED_P(vecteur_succ(ve)));
  }

  return constant_p;
}


/* Test if an expression is an assignment operation.
 */
bool assignment_expression_p(expression e) {
  return operator_expression_p(e, ASSIGN_OPERATOR_NAME);
}


/* Test if an expression is an addition.
 */
bool add_expression_p(expression e) {
  return operator_expression_p(e, PLUS_OPERATOR_NAME)
    || operator_expression_p(e, PLUS_C_OPERATOR_NAME)
    ;
}
bool sub_expression_p(expression e) {
  return operator_expression_p(e, MINUS_OPERATOR_NAME)
    || operator_expression_p(e, MINUS_C_OPERATOR_NAME)
    ;
}


/* Test if an expression is an substraction.
 */
bool substraction_expression_p(expression e) {
  return operator_expression_p(e, MINUS_OPERATOR_NAME)
    || operator_expression_p(e, MINUS_C_OPERATOR_NAME);
}


bool modulo_expression_p(e)
expression e;
{
    return operator_expression_p(e, MODULO_OPERATOR_NAME);
}

bool divide_expression_p(e)
expression e;
{
    return operator_expression_p(e, DIVIDE_OPERATOR_NAME);
}

bool power_expression_p(e)
expression e;
{
    return operator_expression_p(e, POWER_OPERATOR_NAME);
}

bool abs_expression_p(e)
expression e;
{
    return operator_expression_p(e, ABS_OPERATOR_NAME);
}

bool iabs_expression_p(e)
expression e;
{
    return operator_expression_p(e, IABS_OPERATOR_NAME);
}

bool dabs_expression_p(e)
expression e;
{
    return operator_expression_p(e, DABS_OPERATOR_NAME);
}

bool cabs_expression_p(e)
expression e;
{
    return operator_expression_p(e, CABS_OPERATOR_NAME);
}

bool min0_expression_p(e)
expression e;
{
    return operator_expression_p(e, MIN0_OPERATOR_NAME) ||
	operator_expression_p(e, MIN_OPERATOR_NAME);
}

bool max0_expression_p(e)
expression e;
{
    return operator_expression_p(e, MAX0_OPERATOR_NAME) ||
	operator_expression_p(e, MAX_OPERATOR_NAME);
}

bool user_function_call_p(e)
expression e;
{
    syntax s = expression_syntax(e);
    bool user_function_call_p = false;

    if(syntax_call_p(s)) {
	call c = syntax_call(s);
	entity f = call_function(c);
	value v = entity_initial(f);
	user_function_call_p = value_code_p(v);
    }
    else {
	user_function_call_p = false;
    }

    return user_function_call_p;
}

bool operator_expression_p(e, op_name)
expression e;
string op_name;
{
    syntax s = expression_syntax(e);

    if(syntax_call_p(s)) {
	call c = syntax_call(s);
	entity op = call_function(c);

	return strcmp(op_name, entity_local_name(op)) == 0;
    }
    else
	return false;
}

expression make_true_expression()
{
  return make_call_expression(MakeConstant(TRUE_OPERATOR_NAME,is_basic_logical),NIL);
}

expression make_false_expression()
{
  return make_call_expression(MakeConstant(FALSE_OPERATOR_NAME,is_basic_logical),NIL);
}

bool true_expression_p(expression e)
{
  return operator_expression_p(e,TRUE_OPERATOR_NAME);
}

bool false_expression_p(expression e)
{
  return operator_expression_p(e,FALSE_OPERATOR_NAME);
}

/* bool unbounded_dimension_p(dim)
 * input    : a dimension of an array entity.
 * output   : true if the last dimension is unbounded (*),
 *            false otherwise.
 * modifies : nothing
 * comment  :
 */
bool unbounded_dimension_p(dim)
dimension dim;
{
    syntax dim_synt = expression_syntax(dimension_upper(dim));
    bool res = false;

    if (syntax_call_p(dim_synt)) {
	const char* dim_nom = entity_local_name(call_function(syntax_call(dim_synt)));

	if (same_string_p(dim_nom, UNBOUNDED_DIMENSION_NAME))
	    res = true;
    }

    return(res);
}


expression find_ith_argument(list args, int n)
{
    int i;
    pips_assert("find_ith_argument", n > 0);

    for(i=1; i<n && !ENDP(args); i++, POP(args))
	;
    if(i==n && !ENDP(args))
	return EXPRESSION(CAR(args));
    else
	return expression_undefined;
}

/* find_ith_expression() is obsolet; use find_ith_argument() instead */
expression find_ith_expression(list le, int r)
{
    /* the first element is one */
    /* two local variables, useless but for debugging */
    list cle;
    int i;

    pips_assert("find_ith_expression", r > 0);

    for(i=r, cle=le ; i>1 && !ENDP(cle); i--, POP(cle))
	;

    if(ENDP(cle))
	pips_internal_error("not enough elements in expresion list");

    return EXPRESSION(CAR(cle));
}


/* transform an int into an expression and generate the corresponding
   entity if necessary; it is not clear if strdup() is always/sometimes
   necessary and if a memory leak occurs; wait till syntax/expression.c
   is merged with ri-util/expression.c

   Negative constants do not seem to be included in PIPS internal
   representation.
  */
expression int_to_expression(_int i)
{
    bool negative_p = i<0;
    /* special hook for VALUE_MIN: the problem is that VALUE_MIN cannot be represented in the IR because -VALUE_MIN does not fit into and _int, so we replace it by VALUE_MIN -1, which is still big ... */
    if(negative_p)
    while(i==-i) { // ie while we have an integer overflow
        ++i;
    }
    entity e = int_to_entity(abs(i));
    expression exp =  call_to_expression(make_call(e,NIL));
    if(negative_p)
        exp = MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),exp);
    return exp;
}

/* Make a zero expression.
 *
 * It is useful compared to int_to_expression(0) because it is much
 * easier to search in source text.
 */
expression make_zero_expression(void)
{
  return int_to_expression(0);
}

bool zero_expression_p(expression e)
{
  bool zero_p = false;
  if(expression_call_p(e)) {
    call c = expression_call(e);
    entity f = call_function(c);
    if(f==int_to_entity(0))
      zero_p = true;
  }
  return zero_p;
}

expression float_to_expression(float c)
{
    entity e = float_to_entity(c);
    return call_to_expression(make_call(e,NIL));
}
expression complex_to_expression(float re, float im)
{
    return MakeComplexConstantExpression(float_to_expression(re),float_to_expression(im));
}
expression bool_to_expression(bool b)
{
    return MakeNullaryCall
        (MakeConstant(b ? TRUE_OPERATOR_NAME : FALSE_OPERATOR_NAME,
		      is_basic_logical));
}

/* added interface for linear stuff.
 * it is not ok if Value is not an int, but if Value is changed
 * sometime, I guess code that use this function will not need
 * any change.
 * FC.
 */
expression Value_to_expression(Value v)
{
    return(int_to_expression(VALUE_TO_INT(v)));
}

/* conversion of an expression into a list of references; references are
   appended to list lr as they are encountered; array references are
   added before their index expressions are scanned;

   references to functions and constants (which are encoded as null-ary
   functions) are not recorded
*/
list expression_to_reference_list(expression e, list lr)
{
    syntax s = expression_syntax(e);
    lr = syntax_to_reference_list(s, lr);
    return lr;
}

list syntax_to_reference_list(syntax s, list lr)
{
    switch(syntax_tag(s)) {
    case is_syntax_reference:
	lr = gen_nconc(lr, CONS(REFERENCE, syntax_reference(s), NIL));
	MAPL(ce, {
	    expression e = EXPRESSION(CAR(ce));
	    lr = expression_to_reference_list(e, lr);
	    },
	     reference_indices(syntax_reference(s)));
	break;
    case is_syntax_range:
	lr = expression_to_reference_list(range_lower(syntax_range(s)), lr);
	lr = expression_to_reference_list(range_upper(syntax_range(s)), lr);
	lr = expression_to_reference_list(range_increment(syntax_range(s)),
					  lr);
	break;
    case is_syntax_call:
	MAPL(ce, {
	    expression e = EXPRESSION(CAR(ce));
	    lr = expression_to_reference_list(e, lr);
	    },
	     call_arguments(syntax_call(s)));
	break;
    case is_syntax_cast: {
      cast c = syntax_cast(s);
      expression e = cast_expression(c);
      lr = expression_to_reference_list(e, lr);
      break;
    }
    case is_syntax_sizeofexpression: {
      sizeofexpression soe = syntax_sizeofexpression(s);
      if(sizeofexpression_expression_p(soe)) {
	expression e = sizeofexpression_expression(soe);
	lr = expression_to_reference_list(e, lr);
      }
      break;
    }
    case is_syntax_subscript: {
      subscript sub = syntax_subscript(s);
      expression e = subscript_array(sub);
      list il = subscript_indices(sub);
      lr = expression_to_reference_list(e, lr);
      FOREACH(EXPRESSION, i,il) {
      lr = expression_to_reference_list(i, lr);
      }
      break;
    }
    case is_syntax_application: {
      application app = syntax_application(s);
      expression f = application_function(app);
      list al = application_arguments(app);
      lr = expression_to_reference_list(f, lr);
      FOREACH(EXPRESSION, a,al) {
      lr = expression_to_reference_list(a, lr);
      }
      break;
    }
    case is_syntax_va_arg: {
      list two = syntax_va_arg(s);
      FOREACH(SIZEOFEXPRESSION, soe, two) {
	if(sizeofexpression_expression_p(soe)) {
	  expression e = sizeofexpression_expression(soe);
	  lr = expression_to_reference_list(e, lr);
	}
      }
      break;
    }
    default:
	pips_internal_error("illegal tag %d",
		   syntax_tag(s));

    }
    return lr;
}

void fprint_expression(FILE * f, expression e)
{
  print_words(f, words_syntax(expression_syntax(e), NIL));
}

/* no file descriptor is passed to make is easier to use in a debugging
   stage.
   Do not make macros of those printing functions */

void print_expression(expression e)
{
  if(e==expression_undefined)
    (void) fprintf(stderr,"EXPRESSION UNDEFINED\n");
  // For debugging with gdb, dynamic type checking
  else if(expression_domain_number(e)!=expression_domain)
    (void) fprintf(stderr,"Arg. \"e\"is not an expression.\n");
  else {
    normalized n;
    (void) fprintf(stderr,"syntax = ");
    print_syntax(expression_syntax(e));
    (void) fprintf(stderr,"\nnormalized = ");
    if((n=expression_normalized(e))!=normalized_undefined)
      print_normalized(n);
    else
      (void) fprintf(stderr,"NORMALIZED UNDEFINED\n");
  }
}

string expression_to_string(expression e) {
    list l = words_expression(e,NIL) ;
    string out = words_to_string(l);
    FOREACH(STRING,w,l) free(w);
    gen_free_list(l);
    return out;
}


void print_expressions(list le)
{

  MAP(EXPRESSION, e , {
    print_expression(e);
      },
    le);

}

void print_syntax_expressions(list le)
{

  MAP(EXPRESSION, e , {
    print_syntax(expression_syntax(e));
    if(!ENDP(CDR(le))) {
	(void) fprintf(stderr, ", ");
    }
      },
    le);

}

void print_syntax(syntax s)
{
  print_words(stderr,words_syntax(s, NIL));
}

void print_reference(reference r)
{
  if(reference_undefined_p(r))
    fprintf(stderr, "reference undefined\n");
  // For debugging with gdb, dynamic type checking
  else if(reference_domain_number(r)!=reference_domain)
    fprintf(stderr, "Not a Newgen \"reference\" object\n");
  else {
    print_words(stderr,words_reference(r, NIL));
  }
}

void print_reference_list(list lr)
{
    if(ENDP(lr))
	fputs("NIL", stderr);
    else
	MAPL(cr,
	 {
	     reference r = REFERENCE(CAR(cr));
	     entity e = reference_variable(r);
	     (void) fprintf(stderr,"%s, ", entity_local_name(e));
	 },
	     lr);

    (void) putc('\n', stderr);
}

void print_references(list rl)
{
  print_reference_list(rl);
}

void print_normalized(normalized n)
{
    if(normalized_complex_p(n))
	(void) fprintf(stderr,"COMPLEX\n");
    else
	/* should be replaced by a call to expression_fprint() if it's
	   ever added to linear library */
	vect_debug((Pvecteur)normalized_linear(n));
}
bool expression_equal_p(expression e1, expression e2)
{
  syntax s1, s2;

  /* Add expression_undefined tests to avoid segmentation fault */

  if (expression_undefined_p(e1) && expression_undefined_p(e2))
    return true;
  if (expression_undefined_p(e1) || expression_undefined_p(e2))
    return false;

  /* let's assume that every expression has a correct syntax component */
  s1 = expression_syntax(e1);
  s2 = expression_syntax(e2);

  return syntax_equal_p(s1, s2);
}

/* this is slightly different from expression_equal_p, as it will return true for
 * a+b vs b+a
 */
bool same_expression_p(expression e1, expression e2)
{

  /* lazy normalization.
   */
  NORMALIZE_EXPRESSION(e1);
  NORMALIZE_EXPRESSION(e2);

  normalized n1, n2;
  n1 = expression_normalized(e1);
  n2 = expression_normalized(e2);



  if (normalized_linear_p(n1) && normalized_linear_p(n2))
    return vect_equal(normalized_linear(n1), normalized_linear(n2));
  else
    return expression_equal_p(e1, e2);
}

bool sizeofexpression_equal_p(sizeofexpression s0, sizeofexpression s1)
{
    if(sizeofexpression_type_p(s0) && sizeofexpression_type_p(s1))
        return type_equal_p(sizeofexpression_type(s0), sizeofexpression_type(s1));
    if(sizeofexpression_expression_p(s0) && sizeofexpression_expression_p(s1))
        return expression_equal_p(sizeofexpression_expression(s0),sizeofexpression_expression(s1));
    return false;
}

bool cast_equal_p(cast c1, cast c2)
{
  return
    type_equal_p(cast_type(c1), cast_type(c2)) &&
    expression_equal_p(cast_expression(c1), cast_expression(c2));
}

bool syntax_equal_p(syntax s1, syntax s2)
{
  tag t1 = syntax_tag(s1);
  tag t2 = syntax_tag(s2);

  if(t1!=t2)
    return false;

  switch(t1) {
  case is_syntax_reference:
    return reference_equal_p(syntax_reference(s1), syntax_reference(s2));
  case is_syntax_range:
    return range_equal_p(syntax_range(s1), syntax_range(s2));
  case is_syntax_call:
    return call_equal_p(syntax_call(s1), syntax_call(s2));
  case is_syntax_cast:
    return cast_equal_p(syntax_cast(s1), syntax_cast(s2));
  case is_syntax_sizeofexpression:
    return sizeofexpression_equal_p(syntax_sizeofexpression(s1),syntax_sizeofexpression(s2));
  case is_syntax_subscript:
    return subscript_equal_p(syntax_subscript(s1),syntax_subscript(s2));
  case is_syntax_application:
  case is_syntax_va_arg:
    pips_internal_error("Not implemented for syntax tag %d\n", t1);
  default:
    return false;
    break;
  }

  pips_internal_error("illegal. syntax tag %d", t1);
  return false;
}

bool subscript_equal_p(subscript s1, subscript s2) {
            return expression_equal_p(subscript_array(s1),subscript_array(s2))
        && gen_equals(subscript_indices(s1),subscript_indices(s2),(gen_eq_func_t)expression_equal_p);
}

bool reference_equal_p(reference r1, reference r2)
{
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);

  list dims1 = reference_indices(r1);
  list dims2 = reference_indices(r2);

  if(v1 != v2)
    return false;

  return gen_equals(dims1,dims2,(gen_eq_func_t)expression_equal_p);
}


bool range_equal_p(range r1, range r2)
{
  return expression_equal_p(range_lower(r1), range_lower(r2))
    && expression_equal_p(range_upper(r1), range_upper(r2))
    && expression_equal_p(range_increment(r1), range_increment(r2));
}

bool call_equal_p(call c1, call c2)
{
  entity f1 = call_function(c1);
  entity f2 = call_function(c2);
  list args1 = call_arguments(c1);
  list args2 = call_arguments(c2);

  if(f1 != f2)
    return false;

  return gen_equals(args1,args2,(gen_eq_func_t)expression_equal_p);

  return true;
}


int integer_constant_expression_value(expression e)
{
  pips_assert("is constant", integer_constant_expression_p(e));
  return signed_integer_constant_expression_value(e);
}

int signed_integer_constant_expression_value(expression e)
{
  /* could be coded by geting directly the value of the constant entity... */
  /* also available as integer_constant_p() which has *two* arguments */

  normalized n = normalized_undefined;
  int val = 0;

  pips_assert("is signed constant", signed_integer_constant_expression_p(e));

  n = NORMALIZE_EXPRESSION(e);
  if(normalized_linear_p(n)) {
    Pvecteur v = (Pvecteur) normalized_linear(n);

    if(vect_constant_p(v)) {
      Value x = vect_coeff(TCST, v);
      val = VALUE_TO_INT(x);
    }
    else
      pips_internal_error("non constant expression");
  }
  else
    pips_internal_error("non affine expression");

  return val;
}


expression make_max_expression(expression e1, expression e2, enum language_utype lang)
{
  expression new_exp = expression_undefined;
  if (lang == is_language_c)
    {
      expression comp_exp = MakeBinaryCall(entity_intrinsic(C_LESS_THAN_OPERATOR_NAME),
					   copy_expression(e1),
					   copy_expression(e2));
      new_exp = MakeTernaryCall(entity_intrinsic(CONDITIONAL_OPERATOR_NAME),
			       comp_exp,
			       e2,
			       e1);
    }
  else
    {
      new_exp = MakeBinaryCall(entity_intrinsic(MAX_OPERATOR_NAME),
			       e1,e2);
    }
  return  new_exp;
}

expression make_min_expression(expression e1, expression e2, enum language_utype lang)
{
  expression new_exp = expression_undefined;
  if (lang == is_language_c)
    {
      expression comp_exp = MakeBinaryCall(entity_intrinsic(C_LESS_THAN_OPERATOR_NAME),
					   copy_expression(e1),
					   copy_expression(e2));
      new_exp = MakeTernaryCall(entity_intrinsic(CONDITIONAL_OPERATOR_NAME),
			       comp_exp,
			       e1,
			       e2);
    }
  else
    {
      new_exp = MakeBinaryCall(entity_intrinsic(MAX_OPERATOR_NAME),
			       e1,e2);
    }
  return  new_exp;
}




/* Some functions to generate expressions from vectors and constraint
   systems. */


/* expression make_factor_expression(int coeff, entity vari)
 * make the expression "coeff*vari"  where vari is an entity.
 */
expression make_factor_expression(int coeff, entity vari)
{
  expression e1, e2, e3;

  e1 = int_to_expression(coeff);
  if (vari==NULL)
    return(e1);			/* a constant only */
  else {
    e2 = entity_to_expression(vari);
    if (coeff == 1) return(e2);
    else {
      e3 = MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),e1,e2);
      return (e3);
    }
  }
}

/* make expression for vector (Pvecteur)
 */
expression make_vecteur_expression(Pvecteur pv)
{
  /* sort: to insure a deterministic generation of the expression.
   * note: the initial system is *NOT* touched.
   * ??? Sometimes the vectors are shared, so you cant modify them
   *     that easily. Many cores in Hpfc (deducables), Wp65, and so.
   * ok, I'm responsible for some of them:-)
   *
   *  (c) FC 24/11/94
   *  SG: added support for generation of C operator when needed
   */
  Pvecteur
    v_sorted = vect_sort(pv, compare_Pvecteur),
    v = v_sorted;
  expression factor1, factor2;
  entity op_add, op_sub,
         c_op_add, c_op_sub;
  int coef;

  op_add = entity_intrinsic(PLUS_OPERATOR_NAME);
  op_sub = entity_intrinsic(MINUS_OPERATOR_NAME);
  c_op_add = entity_intrinsic(PLUS_C_OPERATOR_NAME);
  c_op_sub = entity_intrinsic(MINUS_C_OPERATOR_NAME);

  if (VECTEUR_NUL_P(v))
    return int_to_expression(0);

  coef = VALUE_TO_INT(vecteur_val(v));

  entity var = (entity) vecteur_var(v);
  bool next_op_is_c = var !=TCST && entity_pointer_p(var);
  if (coef==-1) /* let us avoid -1*var, we prefer -var */
    {
      entity op_ums = entity_intrinsic(UNARY_MINUS_OPERATOR_NAME);
      factor1 = make_factor_expression(1, (entity) vecteur_var(v));
      factor1 = call_to_expression
	(make_call(op_ums, CONS(EXPRESSION, factor1, NIL)));
    }
  else
    factor1 = make_factor_expression(coef, (entity) vecteur_var(v));

  for (v=v->succ; v!=NULL; v=v->succ)
    {
      var = (entity) vecteur_var(v);
      coef = VALUE_TO_INT(vecteur_val(v));
      pips_assert("some coefficient", coef!=0);
      factor2 = make_factor_expression(ABS(coef), var);
      /* choose among C or fortran operator depending on the entity type
       * this limits the use of +C and -C to pointer arithmetic
       */
      entity op =
          ( next_op_is_c ) ?
          ( coef> 0 ? c_op_add : c_op_sub ) :
          ( coef> 0 ? op_add   : op_sub ) ;
      factor1 = MakeBinaryCall(op,factor1,factor2);
      next_op_is_c = var !=TCST && entity_pointer_p(var);
    }

  vect_rm(v_sorted);

  return factor1;
}

/* generates var = linear expression from the Pvecteur.

   var is removed if necessary.

   ??? should manage an (positive remainder) integer divide ?  Have a look
   to make_constraint_expression instead?
 */
statement Pvecteur_to_assign_statement(entity var,
				       Pvecteur v)
{
  statement result;
  Pvecteur vcopy;
  Value coef;

  coef = vect_coeff((Variable) var, v);
  assert(value_le(value_abs(coef),VALUE_ONE));

  vcopy = vect_dup(v);

  if (value_notzero_p(coef)) vect_erase_var(&vcopy, (Variable) var);
  if (value_one_p(coef)) vect_chg_sgn(vcopy);

  result = make_assign_statement(entity_to_expression(var),
				 make_vecteur_expression(vcopy));
  vect_rm(vcopy);

  return result;
}


/* Make an expression from a constraint v for a given index.

  For example: for a constraint of index I : aI + linear_expr(J,K,TCST) <=0
  @return the new expression for I that is -expr_linear(J,K,TCST)/a
 */
expression make_constraint_expression(Pvecteur v, Variable index)
{
  Pvecteur pv;
  expression ex1, ex2, ex;
  entity div;
  Value coeff;

  /*search the couple (var,val) where var is equal to index and extract it */
  pv = vect_dup(v);
  coeff = vect_coeff(index, pv);
  vect_erase_var(&pv, index);

  if (VECTEUR_NUL_P(pv))
    /* If the vector wihout the index is the vector null, we have simply
       index = 0: */
    return int_to_expression(0);

  /* If the coefficient for the index is positive, inverse all the
     vector since the index goes at the other side of "=": */
  if (value_pos_p(coeff))
    vect_chg_sgn(pv);
  else {
    /* If coeff is negative, correct the future division rounding (by
       -coeff) by adding (coeff - 1) to the vector first: */
    value_absolute(coeff);
    vect_add_elem(&pv, TCST, value_minus(coeff, VALUE_ONE));
  }

  if(vect_size(pv) == 1 && vecteur_var(pv) == TCST) {
    /* If the vector is simply coeff.index=c, directly generate and
       return c/coeff: */
    vecteur_val(pv) = value_pdiv(vecteur_val(pv), coeff);
    return make_vecteur_expression(pv);
  }

  /* Generate an expression from the linear vector: */
  ex1 = make_vecteur_expression(pv);

  if (value_gt(coeff, VALUE_ONE)) {
    /* If coeff > 1, divide all the expression by coeff: */
    /* FI->YY: before generating a division, you should test if it could
       not be performed statically; you have to check if ex1 is not a
       constant expression, which is fairly easy since you still have
       its linear form, pv */
    div = gen_find_tabulated("TOP-LEVEL:/",entity_domain);
    pips_assert("Division operator not found", div != entity_undefined);

    ex2 = int_to_expression(VALUE_TO_INT(coeff));
    ex = make_expression(make_syntax(is_syntax_call,
				     make_call(div,
					       CONS(EXPRESSION,ex1,
						    CONS(EXPRESSION,
							 ex2,NIL)))),
			 normalized_undefined);
    return(ex);
  }
  else
    return(ex1);
}


/*  A wrapper around make_constraint_expression() for compatibility.
 */
expression make_contrainte_expression(Pcontrainte pc, Variable index) {
  /* Simply call the function on the vector in the constrain system: */
  return make_constraint_expression(pc->vecteur, index);
}


/* AP, sep 25th 95 : some usefull functions moved from
   static_controlize/utils.c */

/* rather use make_vecteur_expression which was already there */
expression Pvecteur_to_expression(Pvecteur vect)
{
    return make_vecteur_expression(vect);
}


/* Short cut, meaningful only if expression_reference_p(e) holds. */
reference expression_reference(expression e)
{
    pips_assert("e is a reference\n",expression_reference_p(e));
    return syntax_reference(expression_syntax(e));
}

bool expression_subscript_p(expression e) {
    return syntax_subscript_p(expression_syntax(e));
}


subscript expression_subscript(expression e)
{
    pips_assert("is a subscript\n",expression_subscript_p(e));
    return syntax_subscript(expression_syntax(e));
}
bool expression_range_p(expression e)
{
    return syntax_range_p(expression_syntax(e));
}

range expression_range(expression e)
{
    pips_assert("is a range", expression_range_p(e));
    return syntax_range(expression_syntax(e));
}

/* predicates on references */

bool array_reference_p(reference r)
{
  /* two possible meanings:
   * - the referenced variable is declared as an array
   * - the reference is to an array element
   *
   * This makes a difference in procedure calls and IO statements
   *
   * The second interpretation is chosen.
   */

  return reference_indices(r) != NIL;
}

/* If true is returned, the two references cannot conflict unless array
 * bound declarations are violated. If false is returned, the two references
 * may conflict.
 *
 * true is returned if the two references are array references and if
 * the two references entities are equal and if at least one dimension
 * can be used to desambiguate the two references using constant subscript
 * expressions. This test is store independent and certainly does not
 * replace a dependence test. It may beused to compute ude-def chains.
 *
 * If needed, an extra effort could be made for aliased arrays.
 */

bool references_do_not_conflict_p(reference r1, reference r2)
{
  bool do_not_conflict = false;
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);

  if(v1==v2) {
    list s1 = reference_indices(r1);
    list s2 = reference_indices(r2);
    if(!ENDP(s1) && gen_length(s1)==gen_length(s2)) {
      list cs1, cs2;
      for(cs1=s1, cs2=s2; !ENDP(cs1) && !do_not_conflict; POP(cs1), POP(cs2)) {
	expression sub1 = EXPRESSION(CAR(cs1));
	expression sub2 = EXPRESSION(CAR(cs2));
	if(expression_constant_p(sub1) && expression_constant_p(sub2)) {
	  /* FI: OK, it would be better to use their normalized forms */
	  do_not_conflict = (expression_to_int(sub1)!=expression_to_int(sub2));
	}
      }
    }
  }

  return do_not_conflict;
}

/*
 *    Utils from hpfc on 15 May 94, FC
 *
 */
expression expression_list_to_binary_operator_call(list l, entity op)
{
  int
    len = gen_length(l);
  expression
    result = expression_undefined;

  pips_assert("list_to_binary_operator_call", len!=0);

  result = EXPRESSION(CAR(l));

  MAPL(ce,
       {
	 result = MakeBinaryCall(op, EXPRESSION(CAR(ce)), result);
       },
       CDR(l));

  return(result);
}

expression expression_list_to_conjonction(list l)
{
  int	len = gen_length(l);
  entity and = entity_intrinsic(AND_OPERATOR_NAME);
  return(len==0?
	 MakeNullaryCall(entity_intrinsic(".TRUE.")):
	 expression_list_to_binary_operator_call(l, and));
}

/* bool expression_intrinsic_operation_p(expression exp): Returns true
 * if "exp" is an expression with a call to an intrinsic operation.
 */
bool expression_intrinsic_operation_p(expression exp)
{
  entity e;
  syntax syn = expression_syntax(exp);

  if (syntax_tag(syn) != is_syntax_call)
    return (false);

  e = call_function(syntax_call(syn));

  return(value_tag(entity_initial(e)) == is_value_intrinsic);
}

/* bool call_constant_p(call c): Returns true if "c" is a call to a constant,
 * that is, a constant number or a symbolic constant.
 */
bool call_constant_p(call c)
{
  value cv = entity_initial(call_function(c));
  return( (value_tag(cv) == is_value_constant) ||
	  (value_tag(cv) == is_value_symbolic)   );
}


/*=================================================================*/
/* bool expression_equal_integer_p(expression exp, int i): returns true if
 * "exp" is a constant value equal to "i".
 */
bool expression_equal_integer_p(expression exp, int i)
{
  pips_debug(7, "doing\n");
  if(expression_constant_p(exp))
    return(expression_to_int(exp) == i);
  return(false);
}

/*=================================================================*/
/* expression make_op_exp(char *op_name, expression exp1 exp2):
 * Returns an expression containing the operation "op_name" between "exp1" and
 * "exp2".
 * "op_name" must be one of the four classic operations : +, -, * or /.
 *
 * If both expressions are integer constant values and the operation
 * result is an integer then the returned expression contained the
 * calculated result.
 *
 * Else, we treat five special cases :
 *       _ exp1 and exp2 are integer linear and op_name is + or -.
 *         This case is resolved by make_lin_op_exp().
 *       _ exp1 = 0
 *       _ exp1 = 1
 *       _ exp2 = 0
 *       _ exp2 = 1
 *
 * Else, we create a new expression with a binary call.
 *
 * Note: The function MakeBinaryCall() comes from Pips/.../syntax/expression.c
 *       The function int_to_expression() comes from ri-util.
 *
 * Warning: using the same semantic as MakeBinaryCall,
 * make_op_exp owns the pointer exp1 and exp2 after the call,
 * beware of not sharing them !
 */
expression make_op_exp(char *op_name, expression exp1, expression exp2)
{
  expression result_exp = expression_undefined;
  entity op_ent, unary_minus_ent;

  debug( 7, "make_op_exp", "doing\n");
  op_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						   op_name), entity_domain);
  unary_minus_ent =
    gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
					    UNARY_MINUS_OPERATOR_NAME),
		       entity_domain);

  pips_debug(5, "begin OP EXP : %s  %s  %s\n",
	words_to_string(words_expression(exp1, NIL)),
	op_name,
	words_to_string(words_expression(exp2, NIL)));

  if( ! ENTITY_FIVE_OPERATION_P(op_ent) )
    user_error("make_op_exp", "operation must be : +, -, *, MOD, or /");

  intptr_t val1, val2;
  if( expression_integer_value(exp1,&val1) && expression_integer_value(exp2,&val2) )
    {

      debug(6, "make_op_exp", "Constant expressions\n");

      if (ENTITY_PLUS_P(op_ent))
	result_exp = int_to_expression(val1 + val2);
      else if(ENTITY_MINUS_P(op_ent))
	result_exp = int_to_expression(val1 - val2);
      else if(ENTITY_MULTIPLY_P(op_ent))
	result_exp = int_to_expression(val1 * val2);
      else if(ENTITY_MODULO_P(op_ent))
          result_exp = int_to_expression(val1 % val2);
      else /* ENTITY_DIVIDE_P(op_ent) */
	/* we compute here as FORTRAN would do */
	result_exp = int_to_expression((int) (val1 / val2));
      free_expression(exp1);
      free_expression(exp2);
    }
  else
    {
      /* We need to know the integer linearity of both expressions. */
      normalized nor1 = NORMALIZE_EXPRESSION(exp1);
      normalized nor2 = NORMALIZE_EXPRESSION(exp2);

      /* dividing by one or minus one is similar to multiplying */ 
      _int val;
      if(( ENTITY_DIVIDE_P(op_ent) || ENTITY_MULTIPLY_P(op_ent))
              && expression_integer_value(exp2,&val)
              && ( val == -1) ) {
          free_expression(exp2);
          return make_op_exp(MINUS_OPERATOR_NAME,int_to_expression(0),exp1);
      }

      if((normalized_linear_p(nor1) && normalized_linear_p(nor2) ) &&
	 (ENTITY_PLUS_P(op_ent) || ENTITY_MINUS_P(op_ent) ) )
	{
	  pips_debug(6, "Linear operation\n");

      result_exp = make_lin_op_exp(op_ent, copy_expression(exp1), copy_expression(exp2));
	}
      else if(expression_equal_integer_p(exp1, 0))
	{
	  if (ENTITY_PLUS_P(op_ent)) {
	    result_exp = exp2;
        free_expression(exp1);
      }
	  else if(ENTITY_MINUS_P(op_ent)) {
	    result_exp = MakeUnaryCall(unary_minus_ent, exp2);
        free_expression(exp1);
      }
	  else /* ENTITY_MULTIPLY_P(op_ent) || ENTITY_DIVIDE_P(op_ent) */ {
	    result_exp = int_to_expression(0);
        free_expression(exp1);free_expression(exp2);
      }
	}
      else if(expression_equal_integer_p(exp1, 1))
	{
	  if(ENTITY_MULTIPLY_P(op_ent)) {
	    result_exp = exp2;
        free_expression(exp1);
      }
	}
      else if(expression_equal_integer_p(exp2, 0))
	{
      free_expression(exp2);
	  if (ENTITY_PLUS_P(op_ent) || ENTITY_MINUS_P(op_ent))
	    result_exp = exp1;
	  else if (ENTITY_MULTIPLY_P(op_ent)) {
	    result_exp = int_to_expression(0);
        free_expression(exp1);
      }
	  else /* ENTITY_DIVIDE_P(op_ent) */
	    user_error("make_op_exp", "division by zero");
	}
      else if(expression_equal_integer_p(exp2, 1))
	{
	  if(ENTITY_MULTIPLY_P(op_ent) || ENTITY_DIVIDE_P(op_ent)) {
	    result_exp = exp1;
        free_expression(exp2);
      }
	}
    }

  if(result_exp == expression_undefined)
    result_exp = MakeBinaryCall(op_ent, exp1, exp2);

  pips_debug(5, "end   OP EXP : %s\n",
	words_to_string(words_expression(result_exp, NIL)));

  return (result_exp);
}

/// @return a new expression that adds the an expression with an integer
/// @param e, the expression to add
/// @param n, the integer to add
expression add_integer_to_expression (expression exp, int val) {
  return make_op_exp (PLUS_OPERATOR_NAME, exp, int_to_expression(val));
}

/*=================================================================*/
/* expression make_lin_op_exp(entity op_ent, expression exp1 exp2): returns
 * the expression resulting of the linear operation (ie. + or -) "op_ent"
 * between two integer linear expressions "exp1" and "exp2".
 *
 * This function uses the linear library for manipulating Pvecteurs.
 * exp1 and exp2 are freed
 *
 * Pvecteur_to_expression() is a function that rebuilds an expression
 * from a Pvecteur.
 */
expression make_lin_op_exp(entity op_ent, expression exp1, expression exp2)
{
  Pvecteur V1, V2, newV = VECTEUR_NUL;

  debug( 7, "make_lin_op_exp", "doing\n");
  if(normalized_complex_p(expression_normalized(exp1)) ||
     normalized_complex_p(expression_normalized(exp2)) )
    pips_internal_error( "expressions MUST be linear and normalized");

  V1 = (Pvecteur) normalized_linear(expression_normalized(exp1));
  V2 = (Pvecteur) normalized_linear(expression_normalized(exp2));

  if (ENTITY_PLUS_P(op_ent))
    newV = vect_add(V1, V2);
  else if (ENTITY_MINUS_P(op_ent))
    newV = vect_substract(V1, V2);
  else
    pips_internal_error("operation must be : + or -");
  free_expression(exp1);
  free_expression(exp2);

  return(Pvecteur_to_expression(newV));
}


/*=================================================================*/
/* int expression_to_int(expression exp): returns the integer value of
 * "exp" when "exp" is programming language constant or a Fortran
 * symbolic constant.
 *
 * Note: "exp" is supposed to contain an integer value which means that the
 *       function expression_constant_p() has returned true with "exp" as
 *       argument.
 *
 *       This implies that if "exp" is not a "value_constant", it must be
 *       a "value_intrinsic". In that case it is an unary minus operation
 *       upon an expression for which the function expression_constant_p()
 *       returns true (see its comment).
 *
 * SG: improvement: the integer value of a
 *  extended_integer_constant_expression_p is computed too, for
 *  instance 0+5 or 6+3.
 *
 * FI: I am not sure it is an improvement, at best it is an extension;
 * it looks more like a change of semantics (see above comments and
 * expression_constant_p()); other functions exist to evaluate a
 * constant expression... The extension should be useless as this
 * function should not be called unless guarded by a test with
 * expression_constant_p()... The same is true for the extension
 * related to symbolic constants.
 *
 * For an extended static expression evaluation, see
 * EvalExpression(). See also extended_expression_constant_p() for
 * more comments about what is called a "constant" expression.
 *
 * For a cleaner implementation of this function as it was originally
 * intended, see below expression_to_float().
 */
int expression_to_int(expression exp)
{
  int rv = 0;
  pips_debug(7, "doing\n");

  /* use the normalize field first: this is Serge Guelton's improvement... */
  /* Not good for user source code preservation... */
  NORMALIZE_EXPRESSION(exp);
  normalized n = expression_normalized(exp);
  if(normalized_linear_p(n)) {
      Pvecteur pv = normalized_linear(n);
      if(vect_constant_p(pv))
          return (int)vect_coeff(TCST, pv);
  }

  /* This is the initial code */
  if(expression_constant_p(exp)) {
    syntax s = expression_syntax(exp);
    if(syntax_call_p(s)) { // A nullary call is assumed
      call c = syntax_call(s);
      switch(value_tag(entity_initial(call_function(c)))) {
      case is_value_constant:	{
	rv = constant_int(value_constant(entity_initial(call_function(c))));
	break;
      }
      case is_value_intrinsic: {
	entity op = call_function(c);
	if(ENTITY_UNARY_MINUS_P(op))
	  rv = 0 - expression_to_int(binary_call_lhs(c));
	else if(ENTITY_UNARY_PLUS_P(op))
	  rv = expression_to_int(binary_call_lhs(c));
	else
	  pips_internal_error("Unexpected intrinsic \"%s\"\n",
			      entity_local_name(op));
	break;
      }
      default:
	pips_internal_error("expression %s is not an integer constant\n",
			    expression_to_string(exp));
      }
    }
    else if(false && syntax_sizeofexpression(s)) {
      sizeofexpression soe = syntax_sizeofexpression(s);
      // FI: do we want to guard this evaluation with EVAL_SIZEOF?
      // This should be part of eval.c
      value v = EvalSizeofexpression(soe);
      if(value_constant_p(v)) { // Should always be true... but for dependent types
	rv = constant_int(value_constant(v));
      }
      else {
	/* Could be improved checking dependent and not dependent types...*/
	pips_internal_error("expression %s is not an integer constant\n",
			    expression_to_string(exp));
      }
      free_value(v);
    }
  }
  /* This is another useless extension */
  else if(expression_call_p(exp)) {
    /* Is it an integer "parameter" in Fortran, that is symbolic constants?
     *
     * Should this be generalized to C const qualified variables?
     *
     * I have doubt about this piece of code since the call is
     * supposedly guarded by expression_constant_p()
     */
    entity p = call_function(syntax_call(expression_syntax(exp)));
    value v = entity_initial(p);

    if(value_symbolic_p(v) && constant_int_p(symbolic_constant(value_symbolic(v)))) {
      rv = constant_int(symbolic_constant(value_symbolic(v)));
    }
    else {
      pips_internal_error("expression %s is not an integer constant\n",
			  expression_to_string(exp));
    }
  }
  else
      pips_internal_error("expression %s is not an integer constant"
			  " in the sense of expression_constant_p()\n",
			  expression_to_string(exp));
  return(rv);
}



/* Same as above for floating point constants
 *
 * Its calls must be guarded by expression_constant_p()
 */
float expression_to_float(expression exp)
{
  float rv = 0;

  pips_debug( 7, "doing\n");
  if(expression_constant_p(exp)) {
    call c = syntax_call(expression_syntax(exp));
    switch(value_tag(entity_initial(call_function(c)))) {
    case is_value_constant:	{
      rv = atof(entity_user_name(call_function(c)));
      break;
    }
    case is_value_intrinsic: {
      entity op = call_function(c);
      if(ENTITY_UNARY_MINUS_P(op))
	rv = 0 - expression_to_float(binary_call_lhs(c));
      else if(ENTITY_UNARY_PLUS_P(op))
	rv = expression_to_float(binary_call_lhs(c));
      else
	pips_internal_error("Unexpected intrinsic \"%s\"\n",
			    entity_local_name(op));
      break;
    }
    default:
      pips_internal_error("expression is not a constant"
			  " according to expression_constant_p()");
    }
  }
  else
    pips_internal_error("expression is not a constant"
			" according to expression_constant_p()");
  return(rv);
}

/* This function returns a "constant" object if the expression is a
 * constant such as 10, -11 or 2.345 or "foo".
 *
 * Expressions such as "5+0" or "sizeof(int)" or "m", with m defined
 * as "const int m = 2;" or "M" with M defined as PARAMETER do not
 * qualify. However their value is constant, i.e. store independent,
 * and can be evaluated using EvalExpression().
 */
constant expression_constant(expression exp)
{
  syntax s = expression_syntax(exp);
  if(syntax_call_p(s))
    {
      call c = syntax_call(expression_syntax(exp));
      value v = entity_initial(call_function(c));
      switch(value_tag(v))
	{
	case is_value_constant:
	  return value_constant(v);
	case is_value_intrinsic: {
	  entity op = call_function(c);
	  if(ENTITY_UNARY_MINUS_P(op)||ENTITY_UNARY_PLUS_P(op))
	    return expression_constant(binary_call_lhs(c));
	}
	default:
	  ;
	}
    }
  else if(false && syntax_sizeofexpression_p(s)) {
    sizeofexpression soe = syntax_sizeofexpression(s);
    /* FI: difficult to make a decision here. We may have a constant
       expression that cannot be evaluated as a constant statically by
       PIPS. What is the semantics of expression_constant_p()?  */
    if(sizeofexpression_type_p(soe) /* && get_bool_property("EVAL_SIZEOF")*/ ) {
      /* Too bad we need a function located in eval.c ... */
      value v = EvalSizeofexpression(soe);
      if(value_constant_p(v)) { // Should always be true...
	constant c = value_constant(v);
	value_constant(v) = constant_undefined;
	free_value(v);
	return c;
      }
    }
  }
  else if(false && syntax_reference_p(s)) {
    /* Might be a reference to a scalar constant. */
    reference r = syntax_reference(s);
    entity v = reference_variable(r);
    if(const_variable_p(v)) {
      value val = entity_initial(v);
      if(value_constant_p(val)) {
	constant c = copy_constant(value_constant(val));
	return c;
      }
    }
  }
  return constant_undefined;
}

bool expression_string_constant_p(expression exp) {
  if(expression_constant_p(exp) && expression_call_p(exp) ) {
    call c = expression_call(exp);
    entity operator = call_function(c);
    const char * eun = entity_user_name(operator);
    return ( eun[0]=='"' && eun[strlen(eun)-1] == '"' ) ;
  }
  return false;
}

/* returns a newly allocated string! */
char* expression_string_constant(expression exp) {
  pips_assert("is a string constant", expression_string_constant_p(exp));
    call c = expression_call(exp);
    entity operator = call_function(c);
    const char * eun = entity_user_name(operator);
    return strndup(eun+1,strlen(eun)-2);
}

bool expression_integer_constant_p(e)
expression e;
{
    syntax s = expression_syntax(e);
    normalized n = expression_normalized(e);

    if ((n!=normalized_undefined) && (normalized_linear_p(n)))
    {
	Pvecteur v = normalized_linear(n);
	int s = vect_size(v);

	if (s==0) return(true);
	if (s>1) return(false);
	return((s==1) && value_notzero_p(vect_coeff(TCST,v)));
    }
    else
    if (syntax_call_p(s))
    {
	call c = syntax_call(s);
	value v = entity_initial(call_function(c));

	/* I hope a short evaluation is made by the compiler */
	return((value_constant_p(v)) && (constant_int_p(value_constant(v))));
    }
    
    return(false);
}
/*=================================================================*/
/* bool expression_constant_p(expression exp)
 * Returns true if "exp" is an (integer) constant value.
 *
 * Note : A negativePositive constant can be represented with a call to the unary
 *        minus/plus intrinsic function upon a positive value.
 *
 * See below extended_expression_constant_p() for a more general function.
 */
bool expression_constant_p(expression exp)
{
  return !constant_undefined_p(expression_constant(exp));
}

/* Returns true if the value of the expression does not depend
   syntactically on the current store. Returns false when this has not
   been proved. */
bool extended_expression_constant_p(expression exp)
{
  syntax s = expression_syntax(exp);
  bool constant_p = false;

  switch(syntax_tag(s)) {
  case is_syntax_reference: {
    /* The only constant references in PIPS internal representation
       are references to functions and to declared constants. */
    reference r = syntax_reference(s);
    entity v = reference_variable(r);
    type t = ultimate_type(entity_type(v));

    constant_p = type_functional_p(t) || const_variable_p(v);
    break;
  }
  case is_syntax_range: {
    range r = syntax_range(s);
    expression lb = range_lower(r);
    expression ub = range_upper(r);
    expression inc = range_increment(r);

    constant_p = extended_expression_constant_p(lb)
      && extended_expression_constant_p(ub)
      && extended_expression_constant_p(inc);
    break;
  }
  case is_syntax_call: {
    call c = syntax_call(s);
    entity f = call_function(c);
    value v = entity_initial(f);
    switch(value_tag(v)) {
    case is_value_constant:
      constant_p = true;
      break;
    case is_value_intrinsic: {
      /* Check that all arguments are constant */
      list args = call_arguments(c);
      constant_p = true;
      FOREACH(EXPRESSION, sub_exp, args) {
	constant_p = constant_p && extended_expression_constant_p(sub_exp);
      }
      break;
    }
    case is_value_symbolic:
      /* Certainly true for Fortran. Quid for C? */
      constant_p = true;
      break;
    case is_value_expression: {
      /* does not make much sense... for a function! */
      expression sub_exp = value_expression(v);
      constant_p = extended_expression_constant_p(sub_exp);
      break;
    }
    case is_value_unknown:
    case is_value_code:
    default:
      /* Let's be conservative */
      constant_p = false;
    }
    break;
  }
  case is_syntax_cast: {
    /* There might be another case of constant expressions: all that
       are casted to void... Usage? */
    cast c = syntax_cast(s);
    expression sub_exp = cast_expression(c);
    constant_p = extended_expression_constant_p(sub_exp);
    break;
  }
  case is_syntax_sizeofexpression: {
    /* This should be a constant, except for dependent types. */
    sizeofexpression soe = syntax_sizeofexpression(s);
    type t = type_undefined;
    if(sizeofexpression_type_p(soe))
      t = copy_type(sizeofexpression_type(soe));
    else {
      expression se = sizeofexpression_expression(soe);
      t = expression_to_type(se);
    }
    // constant_p = !dependent_type_p(t);
    constant_p = !variable_length_array_type_p(t);
    free_type(t);
    break;
  }
  case is_syntax_subscript:
    break;
  case is_syntax_application:
    break;
  case is_syntax_va_arg:
    break;
  default:
    pips_internal_error("Unexpected syntax tag %d", syntax_tag(s));
  }
  return constant_p;
}

/* Not all expressions can be used as right-hand side (rhs) in C
   assignments.

   PIPS expressions used to encode initializations such as

   "{ 1, 2, 3}"

   in

   "int k[] = { 1, 2, 3 };" cannot be used as rhs.

   There are probably many more cases, especially with
   Fortran-specific expressions, e.g. IO expressions. But we do not
   have a way to know if an intrinsic is a Fortran or a C or a shared
   intrinsic. The information is not carried by PIPS internal
   representation and hence not initialized in bootstrap.

   Note: this test is too restrictive as the condition depends on the
   type of the lhs. A struct can be assigned a brace expression. So a
   type argument should be passed to make such decisions.

   "s = { 1, 2, 3};" is ok if s is a struct with three integer
   compatible fields.
 */
bool expression_is_C_rhs_p(expression exp)
{
  bool is_rhs_p = false;

  is_rhs_p = !brace_expression_p(exp);

  return is_rhs_p;
}

bool expression_one_p(expression exp)
{
  bool one_p = false;

  if(syntax_call_p(expression_syntax(exp))) {
    call c = syntax_call(expression_syntax(exp));
    value v = entity_initial(call_function(c));

    if(value_constant_p(v)) {
      constant c = value_constant(v);
      one_p = constant_int_p(c) && (constant_int(c)==1);
    }
  }
  return one_p;
}

/**
   returns true if the expression is equal to zero or NULL (even if
   there is a cast before such as in (void *) 0).
*/
bool expression_null_p(expression exp)
{
  bool null_p = false;
  if (expression_cast_p(exp))
    null_p = expression_null_p(cast_expression(expression_cast(exp)));
  else if (expression_reference_p(exp))
    {
      null_p = same_string_p(entity_local_name(expression_variable(exp)), "NULL");
    }
  else
    {
      if (expression_constant_p(exp))
	null_p = (expression_to_int(exp) == 0);
    }
  return null_p;
}


/****************************************************** SAME EXPRESSION NAME */

/* compare two entities for their appearance point of view.
 * used for putting common in includes.
 */

bool same_expression_name_p(expression, expression);

bool same_lexpr_name_p(list l1, list l2)
{
  if (gen_length(l1)!=gen_length(l2))
    return false;
  /* else */
  for(; l1 && l2; POP(l1), POP(l2))
    if (!same_expression_name_p(EXPRESSION(CAR(l1)), EXPRESSION(CAR(l2))))
      return false;
  return true;
}

bool same_entity_lname_p(entity e1, entity e2)
{
  return same_string_p(entity_local_name(e1), entity_local_name(e2));
}

bool same_call_name_p(call c1, call c2)
{
  return same_entity_lname_p(call_function(c1), call_function(c2)) &&
    same_lexpr_name_p(call_arguments(c1), call_arguments(c2));
}

bool same_ref_name_p(reference r1, reference r2)
{
  return same_entity_lname_p(reference_variable(r1), reference_variable(r2))
    && same_lexpr_name_p(reference_indices(r1), reference_indices(r2));
}

bool same_range_name_p(range r1, range r2)
{
  return same_expression_name_p(range_lower(r1), range_lower(r2)) &&
    same_expression_name_p(range_upper(r1), range_upper(r2)) &&
    same_expression_name_p(range_increment(r1), range_increment(r2));
}

bool same_type_name_p(const type t0, const type t1) {
    string s0 = string_of_type(t0),
           s1 =string_of_type(t1);
    bool same = same_string_p(s0,s1);
    free(s0);free(s1);
    return same;
}

bool same_sizeofexpression_name_p(sizeofexpression s0, sizeofexpression s1)
{
    if(sizeofexpression_type_p(s0) && sizeofexpression_type_p(s1))
        return same_type_name_p(sizeofexpression_type(s0),sizeofexpression_type(s1));
    if(sizeofexpression_expression_p(s0) && sizeofexpression_expression_p(s1))
        return same_expression_name_p(sizeofexpression_expression(s0),sizeofexpression_expression(s1));
    return false;
}

bool same_subscript_name_p(subscript ss1, subscript ss2)
{
  return same_expression_name_p(subscript_array(ss1), subscript_array(ss2)) 
     && same_lexpr_name_p(subscript_indices(ss1), subscript_indices(ss2));
}

bool same_cast_name_p(cast cs1, cast cs2)
{
  return same_type_name_p(cast_type(cs1), cast_type(cs2)) &&
    same_expression_name_p(cast_expression(cs1), cast_expression(cs2)) ;
}

bool same_application_name_p(application a1, application a2)
{
  return  same_expression_name_p(application_function(a1), application_function(a2)) &&
   same_lexpr_name_p(application_arguments(a1), application_arguments(a2));
}

bool same_va_arg_name_p(list l1, list l2)
{
  if (gen_length(l1)!=gen_length(l2))
    return false;

  for(; l1 && l2; POP(l1), POP(l2)) {
    sizeofexpression s1 = SIZEOFEXPRESSION(CAR(l1));
    sizeofexpression s2 = SIZEOFEXPRESSION(CAR(l2));
    if (!same_sizeofexpression_name_p(s1, s2))
      return false;
  }
  return true;
}


bool same_syntax_name_p(syntax s1, syntax s2)
{
  if (syntax_tag(s1)!=syntax_tag(s2))
    return false;
  /* else */
  switch (syntax_tag(s1))
    {
    case is_syntax_call:
      return same_call_name_p(syntax_call(s1), syntax_call(s2));
    case is_syntax_reference:
      return same_ref_name_p(syntax_reference(s1), syntax_reference(s2));
    case is_syntax_range:
      return same_range_name_p(syntax_range(s1), syntax_range(s2));
    case is_syntax_sizeofexpression:
      return same_sizeofexpression_name_p(syntax_sizeofexpression(s1),syntax_sizeofexpression(s2));
    case is_syntax_subscript:
      return same_subscript_name_p(syntax_subscript(s1), syntax_subscript(s2));
    case is_syntax_cast:
      return same_cast_name_p(syntax_cast(s1), syntax_cast(s2));
    case is_syntax_application:
      return same_application_name_p(syntax_application(s1), syntax_application(s2));
    case is_syntax_va_arg:
      return same_va_arg_name_p(syntax_va_arg(s1), syntax_va_arg(s2));
    default:
      pips_internal_error("unexpected syntax tag: %d", syntax_tag(s1));
    }
  return false;
}

bool same_expression_name_p(expression e1, expression e2)
{
  return same_syntax_name_p(expression_syntax(e1), expression_syntax(e2));
}

/************************************************************* DAVINCI GRAPH */

#define ALREADY_SEEN(node) (hash_defined_p(seen, (char*)node))
#define SEEN(node) (hash_put(seen, (char*) node, (char*) 1))

#define DV_CIRCLE ",a(\"_GO\",\"circle\")"
#define DV_YELLOW ",a(\"COLOR\",\"yellow\")"

static bool  davinci_dump_expression_rc(
    FILE * out, expression e, hash_table seen)
{
  syntax s;
  const char* name, *shape, *color;
  list sons = NIL;
  bool first = true, something = true;

  if (ALREADY_SEEN(e)) return false;
  SEEN(e);

  s = expression_syntax(e);
  switch (syntax_tag(s))
  {
  case is_syntax_call:
    {
      call c = syntax_call(s);
      name = entity_local_name(call_function(c));
      sons = call_arguments(c);
      shape = "";
      color = "";
      break;
    }
  case is_syntax_range:
    name = "::";
    shape = "";
    color = "";
    break;
  case is_syntax_cast:
    pips_user_warning("skipping cast\n");
    break;
  case is_syntax_reference:
    name = entity_local_name(reference_variable(syntax_reference(s)));
    shape = DV_CIRCLE;
    color = DV_YELLOW;
    break;
  default:
    name = "";
    shape = "";
    color = "";
    pips_internal_error("unexpected syntax tag (%d)", syntax_tag(s));
  }

    /* daVinci node prolog. */
  fprintf(out, "l(\"%zx\",n(\"\",[a(\"OBJECT\",\"%s\")%s%s],[",
	  (_uint) e, name, color, shape);

  MAP(EXPRESSION, son,
  {
    if (!first) fprintf(out, ",\n");
    else { fprintf(out, "\n"); first=false; }
    fprintf(out, " l(\"%zx->%zx\",e(\"\",[],r(\"%zx\")))",
	    (_uint) e, (_uint) son, (_uint) son);
  },
    sons);

    /* node epilog */
  fprintf(out, "]))\n");

  MAP(EXPRESSION, son,
  {
    if (something) fprintf(out, ",");
    something = davinci_dump_expression_rc(out, son, seen);
  }, sons);

  return true;
}

/* dump expression e in file out as a davinci graph.
 */
void davinci_dump_expression(FILE * out, expression e)
{
  hash_table seen = hash_table_make(hash_pointer, 0);
  fprintf(out, "[\n");
  davinci_dump_expression_rc(out, e, seen);
  fprintf(out, "]\n\n");
  hash_table_free(seen);
}

static FILE * out_flt = NULL;
static bool expr_flt(expression e)
{
  davinci_dump_expression(out_flt, e);
  return false;
}

/* dump all expressions in s to out.
 */
void davinci_dump_all_expressions(FILE * out, statement s)
{
  out_flt = out;
  gen_recurse(s, expression_domain, expr_flt, gen_null);
  out_flt = NULL;
}

/* This function replaces all the occurences of an old entity in the
 * expression exp by the new entity. It returns the expression modified.
 * I think we  can write this function by using gen_context_multi_recurse  ... * To do .... NN */
expression substitute_entity_in_expression(entity old, entity new, expression e)
{
  syntax s;
  tag t;
  call c;
  range ra;
  reference re;
  list args,tempargs= NIL;
  expression retour = copy_expression(e), exp,temp,low,up,inc;

  s = expression_syntax(e);
  t = syntax_tag(s);
  switch (t){
  case is_syntax_call:
    {
      c = syntax_call(s);
      args = call_arguments(c);
      while (!ENDP(args))
	{
	  exp = EXPRESSION(CAR(args));
	  temp = substitute_entity_in_expression(old,new,exp);
	  tempargs = gen_nconc(tempargs,CONS(EXPRESSION,temp,NIL));
	  args = CDR(args);
	}

      call_arguments(syntax_call(expression_syntax(retour))) = tempargs;

      if (same_entity_p(call_function(c),old))
	call_function(syntax_call(expression_syntax(retour))) = new;
      else
	call_function(syntax_call(expression_syntax(retour))) = call_function(c);

      break;
    }
  case is_syntax_reference:
    {
      re = syntax_reference(s);
      args = reference_indices(re);
      while (!ENDP(args))
	{
	  exp = EXPRESSION(CAR(args));
	  temp = substitute_entity_in_expression(old,new,exp);
	  tempargs = gen_nconc(tempargs,CONS(EXPRESSION,temp,NIL));
	  args = CDR(args);
	}

      reference_indices(syntax_reference(expression_syntax(retour))) = tempargs;

      if (same_entity_p(reference_variable(re),old))
	reference_variable(syntax_reference(expression_syntax(retour))) = new;
      else
	reference_variable(syntax_reference(expression_syntax(retour))) = reference_variable(re);

      break;
    }
  case is_syntax_range:
    {
      ra = syntax_range(s);
      low = range_lower(ra);
      range_lower(syntax_range(expression_syntax(retour))) = substitute_entity_in_expression(old,new,low);

      up = range_upper(ra);
      range_upper(syntax_range(expression_syntax(retour))) = substitute_entity_in_expression(old,new,up);


      inc = range_increment(ra);
      range_increment(syntax_range(expression_syntax(retour))) = substitute_entity_in_expression(old,new,inc);

      break;
    }
  }

  return retour;
}

/* Replace C operators "+C" and "-C" which can handle pointers by
   arithmetic operators "+" and "-" when it is safe to do so, i.e. when no
   pointer arithmetic is involved.

   FI: Also, it might be useful to normalize the expression in order
   not to leave an undefined field in it. But this is a recursive
   function and probably not the right place to cope with this.
 */
bool simplify_C_expression(expression e)
{
  syntax s = expression_syntax(e);
  bool can_be_substituted_p = false;

  pips_debug(9, "Begin\n");

  switch(syntax_tag(s)) {
  case is_syntax_reference:
    {
      entity re = reference_variable(syntax_reference(s));
      type rt = entity_type(re);

      if(type_undefined_p(rt)) {
	/* FI: see C_syntax/block_scope12.c. The source code line
	   number where the problem occurs cannot be given because we
	   are not in the c_syntax library. */
	pips_user_warning("Variable \"%s\" is probably used before it is defined\n",
			  entity_user_name(re));
	can_be_substituted_p = false;
      }
      else {
	basic bt = basic_undefined;

	if(type_variable_p(rt)) { /* FI: What if not? core dump? */
	  /* The variable type can hide a functional type via a
	     typedef */
	  type urt = ultimate_type(rt);
	  if(type_variable_p(urt)) {
	    bt = variable_basic(type_variable(urt));

	    can_be_substituted_p =
	      basic_int_p(bt)
	      || basic_float_p(bt)
	      || basic_overloaded_p(bt) /* Might be wrong, but necessary */
	      || basic_complex_p(bt) /* Should not occur in old C code */
	      || basic_logical_p(bt); /* Should not occur in old C code */

	    pips_debug(9, "Variable %s is an arithmetic variable: %s\n",
		       entity_local_name(re), bool_to_string(can_be_substituted_p));
	  }
	}
      }
      break; /* FI: The index expressions should be simplified too... */
    }
  case is_syntax_call:
    {
      call c = syntax_call(s);

      if(expression_constant_p(e)) {
	/* What is the type of the constant? */
	entity cste = call_function(c);
	basic rb = variable_basic(type_variable(functional_result(type_functional(entity_type(cste)))));

	can_be_substituted_p =
	  basic_int_p(rb)
	  || basic_float_p(rb)
	  || basic_complex_p(rb); /* Should not occur in C, before C99 */
      }
      else if(gen_length(call_arguments(c))==2) {
	/* Check "+C" and "-C" */
	expression e1 = binary_call_lhs(c);
	expression e2 = binary_call_rhs(c);
	bool can_be_substituted_p1 = simplify_C_expression(e1);
	bool can_be_substituted_p2 = simplify_C_expression(e2);
	can_be_substituted_p = can_be_substituted_p1 && can_be_substituted_p2;
	if(can_be_substituted_p) {
	  entity op = call_function(c);
	  if(ENTITY_PLUS_C_P(op)) {
	    call_function(c) = entity_intrinsic(PLUS_OPERATOR_NAME);
	  }
	  else if(ENTITY_MINUS_C_P(op)) {
	    call_function(c) = entity_intrinsic(MINUS_OPERATOR_NAME);
	  }
	}
      }
      else {
	/* Try to simplify the arguments, do not hope much from the result
           type because of overloading. */
	type ft = call_to_functional_type(c, true);
	type rt = ultimate_type(functional_result(type_functional(ft)));

	//pips_assert("The function type is functional", type_functional_p(entity_type(f)));

	MAP(EXPRESSION, se, {
	    (void) simplify_C_expression(se);
	  }, call_arguments(c));

	if(type_variable_p(rt)) {
	  basic rb = variable_basic(type_variable(rt));

	  if(basic_overloaded_p(rb)) {
	    /* a void expression such as (void) 0 results in an undefined basic. */
	    rb = basic_of_expression(e);
	  }
	  else
	    rb = copy_basic(rb);

	  if(!basic_undefined_p(rb)) {
	    /* FI: I guess, typedef equivalent to those could also be declared substituable */
	    can_be_substituted_p =
	      basic_int_p(rb)
	      || basic_float_p(rb)
	      || basic_complex_p(rb); /* Should not occur in C */
	    free_basic(rb);
	  }
	  else {
	    /* e must be a void expression, i.e. an expression returning no value */
	    can_be_substituted_p = false;
	  }
	}
	else {
	  can_be_substituted_p = false;
	}
      }
      break;
    }
  case is_syntax_range:
    {
      range r = syntax_range(s);
      expression le = range_lower(r);
      expression ue = range_upper(r);
      expression ince = range_increment(r);
      (void) simplify_C_expression(le);
      (void) simplify_C_expression(ue);
      (void) simplify_C_expression(ince);
      can_be_substituted_p = false;
      break;
      }
  case is_syntax_cast:
  case is_syntax_sizeofexpression:
  case is_syntax_subscript:
  case is_syntax_application:
  case is_syntax_va_arg:
      can_be_substituted_p = false;
      break;
  default: pips_internal_error("Bad syntax tag");
    can_be_substituted_p = false; /* for gcc */
  }

  pips_debug(9, "End: %s\n", bool_to_string(can_be_substituted_p));
  return can_be_substituted_p;
}

/* Replace a C expression used as FOR bound by a Fortran DO bound
   expression, taking into account the C comparison operator used.
*/
expression convert_bound_expression(expression e, bool upper_p, bool non_strict_p)
{
  expression b = expression_undefined;

  if(non_strict_p) {
    b = copy_expression(e);
  }
  else {
    /* */
    intptr_t ib = 0;
    intptr_t nb = 0;

    /* Fi this test is too strong to preserve source code */
    if(false && expression_integer_value(e, &ib)) {
      /* The offset might not be plus or minus one, unless we know the
	 index is an integer? Does it depend on the step value? More
	 thought needed than available tonight (FI) */
      nb = upper_p? ib-1 : ib+1;
      b = int_to_expression(nb);
    }
    else if(expression_constant_p(e)) {
      /* The offset might not be plus or minus one, unless we know the
	 index is an integer? Does it depend on the step value? More
	 thought needed than available tonight (FI) */
      ib = expression_to_int(e);
      nb = upper_p? ib-1 : ib+1;
      b = int_to_expression(nb);
    }
    else if(NORMALIZE_EXPRESSION(e), expression_linear_p(e)) {
      /* May modify the source code a bit more than necessary, but
	 avoids stupid expressions such as 64-1-1 */
      normalized n = expression_normalized(e);
      Pvecteur v = normalized_linear(n);
      /* This could be generalized to any affine expression. See for
	 instance loop_bound02.c. But the source code is likely to be
	 disturbed if the bound expression is regenerated from v after
	 adding or subtracting 1 from its constant term. */
      if(vect_constant_p(v) || VECTEUR_NUL_P(v)) {
	Value c = vect_coeff(TCST, v);
	ib = (int) c;
	nb = upper_p? ib-1 : ib+1;
	b = int_to_expression(nb);
      }
    }
    if(expression_undefined_p(b)) {
      expression offset = int_to_expression(1);
      entity op = entity_intrinsic(upper_p? MINUS_OPERATOR_NAME : PLUS_OPERATOR_NAME);

      b = MakeBinaryCall(op, copy_expression(e), offset);
    }
  }
  return b;
}

bool reference_with_constant_indices_p(reference r)
{
  list sel = reference_indices(r);
  bool constant_p = true;

  MAP(EXPRESSION, se, {
      if(!extended_integer_constant_expression_p(se)) {
	constant_p = false;
	break;
      }
    }, sel);
  return constant_p;
}

/* Return by side effect a reference whose memory locations includes
   the memory locations of r in case the subcript expressions are
   changed by a store change.

   Constant subscript expressions are preserved.

   Store varying subscript expressions are replaced by unbounded expressions.
 */
reference reference_with_store_independent_indices(reference r)
{
  list sel = reference_indices(r);
  list sec = list_undefined;

  for(sec = sel; !ENDP(sec); POP(sec)) {
    expression se = EXPRESSION(CAR(sec));

    if(!extended_integer_constant_expression_p(se)) {
      free_expression(se);
      EXPRESSION_(CAR(sec)) = make_unbounded_expression();
    }
  }

  return r;
}

/* indices can be constant or unbounded: they are store independent. */
bool reference_with_unbounded_indices_p(reference r)
{
  list sel = reference_indices(r);
  bool unbounded_p = true;

  MAP(EXPRESSION, se, {
      if(!extended_integer_constant_expression_p(se)
	 && !unbounded_expression_p(se)) {
	unbounded_p = false;
	break;
      }
    }, sel);
  return unbounded_p;
}

/* Does this reference define the same set of memory locations
   regardless of the current (environment and) memory state?
 */
bool store_independent_reference_p(reference r)
{
  bool independent_p = true;
  //list ind = reference_indices(r);
  entity v = reference_variable(r);
  type t = ultimate_type(entity_type(v));

  if(pointer_type_p(t)) {
    independent_p = false;
  }
  else {
    independent_p = reference_with_constant_indices_p(r);
  }

  return independent_p;
}

void check_user_call_site(entity func, list args)
{
  /* check the number of parameters */
  list l_formals = module_formal_parameters(func);
  int n_formals = (int) gen_length(l_formals);

  if ((int) gen_length(args) < n_formals)
    {
      /* this is really a user error.
       * if you move this as a user warning, the pips would drop
       * effects about unbounded formals... why not? FC.
       */
      fprintf(stderr,"%d formal arguments for module %s:\n",
	      n_formals,module_local_name(func));
      dump_arguments(l_formals);
      fprintf(stderr,"%zd actual arguments:\n",gen_length(args));
      print_expressions(args);
      pips_user_error("\nCall to module %s: "
		      "insufficient number of actual arguments.\n",
		      module_local_name(func));
    }
  else if ((int) gen_length(args) > n_formals)
    {
      /* This can be survived... */
      fprintf(stderr,"%d formal arguments for module%s:\n",
	      n_formals,module_local_name(func));
      dump_arguments(l_formals);
      fprintf(stderr,"%zd actual arguments:\n",gen_length(args));
      print_expressions(args);

      pips_user_warning("\nCall to module %s: "
			"too many actual arguments.\n",
			module_local_name(func));
    }
  gen_free_list(l_formals), l_formals=NIL;
}

/* just returns the entity of an expression, or entity_undefined
 *
 * The entity returned is either the function called or the variable
 * referenced
 *
 * SG: moved here from hpfc
 */
entity expression_to_entity(expression e)
{
    syntax s = expression_syntax(e);

    switch (syntax_tag(s))
    {
    case is_syntax_call:
	return call_function(syntax_call(s));
    case is_syntax_reference:
	return reference_variable(syntax_reference(s));
    case is_syntax_range:
    case is_syntax_cast:
    case is_syntax_sizeofexpression:
    case is_syntax_subscript:
    case is_syntax_application:
    case is_syntax_va_arg:
    default:
	return entity_undefined;
    }
}
/* map expression_to_entity on expressions */
list expressions_to_entities(list expressions)
{
    list entities =NIL;
    FOREACH(EXPRESSION,exp,expressions)
        entities=CONS(ENTITY,expression_to_entity(exp),entities);
    return gen_nreverse(entities);
}


/**
 * @brief perform the real similarity comparaison between two expressions
 * @a target is matched against @a pattern, and expression <> argument is stored in @a symbols
 *
 * @param target cheked expression
 * @param pattern pattern expression
 * @param symbols map storing entity <> expression tuple
 *
 * @return true if similar
 */
static bool _expression_similar_p(expression target, expression pattern,hash_table symbols)
{
    bool similar=true;
    syntax starget = expression_syntax(target),
           spattern = expression_syntax(pattern);

    /* cast handler */
    if( syntax_cast_p( spattern ) )
    {
        pips_user_warning("cast ignored\n");
        return _expression_similar_p(target, cast_expression(syntax_cast(spattern)),symbols);
    }
    if( syntax_cast_p( starget ) )
    {
        pips_user_warning("cast ignored\n");
        return _expression_similar_p(cast_expression(syntax_cast(starget)), pattern,symbols);
    }

    switch(syntax_tag(spattern) )
    {
        /* we memorize reference from target and pattern in the symbol table
         * similar to \1 in sed
         */
        case is_syntax_reference:
            {
		    reference r = syntax_reference(spattern);
		    /* scalar reference always matches.
             * SG: in fact, you have to check for type compatibility too ...
             * Allows to match malloc(a) with more complex expressions like malloc(1 +
		     * strlen("...")).
             * Interferes incorrectly with commutativity. */
		    if (! expression_scalar_p(pattern))
			    similar = false;
		    else {
                basic bpattern = basic_of_expression(pattern),
                      btarget = basic_of_expression(target);
                basic bmax = basic_maximum(bpattern,btarget);
                if(basic_overloaded_p(bmax) || basic_equal_p(bmax,bpattern)) {
                    hash_put(symbols,entity_name(reference_variable(r)), target);
                }
                else
                    similar = false;
                free_basic(bmax);
                free_basic(bpattern);
                free_basic(btarget);
            }
            } break;
            /* recursively compare each arguments if call do not differ */
        case is_syntax_call:
            if( syntax_call_p(starget) &&
                    same_entity_p( call_function(syntax_call(starget)), call_function(syntax_call(spattern)) ) )
            {
                call cpattern = syntax_call(spattern);
                call ctarget = syntax_call(starget);
                if(commutative_call_p(cpattern))
                {
                    pips_assert("pips commutative call have only two arguments\n",gen_length(call_arguments(cpattern))==2);
                    expression lhs_pattern = binary_call_lhs(cpattern),
                               rhs_pattern = binary_call_rhs(cpattern),
                               lhs_target = binary_call_lhs(ctarget),
                               rhs_target = binary_call_rhs(ctarget);
                    similar = (_expression_similar_p(lhs_target,lhs_pattern,symbols) && _expression_similar_p(rhs_target,rhs_pattern,symbols))
                        ||
                        (_expression_similar_p(lhs_target,rhs_pattern,symbols) && _expression_similar_p(rhs_target,lhs_pattern,symbols))
                        ;

                }
                else
                {
                    list iter = call_arguments(cpattern);
                    FOREACH(EXPRESSION, etarget, call_arguments(ctarget) )
                    {
                        if( ENDP(iter) ) { similar = false; break; }/* could occur with va args */
                        expression epattern = EXPRESSION(CAR(iter));
                        similar&= _expression_similar_p(etarget,epattern,symbols);
                        POP(iter);
                        if(!similar)
                            break;
                    }
                }
            }
            else
            {
                similar =false;
            }
            break;
            /* SG: will this be usefull ?*/
        case is_syntax_range:
            similar = syntax_range_p(starget) &&
                _expression_similar_p(range_lower(syntax_range(starget)),range_lower(syntax_range(spattern)),symbols) &&
                _expression_similar_p(range_upper(syntax_range(starget)),range_upper(syntax_range(spattern)),symbols) &&
                _expression_similar_p(range_increment(syntax_range(starget)),range_increment(syntax_range(spattern)),symbols);
            break;

            /* SG:not supported yet */
        case is_syntax_cast:
            pips_user_warning("cast ignored\n");
            similar = _expression_similar_p(cast_expression(syntax_cast(starget)),pattern,symbols);
            break;

        case is_syntax_sizeofexpression:
            if( syntax_sizeofexpression_p(starget) )
            {
                sizeofexpression seo_target = syntax_sizeofexpression(starget);
                sizeofexpression seo_pattern = syntax_sizeofexpression(spattern);
                if( sizeofexpression_type(seo_pattern) )
                    similar = sizeofexpression_type_p(seo_target) &&
                        type_equal_p( sizeofexpression_type(seo_target), sizeofexpression_type(seo_pattern) );
                else
                    similar = _expression_similar_p(sizeofexpression_expression(seo_target),
                            sizeofexpression_expression(seo_pattern),
                            symbols );
            }
            else
            {
                similar =false;
            }
            break;

        case is_syntax_subscript:
            if( syntax_subscript_p(starget) )
            {
                subscript sub_target = syntax_subscript(starget),
                          sub_pattern = syntax_subscript(spattern);
                similar&= _expression_similar_p( subscript_array(sub_target), subscript_array(sub_pattern),symbols );

                list iter = subscript_indices(sub_pattern);
                FOREACH(EXPRESSION, etarget, subscript_indices(sub_target) )
                {
                    if( ENDP(iter) ) { similar = false; break; }/* could occur with va args */
                    expression epattern = EXPRESSION(CAR(iter));
                    similar&= _expression_similar_p(etarget,epattern,symbols);
                    POP(iter);
                }
            }
            else
                similar =false;
            break;
        case is_syntax_application:
            pips_user_warning("application similarity not implemented yet\n");
            similar=false;
            break;
        case is_syntax_va_arg:
            pips_user_warning("va_arg similarity not implemented yet\n");
            similar=false;
            break;
    };
    return similar;
}
/**
 * @brief similar to expression_similar_p but the hash_map
 * containing the crossref value is retured for further use
 *
 * @param target expression to compare
 * @param pattern expression serving as pattern
 * @param symbol_table pointer to unallocated hash_map, in the end it will contain a set of pair (original syntax reference name, substituted syntax element). Must be freed, but only if expressions are similar
 *
 * @return true if expressions are similar
 */
bool expression_similar_get_context_p(expression target, expression pattern, hash_table* symbol_table)
{
    *symbol_table = hash_table_make(hash_string,HASH_DEFAULT_SIZE);
    bool similar = _expression_similar_p(target,pattern,*symbol_table);
    if( !similar) hash_table_free(*symbol_table);
    return similar;
}

/**
 * @brief compare if two expressions are similar
 * that is can we exchange target and pattern by substituing
 * variables
 * examples:
 *  1+2 ~  a+b
 *  a+b !~ a+2
 *  1+b ~  1+c
 * @param target expression that sould match with pattern
 * @param pattern the pattern to match
 *
 * @return true if expressions are similar
 */
bool expression_similar_p(expression target, expression pattern)
{
  hash_table symbol_table = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
  bool similar = _expression_similar_p(target,pattern,symbol_table);
  hash_table_free(symbol_table);
  return similar;
}

list /* of expression */
make_list_of_constant(int val,    /* the constant value */
		      int number) /* the length of the created list */
{
  list l=NIL;

  pips_assert("valid number", number>=0);
  for(; number; number--)
    l = CONS(EXPRESSION, int_to_expression(val), l);

  return l;
}

/**
 * Return bool indicating if expression e is a brace expression
 */
bool brace_expression_p(expression e)
{
    if (expression_call_p(e))
    {
        entity f = call_function(syntax_call(expression_syntax(e)));
        if (ENTITY_BRACE_INTRINSIC_P(f))
            return true;
    }
    return false;
}

/* helper for brace_expression_to_statements */
static list do_brace_expression_to_statements(entity arr,expression e, list curr_indices) {
  int new_index = 0;
  list out = NIL;
  FOREACH(EXPRESSION,arg,call_arguments(expression_call(e))) {
    expression ee = int_to_expression(new_index);
    list ind = gen_append(gen_full_copy_list(curr_indices),make_expression_list(ee));
    if(brace_expression_p(arg)) {
      list out_bis = do_brace_expression_to_statements(arr,arg,ind);
      gen_full_free_list(ind);
      out=gen_append(out,out_bis);
    }
    else {
      out=gen_append(out,make_statement_list(
            make_assign_statement(
              reference_to_expression(
                make_reference(arr,ind)
                ),
              copy_expression(arg)
              )
            )
          );
    }
    ++new_index;
  }
  return out;
}

/* converts a brace expression used to initialize an array (not a struct yet)
 * into a statement sequence
 */
list brace_expression_to_statements(entity arr, expression e) {
  pips_assert("is a brace expression\n",brace_expression_p(e));
  pips_assert("is an array\n",array_entity_p(arr));//SG: needs work to support structures
  list curr_index = NIL;
  list out = do_brace_expression_to_statements(arr,e,curr_index);
  return out;
}
/* This function returns true if Reference r is scalar
*/

bool reference_scalar_p(reference r)
{
  entity v = reference_variable(r);
  assert(!reference_undefined_p(r) && r!=NULL && v!=NULL);
  return (reference_indices(r) == NIL &&  entity_scalar_p(v));
}

/**
   @brief take a list of expression and apply a binary operator between all
   of them and return it as an expression
   @return the  operations as an expression
   @param l_exprs, the list of expressions to compute with the operator
   @param op, the binary operator to apply
 **/
expression expressions_to_operation (const list l_exprs, entity op) {
  expression result = expression_undefined;
  if ((l_exprs != NIL) && (op != entity_undefined)){
    list l_src = l_exprs;
    result = EXPRESSION (CAR (l_src));
    POP(l_src);
    FOREACH (EXPRESSION, ex, l_src) {
      list args =  gen_expression_cons (result, NIL);
      args = gen_expression_cons (ex, args);
      call c = make_call (op, args);
      result = call_to_expression (c);
    }
  }
  return result;
}

/**
 *  frees expression syntax of @p e
 *  and replace it by the new syntax @p s
 */
void update_expression_syntax(expression e, syntax s)
{
    unnormalize_expression(e);
    free_syntax(expression_syntax(e));
    expression_syntax(e)=s;
}
/* replace expression @p caller by expression @p field , where @p field is contained by @p caller */
void local_assign_expression(expression caller, expression field)
{
     syntax s = expression_syntax(field) ;
     expression_syntax(field)=syntax_undefined;
     free_syntax(expression_syntax(caller));
     expression_syntax(caller)=s;
     free_normalized(expression_normalized(caller));
}

/* generates an expression from a syntax */
expression syntax_to_expression(syntax s) {
  return make_expression(
      s,
      normalized_undefined
      );
}

/**
 * very simple conversion from string to expression
 * only handles entities and numeric values at the time being
 */
entity string_to_entity(const char * s,entity module)
{
    if(empty_string_p(s)) return entity_undefined;

    /* try float conversion */
    string endptr;
    const char *module_name=module_local_name(module);
    long int l = strtol(s,&endptr,10);
    if(!*endptr) {
        if(l>=0)
            return int_to_entity(l);
        else /* no negative integer entity in pips */
            return entity_undefined;
    }
    float f = strtof(s,&endptr);
    if(!*endptr) return float_to_entity(f);

    entity candidate = entity_undefined;
    /* first find all relevant entities */
    FOREACH(ENTITY,e,entity_declarations(module))
    {
        /* this an heuristic to find the one with a suiting scope
         * error prone*/
        if(same_string_p(entity_user_name(e),s) )
            if(entity_undefined_p(candidate) ||
                    strlen(entity_name(candidate)) > strlen(entity_name(e)))
                candidate=e;
    }
    /* try at the compilation unit level */
    if(entity_undefined_p(candidate))
        candidate=FindEntity(compilation_unit_of_module(module_name),s);
    /* try at top level */
    if(entity_undefined_p(candidate))
        candidate=FindEntity(TOP_LEVEL_MODULE_NAME,s);
    return candidate;
}

/* try to parse @p s in the context of module @p module
 * only simple expressions are found */
expression string_to_expression(const char * s,entity module)
{
    entity e = string_to_entity(s,module);
    if(entity_undefined_p(e)) {
        /* try to find simple expression */
        /* unary operators */
        for(const char *iter = s ; *iter ; iter++) {
            if(isspace(*iter)) continue;
            if(*iter=='-') {
                expression etmp = string_to_expression(iter+1, module);
                if(!expression_undefined_p(etmp)) {
                    return MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),
                            etmp);
                }
            }
        }
        
        /*binary operators*/
        static const char* seeds[] = {   PLUS_OPERATOR_NAME, MINUS_OPERATOR_NAME,MULTIPLY_OPERATOR_NAME, DIVIDE_OPERATOR_NAME};
	for(int i=0; i < (int) (sizeof(seeds)/sizeof(seeds[0])); i++) {
            char *where = strchr(s,seeds[i][0]);
            if(where) {
                char * head = strdup(s);
                char * tail = head + (where -s) +1 ;
                head[where-s]='\0';
                expression e0 = string_to_expression(head,module);
                expression e1 = string_to_expression(tail,module);
                free(head);
                if(!expression_undefined_p(e0) &&!expression_undefined_p(e1)) {
                    return MakeBinaryCall(
                            entity_intrinsic(seeds[i]),
                            e0,
                            e1
                            );
                }
                else {
                    free_expression(e0);
                    free_expression(e1);
                }
            }
        }
        return expression_undefined;
    }
    else
        return entity_to_expression(e);
}
/* split a string using @p seed as separator
 * and call string_to_expression on each chunk */
list string_to_expressions(const char * str, const char * seed, entity module) {
    list strings = strsplit(str,seed);
    list expressions = NIL;
    FOREACH(STRING,s,strings) {
        expression expr = string_to_expression(s,module);
        if(!expression_undefined_p(expr)) {
            expressions = CONS(EXPRESSION,
                    expr,
                    expressions);
        }
    }
    gen_map(free,strings);
    gen_free_list(strings);
    return gen_nreverse(expressions);
}
/* split a string using @p seed as separator
 * and call string_to_entity on each chunk */
list string_to_entities(const char * str, const char * seed, entity module) {
    list strings = strsplit(str,seed);
    list entities = NIL;
    FOREACH(STRING,s,strings) {
        entity e = string_to_entity(s,module);
        if(!entity_undefined_p(e)) {
            entities = CONS(ENTITY,
                    e,
                    entities);
        }
    }
    gen_map(free,strings);
    gen_free_list(strings);
    return gen_nreverse(entities);

}

/* converts a monome to an expression */
expression monome_to_expression(Pmonome pm)
{
    if (MONOME_UNDEFINED_P(pm))
        return expression_undefined;
    else {
        expression coeff;
        float x= monome_coeff(pm);
        if(x == (int)x)
            coeff = (x==1.f)? expression_undefined:int_to_expression((int)x);
        else
            coeff = float_to_expression(x);
        expression term = expression_undefined;
        for(Pvecteur v = monome_term(pm);!VECTEUR_NUL_P(v);v=vecteur_succ(v)) {
            Value exp = vecteur_val(v);
            Variable var = vecteur_var(v);
            expression tmp ;
            if(exp==0||var==TCST) tmp = int_to_expression(1); 
            else {
                Value val = exp>0 ? exp: -exp;
                tmp = entity_to_expression((entity)var);
                while(--val)
                    tmp=make_op_exp(MULTIPLY_OPERATOR_NAME,tmp,entity_to_expression((entity)var));
                if(exp<0)
                    tmp=make_op_exp(DIVIDE_OPERATOR_NAME,int_to_expression(1),tmp);
            }
            term=expression_undefined_p(term)?tmp:make_op_exp(MULTIPLY_OPERATOR_NAME,term,tmp);
        }
        return expression_undefined_p(coeff)?term:
            make_op_exp(MULTIPLY_OPERATOR_NAME, coeff, term);
    }
}

/* converts a polynomial to expression */
expression polynome_to_expression(Ppolynome pp)
{
    expression r =expression_undefined;

    if (POLYNOME_UNDEFINED_P(pp))
        r = expression_undefined;
    else if (POLYNOME_NUL_P(pp))
        r = int_to_expression(0);
    else {
        while (!POLYNOME_NUL_P(pp)) {
            expression s =	monome_to_expression(polynome_monome(pp));
            if(expression_undefined_p(r)) r=s;
            else
                r=make_op_exp(PLUS_OPERATOR_NAME, r, s);
            pp = polynome_succ(pp);
        }
    }
    return r;
}
/*============================================================================*/
/* Ppolynome expression_to_polynome(expression exp): translates "exp" into a
 * polynome. This transformation is feasible if "exp" contains only scalars and
 * the four classical operations (+, -, *, /).
 *
 * The translation is done straightforwardly and recursively.
 *
 * it returns a POLYNOME_UNDEFINED if the conversion failed
 */
Ppolynome expression_to_polynome(expression exp)
{
    Ppolynome pp_new=POLYNOME_UNDEFINED; /* This the resulting polynome */
    syntax sy = expression_syntax(exp);

    switch(syntax_tag(sy))
    {
        case is_syntax_reference:
            {
                reference r = syntax_reference(sy);
                entity en = reference_variable(r);

                if(entity_scalar_p(en)||(entity_pointer_p(en)&&ENDP(reference_indices(r))))
                    pp_new = make_polynome(1.0, (Variable) en, (Value) 1);
                break;
            }
        case is_syntax_call:
            {
                /* Two cases : _ a constant
                 *             _ a "real" call, ie an intrinsic or external function
                 */
                if (expression_constant_p(exp)) {
                    float etof = expression_to_float(exp);
                    pp_new = make_polynome( etof,
                            (Variable) entity_undefined, (Value) 0);
                    /* We should have a real null polynome : 0*TCST^1 AL, AC 04 11 93
                     *else  {
                     *  Pmonome pm = (Pmonome) malloc(sizeof(Smonome));
                     * monome_coeff(pm) = 0;
                     *  monome_term(pm) = vect_new(TCST, 1);
                     *  pp_new = monome_to_new_polynome(pm);
                     *}
                     */
                }
                else
                {
                    int cl;
                    expression arg1, arg2 = expression_undefined;
                    entity op_ent = call_function(syntax_call(sy));

                    /* The call must be one of the four classical operations:
                     *	+, - (unary or binary), *, /
                     */
                    if(ENTITY_FIVE_OPERATION_P(op_ent))
                    {
                        /* This call has one (unary minus) or two (binary plus, minus,
                         * multiply or divide) arguments, no less and no more.
                         */
                        cl = gen_length(call_arguments(syntax_call(sy)));
                        if( (cl != 2) && (cl != 1) )
                            pips_internal_error("%s call with %d argument(s)",
                                    entity_local_name(op_ent), cl);

                        arg1 = EXPRESSION(CAR(call_arguments(syntax_call(sy))));
                        if(cl == 2)
                            arg2 = EXPRESSION(CAR(CDR(call_arguments(syntax_call(sy)))));

                        if (ENTITY_PLUS_P(op_ent)||ENTITY_PLUS_C_P(op_ent)) /* arg1 + arg2 */
                        {
                            pp_new = expression_to_polynome(arg1);
                            if(!POLYNOME_UNDEFINED_P(pp_new))
                                polynome_add(&pp_new, expression_to_polynome(arg2));
                        }
                        else if(ENTITY_MINUS_P(op_ent)||ENTITY_MINUS_C_P(op_ent)) /* -arg2 + arg1 */
                        {
                            pp_new = expression_to_polynome(arg2);
                            if(!POLYNOME_UNDEFINED_P(pp_new)) {
                                polynome_negate(&pp_new);
                                Ppolynome parg1 = expression_to_polynome(arg1);
                                if(!POLYNOME_UNDEFINED_P(parg1))
                                    polynome_add(&pp_new, parg1);
                                else {
                                    polynome_rm(&pp_new);
                                    pp_new=POLYNOME_UNDEFINED;
                                }
                            }
                        }
                        else if(ENTITY_MULTIPLY_P(op_ent)) /* arg1 * arg2 */ {
                            Ppolynome p1 = expression_to_polynome(arg1);
                            if(!POLYNOME_UNDEFINED_P(p1)) {
                                Ppolynome p2 = expression_to_polynome(arg2);
                                if(!POLYNOME_UNDEFINED_P(p2)) {
                                    pp_new = polynome_mult(p1,p2);
                                }
                            }
                        }
                        else if(ENTITY_DIVIDE_P(op_ent)) /* arg1 / arg2 */ {
                            Ppolynome p1 = expression_to_polynome(arg1);
                            if(!POLYNOME_UNDEFINED_P(p1)) {
                                Ppolynome p2 = expression_to_polynome(arg2);
                                if(!POLYNOME_UNDEFINED_P(p2)) {
                                    pp_new = polynome_div(p1,p2);
                                }
                            }
                        }
                        else /* (ENTITY_UNARY_MINUS_P(op_ent)) : -arg1 */
                        {
                            pp_new = expression_to_polynome(arg1);
                            if(!POLYNOME_UNDEFINED_P(pp_new)) {
                                polynome_negate(&pp_new);
                            }
                        }
                    }
                }

                break;
            }
        default :
            {
                pp_new=POLYNOME_UNDEFINED;
            }
    }
    return(pp_new);
}

/* use polynomials to simplify an expression
 * in some cases this operation can change the basic of the expression. 
 * E.g. n/4 -> .25 * n
 * In that case we just undo the simplification
 */
bool simplify_expression(expression * pexp) {
    expression exp = *pexp;
    bool result =false;
    if(!expression_undefined_p(exp)) {
        basic oldb = basic_of_expression(exp);
        Ppolynome pu = expression_to_polynome(exp);
        if((result=!POLYNOME_UNDEFINED_P(pu))) {
            expression pue = polynome_to_expression(pu);
            basic nbasic = basic_of_expression(pue);
            if(basic_equal_p(oldb,nbasic)) {
                free_expression(exp);
                *pexp=pue;
            }
            else {
                free_expression(pue);
            }
            free_basic(nbasic);
            polynome_rm(&pu);
        }
        free_basic(oldb);
    }
    return result;
}

static void do_simplify_expressions(call c) {
    for(list iter = call_arguments(c);!ENDP(iter);POP(iter)) {
        expression *pexp = (expression*)REFCAR(iter);
        simplify_expression(pexp);
    }
}

void simplify_expressions(void *obj) {
    gen_recurse(obj,call_domain,gen_true, do_simplify_expressions);
}

/* call maxima to simplify an expression 
 * prefer simplify_expression !*/
bool maxima_simplify(expression *presult) {
    bool success = true;
    expression result = *presult;
    /* try to call maxima to simplify this expression */
    if(!expression_undefined_p(result) ) {
        list w = words_expression(result,NIL);
        string str = words_to_string(w);
        gen_free_list(w);
        char * cmd;
        asprintf(&cmd,"maxima -q --batch-string \"string(fullratsimp(%s));\"\n",str);
        free(str);
        FILE* pout = popen(cmd,"r");
        if(pout) {
            /* strip out banner */
            fgetc(pout);fgetc(pout);
            /* look for first % */
            while(!feof(pout) && fgetc(pout)!='%');
            if(!feof(pout)) {
                /* skip the three next chars */
                fgetc(pout);fgetc(pout);fgetc(pout);
                /* parse the output */
                char bufline[strlen(cmd)];
                if(fscanf(pout," %s\n",&bufline[0]) == 1 ) {
                    expression exp = string_to_expression(bufline,get_current_module_entity());
                    if(!expression_undefined_p(exp)) {
                        free_expression(result);
                        *presult=exp;
                    }
                    else
                        success= false;
                }
            }
            else
                success= false;
            fclose(pout);
        }
        else
            success= false;
        free(cmd);
    }
    return success;
}

/* computes the offset of a C reference with its origin
 */
expression reference_offset(reference ref)
{
    if(ENDP(reference_indices(ref))) return int_to_expression(0);
    else {
        expression address_computation = copy_expression(EXPRESSION(CAR(reference_indices(ref))));

        /* iterate on the dimensions & indices to create the index expression */
        list dims = variable_dimensions(type_variable(ultimate_type(entity_type(reference_variable(ref)))));
        list indices = reference_indices(ref);
        POP(indices);
        if(!ENDP(dims)) POP(dims); // the first dimension is unused
        FOREACH(DIMENSION,dim,dims)
        {
            expression dimension_size = make_op_exp(
                    PLUS_OPERATOR_NAME,
                    make_op_exp(
                        MINUS_OPERATOR_NAME,
                        copy_expression(dimension_upper(dim)),
                        copy_expression(dimension_lower(dim))
                        ),
                    int_to_expression(1));

            if( !ENDP(indices) ) { /* there may be more dimensions than indices */
                expression index_expression = EXPRESSION(CAR(indices));
                address_computation = make_op_exp(
                        PLUS_OPERATOR_NAME,
                        copy_expression(index_expression),
                        make_op_exp(
                            MULTIPLY_OPERATOR_NAME,
                            dimension_size,address_computation
                            )
                        );
                POP(indices);
            }
            else {
                address_computation = make_op_exp(
                        MULTIPLY_OPERATOR_NAME,
                        dimension_size,address_computation
                        );
            }
        }

        /* there may be more indices than dimensions */
        FOREACH(EXPRESSION,e,indices)
        {
            address_computation = make_op_exp(
                    PLUS_OPERATOR_NAME,
                    address_computation,copy_expression(e)
                    );
        }
        return address_computation ;
    }
}

/* Use side effects to move the content of e2, s2 and n2, into e1; s1
   and n1 are freed, as well as e2. This is useful if you need to
   keep the handle on e1. e1 is returned, although it is redundant. */
expression replace_expression_content(expression e1, expression e2)
{
  syntax s1 = expression_syntax(e1);
  normalized n1 = expression_normalized(e1);
  syntax s2 = expression_syntax(e2);
  normalized n2 = expression_normalized(e2);

  expression_syntax(e1) = s2;
  expression_normalized(e1) = n2;
  expression_syntax(e2) = syntax_undefined;
  expression_normalized(e2) = normalized_undefined;
  free_syntax(s1);
  free_normalized(n1);
  free_expression(e2);

  return e1;
}
/* @return true if expression @p e is a min or a max */
bool expression_minmax_p(expression e)
{
    if(expression_call_p(e))
    {
        entity op = call_function(expression_call(e));
        return ENTITY_MIN_P(op) || ENTITY_MAX_P(op);
    }
    return false;
}

/******************* EXPRESSIONS **********************
 * moved there from c_syntax by SG
 */

expression MakeSizeofExpression(expression e)
{

  syntax s = make_syntax_sizeofexpression(make_sizeofexpression_expression(e));
  expression exp =  make_expression(s,normalized_undefined);
  return exp; /* exp = sizeof(e)*/
}

expression MakeSizeofType(type t)
{
  syntax s = make_syntax_sizeofexpression(make_sizeofexpression_type(t));
  expression exp =  make_expression(s,normalized_undefined);
  return exp;  /* exp = sizeof(t) */
}

expression MakeCastExpression(type t, expression e)
{
  syntax s = make_syntax_cast(make_cast(t,e));
  expression exp = make_expression(s,normalized_undefined);
  return exp; /* exp = (t) e */
}

expression MakeCommaExpression(list l)
{
  if (ENDP(l))
    return expression_undefined;
  if (gen_length(l)==1)
    return EXPRESSION(CAR(l));
  return make_call_expression(CreateIntrinsic(COMMA_OPERATOR_NAME),l);
}

expression MakeBraceExpression(list l)
{
  return make_call_expression(CreateIntrinsic(BRACE_INTRINSIC),l);
}

/* generate a newly allocated expression for *(e)
 */
expression dereference_expression(expression e)
{
  if (expression_call_p(e))
  {
    call c = expression_call(e);
    if (ENTITY_ADDRESS_OF_P(call_function(c))) // e is "&x"
    {
      pips_assert("one arg to address operator (&)",
                  gen_length(call_arguments(c))==1);

      // result is simply "x"
      return copy_expression(EXPRESSION(CAR(call_arguments(c))));
    }
  }

  // result is "*e"
  return MakeUnaryCall(CreateIntrinsic(DEREFERENCING_OPERATOR_NAME),
                       copy_expression(e));
}

/* generate a newly allocated expression for &(e)
 */
expression make_address_of_expression(expression e)
{
  if (expression_call_p(e))
  {
    call c = expression_call(e);
    if (ENTITY_DEREFERENCING_P(call_function(c))) // e is "*x"
    {
      pips_assert("one arg to address operator (&)",
                  gen_length(call_arguments(c))==1);

      // result is simply "x"
      return copy_expression(EXPRESSION(CAR(call_arguments(c))));
    }
  }

  // result is "*e"
  return MakeUnaryCall(CreateIntrinsic(ADDRESS_OF_OPERATOR_NAME),
                       copy_expression(e));
}



/* make a full copy of the subscript expression list, preserve
   constant subscripts, replace non-constant subscript by the star
   subscript expression. */
list subscript_expressions_to_constant_subscript_expressions(list sl)
{
  list nsl = NIL;

  FOREACH(EXPRESSION, s, sl){
    expression ni = expression_undefined;
    value v = EvalExpression(s);
    if(value_constant_p(v) && constant_int_p(value_constant(v))) {
      int i = constant_int(value_constant(v));
      ni = int_to_expression(i);
    }
    else {
      ni = make_unbounded_expression();
    }
    nsl = CONS(EXPRESSION, ni, nsl);
  }
  nsl = gen_nreverse(nsl);
  return nsl;
}


/* Assume p is a pointer. Compute expression "*(p+i)" from reference
   r = "p[i]". */
expression pointer_reference_to_expression(reference r)
{
  entity p = reference_variable(r);
  type t = entity_basic_concrete_type(p);
  list rsl = reference_indices(r);
  int p_d = variable_dimension_number(type_variable(t)); // pointer dimension
  int r_d = (int) gen_length(rsl); // reference dimension

  pips_assert("The reference dimension is strictly greater than "
	      "the array of pointers dimension", r_d>p_d);

  /* rsl is fully copied into two sub-lists: the effective array
     indices and then the pointer indices. */
  list esl = NIL;
  list psl = NIL;
  list crsl = rsl;
  int i;
  for(i = 0; i<r_d; i++) {
    expression se = EXPRESSION(CAR(crsl));
    i<p_d? (esl = CONS(EXPRESSION, copy_expression(se), esl))
      :  (psl = CONS(EXPRESSION, copy_expression(se), psl));
    POP(crsl);
  }
  esl = gen_nreverse(esl), psl = gen_nreverse(psl);

  pips_assert("The pointer index list is not empty", !ENDP(psl));

  /* We build a proper reference to an element of p */
  reference nr = make_reference(p, esl);
  expression nre = reference_to_expression(nr);

  /* We build the equivalent pointer arithmetic expression */
  expression pae = nre;
  // FI: would be better to compute the two operators before entering the loop
  // entity plus = ;
  // entity indirection = ;
  FOREACH(EXPRESSION, pse, psl) {
    pae = binary_intrinsic_expression(PLUS_C_OPERATOR_NAME, pae, pse);
    pae = unary_intrinsic_expression(DEREFERENCING_OPERATOR_NAME, pae);
  }

  return pae;
}

bool C_initialization_expression_p(expression e)
{
  bool initialization_p = false;
  syntax s = expression_syntax(e);
  if(syntax_call_p(s)) {
    call c = syntax_call(s);
    entity f = call_function(c);
    if(ENTITY_BRACE_INTRINSIC_P(f))
      initialization_p = true;
  }
  return initialization_p;
}
