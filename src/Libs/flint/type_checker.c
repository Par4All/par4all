/*
 * $Id$
 *
 * Typecheck Fortran code.
 * by Son PhamDinh 03-05/2000
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"

#include "bootstrap.h" /* type of intrinsics stuff... */

/* Working with hash_table of basic
 */
#define GET_TYPE(h, e) ((basic)hash_get(h, (char*)(e)))
#define PUT_TYPE(h, e, b) hash_put(h, (char*)(e), (char*)(b))

/* ENTITY
 * It should be added in file "ri-util.h"
 */
#define CONCAT_OPERATOR_NAME "//"
#define ENTITY_CONCAT_P(e) (entity_an_operator_p(e, CONCAT))
#define ENTITY_EXTERNAL_P(e) (value_code_p(entity_initial(e)))
#define ENTITY_INTRINSIC_P(e) (value_intrinsic_p(entity_initial(e)))

#define ENTITY_CONVERSION_P(e,name) \
  (strcmp(entity_local_name(e), name##_GENERIC_CONVERSION_NAME)==0)
#define ENTITY_CONVERSION_CMPLX_P(e) ENTITY_CONVERSION_P(e, CMPLX)
#define ENTITY_CONVERSION_DCMPLX_P(e) ENTITY_CONVERSION_P(e, DCMPLX)

/* should be some properties to accomodate cray codes?? */
#define INT_LENGTH 4
#define REAL_LENGTH 4
#define DOUBLE_LENGTH 8
#define COMPLEX_LENGTH 8
#define DCOMPLEX_LENGTH 16
 
static void type_this_entity_if_needed(entity, type_context_p);

/************************************************************************** 
 * Convert a constant from INT to REAL
 * e.g: REAL(10) --> 10.0
 */
call
convert_constant_from_int_to_real(call c)
{
  char s[255];
  strcpy(s, entity_local_name(call_function(c)));
  strcat(s,".0E0");
  return make_call(make_constant_entity((string)s, 
					is_basic_float, REAL_LENGTH),
		   NIL);
}
/* INT -> DOUBLE
 * e.g: DBLE(10) => 10.0
 */
call
convert_constant_from_int_to_double(call c)
{
  char s[255];
  strcpy(s, entity_local_name(call_function(c)));
  strcat(s,".0D0");
  return make_call(make_constant_entity((string)s, 
					is_basic_float, DOUBLE_LENGTH),
		   NIL);
}
/* REAL -> INT
 * e.g: INT(-5.9E2) => -590
 */
call
convert_constant_from_real_to_int(call c)
{
  long l;
  float r;
  char s[255];
  sscanf(entity_local_name(call_function(c)), "%f",&r);
  l = (long)r;
  sprintf(s, "%ld", l);
  return make_call(make_constant_entity((string)s, is_basic_int, INT_LENGTH),
		   NIL);
}
/* REAL -> DOUBLE
 * e.g: DBLE(-5.9E-2) => -5.9D-2
 */
call
convert_constant_from_real_to_double(call c)
{
  char s[255];
  int i, len;
  strcpy(s, entity_local_name(call_function(c)));
  len = strlen(s);
  for(i = 0; i < len; i++)
  {
    if (s[i]=='E' || s[i]=='e')
    {
      s[i]='D';
      break;
    }
  }
  /* There is not E, e.g: 5.496 */
  if (i == len)
  {
    strcat(s,"D0");
  }
  return make_call(make_constant_entity((string)s, 
					is_basic_float, DOUBLE_LENGTH),
		   NIL);
}
/* DOUBLE -> REAL
 * e.g: REAL(-5.9D-2) => -5.9E-2
 */
call
convert_constant_from_double_to_real(call c)
{
  char s[255];
  int i, len;
  strcpy(s, entity_local_name(call_function(c)));
  len = strlen(s);
  for(i = 0; i < len; i++)
  {
    if (s[i]=='D' || s[i]=='d')
    {
      s[i]='E';
      break;
    }
  }
  /* There is not D, e.g: 5.496 */
  if (i == len)
  {
    strcat(s,"E0");
  }
  return make_call(make_constant_entity((string)s, 
					is_basic_float, REAL_LENGTH),
		   NIL);
}
/* DOUBLE -> INT
 * e.g: INT(-5.9D2) => -590
 */
call
convert_constant_from_double_to_int(call c)
{
  call c_result, c_real = convert_constant_from_double_to_real(c);
  c_result = convert_constant_from_real_to_int(c_real);
  free_call(c_real);
  return c_result;
}
/* REAL -> COMPLEX
 * e.g: CMPLX(-5.9E5) => (-5.9E5, 0.0)
 */
call
convert_constant_from_real_to_complex(call c)
{
  expression exp_real, exp_imag;
  call c_imag;
  list args;
  exp_real = make_expression(make_syntax(is_syntax_call, copy_call(c)), 
			     normalized_undefined);
  c_imag = make_call(make_constant_entity("0.0E0", 
					  is_basic_float, REAL_LENGTH),
		     NIL);
  exp_imag = make_expression(make_syntax(is_syntax_call, c_imag), 
			     normalized_undefined);
  args = CONS(EXPRESSION, exp_real, CONS(EXPRESSION, exp_imag, NIL));
  /* Conversion explicit */
  if (get_bool_property("TYPE_CHECKER_EXPLICIT_COMPLEX_CONSTANTS"))
  {
    return make_call(CreateIntrinsic(CMPLX_GENERIC_CONVERSION_NAME), args);
  }
  /* Conversion inplicit */
  return make_call(CreateIntrinsic(IMPLIED_COMPLEX_NAME), args);
}
/* DOUBLE -> COMPLEX
 * e.g: CMPLX(-5.9D5) => (-5.9E5, 0.0)
 */
call
convert_constant_from_double_to_complex(call c)
{
  call c_result, c_real = convert_constant_from_double_to_real(c);
  c_result = convert_constant_from_real_to_complex(c_real);
  free_call(c_real);
  return c_result;
}
/* INT -> COMPLEX
 * e.g: CMPLX(-5) => (-5.0, 0.0)
 */
call
convert_constant_from_int_to_complex(call c)
{
  call c_result, c_real = convert_constant_from_int_to_real(c);
  c_result = convert_constant_from_real_to_complex(c_real);
  free_call(c_real);
  return c_result;
}
/* DOUBLE -> DCOMPLEX
 * e.g: DCMPLX(-5.9D5) => (-5.9D5, 0.0)
 */
call
convert_constant_from_double_to_dcomplex(call c)
{
  expression exp_real, exp_imag;
  call c_imag;
  list args;
  exp_real = make_expression(make_syntax(is_syntax_call, copy_call(c)), 
			     normalized_undefined);
  c_imag = make_call(make_constant_entity("0.0D0", 
					  is_basic_float, DOUBLE_LENGTH),
		     NIL);
  exp_imag = make_expression(make_syntax(is_syntax_call, c_imag), 
			     normalized_undefined);
  args = CONS(EXPRESSION, exp_real, CONS(EXPRESSION, exp_imag, NIL));
  
  /* Conversion explicit */
  if (get_bool_property("TYPE_CHECKER_EXPLICIT_COMPLEX_CONSTANTS"))
  {
    return make_call(CreateIntrinsic(DCMPLX_GENERIC_CONVERSION_NAME), args);
  }
  /* Conversion inplicit */
  return make_call(CreateIntrinsic(IMPLIED_DCOMPLEX_NAME), args);
}
/* REAL -> DCOMPLEX
 * e.g: DCMPLX(-5.9E5) => (-5.9D5, 0.0D0)
 */
call
convert_constant_from_real_to_dcomplex(call c)
{
  call c_result, c_double = convert_constant_from_real_to_double(c);
  c_result = convert_constant_from_double_to_dcomplex(c_double);
  free_call(c_double);
  return c_result;
}
/* INT -> DCOMPLEX
 * e.g: DCMPLX(-5) => (-5D0, 0.0D0)
 */
call
convert_constant_from_int_to_dcomplex(call c)
{
  call c_result, c_double = convert_constant_from_int_to_double(c);
  c_result = convert_constant_from_double_to_dcomplex(c_double);
  free_call(c_double);
  return c_result;
}

/***************************************************************************** 
 * Convert constant C to basic naming to_basic
 */
call
convert_constant(call c, basic to_basic)
{
  basic b;
  entity function_called = call_function(c);
  if(entity_constant_p(function_called))
  {
    b = entity_basic(function_called);
    if(basic_equal_p(b, to_basic))
    {
      //return NULL;
      return copy_call(c);
    }
    else if (basic_int_p(b))
    {
      /* INT -> REAL */
      if (basic_float_p(to_basic) && basic_float(to_basic)==4)
      {
	return convert_constant_from_int_to_real(c);
      }
      /* INT -> DOUBLE */
      if (basic_float_p(to_basic) && basic_float(to_basic)==8)
      {
	return convert_constant_from_int_to_double(c);
      }
      /* INT -> COMPLEX */
      if (basic_complex_p(to_basic) && basic_complex(to_basic)==8)
      {
	return convert_constant_from_int_to_complex(c);
      }
      /* INT -> DCOMPLEX */
      if (basic_complex_p(to_basic) && basic_complex(to_basic)==16)
      {
	return convert_constant_from_int_to_dcomplex(c);
      }
    }
    else if (basic_float_p(b) && basic_float(b)==4)
    {
      /* REAL -> INT */
      if (basic_int_p(to_basic))
      {
	return convert_constant_from_real_to_int(c);
      }
      /* REAL -> DOUBLE */
      else if (basic_float_p(to_basic) && basic_float(to_basic)==8)
      {
	return convert_constant_from_real_to_double(c);
      }
      /* REAL -> COMPLEX */
      else if (basic_complex_p(to_basic) && basic_complex(to_basic)==8)
      {
	return convert_constant_from_real_to_complex(c);
      }
      /* REAL -> DCOMPLEX */
      else if (basic_complex_p(to_basic) && basic_complex(to_basic)==16)
      {
	return convert_constant_from_real_to_dcomplex(c);
      }
    }
    else if (basic_float_p(b) && basic_float(b)==8)
    {
      /* DOUBLE -> INT */
      if (basic_int_p(to_basic))
      {
	return convert_constant_from_double_to_int(c);
      }
      /* DOUBLE -> REAL */
      else if (basic_float_p(to_basic) && basic_float(to_basic)==4)
      {
	return convert_constant_from_double_to_real(c);
      }
      /* DOUBLE -> COMPLEX */
      else if (basic_complex_p(to_basic) && basic_complex(to_basic)==8)
      {
	return convert_constant_from_double_to_complex(c);
      }
      /* DOUBLE -> DCOMPLEX */
      else if (basic_complex_p(to_basic) && basic_complex(to_basic)==16)
      {
	return convert_constant_from_double_to_dcomplex(c);
      }
    }
  }
  return NULL;
}

/***************************************************************************** 
 * Cast an expression constant to the basic to_basic.
 * Return TRUE if OK
 */
static expression
cast_constant(expression exp_constant, basic to_basic, type_context_p context)
{
  entity function_called;
  call c;
  basic b = NULL;
  expression exp, exp_real, exp_imag, exp_real2, exp_imag2;
  syntax s = expression_syntax(exp_constant);
  if(syntax_call_p(s))
  {
    function_called = call_function(syntax_call(s));
    if(entity_constant_p(function_called))
    {
      /* Convert if necessary */
      c = convert_constant(syntax_call(s), to_basic);
      if (c != NULL)
      {	
	context->number_of_simplication++;
	return make_expression(make_syntax(is_syntax_call, c), 
			       normalized_undefined);	
      }
    }
    else if(ENTITY_UNARY_MINUS_P(function_called))
    {
      exp = cast_constant(EXPRESSION(CAR(call_arguments(syntax_call(s)))),
			  to_basic, context);
      if (exp != NULL)
      {
	c = make_call(copy_entity(function_called),
		      CONS(EXPRESSION, exp, NIL));
	return make_expression(make_syntax(is_syntax_call, c),
			       normalized_undefined);
      }
    }
    else if(ENTITY_IMPLIED_CMPLX_P(function_called) ||
	    ENTITY_CONVERSION_CMPLX_P(function_called) ||
	    ENTITY_IMPLIED_DCMPLX_P(function_called) ||
	    ENTITY_CONVERSION_DCMPLX_P(function_called))
    {
      exp_real = EXPRESSION(CAR(call_arguments(syntax_call(s))));
      /* Two arguments, with imagine party */
      if (CDR(call_arguments(syntax_call(s))) != NIL )
      {
	exp_imag = EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
      }
      /* One argument, no imagine party */
      else
      {
	exp_imag = NULL;
      }
      if (!basic_complex_p(to_basic))
      {
	return cast_constant(exp_real, to_basic, context);
      }
      /* DCOMPLEX -> COMPLEX */
      else if (basic_complex(to_basic) == 8 &&
	       (ENTITY_IMPLIED_DCMPLX_P(function_called) ||
		ENTITY_CONVERSION_DCMPLX_P(function_called)))
      {
	b = make_basic_float(4);
	if ((exp_real2 = cast_constant(exp_real, b, context)) == NULL)
	{
	  exp_real2 = exp_real;
	}
	
	if (exp_imag != NULL)
	{
	  if ((exp_imag2 = cast_constant(exp_imag, b, context)) == NULL)
	  {
	    exp_imag2 = exp_imag;
	  }
	}
	else
	{
	  c =  make_call(make_constant_entity("0.0E0", is_basic_float, 
					      REAL_LENGTH),
			 NIL);
	  exp_imag2 = make_expression(make_syntax(is_syntax_call, c),
				      normalized_undefined);
	}
	/* Conversion implicit */
	if (!get_bool_property("TYPE_CHECKER_EXPLICIT_COMPLEX_CONSTANTS") &&
	    ENTITY_IMPLIED_DCMPLX_P(function_called))
	{
	  c = make_call(CreateIntrinsic(IMPLIED_COMPLEX_NAME),
			CONS(EXPRESSION, exp_real2,
			     CONS(EXPRESSION, exp_imag2, NIL)));	
	}
	/* Conversion explicit */
	else
	{
	  c = make_call(CreateIntrinsic(CMPLX_GENERIC_CONVERSION_NAME),
			CONS(EXPRESSION, exp_real2,
			     CONS(EXPRESSION, exp_imag2, NIL)));	
	}
	return make_expression(make_syntax(is_syntax_call, c),
			       normalized_undefined);
      }
      /* COMPLEX -> DCOMPLEX */
      else if (basic_complex(to_basic) == 16 &&
	       (ENTITY_IMPLIED_CMPLX_P(function_called) ||
		ENTITY_CONVERSION_CMPLX_P(function_called)))
      {
	b = make_basic_float(8);
	if ((exp_real2 = cast_constant(exp_real, b, context)) == NULL)
	{
	  exp_real2 = exp_real;
	}
	if (exp_imag != NULL)
	{
	  if ((exp_imag2 = cast_constant(exp_imag, b, context)) == NULL)
	  {
	    exp_imag2 = exp_imag;
	  }
	}
	else
	{
	  c =  make_call(make_constant_entity("0.0D0", is_basic_float, 
					      DOUBLE_LENGTH),
			 NIL);
	  exp_imag2 = make_expression(make_syntax(is_syntax_call, c),
				      normalized_undefined);
	}
	/* Conversion implicit */
	if (!get_bool_property("TYPE_CHECKER_EXPLICIT_COMPLEX_CONSTANTS") &&
	    ENTITY_IMPLIED_CMPLX_P(function_called))
	{
	  c = make_call(CreateIntrinsic(IMPLIED_DCOMPLEX_NAME),
			CONS(EXPRESSION, exp_real2,
			     CONS(EXPRESSION, exp_imag2, NIL)));	
	}
	/* Conversion explicit */
	else
	{
	  c = make_call(CreateIntrinsic(DCMPLX_GENERIC_CONVERSION_NAME),
			CONS(EXPRESSION, exp_real2,
			     CONS(EXPRESSION, exp_imag2, NIL)));	
	}
	return make_expression(make_syntax(is_syntax_call, c),
			       normalized_undefined);
      }
    }
  }
  if (b != NULL)
  {
    free_basic(b);
  }
  return NULL;
}
/***************************************************************************** 
 * Specify a cast for converting from a basic ('from') to another basic (cast)
 */
static entity 
get_cast_function_for_basic(basic cast, basic from)
{
  switch (basic_tag(cast))
  {
  case is_basic_int:
    if (from!=NULL && from!=basic_undefined && basic_string_p(from))
    {
      return CreateIntrinsic(CHAR_TO_INT_CONVERSION_NAME);
    }
    else
    {
      switch(basic_int(cast))
      {
      case 2: 
      case 4: 
      case 8: 
	return CreateIntrinsic(INT_GENERIC_CONVERSION_NAME) ;
      default: 
	pips_internal_error("Unexpected integer size %d\n", basic_int(cast));
      }
    }
    break;
  case is_basic_float:
    switch(basic_float(cast))
    {
    case 4: 
      return CreateIntrinsic(REAL_GENERIC_CONVERSION_NAME) ;
    case 8: 
      return CreateIntrinsic(DBLE_GENERIC_CONVERSION_NAME) ;
    default: 
      pips_internal_error("Unexpected float size %d\n", basic_float(cast));
    }
    break;
  case is_basic_logical:
    switch(basic_logical(cast))
    {
    case 1: 
    case 2: 
    case 4: 
    case 8: 
      return entity_undefined;
    default: 
      pips_internal_error("Unexpected logical size %d\n", basic_logical(cast));
    }
    break;
  case is_basic_complex:
    switch(basic_complex(cast))
    {
    case 8: 
      return CreateIntrinsic(CMPLX_GENERIC_CONVERSION_NAME) ;
    case 16: 
      return CreateIntrinsic(DCMPLX_GENERIC_CONVERSION_NAME) ;
    default: 
      pips_internal_error("Unexpected complex size %d\n", basic_complex(cast));
    }
    break;
  case is_basic_string:
    return CreateIntrinsic(INT_TO_CHAR_CONVERSION_NAME);
  case is_basic_overloaded:
    pips_internal_error("Should never convert to overloaded...");
  default: 
    pips_internal_error("Unexpected basic tag (%d)", basic_tag(cast));
  }
  
  return NULL;
}

/***************************************************************************** 
 * Cast an expression
 * e.g: x --> INT(x)
 */
static expression 
insert_cast(basic cast, basic from, expression exp, type_context_p context)
{
  call c;
  syntax s;
  entity cast_function;
  expression exp_constant;
  
  s = expression_syntax(exp);
  
  /* If exp is a constant -> Convert it 
   * e.g: 5.9 -> 5 (REAL -> INT)
   */ 
  if ((exp_constant = cast_constant(exp, cast, context)) != NULL)
  {
    return exp_constant;
  }
  
  /* If not */
  cast_function = get_cast_function_for_basic(cast, from);    
  if (cast_function == NULL)
  {
    pips_internal_error("Cast function is not verified!\n");      
  }
  if (cast_function == entity_undefined)
  {
    pips_internal_error("Can not convert to LOGICAL!\n");      
  }
  c = make_call(cast_function, CONS(EXPRESSION, exp, NIL));
  s = make_syntax(is_syntax_call, c);
  
  /* Count the number of conversions */
  context->number_of_conversion++;
  
  return make_expression(s, normalized_undefined);
}
/**************************************************************************
 * Typing all the arguments of user-defined function C to its parameters
 * correspondent.
 *
 * Note: The call C must be an user-defined function
 */
static basic
typing_arguments_of_user_function(call c, type_context_p context)
{
  list            args = call_arguments(c);
  type            the_tp = entity_type(call_function(c));
  functional      ft = type_functional(the_tp);
  list            params = functional_parameters(ft);
  type            result = functional_result(ft);
  int             na = gen_length(args);
  int             nt = gen_length(params);
  parameter       param;
  basic b, b1;
  int argnumber = 0;
  
  if (na == nt || 
      (nt<=na && 
       type_varargs_p(parameter_type(PARAMETER(CAR(gen_last(params)))))))
  {
    while (args != NIL)
    {
      argnumber++;
      /* Here, parameter is always a variable */
      param = PARAMETER(CAR(params));
      b = variable_basic(type_variable(parameter_type(param)));
      b1 = GET_TYPE(context->types, EXPRESSION(CAR(args)));
      if (!basic_equal_p(b, b1))
      {
	add_one_line_of_comment((statement) stack_head(context->stats),
				"invalid arg #%d to '%s', %s instead of %s!",
				argnumber,
				entity_local_name(call_function(c)),
				basic_to_string(b1),
				basic_to_string(b));
	context->number_of_error++;
      }
      args = CDR(args);
      params = CDR(params);
    }	
  }
  else if (na < nt)
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "Too few argument(s) to '%s' (%d<%d)!",
			    entity_local_name(call_function(c)), na, nt);
    context->number_of_error++;
  }
  else
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "Too many argument(s) to '%s' (%d>%d)!",
			    entity_local_name(call_function(c)), na, nt);
    context->number_of_error++;
  }
  
  /* Subroutine */
  if (type_void_p(result))
  {
    pips_debug(7, "type of %s is overloaded\n", entity_name(call_function(c)));
    b = make_basic_overloaded();
  }
  /* Function */
  else
  {
    pips_debug(7, "type of %s is a function\n", entity_name(call_function(c)));
    b = copy_basic(variable_basic(type_variable(result)));
  }
  return b;
}

/***************************************************************************** 
 * Make typing an expression of type CALL
 * WARNING: The interpretion of COMPLEX !!!
 */
static basic 
type_this_call(expression exp, type_context_p context)
{
  typing_function_t dotype;
  switch_name_function simplifier;
  call c = syntax_call(expression_syntax(exp));
  entity function_called = call_function(c);
  basic b;
  b = basic_undefined;
  
  pips_debug(2, "Call to %s; Its type is %s \n", entity_name(function_called), 
	     type_to_string(entity_type(function_called)));
  
  /* Labels */
  if (entity_label_p(function_called))
  {
    b = make_basic_overloaded();
  }

  /* Constants */
  else if (entity_constant_p(function_called))
  {
    b = basic_of_call(c);
  }
  
  /* User-defined functions */
  else if (ENTITY_EXTERNAL_P(function_called))
  {
    b = typing_arguments_of_user_function(c, context);
  }
  
  /* All intrinsics */
  else if (ENTITY_INTRINSIC_P(function_called))
  {
    /* Typing intrinsics */
    dotype = get_typing_function_for_intrinsic(
				   entity_local_name(function_called));
    if (dotype != 0)
    {
      b = dotype(c, context);
    }
    
    /* Simplification */
    simplifier = get_switch_name_function_for_intrinsic(
				   entity_local_name(function_called));
    if (simplifier != 0)
    {
      simplifier(exp, context);
    }
  }
  else if (value_symbolic_p(entity_initial(function_called)))
  {
    /* lazy type entity contents... */
    type_this_entity_if_needed(function_called, context);
    b = GET_TYPE(context->types, 
       symbolic_expression(value_symbolic(entity_initial(function_called))));
    b = copy_basic(b);
  }

  pips_debug(7, "Call to %s typed as %s\n", entity_name(function_called), 
	     basic_to_string(b));

  return b;
}

/***************************************************************************** 
 * Make typing an instruction 
 * (Assignment statement (=) is the only instruction that is typed here)
 */
static void 
type_this_instruction(instruction i, type_context_p context)
{
  basic b1;
  call c;
  typing_function_t dotype;
  
  if (instruction_call_p(i))
  {
    c = instruction_call(i);
    pips_debug(1, "Call to %s; Its type is %s \n", 
	       entity_name(call_function(c)), 
	       type_to_string(entity_type(call_function(c))));

    /* type check a SUBROUTINE call. */
    if (ENTITY_EXTERNAL_P(call_function(c)))
    {
      b1 = typing_arguments_of_user_function(c, context);

      if (!basic_overloaded_p(b1)) 
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Ignored %s value returned by '%s'",
				basic_to_string(b1), 
				entity_local_name(call_function(c)));
	/* Count the number of errors */
	context->number_of_error++;
      }
      free_basic(b1);

      return;
    }

    /* Typing intrinsics: 
     * Assignment, control statement, IO statement
     */
    dotype = get_typing_function_for_intrinsic(
				   entity_local_name(call_function(c)));
    if (dotype != 0)
    {
      b1 = dotype(c, context);
    }
  }
}

static void 
check_this_test(test t, type_context_p context)
{
  basic b = GET_TYPE(context->types, test_condition(t));
  if (!basic_logical_p(b)) 
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Test condition must be a logical expression!");
    context->number_of_error++;
  }
}

static void 
check_this_whileloop(whileloop w, type_context_p context)
{
  basic b = GET_TYPE(context->types, whileloop_condition(w));
  if (!basic_logical_p(b)) 
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "While condition must be a logical expression!");
    context->number_of_error++;
  }
}

/***************************************************************************** 
 * Range of loop (lower, upper, increment), all must be Integer, Real or Double
 * (According to ANSI X3.9-1978, FORTRAN 77; Page 11-5)
 *
 * Return TRUE if type of range is correct, otherwise FALSE
 */
bool
check_loop_range(range r, hash_table types)
{
  basic lower, upper, incr;
  lower = GET_TYPE(types, range_lower(r));
  upper = GET_TYPE(types, range_upper(r));
  incr = GET_TYPE(types, range_increment(r));
  if( (basic_int_p(lower) || basic_float_p(lower)) &&
      (basic_int_p(upper) || basic_float_p(upper)) &&
      (basic_int_p(incr) || basic_float_p(incr)))
  {
    return TRUE;
  }
  return FALSE;
}
/***************************************************************************** 
 * Typing range of loop to the type of index loop. 
 * This range is already verified 
 */
static void
type_loop_range(basic index, range r, type_context_p context)
{
  basic lower, upper, incr;
  lower = GET_TYPE(context->types, range_lower(r));
  upper = GET_TYPE(context->types, range_upper(r));
  incr = GET_TYPE(context->types, range_increment(r));
  
  if(!basic_equal_p(index, lower))
  {
    range_lower(r) = insert_cast(index, lower, range_lower(r), context);
  }
  if(!basic_equal_p(index, upper))
  {
    range_upper(r) = insert_cast(index, upper, range_upper(r), context);
  }
  if(!basic_equal_p(index, incr))
  {
    range_increment(r) = insert_cast(index, incr, range_increment(r), context);
  }
}
/***************************************************************************** 
 * Typing the loop if necessary
 */
static void 
check_this_loop(loop l, type_context_p context)
{
  basic ind = entity_basic(loop_index(l));

  /* ok for F77, but not in F90? */
  if (!basic_int_p(ind)) 
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Obsolescent non integer loop index '%s'"
			    " (R822 ISO/IEC 1539:1991 (E))",
			    entity_local_name(loop_index(l)));
    context->number_of_error++;
  }

  if( !(basic_int_p(ind) || basic_float_p(ind)) )
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "Index '%s' must be Integer, Real or Double!", 
			    entity_local_name(loop_index(l)));
    context->number_of_error++;
  }
  else if (!check_loop_range(loop_range(l), context->types))
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
		    "Range of index '%s' must be Integer, Real or Double!", 
			    entity_local_name(loop_index(l)));
    context->number_of_error++;
  }
  else
  {
    type_loop_range(ind, loop_range(l), context);
  }
}

/***************************************************************************** 
 * This function will be called in the function 
 * gen_context_recurse(...) of Newgen as its parameter
 */
static void 
type_this_expression(expression e, type_context_p context)
{
  syntax s = expression_syntax(e);    
  basic b = basic_undefined;
  
  /* Specify the basic of the expression e  */
  switch (syntax_tag(s))
  {
  case is_syntax_call:
    b = type_this_call(e, context);
    break;
    
  case is_syntax_reference:
    b = copy_basic(entity_basic(reference_variable(syntax_reference(s))));
    pips_debug(2,"Reference: %s; Type: %s\n", 
	       entity_name(reference_variable(syntax_reference(s))), 
	       basic_to_string(b));
    break;
    
  case is_syntax_range:
    /* PDSon: For the range alone (not in loop), 
     * I only check lower, upper and step, they must be all INT, REAL or DBLE
     */
    if (!check_loop_range(syntax_range(s), context->types))
    {
      add_one_line_of_comment((statement)stack_head(context->stats),
			      "Range must be INT, REAL or DBLE!");
      context->number_of_error++;
    }
    break;
    
  default:
    pips_internal_error("unexpected syntax tag (%d)", syntax_tag(s));
  }
  
  /* Push the basic in hash table "types" */
  if (!basic_undefined_p(b))
  {
    PUT_TYPE(context->types, e, b);
  }
}

static void check_this_reference(reference r, type_context_p context)
{
  MAP(EXPRESSION, ind,
  {
    /* cast expressions to INT if not already an int... ? */
    /* ??? maybe should update context->types ??? */
    
    basic b = GET_TYPE(context->types, ind);
    if (!basic_int_p(b))
    {
      basic bint = make_basic_int(4);
      insert_cast(bint, b, ind, context); /* and simplifies! */
      free_basic(bint);
    }
  },
      reference_indices(r));
}

static bool 
stmt_flt(statement s, type_context_p context)
{
  stack_push(s, context->stats);
  return TRUE;
}

static void 
stmt_rwt(statement s, type_context_p context)
{
  pips_assert("pop same push", stack_head(context->stats)==s);
  stack_pop(context->stats);
}

static void type_this_chunk(void * c, type_context_p context)
{
  gen_context_multi_recurse
    (c, context, 
     statement_domain, stmt_flt, stmt_rwt,
     instruction_domain, gen_true, type_this_instruction,
     test_domain, gen_true, check_this_test,
     whileloop_domain, gen_true, check_this_whileloop,
     loop_domain, gen_true, check_this_loop,
     expression_domain, gen_true, type_this_expression,
     reference_domain, gen_true, check_this_reference,
     NULL);
}

static void type_this_entity_if_needed(entity e, type_context_p context)
{
  value v = entity_initial(e);

  /* ??? TODO: type->variable->dimensions->dimension->expression */

  if (value_symbolic_p(v))
  {
    symbolic sy = value_symbolic(v);
    expression s = symbolic_expression(sy);
    basic b1, b2;

    if (hash_defined_p(context->types, s))
      return;

    type_this_chunk((void *) s, context);

    /* type as "e = s" */
    b1 = entity_basic(e);
    b2 = GET_TYPE(context->types, s);

    if (!basic_compatible_p(b1, b2))
    {
      add_one_line_of_comment((statement) stack_head(context->stats), 
		   "%s parameter '%s' definition from incompatible type %s", 
			      basic_to_string(b1),
			      entity_local_name(e),
			      basic_to_string(b2)); 
      context->number_of_error++;
      return;
    }

    if (!basic_equal_p(b1, b2))
    {
      symbolic_expression(sy) = insert_cast(b1, b2, s, context);
      PUT_TYPE(context->types, symbolic_expression(sy), copy_basic(b1));
    }
}
}

static void put_summary(string name, type_context_p context)
{
  user_log("Type Checker Summary\n"
	   "\t%d errors found\n"
	   "\t%d conversions inserted\n"
	   "\t%d simplifications performed\n",
	   context->number_of_error,
	   context->number_of_conversion,
	   context->number_of_simplication);

  pips_user_warning("summary of '%s': "
		    "%d errors, %d convertions., %d simplifications\n",
		    name, 
		    context->number_of_error,
		    context->number_of_conversion,
		    context->number_of_simplication);

  if (name && get_bool_property("TYPE_CHECKER_ADD_SUMMARY"))
  {
    entity module = local_name_to_top_level_entity(name);
    code c;
    char buf[100];

    pips_assert("entity is a module", entity_module_p(module));

    c = value_code(entity_initial(module));

    sprintf(buf, 
	    "!PIPS TYPER: %d errors, %d conversions, %d simplifications\n",
	    context->number_of_error,
	    context->number_of_conversion,
	    context->number_of_simplication);

    if (!code_decls_text(c) || string_undefined_p(code_decls_text(c)))
      code_decls_text(c) = strdup(buf);
    else
    {
      string tmp = code_decls_text(c);
      code_decls_text(c) = strdup(concatenate(buf, tmp, NULL));
      free(tmp);
    }
  }
}

/************************************************************************** 
 * Type check all expressions in statements.
 * Returns false if type errors are detected.
 */
void typing_of_expressions(string name, statement s)
{
  type_context_t context;
  
  context.types = hash_table_make(hash_pointer, 0);
  context.stats = stack_make(statement_domain, 0, 0);
  context.number_of_error = 0;
  context.number_of_conversion = 0;
  context.number_of_simplication = 0;
  
  /* Bottom-up typing */
  type_this_chunk((void *) s, &context);
  
  /* Summary */
  put_summary(name, &context);
  
  /* Type checking ... */
  HASH_MAP(st, ba, free_basic(ba), context.types);
  hash_table_free(context.types);
  stack_free(&context.stats);
}

bool 
type_checker(string name)
{
  statement stat;
  debug_on("TYPE_CHECKER_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);

  stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE);
  set_current_module_statement(stat);

  typing_of_expressions(name, stat);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, name, stat);
  reset_current_module_statement();

  pips_debug(1, "done");
  debug_off();
  return TRUE;
}
