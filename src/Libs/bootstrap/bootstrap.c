/*
  $Id$

  Symbol table initialization with Fortran operators, commands and 
  intrinsics
   
  More information is provided in effects/effects.c
   
  Remi Triolet
  
  Modifications:
  - add intrinsics according to Fortran standard Table 5, pp. 15.22-15-25,
  Francois Irigoin, 02/06/90
  - add .SEQ. to handle ranges outside of arrays [pj]
  - add intrinsic DFLOAT. bc. 13/1/96.
  - add pseudo-intrinsics SUBSTR and ASSIGN_SUBSTR to handle strings,
    FI, 25/12/96
  - Fortran specification conformant typing of expressions...

   Bugs:
  - intrinsics are not properly typed

  $Log: bootstrap.c,v $
  Revision 1.73  2002/06/13 13:13:13  irigoin
  Pseudo-intrinsic REPEAT-VALUE added

  Revision 1.72  2002/06/10 12:00:37  irigoin
  $Log: bootstrap.c,v $
  Revision 1.73  2002/06/13 13:13:13  irigoin
  Pseudo-intrinsic REPEAT-VALUE added
 added to keep track of changes

*/

#include <stdio.h>
#include <string.h>
/* #include <values.h> */
#include <limits.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "makefile.h"
#include "database.h"

#include "bootstrap.h"

#include "misc.h"
#include "pipsdbm.h"
#include "parser_private.h"
#include "syntax.h"
#include "constants.h"
#include "resources.h"

#include "properties.h"

#define LOCAL static

/* Working with hash_table of basic
 */
#define GET_TYPE(h, e) ((basic)hash_get(h, (char*)(e)))
#define PUT_TYPE(h, e, b) hash_put(h, (char*)(e), (char*)(b))

/* Function in type_checker.c */
extern expression 
insert_cast(basic cast, basic from, expression exp, type_context_p);
extern expression
cast_constant(expression exp_constant, basic to_basic, 
	      type_context_p context);
extern bool
check_loop_range(range, hash_table);
extern void
type_loop_range(basic, range, type_context_p);


void 
CreateAreas()
{
  make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME, 
			       DYNAMIC_AREA_LOCAL_NAME),
	      make_type(is_type_area, make_area(0, NIL)),
	      make_storage(is_storage_rom, UU),
	      make_value(is_value_unknown, UU));
  
  
  make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME, 
			       STATIC_AREA_LOCAL_NAME),
	      make_type(is_type_area, make_area(0, NIL)),
	      make_storage(is_storage_rom, UU),
	      make_value(is_value_unknown, UU));
}

void 
CreateArrays()
{
  /* First a dummy function - close to C one "crt0()" - in order to
     - link the next entity to its ram
     - make an unbounded dimension for this entity
  */
  
  entity ent;
  
  ent = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,
				     IO_EFFECTS_PACKAGE_NAME),
		    make_type(is_type_functional,
			      make_functional(NIL,make_type(is_type_void,
							    NIL))),
		    make_storage(is_storage_rom, UU),
		    make_value(is_value_code,make_code(NIL, strdup(""), make_sequence(NIL))));
  
  set_current_module_entity(ent);
  
  make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME, 
			       STATIC_AREA_LOCAL_NAME),
	      make_type(is_type_area, make_area(0, NIL)),
	      make_storage(is_storage_rom, UU),
	      make_value(is_value_unknown, UU));
  
  /* GO: entity for io logical units: It is an array which*/
  make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
			       IO_EFFECTS_ARRAY_NAME),
	  MakeTypeArray(make_basic_int(IO_EFFECTS_UNIT_SPECIFIER_LENGTH),
			CONS(DIMENSION,
			     make_dimension
			     (MakeIntegerConstantExpression("0"),
				/*
				  MakeNullaryCall
				  (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
				*/
			      MakeIntegerConstantExpression("2000")
			      ),
			     NIL)),
	      /* make_storage(is_storage_ram,
		 make_ram(entity_undefined, DynamicArea, 0, NIL))
	      */
	      make_storage(is_storage_ram,
			   make_ram(ent,
			   global_name_to_entity(IO_EFFECTS_PACKAGE_NAME, 
						 STATIC_AREA_LOCAL_NAME),
				    0, NIL)),
	      make_value(is_value_unknown, UU));
  
  /* GO: entity for io logical units: It is an array which*/
  make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
			       IO_EOF_ARRAY_NAME),
	MakeTypeArray(make_basic_logical(IO_EFFECTS_UNIT_SPECIFIER_LENGTH),
		      CONS(DIMENSION,
			   make_dimension
			   (MakeIntegerConstantExpression("0"),
			    /*
			      MakeNullaryCall
			      (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
			    */
			    MakeIntegerConstantExpression("2000")
			    ),
			   NIL)),
	      /* make_storage(is_storage_ram,
		 make_ram(entity_undefined, DynamicArea, 0, NIL))
	      */
	      make_storage(is_storage_ram,
		    make_ram(ent,
			     global_name_to_entity(IO_EFFECTS_PACKAGE_NAME, 
						   STATIC_AREA_LOCAL_NAME),
			     0, NIL)),
	      make_value(is_value_unknown, UU));
  
  /* GO: entity for io logical units: It is an array which*/
  make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
			       IO_ERROR_ARRAY_NAME),
	MakeTypeArray(make_basic_logical(IO_EFFECTS_UNIT_SPECIFIER_LENGTH),
		      CONS(DIMENSION,
			   make_dimension
			   (MakeIntegerConstantExpression("0"),
			    /*
			      MakeNullaryCall
			      (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
			    */
			    MakeIntegerConstantExpression("2000")
			    ),
			   NIL)),
	      /* make_storage(is_storage_ram,
		 make_ram(entity_undefined, DynamicArea, 0, NIL))
	      */
	      make_storage(is_storage_ram,
			   make_ram(ent,
			   global_name_to_entity(IO_EFFECTS_PACKAGE_NAME, 
						 STATIC_AREA_LOCAL_NAME),
				    0, NIL)),
	      make_value(is_value_unknown, UU));
  
  reset_current_module_entity();
}

static list
make_parameter_list(int n, parameter (* mkprm)(void))
{
  list l = NIL;
  
  if (n < (INT_MAX)) 
  {
    int i = n;
    while (i-- > 0) 
    {
      l = CONS(PARAMETER, mkprm(), l);
    }
  }
  else 
  {
    /* varargs */
    parameter p = mkprm();
    type pt = copy_type(parameter_type(p));
    type v = make_type(is_type_varargs, pt);
    parameter vp = make_parameter(v, make_mode(is_mode_reference, UU));
    
    l = CONS(PARAMETER, vp, l);
    free_parameter(p);
  }
  return l;
}

/* The default intrinsic type is a functional type with n overloaded
 * arguments returning an overloaded result if the arity is known.
 * If the arity is unknown, the default intrinsic type is a 0-ary
 * functional type returning an overloaded result.
 */

static type 
default_intrinsic_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeOverloadedResult());
  t = make_type(is_type_functional, ft);
  
  functional_parameters(ft) = 
    make_parameter_list(n, MakeOverloadedParameter);
  return t;
}

static type 
overloaded_to_integer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
overloaded_to_real_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeRealResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
overloaded_to_double_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeDoubleprecisionResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
overloaded_to_complex_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeComplexResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
overloaded_to_doublecomplex_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeDoublecomplexResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
overloaded_to_logical_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeLogicalResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
integer_to_integer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeIntegerParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
integer_to_real_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeRealResult());
  functional_parameters(ft) = make_parameter_list(n, MakeIntegerParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

/*
static type 
integer_to_double_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeDoubleprecisionResult());
  functional_parameters(ft) = make_parameter_list(n, MakeIntegerParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}
*/

static type 
real_to_integer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) = make_parameter_list(n, MakeRealParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
real_to_real_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeRealResult());
  functional_parameters(ft) = make_parameter_list(n, MakeRealParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
real_to_double_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeDoubleprecisionResult());
  functional_parameters(ft) = make_parameter_list(n, MakeRealParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
double_to_integer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeDoubleprecisionParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

/*
static type 
double_to_real_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeRealResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeDoubleprecisionParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}
*/

static type 
double_to_double_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeDoubleprecisionResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeDoubleprecisionParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
complex_to_real_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeRealResult());
  functional_parameters(ft) = make_parameter_list(n, MakeComplexParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
doublecomplex_to_double_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeDoubleprecisionResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeDoublecomplexParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
complex_to_complex_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeComplexResult());
  functional_parameters(ft) = make_parameter_list(n, MakeComplexParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
doublecomplex_to_doublecomplex_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeDoublecomplexResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeDoublecomplexParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
character_to_integer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeCharacterParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
character_to_logical_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeLogicalResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeCharacterParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
character_to_character_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeCharacterResult());
  functional_parameters(ft) = 
    make_parameter_list(n, MakeCharacterParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

static type 
substring_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeCharacterResult());
  functional_parameters(ft) = 
    CONS(PARAMETER, MakeIntegerParameter(), NIL);
  functional_parameters(ft) = 
    CONS(PARAMETER, MakeIntegerParameter(),
	 functional_parameters(ft));
  functional_parameters(ft) = 
    CONS(PARAMETER, MakeCharacterParameter(),
	 functional_parameters(ft));
  t = make_type(is_type_functional, ft);
  
  pips_assert("valid arity", gen_length(functional_parameters(ft))==n);
  
  return t;
}

static type 
assign_substring_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeCharacterResult());
  functional_parameters(ft) = 
    CONS(PARAMETER, MakeCharacterParameter(), NIL);
  functional_parameters(ft) = 
    CONS(PARAMETER, MakeIntegerParameter(),
	 functional_parameters(ft));
  functional_parameters(ft) = 
    CONS(PARAMETER, MakeIntegerParameter(),
	 functional_parameters(ft));
  functional_parameters(ft) = 
    CONS(PARAMETER, MakeCharacterParameter(),
	 functional_parameters(ft));
  t = make_type(is_type_functional, ft);

  pips_assert("valid arity", gen_length(functional_parameters(ft))==n);
  
  return t;
}

static type 
logical_to_logical_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;
  
  ft = make_functional(NIL, MakeLogicalResult());
  functional_parameters(ft) = make_parameter_list(n, MakeCharacterParameter);
  t = make_type(is_type_functional, ft);
  
  return t;
}

/***************************** TYPE A CALL FUNCTIONS **********************/

/***************************************************************************** 
 * Typing range of loop to the type of index loop. 
 * This range is already verified 
 */
void type_loop_range(basic index, range r, type_context_p context)
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

/************************************************************************** 
 * Convert a constant from INT to REAL
 * e.g: REAL(10) --> 10.0
 */
static call convert_constant_from_int_to_real(call c)
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
static call convert_constant_from_int_to_double(call c)
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
static call convert_constant_from_real_to_int(call c)
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
static call convert_constant_from_real_to_double(call c)
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
static call convert_constant_from_double_to_real(call c)
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
static call convert_constant_from_double_to_int(call c)
{
  call c_result, c_real = convert_constant_from_double_to_real(c);
  c_result = convert_constant_from_real_to_int(c_real);
  free_call(c_real);
  return c_result;
}
/* REAL -> COMPLEX
 * e.g: CMPLX(-5.9E5) => (-5.9E5, 0.0)
 */
static call convert_constant_from_real_to_complex(call c)
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
static call convert_constant_from_double_to_complex(call c)
{
  call c_result, c_real = convert_constant_from_double_to_real(c);
  c_result = convert_constant_from_real_to_complex(c_real);
  free_call(c_real);
  return c_result;
}
/* INT -> COMPLEX
 * e.g: CMPLX(-5) => (-5.0, 0.0)
 */
static call convert_constant_from_int_to_complex(call c)
{
  call c_result, c_real = convert_constant_from_int_to_real(c);
  c_result = convert_constant_from_real_to_complex(c_real);
  free_call(c_real);
  return c_result;
}
/* DOUBLE -> DCOMPLEX
 * e.g: DCMPLX(-5.9D5) => (-5.9D5, 0.0)
 */
static call convert_constant_from_double_to_dcomplex(call c)
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
static call convert_constant_from_real_to_dcomplex(call c)
{
  call c_result, c_double = convert_constant_from_real_to_double(c);
  c_result = convert_constant_from_double_to_dcomplex(c_double);
  free_call(c_double);
  return c_result;
}
/* INT -> DCOMPLEX
 * e.g: DCMPLX(-5) => (-5D0, 0.0D0)
 */
static call convert_constant_from_int_to_dcomplex(call c)
{
  call c_result, c_double = convert_constant_from_int_to_double(c);
  c_result = convert_constant_from_double_to_dcomplex(c_double);
  free_call(c_double);
  return c_result;
}

/***************************************************************************** 
 * Convert constant C to basic naming to_basic
 */
call convert_constant(call c, basic to_basic)
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
expression
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
expression 
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

/* Type check double complex? */

#define TC_DCOMPLEX \
get_bool_property("TYPE_CHECKER_DOUBLE_COMPLEX_EXTENSION")

/* Determine the longest basic among the arguments of c
 */
static basic 
basic_union_arguments(call c, hash_table types)
{
  basic b2, b1 = basic_undefined;
  
  MAP(EXPRESSION, e,
  {
    if (b1==basic_undefined)
    {
      /* first time */
      b1 = GET_TYPE(types, e);
    }
    else 
    {
      /* after first argument */
      b2 = GET_TYPE(types, e);
      if (is_inferior_basic(b1, b2))
	b1 = b2;
    }
  },
      call_arguments(c));
  
  return b1==basic_undefined? b1: copy_basic(b1);
}

/************* CHECK THE VALIDE OF ARGUMENTS BASIC OF FUNCTION ************/
/* Verify if all the arguments basic of function C are INTEGER
 * If there is no argument, I return TRUE
 */
static bool 
check_if_basics_ok(list le, hash_table types, bool(*basic_ok)(basic))
{
  MAP(EXPRESSION, e, 
  {
    if (!basic_ok(GET_TYPE(types, e))) 
    {
      return FALSE;
    }
  }
      , le);
  
  return TRUE;
}

static bool 
is_basic_int_p(basic b) 
{ 
  return basic_int_p(b); 
}
static bool 
is_basic_real_p(basic b) 
{ 
  return basic_float_p(b) && basic_float(b)==4; 
}
static bool 
is_basic_double_p(basic b) 
{ 
  return basic_float_p(b) && basic_float(b)==8; 
}
static bool 
is_basic_complex_p(basic b) 
{ 
  return basic_complex_p(b) && basic_complex(b)==8; 
}
static bool 
is_basic_dcomplex_p(basic b) 
{ 
  return basic_complex_p(b) && basic_complex(b)==16; 
}

static bool 
arguments_are_integer(call c, hash_table types)
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_int_p);
}

static bool 
arguments_are_real(call c, hash_table types)
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_real_p);
}
static bool 
arguments_are_double(call c, hash_table types)
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_double_p);
}
static bool 
arguments_are_complex(call c, hash_table types)
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_complex_p);
}
static bool 
arguments_are_dcomplex(call c, hash_table types)
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_dcomplex_p);
}

/************************************************************************** 
 * Verify if all the arguments basic of function C
 * If there is no argument, I return TRUE
 *
 * Note: I - Integer; R - Real; D - Double; C - Complex
 */


static bool 
arguments_are_something(
    call c, 
    type_context_p context,
    bool integer_ok,
    bool real_ok,
    bool double_ok,
    bool complex_ok,
    bool dcomplex_ok,
    bool logical_ok,
    bool character_ok)
{
  basic b;
  int argnumber = 0;
  bool 
    okay = TRUE,  
    arg_double = FALSE,
    arg_cmplx = FALSE;

  list args = call_arguments(c);
  
  MAP(EXPRESSION, e,
  {
    argnumber++;

    pips_assert("type is defined", hash_defined_p(context->types, e));
    
    b = GET_TYPE(context->types, e);

    /* Subroutine maybe be used as a function */
    if (basic_overloaded_p(b))
    {
      syntax s = expression_syntax(e);
      string what = NULL;
      switch (syntax_tag(s)) {
      case is_syntax_call: 
	what = entity_local_name(call_function(syntax_call(s)));
	break;
      case is_syntax_reference:
	what = entity_local_name(reference_variable(syntax_reference(s)));
	break;
      case is_syntax_range:
	what = "**RANGE**";
	break;
      default: pips_internal_error("unexpected syntax tag");
      }

      add_one_line_of_comment((statement) stack_head(context->stats),
			      "not typed '%s' used as a function.",
			      what, entity_local_name(call_function(c)));
      context->number_of_error++;
      okay = FALSE;
    }
    else if (!((integer_ok && basic_int_p(b)) ||
	  (real_ok && basic_float_p(b) && basic_float(b)==4) ||
	  (double_ok &&  basic_float_p(b) && basic_float(b)==8) ||
	  (complex_ok && basic_complex_p(b) && basic_complex(b)==8) ||
	  (dcomplex_ok && basic_complex_p(b) && basic_complex(b)==16) ||
	  (logical_ok && basic_logical_p(b)) ||
	  (character_ok && basic_string_p(b))))
    {
      add_one_line_of_comment((statement) stack_head(context->stats),
	   "#%d argument of '%s' must be %s%s%s%s%s%s%s but not %s",
			      argnumber,
			      entity_local_name(call_function(c)),
			      integer_ok? "INT, ": "",
			      real_ok? "REAL, ": "",
			      double_ok? "DOUBLE, ": "",
			      complex_ok? "COMPLEX, ": "",
			      dcomplex_ok? "DCOMPLEX, ": "",
			      logical_ok? "LOGICAL, ": "",
			      character_ok? "CHARACTER, ": "",
			      basic_to_string(b));
      context->number_of_error++;
      okay = FALSE;
    }

    /* if TC_DCOMPLEX, maybe they should not be incompatible? */
    arg_cmplx = arg_cmplx ||
      (complex_ok && basic_complex_p(b) && basic_complex(b)==8);

    arg_double = arg_double || 
      (double_ok &&  basic_float_p(b) && basic_float(b)==8);
  }
      , args);

  if (arg_cmplx && arg_double)
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
		    "mixed complex and double arguments of '%s' forbidden",
			    entity_local_name(call_function(c)));
    context->number_of_error++;
    okay = FALSE;
  }
  
  return okay;
}

static bool 
arguments_are_IRDCS(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, TRUE, TRUE, TRUE, TRUE, TC_DCOMPLEX, FALSE, TRUE);
}

static bool 
arguments_are_IRDC(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, TRUE, TRUE, TRUE, TRUE, TC_DCOMPLEX, FALSE, FALSE);
}
static bool
arguments_are_character(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE);
}
static bool
arguments_are_logical(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE);
}
static bool 
arguments_are_RD(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE);
}

static bool 
arguments_are_IR(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE);
}
static bool 
arguments_are_IRD(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE);
}
/************************************************************************** 
 * Verify if all the arguments basic of function C are REAL, DOUBLE 
 * and COMPLEX
 * According to (ANSI X3.9-1978 FORTRAN 77, Table 2 & 3, Page 6-5 & 6-6),
 * it is prohibited an arithetic operator operaters on 
 * a pair of DOUBLE and COMPLEX, so that I return FALSE in that case.
 *
 * PDSon: If there is no argument, I return TRUE
 */
static bool 
arguments_are_RDC(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, FALSE, TRUE, TRUE, TRUE, TC_DCOMPLEX, FALSE, FALSE);
}

/************************************************************************** 
 * Verification if all the arguments are compatible
 * PDSon: If #arguments <=1, I return true
 */
bool
arguments_are_compatible(call c, hash_table types)
{
  basic b1, b2;
  b1 = basic_undefined;
  
  MAP(EXPRESSION, e,
  { 
    /* First item */
    if(basic_undefined_p(b1))
    {
      b1 = GET_TYPE(types, e);
    }
    /* Next item */
    else
    {
      b2 = GET_TYPE(types, e);
      if(!basic_compatible_p(b1, b2))
      {
	return FALSE;
      }
    }
  }
      , call_arguments(c));
  
  return TRUE;
}

/************************************************************************** 
 * Typing all the arguments of c to basic b if their basic <> b
 */
static void 
typing_arguments(call c, type_context_p context, basic b)
{
  basic b1;
  list args = call_arguments(c);
  
  while (args != NIL)
  {
    b1 = GET_TYPE(context->types, EXPRESSION(CAR(args)));
    if (!basic_equal_p(b, b1))
    {
      EXPRESSION(CAR(args)) =
	insert_cast(b, b1, EXPRESSION(CAR(args)), context);
      /* Update hash table */
      PUT_TYPE(context->types, EXPRESSION(CAR(args)), copy_basic(b));
    }
    args = CDR(args);
  }
}

/************************************************************************** 
 *                           TYPING THE INTRINSIC FUNCTIONS
 * Typing arithmetic operator (+, -, --, *, /), except **
 */
static basic
typing_arithmetic_operator(call c, type_context_p context)
{
  basic b;
  
  if(!arguments_are_IRDC(c, context))
  {
    /* Just for return a result */
    return make_basic_float(4); 
  }
  /* Find the longest type amongs all arguments */
  b = basic_union_arguments(c, context->types);
  
  /* Typing all arguments to b if necessary */
  typing_arguments(c, context, b);
  
  return b;    
}
/************************************************************************** 
 * Typing power operator (**)
 */
static basic
typing_power_operator(call c, type_context_p context)
{
  basic b, b1, b2;
  list /* of expression */ args = call_arguments(c);
  b = basic_undefined;
  
  if(!arguments_are_IRDC(c, context))
  {
    /* Just for return a result */
    return make_basic_float(4); 
  }
  
  b1 = GET_TYPE(context->types, EXPRESSION(CAR(args)));
  b2 = GET_TYPE(context->types, EXPRESSION(CAR(CDR(args))));
  
  if (is_inferior_basic(b1, b2))
  {
    b = b2;
  }
  else
  {
    b = b1;
  }
  
  if (!basic_equal_p(b, b1))
  {
    EXPRESSION(CAR(args)) = 
      insert_cast(b, b1, EXPRESSION(CAR(args)), context);
  }
  /* Fortran prefers: (ANSI X3.9-1978, FORTRAN 77, PAGE 6-6, TABLE 3)
   * "var_double = var_double ** var_int" instead of
   * "var_double = var_double ** DBLE(var_int)"
   */
  if (!basic_equal_p(b, b2) && !basic_int_p(b2))
  {
    EXPRESSION(CAR(CDR(args))) = 
      insert_cast(b, b2, EXPRESSION(CAR(CDR(args))), context);
  }
  return copy_basic(b);
}
/************************************************************************** 
 * Typing relational operator (LT, LE, EQ, GT, GE) 
 */
static basic
typing_relational_operator(call c, type_context_p context)
{
  basic b;
  
  if(!arguments_are_IRDCS(c, context))
  {
    /* Just for return a result */
    return make_basic_logical(4);
  }
  /* Find the longest type amongs all arguments */
  b = basic_union_arguments(c, context->types);
  
  /* Typing all arguments to b if necessary */
  typing_arguments(c, context, b);
  
  free_basic(b);
  return make_basic_logical(4);
}
/************************************************************************** 
 * Typing logical operator (NOT, AND, OR, EQV, NEQV)
 */
static basic
typing_logical_operator(call c, type_context_p context)
{
  if(!arguments_are_logical(c, context))
  {
    /* Just for return a result */
    return make_basic_logical(4);
  }
  return make_basic_logical(4);
}
/************************************************************************** 
 * Typing concatenate operator (//)
 */
static basic
typing_concat_operator(call c, type_context_p context)
{
  if(!arguments_are_character(c, context))
  {
    /* Just for return a result */
    return make_basic_string(value_undefined);
  }
  return make_basic_string(value_undefined);
}

/************************************************************************** 
 * Typing function C whose argument type is from_type and
 * whose return type is to_type
 */
static basic 
typing_function_argument_type_to_return_type(call c, type_context_p context,
					     basic from_type, basic to_type)
{
  bool check_arg = FALSE;
  
  /* INT */
  if(basic_int_p(from_type))
  {
    check_arg = arguments_are_integer(c, context->types);
  }
  /* REAL */
  else if(basic_float_p(from_type) && basic_float(from_type) == 4)
  {
    check_arg = arguments_are_real(c, context->types);
  }
  /* DOUBLE */
  else if(basic_float_p(from_type) && basic_float(from_type) == 8)
  {
    check_arg = arguments_are_double(c, context->types);
  }
  /* COMPLEX */
  else if(basic_complex_p(from_type) && basic_complex(from_type) == 8)
  {
    check_arg = arguments_are_complex(c, context->types);
  }
  /* DOUBLE COMPLEX */
  else if(basic_complex_p(from_type) && basic_complex(from_type) == 16)
  {
    if (TC_DCOMPLEX)
      check_arg = arguments_are_dcomplex(c, context->types);
  }
  /* CHAR */
  else if(basic_string_p(from_type))
  {
    check_arg = arguments_are_character(c, context);
  }
  /* LOGICAL */
  else if(basic_logical_p(from_type))
  {
    check_arg = arguments_are_logical(c, context);
  }
  /* UNEXPECTED */
  else
  {
    pips_internal_error("Unexpected basic: %s \n", 
			basic_to_string(from_type));
  }

  /* ERROR: Invalide of argument type */
  if(check_arg == FALSE)
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "Invalid argument(s) to '%s'!",
			    entity_local_name(call_function(c))); 
    
    /* Count the number of errors */
    context->number_of_error++;
  }
  
  return copy_basic(to_type);
}

static basic
typing_function_int_to_int(call c, type_context_p context)
{
  basic result, type_INT = make_basic_int(4);
  result = typing_function_argument_type_to_return_type(c, context, 
						     type_INT, type_INT);
  free_basic(type_INT);
  return result;
}
static basic
typing_function_real_to_real(call c, type_context_p context)
{
  basic result, type_REAL = make_basic_float(4);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_REAL, type_REAL);
  free_basic(type_REAL);
  return result;
}
static basic
typing_function_double_to_double(call c, type_context_p context)
{
  basic result, type_DBLE = make_basic_float(8);
  result = typing_function_argument_type_to_return_type(c, context, 
						      type_DBLE, type_DBLE);
  free_basic(type_DBLE);
  return result;
}
static basic
typing_function_complex_to_complex(call c, type_context_p context)
{
  basic result, type_CMPLX = make_basic_complex(8);
  result = typing_function_argument_type_to_return_type(c, context, 
						     type_CMPLX, type_CMPLX);
  free_basic(type_CMPLX);
  return result;
}
static basic
typing_function_dcomplex_to_dcomplex(call c, type_context_p context)
{
  basic result, type_DCMPLX = make_basic_complex(16);
  result = typing_function_argument_type_to_return_type(c, context, 
						    type_DCMPLX, type_DCMPLX);
  free_basic(type_DCMPLX);
  return result;
}
static basic
typing_function_char_to_int(call c, type_context_p context)
{
  basic result, type_INT = make_basic_int(4);
  basic type_CHAR = make_basic_string(value_undefined);
  result = typing_function_argument_type_to_return_type(c, context, type_CHAR, 
						      type_INT);
  free_basic(type_INT);
  free_basic(type_CHAR);
  return result;
}
static basic
typing_function_int_to_char(call c, type_context_p context)
{
  basic result, type_INT = make_basic_int(4);
  basic type_CHAR = make_basic_string(value_undefined);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_INT, 
							type_CHAR);
  free_basic(type_INT);
  free_basic(type_CHAR);
  return result;
}
static basic
typing_function_real_to_int(call c, type_context_p context)
{
  basic result, type_INT = make_basic_int(4);
  basic type_REAL = make_basic_float(4);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_REAL,
							type_INT);
  free_basic(type_INT);
  free_basic(type_REAL);
  return result;
}
static basic
typing_function_int_to_real(call c, type_context_p context)
{
  basic result, type_INT = make_basic_int(4);
  basic type_REAL = make_basic_float(4);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_INT, 
							type_REAL);
  free_basic(type_INT);
  free_basic(type_REAL);
  return result;
}
static basic
typing_function_double_to_int(call c, type_context_p context)
{
  basic result, type_INT = make_basic_int(4);
  basic type_DBLE = make_basic_float(8);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_DBLE, 
							type_INT);
  free_basic(type_INT);
  free_basic(type_DBLE);
  return result;
}
static basic
typing_function_real_to_double(call c, type_context_p context)
{
  basic result, type_REAL = make_basic_float(4);
  basic type_DBLE = make_basic_float(8);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_REAL, 
							type_DBLE);
  free_basic(type_REAL);
  free_basic(type_DBLE);
  return result;
}
static basic
typing_function_complex_to_real(call c, type_context_p context)
{
  basic result, type_REAL = make_basic_float(4);
  basic type_CMPLX = make_basic_complex(8);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_CMPLX, 
							type_REAL);
  free_basic(type_REAL);
  free_basic(type_CMPLX);
  return result;
}
static basic
typing_function_dcomplex_to_double(call c, type_context_p context)
{
  basic result, type_DBLE = make_basic_float(8);
  basic type_DCMPLX = make_basic_complex(16);
  result = typing_function_argument_type_to_return_type(c, context,
							type_DCMPLX, 
							type_DBLE);
  free_basic(type_DBLE);
  free_basic(type_DCMPLX);
  return result;
}
static basic
typing_function_char_to_logical(call c, type_context_p context)
{
  basic result, type_LOGICAL = make_basic_logical(4);
  basic type_CHAR = make_basic_string(value_undefined);
  result = typing_function_argument_type_to_return_type(c, context, 
							type_CHAR, 
							type_LOGICAL);
  free_basic(type_LOGICAL);
  free_basic(type_CHAR);
  return result;
}

/************************************************************************** 
 * Arguments are REAL (or DOUBLE); and the return is the same with argument
 */
static basic
typing_function_RealDouble_to_RealDouble(call c, type_context_p context)
{
  basic b;
  
  if(!arguments_are_RD(c, context))
  {
    return make_basic_float(4); /* Just for return a result */
  }
  /* Find the longest type amongs all arguments */
  b = basic_union_arguments(c, context->types);
  
  /* Typing all arguments to b if necessary */
  typing_arguments(c, context, b);
  
  return b;    
}
static basic
typing_function_RealDouble_to_Integer(call c, type_context_p context)
{
  basic b;
  
  if(!arguments_are_RD(c, context))
  {
    return make_basic_float(4); /* Just for return a result */
  }
  /* Find the longest type amongs all arguments */
  b = basic_union_arguments(c, context->types);

  /* Typing all arguments to b if necessary */
  typing_arguments(c, context, b);

  free_basic(b);
  return make_basic_int(4);
}
static basic
typing_function_RealDoubleComplex_to_RealDoubleComplex(call c, 
						  type_context_p context)
{
  basic b;
  
  if(!arguments_are_RDC(c, context))
  {
    return make_basic_float(4); /* Just for return a result  */
  }
  /* Find the longest type amongs all arguments */
  b = basic_union_arguments(c, context->types);
  
  /* Typing all arguments to b if necessary */
  typing_arguments(c, context, b);
  
  return b;
}
static basic
typing_function_IntegerRealDouble_to_IntegerRealDouble(call c, 
					           type_context_p context)
{
  basic b;
  
  if(!arguments_are_IRD(c, context))
  {
    return make_basic_float(4); /* Just for return a result */
  }
  /* Find the longest type amongs all arguments */
  b = basic_union_arguments(c, context->types);

  // Typing all arguments to b if necessary
  typing_arguments(c, context, b);
  
  return b;    
}
/************************************************************************** 
 * The arguments are INT, REAL, DOUBLE or COMPLEX. The return is the same 
 * with the argument except case argument are COMPLEX, return is REAL
 *
 * Note: Only for Intrinsic ABS(): ABS(CMPLX(x)) --> REAL
 */
static basic
typing_function_IntegerRealDoubleComplex_to_IntegerRealDoubleReal(call c, 
					            type_context_p context)
{
  basic b;
  
  if(!arguments_are_IRDC(c, context))
  {
    return make_basic_float(4); /* Just for return result */
  }
  /* Find the longest type amongs all arguments */
  b = basic_union_arguments(c, context->types);
  
  /* Typing all arguments to b if necessary */
  typing_arguments(c, context, b);

  if (basic_complex_p(b) )
  {
    if (basic_complex(b)==8) 
    {
      free_basic(b);
      b = make_basic_float(4); /* CMPLX --> REAL */
    } else if (basic_complex(b)==16 && TC_DCOMPLEX) {
      free_basic(b);
      b = make_basic_float(8); /* DCMPLX -> DOUBLE */
    } /* else? */
  }
  return b;
}

/************************************************************************** 
 * Intrinsic conversion to a numeric
 *
 * Note: argument must be numeric
 */
static basic
typing_function_conversion_to_numeric(call c, type_context_p context, 
				      basic to_type)
{
  if(!arguments_are_IRDC(c, context))
  {
    return copy_basic(to_type);
  }
  return copy_basic(to_type);
}
static basic
typing_function_conversion_to_integer(call c, type_context_p context)
{
  basic result, b = make_basic_int(4);
  result = typing_function_conversion_to_numeric(c, context, b);
  free_basic(b);
  return result;
}
static basic
typing_function_conversion_to_real(call c, type_context_p context)
{
  basic result, b =  make_basic_float(4);
  result = typing_function_conversion_to_numeric(c, context, b);
  free_basic(b);
  return result;
}
static basic
typing_function_conversion_to_double(call c, type_context_p context)
{
  basic result, b =  make_basic_float(8);
  result = typing_function_conversion_to_numeric(c, context, b);
  free_basic(b);
  return result;
}
static basic
typing_function_conversion_to_complex(call c, type_context_p context)
{
  basic b;
  expression arg;
  if(!arguments_are_IRDC(c, context))
  {
    return make_basic_float(4); /* Just for return result */
  }

  arg = EXPRESSION(CAR(call_arguments(c)));
  if (CDR(call_arguments(c)) == NIL &&
      basic_complex_p(GET_TYPE(context->types, arg)))
  {
    syntax ss = expression_syntax(arg);
    if (syntax_call_p(ss))
    {
      call_arguments(c) = call_arguments(syntax_call(ss));
      context->number_of_conversion++;
    }
    else /* Argument is a varibale */
    {
      return make_basic_complex(8);
    }
    /* Free memory occupied by old argument list*/	  
  }

  /* Typing all arguments to REAL if necessary */
  b = make_basic_float(4);
  typing_arguments(c, context, b);
  free_basic(b);
  
  return make_basic_complex(8);
}
static basic
typing_function_conversion_to_dcomplex(call c, type_context_p context)
{
  basic b;
  expression arg;
  if(!arguments_are_IRDC(c, context))
  {
    return make_basic_float(4); /* Just for return result */
  }

  arg = EXPRESSION(CAR(call_arguments(c)));
  if (CDR(call_arguments(c)) == NIL &&
      basic_complex_p(GET_TYPE(context->types, arg)))
  {
    syntax ss = expression_syntax(arg);
    if (syntax_call_p(ss))
    {
      call_arguments(c) = call_arguments(syntax_call(ss));
      context->number_of_conversion++;
    }    
    else /* Argument is a varibale */
    {
      return make_basic_complex(16);
    }
    /* Free memory occupied by old argument list */
  }

  /* Typing all arguments to DBLE if necessary */
  b = make_basic_float(8);
  typing_arguments(c, context, b);
  free_basic(b);
  
  return make_basic_complex(16);
}
/* CMPLX_ */
static basic
typing_function_constant_complex(call c, type_context_p context)
{
  basic b;
  if(!arguments_are_IR(c, context))
  {
    return make_basic_float(4); /* Just for return result */
  }

  /* Typing all arguments to REAL if necessary */
  b = make_basic_float(4);
  typing_arguments(c, context, b);
  free_basic(b);
  
  return make_basic_complex(8);
}
/* DCMPLX_ */
static basic
typing_function_constant_dcomplex(call c, type_context_p context)
{
  basic b;
  if(!arguments_are_IRD(c, context))
  {
    return make_basic_float(4); /* Just for return result */
  }

  /* Typing all arguments to DOUBLE if necessary */
  b = make_basic_float(8);
  typing_arguments(c, context, b);
  free_basic(b);
  
  return make_basic_complex(16);
}

static basic
typing_function_overloaded(call c, type_context_p context)
{
  return make_basic_overloaded();
}

static basic
typing_function_format_name(call c, type_context_p context)
{
  return make_basic_int(4);
}

static basic 
typing_of_assign(call c, type_context_p context)
{
  list args = call_arguments(c);
  basic b1, b2;
    
  if(!arguments_are_compatible(c, context->types))
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
		       "Arguments of assignment '%s' are not compatible", 
			    entity_local_name(call_function(c))); 
    /* Count the number of errors */
    context->number_of_error++;
  }
  else
  {
    b1 = GET_TYPE(context->types, EXPRESSION(CAR(args)));
    b2 = GET_TYPE(context->types, EXPRESSION(CAR(CDR(args))));
    if (!basic_equal_p(b1, b2))
    {
      EXPRESSION(CAR(CDR(args))) = 
	insert_cast(b1, b2, EXPRESSION(CAR(CDR(args))), context);
      }
  }

  /* Here, we aren't interested in the type of return */
  return basic_undefined;
}

static basic 
typing_substring(call c, type_context_p context)
{
  int count = 0;

  MAP(EXPRESSION, e,
  {
    count++;
    switch (count)
    {
    case 1:
      if( !basic_string_p(GET_TYPE(context->types, e)) ||
	  !syntax_reference_p(expression_syntax(e)) )
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #1 must be a reference to string");
	/* Count the number of errors */
	context->number_of_error++;
      }
      break;
    case 2:
    case 3:
      if( !basic_int_p(GET_TYPE(context->types, e)))
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #%d must be an integer expression",
				count);
	/* Count the number of errors */
	context->number_of_error++;
      }
      break;
    default: /* count > 3 */
      add_one_line_of_comment((statement) stack_head(context->stats), 
		   "Too many of arguments for sub-string function");
      /* Count the number of errors */
      context->number_of_error++;
      return make_basic_string(value_undefined);
    }
  },
      call_arguments(c));
  if (count < 3)
  {
      add_one_line_of_comment((statement) stack_head(context->stats), 
		   "Lack of %d argument(s) for sub-string function",
			      3-count);
      /* Count the number of errors */
      context->number_of_error++;
  }

  return make_basic_string(value_undefined);
}

static basic
typing_assign_substring(call c, type_context_p context)
{
  int count = 0;
  MAP(EXPRESSION, e,
  {
    count++;
    switch (count)
    {
    case 1:
      if( !basic_string_p(GET_TYPE(context->types, e)) ||
	  !syntax_reference_p(expression_syntax(e)) )
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #1 must be a reference to string");
	/* Count the number of errors */
	context->number_of_error++;
      }
      break;
    case 2:
    case 3:
      if( !basic_int_p(GET_TYPE(context->types, e)))
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #%d must be an integer expression",
				count);
	/* Count the number of errors */
	context->number_of_error++;
      }
      break;
    case 4:
      if( !basic_string_p(GET_TYPE(context->types, e)))
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #4 must be a string expression");
	/* Count the number of errors */
	context->number_of_error++;
      }
      break;
    default: /* count > 4 */
      add_one_line_of_comment((statement) stack_head(context->stats), 
		   "Too many of arguments for assign sub-string function");
      /* Count the number of errors */
      context->number_of_error++;
      return basic_undefined;
    }
  },
      call_arguments(c));
  if (count < 4)
  {
      add_one_line_of_comment((statement) stack_head(context->stats), 
		   "Lack of %d argument(s) for assign sub-string function",
			      4-count);
      /* Count the number of errors */
      context->number_of_error++;
  }
  return basic_undefined;
}

static basic
typing_buffer_inout(call c, type_context_p context)
{
  int count = 0;

  MAP(EXPRESSION, e,
  {
    count++;
    switch (count)
    {
    case 1:
      if( !basic_int_p(GET_TYPE(context->types, e)) )
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #1 must be an integer expression");
	/* Count the number of errors */
	context->number_of_error++;
      }
      break;
    case 2:
    case 3:
    case 4:
      /* PDSon: Nobody knows the type of 3 last arguments, I do nothing here */
      break;
    default: /* count > 4 */
      add_one_line_of_comment((statement) stack_head(context->stats), 
			      "Too many of arguments for function '%s'",
			      entity_local_name(call_function(c)));
      /* Count the number of errors */
      context->number_of_error++;
      return basic_undefined;
    }
  },
      call_arguments(c));

  if (count < 4)
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "Lack of %d argument(s) for function '%s'",
			    4-count, entity_local_name(call_function(c)));
    /* Count the number of errors */
    context->number_of_error++;
  }
  return basic_undefined;
}

static basic no_typing(call c, type_context_p context)
{
  basic bt = basic_undefined;
  pips_internal_error("This should not be type-checked because it is not Fortran function\n");
  return bt; /* To please the compiler */
}

static basic
typing_implied_do(call c, type_context_p context)
{
  basic b_int = NULL;
  int count = 0;

  MAP(EXPRESSION, e,
  {
    count++;
    switch (count)
    {
    case 1:
      b_int = GET_TYPE(context->types, e);
      if( !basic_int_p(GET_TYPE(context->types, e)) ||
	  !syntax_reference_p(expression_syntax(e)) )
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #1 must be a reference to integer");
	/* Count the number of errors */
	context->number_of_error++;
      }
      break;
      
    case 2: /* range */
      if(!syntax_range_p(expression_syntax(e)))
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
				"Argument #2 must be a range of integer");
	/* Count the number of errors */
	context->number_of_error++;
	return basic_undefined;
      }
      else
      {
	range r = syntax_range(expression_syntax(e));
	if (!check_loop_range(r, context->types))
	{
	  add_one_line_of_comment((statement) stack_head(context->stats),
				  "Range must be Integer, Real or Double!");
	  context->number_of_error++;
	}
	else
	{
	  type_loop_range(b_int, r, context);
	}
      }
      return basic_undefined;
    }
  },
      call_arguments(c));

  if (count < 2)
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "Lack of %d argument(s) for function '%s'",
			    2-count, entity_local_name(call_function(c)));
    /* Count the number of errors */
    context->number_of_error++;
  }
  return basic_undefined;
}

/******************* VERIFICATION SYNTAX FOR STATEMENTS ********************/
/* Verify if an expression is a constant:
 * YES : return TRUE; otherwise, return FALSE 
 */
static bool
is_constant(expression exp)
{
  syntax s = expression_syntax(exp);
  if (!syntax_call_p(s))
  {
    return FALSE;
  }
  return (entity_constant_p(call_function(syntax_call(s))));
}

/* Verify if an expression is a constant of basic b:
 * YES : return TRUE; otherwise, return FALSE 
 */
static bool
is_constant_of_basic(expression exp, basic b)
{
  type call_type, return_type;
  basic bb;
  if (!is_constant(exp))
  {
    return FALSE;
  }
  call_type = entity_type(call_function(syntax_call(expression_syntax(exp))));
  return_type = functional_result(type_functional(call_type));
  bb = variable_basic(type_variable(return_type));
  if (basic_undefined_p(bb) || !basic_equal_p(b, bb))
  {
    return FALSE;
  }
  return TRUE;
}

static basic
statement_without_argument(call c, type_context_p context)
{
  if (call_arguments(c) != NIL)
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "Statement '%s' doesn't need any argument", 
			    entity_local_name(call_function(c))); 
    /* Count the number of errors */
    context->number_of_error++;
  }

  /* Here, we are not interested in the basic returned */
  return basic_undefined;
}
/***************************************************************************
 * Statement with at most one argument: integer or character constant, 
 * like PAUSE, STOP.
 * Attention: Integer value must be <= 99999 (at most 5 digits)
 * (According to ANSI X3.9-1978 FORTRAN 77; PAGE 11-9)
 */
static basic
statement_with_at_most_one_integer_or_character(call c, 
					       type_context_p context)
{
  basic b_int, b_char;
  expression arg1;
  entity en;
  int l;
  list args = call_arguments(c);
  if (args != NIL)
  {
    b_int = make_basic_int(4);
    b_char = make_basic_string(0);
    arg1 = EXPRESSION(CAR(args));
    if ( !is_constant_of_basic(arg1, b_int) &&
	 !is_constant_of_basic(arg1, b_char))
    {
      add_one_line_of_comment((statement) stack_head(context->stats), 
	   "Argument #1 of '%s' must be an integer or character constant",
			      entity_local_name(call_function(c))); 
      /* Count the number of errors */
      context->number_of_error++;
    }
    else if ( is_constant_of_basic(arg1, b_int) )
    {
      en = call_function(syntax_call(expression_syntax(arg1)));
      l = strlen(entity_local_name(en));
      if (l > 5)
      {
	add_one_line_of_comment((statement) stack_head(context->stats), 
	 "Argument must be an integer of at most 5 digits (instead of '%d')",
				l); 
      /* Count the number of errors */
      context->number_of_error++;
      }
    }

    if (CDR(args) != NIL)
    {
      add_one_line_of_comment((statement) stack_head(context->stats), 
			      "Statement '%s' needs at most an argument, " \
			      "neight integer constant nor character constant",
			      entity_local_name(call_function(c))); 
      /* Count the number of errors */
      context->number_of_error++;
    }
    free_basic(b_int);
    free_basic(b_char);
  }
  /* Here, we are not interested in the basic returned */
  return basic_undefined;
}

static basic
statement_with_at_most_one_expression_integer(call c, 
					     type_context_p context)
{
  basic b;
  list args = call_arguments(c);
  if (args != NIL)
  {
    expression arg1 = EXPRESSION(CAR(args));
    b = GET_TYPE(context->types, arg1);

    if ( !basic_int_p(b) )
    {
      add_one_line_of_comment((statement) stack_head(context->stats), 
			  "Argument #1 of '%s' must be an integer expression",
			      entity_local_name(call_function(c))); 
      /* Count the number of errors */
      context->number_of_error++;
    }

    if (CDR(args) != NIL)
    {
      add_one_line_of_comment((statement) stack_head(context->stats), 
			 "Statement '%s' needs at most an integer expression",
			      entity_local_name(call_function(c))); 
      /* Count the number of errors */
      context->number_of_error++;
    }
  }

  /* Here, we are not interested in the basic returned */
  return basic_undefined;
}

/************************************************ VERIFICATION OF SPECIFIERS */

static bool
is_label_statement(expression exp)
{
  call c;
  entity fc;
  syntax s = expression_syntax(exp);
  if (!syntax_call_p(s))
  {
    return FALSE;
  }
  c = syntax_call(s);
  fc = call_function(c);
  return entity_label_p(fc);
}

static bool 
is_label_specifier(string s, expression e, type_context_p context)
{
  if (!is_label_statement(e))
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "%s specifier must be a label statement", s);
    
    /* Count the number of errors */
    context->number_of_error++;
    return FALSE;
  }
  return TRUE;
}

static bool 
is_integer_specifier(string s, expression e, type_context_p context)
{
  basic b = GET_TYPE(context->types, e);

  if (!basic_int_p(b))
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "%s specifier must be an integer expression", s);
    
    /* Count the number of errors */
    context->number_of_error++;
    return FALSE;
  }
  return TRUE;
}

static bool 
is_string_specifier(string s,expression e, type_context_p context)
{  
  basic b = GET_TYPE(context->types, e);
  if (!basic_string_p(b))
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
			    "%s specifier must be a character expression", s);
    
    /* Count the number of errors */
    context->number_of_error++;
    return FALSE;
  }
  return TRUE;
}

static bool 
is_label_integer_string_specifier(string s, expression e, 
				  type_context_p context)
{  
  basic b = GET_TYPE(context->types, e);
  if (!is_label_statement(e) && !basic_int_p(b) && !basic_string_p(b))
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
       "%s specifier must be a label, an integer or character expression", s);

    /* Count the number of errors */
    context->number_of_error++;
    return FALSE;
  }
  
  return TRUE;
}

static bool
is_varibale_array_element_specifier(string s, expression e, basic b,
				    type_context_p context)
{
  if (!basic_equal_p(GET_TYPE(context->types, e), b) ||
      !syntax_reference_p(expression_syntax(e)))
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
     "%s specifier must be a variable or an array element of %s", s,
			    basic_string_p(b)? "STRING":basic_to_string(b));

    /* Count the number of errors */
    context->number_of_error++;
    return FALSE;
  }
  return TRUE;

}
/* This function verifies the unit specifier; that is integer positive 
 * expression or character expression
 * (According to ANSI X3.9-1978 FORTRAN 77; PAGE 12-7)
 */
static bool
is_unit_specifier(expression exp, type_context_p context)
{
  basic b;
  b = GET_TYPE(context->types, exp);
  if (!basic_int_p(b) && !basic_string_p(b))
  {
    add_one_line_of_comment((statement) stack_head(context->stats), 
	    "UNIT specifier must be an integer or character expression");

    /* Count the number of errors */
    context->number_of_error++;
    return FALSE;
  }

  return TRUE;
}

static bool
is_format_specifier(expression exp, type_context_p context)
{
  return is_label_integer_string_specifier("FORMAT", exp, context);
}

static bool
is_record_specifier(expression exp, type_context_p context)
{
  return is_label_integer_string_specifier("RECORD", exp, context);
}

/* Integer variable or integer array element which is maybe modified
 */
static bool
is_iostat_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_int(4);
  bool r = is_varibale_array_element_specifier("IOSTAT", exp, b, context);
  free_basic(b);
  return r;
}


/* Error specifier is a label statement
 */
static bool
is_err_specifier(expression exp, type_context_p context)
{
  return is_label_specifier("ERR", exp, context);
}

static bool
is_end_specifier(expression exp, type_context_p context)
{
  return is_label_specifier("END", exp, context);
}

static bool
is_file_specifier(expression exp, type_context_p context)
{
  return is_string_specifier("FILE", exp, context);
}

static bool
is_status_specifier(expression exp, type_context_p context)
{
  return is_string_specifier("STATUS", exp, context);
}

static bool
is_access_specifier(expression exp, type_context_p context)
{
  return is_string_specifier("ACCESS", exp, context);
}

static bool
is_form_specifier(expression exp, type_context_p context)
{
  return is_string_specifier("FORM", exp, context);
}

static bool
is_recl_specifier(expression exp, type_context_p context)
{
  return is_integer_specifier("RECL", exp, context);
}

static bool
is_blank_specifier(expression exp, type_context_p context)
{
  return is_string_specifier("BLANK", exp, context);
}

static bool 
is_exist_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_logical(4);
  bool r = is_varibale_array_element_specifier("IOSTAT", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_opened_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_logical(4);
  bool r = is_varibale_array_element_specifier("OPENED", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_number_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_int(4);
  bool r = is_varibale_array_element_specifier("NUMBER", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_named_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_logical(4);
  bool r = is_varibale_array_element_specifier("NAMED", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_name_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_string(0);
  bool r = is_varibale_array_element_specifier("NAME", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_sequential_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_string(0);
  bool r = is_varibale_array_element_specifier("SEQUENTIAL", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_direct_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_string(0);
  bool r = is_varibale_array_element_specifier("DIRECT", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_formated_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_string(0);
  bool r = is_varibale_array_element_specifier("FORMATTED", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_unformated_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_string(0);
  bool r = is_varibale_array_element_specifier("UNFORMATTED", exp, b, context);
  free_basic(b);
  return r;
}

static bool
is_nextrec_specifier(expression exp, type_context_p context)
{
  basic b = make_basic_int(4);
  bool r = is_varibale_array_element_specifier("NEXTREC", exp, b, context);
  free_basic(b);
  return r;
}

static bool 
check_spec (string name,
	    bool allowed,
	    string specifier,
	    expression contents,
	    type_context_p context,
	    bool (*check_contents)(expression, type_context_p))
{ 
  if (same_string_p(name, specifier))
  {
    if (allowed)
    {
      if (check_contents(contents, context))
      {
	return TRUE;
      }
      /* else ok */
    }
    else /* not allowed */
    {
      add_one_line_of_comment((statement) stack_head(context->stats),
			      "Specifier '%s' is not allowed", name);
      context->number_of_error++;
    }
    return TRUE; /* checked! */
  }
  
  return FALSE;
}

static bool 
check_io_list(list /* of expression */ args,
	      type_context_p ctxt,
	      bool a_unit,			  
	      bool a_fmt,
	      bool a_rec,
	      bool a_iostat,
	      bool a_err,
	      bool a_end,
	      bool a_iolist,
	      bool a_file,
	      bool a_status,
	      bool a_access,
	      bool a_form,
	      bool a_blank,
	      bool a_recl,
	      bool a_exist,
	      bool a_opened,
	      bool a_number,
	      bool a_named,
	      bool a_name,
	      bool a_sequential,
	      bool a_direct,
	      bool a_formatted,
	      bool a_unformatted,
	      bool a_nextrec)
{
  string spec;
  pips_assert("Even number of arguments", gen_length(args)%2==0);
  
  for (; args; args = CDR(CDR(args)))
  {
    expression specifier = EXPRESSION(CAR(args));
    expression cont = EXPRESSION(CAR(CDR(args)));
    
    syntax s = expression_syntax(specifier);
    pips_assert("Specifier must be a call with arguments", 
		!syntax_call_p(s) || !call_arguments(syntax_call(s)));
    
    spec = entity_local_name(call_function(syntax_call(s)));
    
    /* specifier must be UNIT= FMT=... */
    if (!check_spec("UNIT=", a_unit, spec, cont, ctxt, 
		    is_unit_specifier) &&
	!check_spec("FMT=", a_fmt, spec, cont, ctxt, 
		    is_format_specifier) &&
	!check_spec("IOSTAT=", a_iostat, spec, cont, ctxt, 
		    is_iostat_specifier) &&
	!check_spec("REC=", a_rec, spec, cont, ctxt, 
		    is_record_specifier) &&
	!check_spec("ERR=", a_err, spec, cont, ctxt, 
		    is_err_specifier) &&
	!check_spec("END=", a_end, spec, cont, ctxt, 
		    is_end_specifier) &&
	!check_spec("IOLIST=", a_iolist, spec, cont, ctxt, 
		    (bool (*)(expression, type_context_p)) gen_true) &&
	!check_spec("FILE=", a_file, spec, cont, ctxt, 
		    is_file_specifier) &&
	!check_spec("STATUS=", a_status, spec, cont, ctxt, 
		    is_status_specifier) &&
	!check_spec("ACCESS=", a_access, spec, cont, ctxt, 
		    is_access_specifier) &&
	!check_spec("FORM=", a_form, spec, cont, ctxt, 
		    is_form_specifier) &&
	!check_spec("BLANK=", a_blank, spec, cont, ctxt, 
		    is_blank_specifier) &&
	!check_spec("RECL=", a_recl, spec, cont, ctxt, 
		    is_recl_specifier) &&
	!check_spec("EXIST=", a_exist, spec, cont, ctxt, 
		    is_exist_specifier) &&
	!check_spec("OPENED=", a_opened, spec, cont, ctxt, 
		    is_opened_specifier) &&
	!check_spec("NUMBER=", a_number, spec, cont, ctxt, 
		    is_number_specifier) &&
	!check_spec("NAMED=", a_named, spec, cont, ctxt, 
		    is_named_specifier) &&
	!check_spec("NAME=", a_name, spec, cont, ctxt, 
		    is_name_specifier) &&
	!check_spec("SEQUENTIAL=", a_sequential, spec, cont, ctxt, 
		    is_sequential_specifier) &&
	!check_spec("DIRECT=", a_direct, spec, cont, ctxt, 
		    is_direct_specifier) &&
	!check_spec("FORMATED=", a_formatted, spec, cont, ctxt, 
		    is_formated_specifier) &&
	!check_spec("UNFORMATED=", a_unformatted, spec, cont, ctxt, 
		    is_unformated_specifier) &&
	!check_spec("NEXTREC=", a_nextrec, spec, cont, ctxt, 
		    is_nextrec_specifier))
    {
      add_one_line_of_comment((statement) stack_head(ctxt->stats),
			      "Unexpected specifier '%s'", spec);
      ctxt->number_of_error++;
      return FALSE;
    }
  }
  
  return TRUE;
}

static basic 
check_read_write(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
		/* UNIT FMT REC IOSTAT ERR END IOLIST */
		TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
		/* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* UNFORMATTED NEXTREC */
		FALSE, FALSE);

  return basic_undefined;
}

static basic
check_open(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
		/* UNIT FMT REC IOSTAT ERR END IOLIST */
		TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE,
		/* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
		TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE,
		/* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* UNFORMATTED NEXTREC */
		FALSE, FALSE);
  return basic_undefined;
}

static basic
check_close(call c, type_context_p context)
{
  list args = call_arguments(c);

  check_io_list(args, context,
		/* UNIT FMT REC IOSTAT ERR END IOLIST */
		TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE,
		/* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
		FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* UNFORMATTED NEXTREC */
		FALSE, FALSE);
  return basic_undefined;
}

static basic
check_inquire(call c, type_context_p context)
{
  list args = call_arguments(c);

  check_io_list(args, context,
		/* UNIT FMT REC IOSTAT ERR END IOLIST */
		TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE,
		/* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
		TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
		/* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
		TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
		/* UNFORMATTED NEXTREC */
		TRUE, TRUE);

  return basic_undefined;
}

static basic
check_backspace(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
		/* UNIT FMT REC IOSTAT ERR END IOLIST */
		TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE,
		/* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* UNFORMATTED NEXTREC */
		FALSE, FALSE);
  return basic_undefined;
}

static basic
check_endfile(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
		/* UNIT FMT REC IOSTAT ERR END IOLIST */
		TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE,
		/* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* UNFORMATTED NEXTREC */
		FALSE, FALSE);
  return basic_undefined;
}

static basic
check_rewind(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
		/* UNIT FMT REC IOSTAT ERR END IOLIST */
		TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE,
		/* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
		FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
		/* UNFORMATTED NEXTREC */
		FALSE, FALSE);
  return basic_undefined;
}

static basic
check_format(call c, type_context_p context)
{
  list args = call_arguments(c);
  if (args == NIL)
  {
    add_one_line_of_comment((statement) stack_head(context->stats),
			    "FORMAT statement needs a format specification");
    context->number_of_error++;    
  }
  else
  {
    expression exp = EXPRESSION(CAR(args));
    if (!basic_string_p(GET_TYPE(context->types, exp)))
    {
      add_one_line_of_comment((statement) stack_head(context->stats),
			      "Format specification must be a string");
      context->number_of_error++;    
    }
  }
  return basic_undefined;
}

/*********************** SIMPLIFICATION DES EXPRESSIONS **********************/
/* Find the specific name from the specific argument
 * ---
 * Each intrinsic of name generic have a function for switching to the
 * specific name correspondent with the argument
 */
static void
switch_generic_to_specific(expression exp, type_context_p context,
			   string arg_int_name,
			   string arg_real_name,
			   string arg_double_name,
			   string arg_complex_name,
			   string arg_dcomplex_name)
{
  call c;
  list args;
  basic arg_basic;
  string specific_name = NULL;
  /* Here, expression_syntax(exp) is always a call */
  syntax s = expression_syntax(exp);
  c = syntax_call(s);  
  args = call_arguments(c);
  arg_basic = GET_TYPE(context->types, EXPRESSION(CAR(args)));
  
  if (basic_int_p(arg_basic))
  {
    specific_name = arg_int_name;
  }
  else if (basic_float_p(arg_basic) && basic_float(arg_basic) == 4)
  {
    specific_name = arg_real_name;
  }
  else if (basic_float_p(arg_basic) && basic_float(arg_basic) == 8)
  {
    specific_name = arg_double_name;
  }
  else if (basic_complex_p(arg_basic) && basic_complex(arg_basic) == 8)
  {
    specific_name = arg_complex_name;
  }
  else if (basic_complex_p(arg_basic) && basic_complex(arg_basic) == 16)
  {
    if (TC_DCOMPLEX)
      specific_name = arg_dcomplex_name;
    /* else generic name is kept... */
  }
  
  /* Modify the (function:entity) of the call c if necessary
   * NOTE: If specific_name == NULL: Invalid argument or 
   * argument basic unknown
   */
  if(specific_name != NULL && 
     strcmp(specific_name, entity_local_name(call_function(c))) != 0)
  {
    call_function(c) = CreateIntrinsic(specific_name);
    
    /* Count number of simplifications */
    context->number_of_simplication++;
  }
}

/* AINT */
static void
switch_specific_aint(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "AINT", "DINT", NULL, NULL);
}
/* ANINT */
static void
switch_specific_anint(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "ANINT", "DNINT", NULL, NULL);
}
/* NINT */
static void
switch_specific_nint(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "NINT", "IDNINT", NULL, NULL);
}
/* ABS */
static void
switch_specific_abs(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     "IABS", "ABS", "DABS", "CABS", "CDABS");
}
/* MOD */
static void
switch_specific_mod(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     "MOD", "AMOD", "DMOD", NULL, NULL);
}
/* SIGN */
static void
switch_specific_sign(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     "ISIGN", "SIGN", "DSIGN", NULL, NULL);
}
/* DIM */
static void
switch_specific_dim(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     "IDIM", "DIM", "DDIM", NULL, NULL);
}
/* MAX */
static void
switch_specific_max(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     "MAX0", "AMAX1", "DMAX1", NULL, NULL);
}
/* MIN */
static void
switch_specific_min(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     "MIN0", "AMIN1", "DMIN1", NULL, NULL);
}
/* SQRT */
static void
switch_specific_sqrt(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "SQRT", "DSQRT", "CSQRT", "CDSQRT");
}
/* EXP */
static void
switch_specific_exp(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "EXP", "DEXP", "CEXP", "CDEXP");
}
/* LOG */
static void
switch_specific_log(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "ALOG", "DLOG", "CLOG", "CDLOG");
}
/* LOG10 */
static void
switch_specific_log10(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "ALOG10", "DLOG10", NULL, NULL);
}
/* SIN */
static void
switch_specific_sin(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL,"SIN","DSIN", "CSIN", "CDSIN");
}
/* COS */
static void
switch_specific_cos(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "COS", "DCOS", "CCOS", "CDCOS");
}
/* TAN */
static void
switch_specific_tan(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "TAN", "DTAN", NULL, NULL);
}
/* ASIN */
static void
switch_specific_asin(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "ASIN", "DASIN", NULL, NULL);
}
/* ACOS */
static void
switch_specific_acos(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "ACOS", "DACOS", NULL, NULL);
}
/* ATAN */
static void
switch_specific_atan(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "ATAN", "DATAN", NULL, NULL);
}
/* ATAN2 */
static void
switch_specific_atan2(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "ATAN2", "DATAN2", NULL, NULL);
}
/* SINH */
static void
switch_specific_sinh(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "SINH", "DSINH", NULL, NULL);
}
/* COSH */
static void
switch_specific_cosh(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "COSH", "DCOSH", NULL, NULL);
}
/* TANH */
static void
switch_specific_tanh(expression exp, type_context_p context)
{
  switch_generic_to_specific(exp, context,
			     NULL, "TANH", "DTANH", NULL, NULL);
}

/* forward declarations */
static void simplification_complex(expression, type_context_p);
static void simplification_dcomplex(expression, type_context_p);

static void 
switch_specific_cmplx(expression exp, type_context_p context)
{
  if (get_bool_property("TYPE_CHECKER_EXPLICIT_COMPLEX_CONSTANTS"))
  {
    pips_assert("expression is a call", expression_call_p(exp));
    call_function(syntax_call(expression_syntax(exp))) = 
      CreateIntrinsic("CMPLX");
  }
  simplification_complex(exp, context);
}

static void 
switch_specific_dcmplx(expression exp, type_context_p context)
{
  if (get_bool_property("TYPE_CHECKER_EXPLICIT_COMPLEX_CONSTANTS") &&
      TC_DCOMPLEX)
  {
    pips_assert("expression is a call", expression_call_p(exp));
    call_function(syntax_call(expression_syntax(exp))) = 
      CreateIntrinsic("DCMPLX");
  }
  simplification_dcomplex(exp, context);
}

/************************* SIMPLIFICATION THE CONVERSION CALL *************
 * e.g: INT(INT(R)) -> INT(R)
 *      INT(2.9) -> 2
 *      INT(I) -> I
 */
static void
simplification_conversion(expression exp, basic to_basic,
			  type_context_p context)
{
  syntax s_arg;
  expression arg, exp_tmp = NULL;
  basic b;
  call c = syntax_call(expression_syntax(exp));
  arg = EXPRESSION(CAR(call_arguments(c)));
  s_arg = expression_syntax(arg);
  /* Argument is a variable */
  if (syntax_reference_p(s_arg))
  {
    /* e.g: INT(I) -> I */
    if (basic_equal_p(to_basic, 
		      entity_basic(reference_variable(
				   syntax_reference(s_arg)))) &&
	CDR(call_arguments(c)) == NIL)
    {
      exp_tmp = copy_expression(arg);      
      context->number_of_simplication++;
    }
    /* e.g: CMPLX(R) -> CMPLX(R, 0.0E0) */
    else if (ENTITY_CONVERSION_CMPLX_P(call_function(c)) &&
	     ! basic_complex_p(entity_basic
		   (reference_variable(syntax_reference(s_arg)))) &&
	     CDR(call_arguments(c)) == NIL)
    {
      call c_imag;
      expression exp_imag;
      c_imag = make_call(make_constant_entity("0.0E0", 
					      is_basic_float, 8),
			 NIL);
      exp_imag = make_expression(make_syntax(is_syntax_call, c_imag), 
				 normalized_undefined);
      call_arguments(c) 
	= CONS(EXPRESSION, arg, CONS(EXPRESSION, exp_imag, NIL));	    
      context->number_of_simplication++;
    }
    /* e.g: DCMPLX(D) -> DCMPLX(D, 0.0D0) */
    else if (ENTITY_CONVERSION_DCMPLX_P(call_function(c)) &&
	     ! basic_complex_p(entity_basic
		   (reference_variable(syntax_reference(s_arg)))) &&
	     CDR(call_arguments(c)) == NIL)
    {
      call c_imag;
      expression exp_imag;
      c_imag = make_call(make_constant_entity("0.0D0",
					      is_basic_float, 16),
			 NIL);
      exp_imag = make_expression(make_syntax(is_syntax_call, c_imag), 
				 normalized_undefined);
      call_arguments(c) 
	= CONS(EXPRESSION, arg, CONS(EXPRESSION, exp_imag, NIL));	    
      context->number_of_simplication++;
    }
  }
  /* Argument is a call */
  else if(syntax_call_p(s_arg))
  {
    b = GET_TYPE(context->types, arg);
    /* e.g: INT(INT(R)) -> INT(R) */
    if (basic_equal_p(b, to_basic))
    {
      exp_tmp = copy_expression(arg);

      context->number_of_simplication++;
    }
    /* Cast constant if necessary */
    /* Conversion: CMPLX, CMPLX_, DCMPLX, DCMPLX_ */
    else if (ENTITY_IMPLIED_CMPLX_P(call_function(c)) ||
	     ENTITY_CONVERSION_CMPLX_P(call_function(c)) ||
	     ENTITY_IMPLIED_DCMPLX_P(call_function(c)) ||
	     ENTITY_CONVERSION_DCMPLX_P(call_function(c)))
    {
      list args = call_arguments(c);
      basic b_arg = GET_TYPE(context->types, arg);
      /* Imagine party is empty */
      if (CDR(args) == NIL)
      {
	/* Argument is NOT complex or double complex */
	if (!basic_complex_p(b_arg))
	{
	  call c_imag;
	  expression exp_imag;
	  /* CMPLX */
	  if ((ENTITY_IMPLIED_CMPLX_P(call_function(c)) ||
	       ENTITY_CONVERSION_CMPLX_P(call_function(c))))
	  {
	    c_imag = make_call(make_constant_entity("0.0E0", 
						    is_basic_float, 8),
			       NIL);
	  }
	  /* DCMPLX */
	  else
	  {
	    c_imag = make_call(make_constant_entity("0.0D0",
						    is_basic_float, 16),
			       NIL);
	  }	  
	  exp_imag = make_expression(make_syntax(is_syntax_call, c_imag), 
				     normalized_undefined);
	  call_arguments(c) 
	    = CONS(EXPRESSION, arg, CONS(EXPRESSION, exp_imag, NIL));	    
	  context->number_of_simplication++;
	}
	/* CMPLX(C) -> C;  DCMPLX(DC) -> DC */
	else if( (basic_complex(b_arg) == 8 && 
		 ENTITY_CONVERSION_CMPLX_P(call_function(c))) || 
		 (basic_complex(b_arg) == 16 && 
		  ENTITY_CONVERSION_DCMPLX_P(call_function(c))))
	{
	  syntax s_tmp;
	  normalized n_tmp;
	  /* Argument being a call is examined in typing function */
	  pips_assert("Argument is a call ",
		      syntax_call_p(expression_syntax(arg)));

	  s_tmp = expression_syntax(exp);
	  n_tmp = expression_normalized(exp);
	  expression_syntax(exp) 
	    = copy_syntax(expression_syntax(arg));
	  expression_normalized(exp) 
	    = copy_normalized(expression_normalized(arg));

	  free_syntax(s_tmp);
	  free_normalized(n_tmp);
	  context->number_of_simplication++;
	  return;
	}
      }
      /* Cast constants if necessary */
      exp_tmp = cast_constant(exp, to_basic, context); 
      /* Number of simplifications is already counted in cast_constant() */
    }
    /* Conversion: INT (IFIX, ...), REAL (FLOAT, ...), DBLE 
     * e.g: INT(2.9) -> 2
     */
    else
    {
      exp_tmp = cast_constant(arg, to_basic, context);
      /* Number of simplifications is already counted in cast_constant() */
    }
  }
  /* Update exp */
  if (exp_tmp != NULL)
  {
    free_syntax(expression_syntax(exp));
    free_normalized(expression_normalized(exp));
    expression_syntax(exp) = copy_syntax(expression_syntax(exp_tmp));
    expression_normalized(exp) = copy_normalized(
					   expression_normalized(exp_tmp));
    free_expression(exp_tmp);
  }
}

static void
simplification_int(expression exp, type_context_p context)
{
  basic b = make_basic_int(4);
  simplification_conversion(exp, b, context);
  free_basic(b);
}
static void
simplification_real(expression exp, type_context_p context)
{
  basic b = make_basic_float(4);
  simplification_conversion(exp, b, context);
  free_basic(b);
}
static void
simplification_double(expression exp, type_context_p context)
{
  basic b = make_basic_float(8);
  simplification_conversion(exp, b, context);
  free_basic(b);
}
static void
simplification_complex(expression exp, type_context_p context)
{
  basic b = make_basic_complex(8);
  simplification_conversion(exp, b, context);
  free_basic(b);
}
static void
simplification_dcomplex(expression exp, type_context_p context)
{
  basic b = make_basic_complex(16);
  simplification_conversion(exp, b, context);
  free_basic(b);
}

/******************************************************** INTRINSICS LIST */

/* The following data structure describes an intrinsic function: its
   name and its arity and its type. */

typedef struct IntrinsicDescriptor 
{
  string name;
  int nbargs;
  type (*intrinsic_type)(int);
  typing_function_t type_function;
  switch_name_function name_function;
} IntrinsicDescriptor;

/* The table of intrinsic functions. this table is used at the begining
   of linking to create Fortran operators, commands and intrinsic functions.

   Functions with a variable number of arguments are declared with INT_MAX
   arguments.
*/

static IntrinsicDescriptor IntrinsicDescriptorTable[] = 
{
  {"+", 2, default_intrinsic_type, typing_arithmetic_operator, 0},
  {"-", 2, default_intrinsic_type, typing_arithmetic_operator, 0},
  {"/", 2, default_intrinsic_type, typing_arithmetic_operator, 0},
  {"*", 2, default_intrinsic_type, typing_arithmetic_operator, 0},
  {"--", 1, default_intrinsic_type, typing_arithmetic_operator, 0},
  {"**", 2, default_intrinsic_type, typing_power_operator, 0},

  /* internal inverse operator... */
  {"INV", 1, real_to_real_type, 
   typing_function_RealDoubleComplex_to_RealDoubleComplex, 0},
  
  {"=", 2, default_intrinsic_type, typing_of_assign, 0},
  
  {".EQV.", 2, overloaded_to_logical_type, typing_logical_operator, 0},
  {".NEQV.", 2, overloaded_to_logical_type, typing_logical_operator, 0},
  
  {".OR.", 2, logical_to_logical_type, typing_logical_operator, 0},
  {".AND.", 2, logical_to_logical_type, typing_logical_operator, 0},
  {".NOT.", 1, logical_to_logical_type, typing_logical_operator, 0},
  
  {".LT.", 2, overloaded_to_logical_type, typing_relational_operator, 0},
  {".GT.", 2, overloaded_to_logical_type, typing_relational_operator, 0},
  {".LE.", 2, overloaded_to_logical_type, typing_relational_operator, 0},
  {".GE.", 2, overloaded_to_logical_type, typing_relational_operator, 0},
  {".EQ.", 2, overloaded_to_logical_type, typing_relational_operator, 0},
  {".NE.", 2, overloaded_to_logical_type, typing_relational_operator, 0},
  
  {"//", 2, character_to_character_type, typing_concat_operator, 0},

  /* IO statement */
  {"WRITE", (INT_MAX), default_intrinsic_type, check_read_write, 0},
  {"REWIND", (INT_MAX), default_intrinsic_type, check_rewind, 0},
  {"BACKSPACE", (INT_MAX), default_intrinsic_type, check_backspace, 0},
  {"OPEN", (INT_MAX), default_intrinsic_type, check_open, 0},
  {"CLOSE", (INT_MAX), default_intrinsic_type, check_close, 0},
  {"READ", (INT_MAX), default_intrinsic_type, check_read_write, 0},
  {"BUFFERIN", (INT_MAX), default_intrinsic_type, typing_buffer_inout, 0},
  {"BUFFEROUT", (INT_MAX), default_intrinsic_type, typing_buffer_inout, 0},
  {"ENDFILE", (INT_MAX), default_intrinsic_type, check_endfile, 0},
  {IMPLIED_DO_NAME, (INT_MAX), default_intrinsic_type, typing_implied_do, 0},
  {REPEAT_VALUE_NAME, 2, default_intrinsic_type, no_typing, 0},
  {FORMAT_FUNCTION_NAME, 1, default_intrinsic_type, check_format, 0},
  {"INQUIRE", (INT_MAX), default_intrinsic_type, check_inquire, 0},
  

  {SUBSTRING_FUNCTION_NAME, 3, substring_type, typing_substring, 0},
  {ASSIGN_SUBSTRING_FUNCTION_NAME, 4, assign_substring_type, 
   typing_assign_substring, 0},

  /* Control statement */
  {"CONTINUE", 0, default_intrinsic_type, statement_without_argument, 0},
  {"ENDDO", 0, default_intrinsic_type, 0, 0},
  {"PAUSE", 1, default_intrinsic_type,
   statement_with_at_most_one_integer_or_character, 0},
  {"RETURN", 0, default_intrinsic_type, 
   statement_with_at_most_one_expression_integer, 0},
  {"STOP", 0, default_intrinsic_type,
   statement_with_at_most_one_integer_or_character, 0},
  {"END", 0, default_intrinsic_type, statement_without_argument, 0},
  

  {"INT", 1, overloaded_to_integer_type, 
   typing_function_conversion_to_integer, simplification_int},
  {"IFIX", 1, real_to_integer_type, typing_function_real_to_int, 
   simplification_int},
  {"IDINT", 1, double_to_integer_type, typing_function_double_to_int, 
   simplification_int},
  {"REAL", 1, overloaded_to_real_type, typing_function_conversion_to_real, 
   simplification_real},
  {"FLOAT", 1, overloaded_to_real_type, typing_function_conversion_to_real, 
   simplification_real},
  {"DFLOAT", 1, overloaded_to_double_type, 
   typing_function_conversion_to_double, simplification_double},
  {"SNGL", 1, overloaded_to_real_type, typing_function_conversion_to_real, 
   simplification_real},
  {"DBLE", 1, overloaded_to_double_type, 
   typing_function_conversion_to_double, simplification_double},
  {"DREAL", 1, overloaded_to_double_type, /* Arnauld Leservot, code CEA */
   typing_function_conversion_to_double, simplification_double}, 
  {"CMPLX", (INT_MAX), overloaded_to_complex_type, 
   typing_function_conversion_to_complex, simplification_complex},
  
  {"DCMPLX", (INT_MAX), overloaded_to_doublecomplex_type, 
   typing_function_conversion_to_dcomplex, simplification_dcomplex},
  
  /* (0.,1.) -> switched to a function call... */
  { IMPLIED_COMPLEX_NAME, 2, overloaded_to_complex_type, 
    typing_function_constant_complex, switch_specific_cmplx },
  { IMPLIED_DCOMPLEX_NAME, 2, overloaded_to_doublecomplex_type, 
    typing_function_constant_dcomplex, switch_specific_dcmplx },
  
  {"ICHAR", 1, default_intrinsic_type, typing_function_char_to_int, 0},
  {"CHAR", 1, default_intrinsic_type, typing_function_int_to_char, 0},

  {"AINT", 1, real_to_real_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_aint},
  {"DINT", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"ANINT", 1, real_to_real_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_anint},
  {"DNINT", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"NINT", 1, real_to_integer_type, 
   typing_function_RealDouble_to_Integer, switch_specific_nint},
  {"IDNINT", 1, double_to_integer_type, typing_function_double_to_int, 0},

  {"IABS", 1, integer_to_integer_type, typing_function_int_to_int, 0},
  {"ABS", 1, real_to_real_type, 
   typing_function_IntegerRealDoubleComplex_to_IntegerRealDoubleReal, 
   switch_specific_abs},
  {"DABS", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"CABS", 1, complex_to_real_type, typing_function_complex_to_real, 0},
  {"CDABS", 1, doublecomplex_to_double_type, 
   typing_function_dcomplex_to_double, 0},
  
  {"MOD", 2, default_intrinsic_type, 
   typing_function_IntegerRealDouble_to_IntegerRealDouble, 
   switch_specific_mod},
  {"AMOD", 2, real_to_real_type, typing_function_real_to_real, 0},
  {"DMOD", 2, double_to_double_type, typing_function_double_to_double, 0},

  {"ISIGN", 2, integer_to_integer_type, typing_function_int_to_int, 0},
  {"SIGN", 2, default_intrinsic_type, 
   typing_function_IntegerRealDouble_to_IntegerRealDouble, 
   switch_specific_sign},
  {"DSIGN", 2, double_to_double_type, typing_function_double_to_double, 0},

  {"IDIM", 2, integer_to_integer_type, typing_function_int_to_int, 0},
  {"DIM", 2, default_intrinsic_type, 
   typing_function_IntegerRealDouble_to_IntegerRealDouble, 
   switch_specific_dim},
  {"DDIM", 2, double_to_double_type, typing_function_double_to_double, 0},

  {"DPROD", 2, real_to_double_type, typing_function_real_to_double, 0},

  {"MAX", (INT_MAX), default_intrinsic_type, 
   typing_function_IntegerRealDouble_to_IntegerRealDouble, 
   switch_specific_max},
  {"MAX0", (INT_MAX), integer_to_integer_type, 
   typing_function_int_to_int, 0},
  {"AMAX1", (INT_MAX), real_to_real_type, typing_function_real_to_real, 0},
  {"DMAX1", (INT_MAX), double_to_double_type, 
   typing_function_double_to_double, 0},
  {"AMAX0", (INT_MAX), integer_to_real_type, 
   typing_function_int_to_real, 0},
  {"MAX1", (INT_MAX), real_to_integer_type, typing_function_real_to_int, 0},

  {"MIN", (INT_MAX), default_intrinsic_type, 
   typing_function_IntegerRealDouble_to_IntegerRealDouble, 
   switch_specific_min},
  {"MIN0", (INT_MAX), integer_to_integer_type, 
   typing_function_int_to_int, 0},
  {"AMIN1", (INT_MAX), real_to_real_type, typing_function_real_to_real, 0},
  {"DMIN1", (INT_MAX), double_to_double_type, 
   typing_function_double_to_double, 0},
  {"AMIN0", (INT_MAX), integer_to_real_type, 
   typing_function_int_to_real, 0},
  {"MIN1", (INT_MAX), real_to_integer_type, typing_function_real_to_int, 0},

  {"LEN", 1, character_to_integer_type, typing_function_char_to_int, 0},
  {"INDEX", 2, character_to_integer_type, typing_function_char_to_int, 0},

  {"AIMAG", 1, complex_to_real_type, typing_function_complex_to_real, 0},
  {"DIMAG", 1, doublecomplex_to_double_type, 
   typing_function_dcomplex_to_double, 0},

  {"CONJG", 1, complex_to_complex_type, 
   typing_function_complex_to_complex, 0},
  {"DCONJG", 1, doublecomplex_to_doublecomplex_type, 
   typing_function_dcomplex_to_dcomplex, 0},

  {"SQRT", 1, default_intrinsic_type, 
   typing_function_RealDoubleComplex_to_RealDoubleComplex, 
   switch_specific_sqrt},
  {"DSQRT", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"CSQRT", 1, complex_to_complex_type, 
   typing_function_complex_to_complex, 0},
  {"CDSQRT", 1, doublecomplex_to_doublecomplex_type, 
                typing_function_dcomplex_to_dcomplex, 0},
  
  {"EXP", 1, default_intrinsic_type, 
   typing_function_RealDoubleComplex_to_RealDoubleComplex, 
   switch_specific_exp},
  {"DEXP", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"CEXP", 1, complex_to_complex_type, 
   typing_function_complex_to_complex, 0},
  {"CDEXP", 1, doublecomplex_to_doublecomplex_type, 
               typing_function_dcomplex_to_dcomplex, 0},

  {"LOG", 1, default_intrinsic_type, 
   typing_function_RealDoubleComplex_to_RealDoubleComplex, 
   switch_specific_log},
  {"ALOG", 1, real_to_real_type, typing_function_real_to_real, 0},
  {"DLOG", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"CLOG", 1, complex_to_complex_type, 
   typing_function_complex_to_complex, 0},
  {"CDLOG", 1, doublecomplex_to_doublecomplex_type, 
               typing_function_dcomplex_to_dcomplex, 0},

  {"LOG10", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_log10},
  {"ALOG10", 1, real_to_real_type, typing_function_real_to_real, 0},
  {"DLOG10", 1, double_to_double_type, typing_function_double_to_double, 0},

  {"SIN", 1, default_intrinsic_type, 
   typing_function_RealDoubleComplex_to_RealDoubleComplex, 
   switch_specific_sin},
  {"DSIN", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"CSIN", 1, complex_to_complex_type, 
   typing_function_complex_to_complex, 0},
  {"CDSIN", 1, doublecomplex_to_doublecomplex_type, 
               typing_function_dcomplex_to_dcomplex, 0},

  {"COS", 1, default_intrinsic_type, 
   typing_function_RealDoubleComplex_to_RealDoubleComplex, 
   switch_specific_cos},
  {"DCOS", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"CCOS", 1, complex_to_complex_type, 
   typing_function_complex_to_complex, 0},
  {"CDCOS", 1, doublecomplex_to_doublecomplex_type, 
               typing_function_dcomplex_to_dcomplex, 0},

  {"TAN", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_tan},
  {"DTAN", 1, double_to_double_type, typing_function_double_to_double, 0},

  {"ASIN", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_asin},
  {"DASIN", 1, double_to_double_type, typing_function_double_to_double, 0},

  {"ACOS", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_acos},
  {"DACOS", 1, double_to_double_type, typing_function_double_to_double, 0},

  {"ATAN", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_atan},
  {"DATAN", 1, double_to_double_type, typing_function_double_to_double, 0},
  {"ATAN2", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_atan2},
  {"DATAN2", 1, double_to_double_type, typing_function_double_to_double, 0},

  {"SINH", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_sinh},
  {"DSINH", 1, double_to_double_type, typing_function_double_to_double, 0},

  {"COSH", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_cosh},
  {"DCOSH", 1, double_to_double_type, typing_function_double_to_double, 0},

  {"TANH", 1, default_intrinsic_type, 
   typing_function_RealDouble_to_RealDouble, switch_specific_tanh},
  {"DTANH", 1, double_to_double_type, typing_function_double_to_double, 0},
  
  {"LGE", 2, character_to_logical_type, typing_function_char_to_logical, 0},
  {"LGT", 2, character_to_logical_type, typing_function_char_to_logical, 0},
  {"LLE", 2, character_to_logical_type, typing_function_char_to_logical, 0},
  {"LLT", 2, character_to_logical_type, typing_function_char_to_logical, 0},
  
  {LIST_DIRECTED_FORMAT_NAME, 0, default_intrinsic_type, 
   typing_function_format_name, 0},
  {UNBOUNDED_DIMENSION_NAME, 0, default_intrinsic_type, 
   typing_function_overloaded, 0},
  
  /* These operators are used within the OPTIMIZE transformation in
     order to manipulate operators such as n-ary add and multiply or
     multiply-add operators ( JZ - sept 98) */
  {EOLE_SUM_OPERATOR_NAME, (INT_MAX), default_intrinsic_type , 
   typing_arithmetic_operator, 0},
  {EOLE_PROD_OPERATOR_NAME, (INT_MAX), default_intrinsic_type , 
   typing_arithmetic_operator, 0},
  {EOLE_FMA_OPERATOR_NAME, 3, default_intrinsic_type , 
   typing_arithmetic_operator, 0},
  {EOLE_FMS_OPERATOR_NAME, 3, default_intrinsic_type , 
   typing_arithmetic_operator, 0},
  
  {NULL, 0, 0, 0, 0}
};

/************************************************************************** 
 * Get the function for typing the specified intrinsic
 *
 */
typing_function_t get_typing_function_for_intrinsic(string name)
{
  static hash_table name_to_type_function = NULL;
  
  /* Initialize first time */
  if (!name_to_type_function) 
  {
    IntrinsicDescriptor * pdt = IntrinsicDescriptorTable;
    
    name_to_type_function = hash_table_make(hash_string, 0);
    
    for(; pdt->name; pdt++)
    {
      hash_put(name_to_type_function, 
	       (void*)pdt->name, (void*)pdt->type_function);
    }
  }
  
  pips_assert("typing function is defined", 
	      hash_defined_p(name_to_type_function, name));
  
  return (typing_function_t) hash_get(name_to_type_function, name);
}
/************************************************************************** 
 * Get the function for switching to specific name from generic name 
 *
 */
switch_name_function get_switch_name_function_for_intrinsic(string name)
{
  static hash_table name_to_switch_function = NULL;
  
  /* Initialize first time */
  if (!name_to_switch_function) 
  {
    IntrinsicDescriptor * pdt = IntrinsicDescriptorTable;
    
    name_to_switch_function = hash_table_make(hash_string, 0);
    
    for(; pdt->name; pdt++)
    {
      hash_put(name_to_switch_function, 
	       (void*)pdt->name, (void*)pdt->name_function);
    }
  }
  
  pips_assert("switch function is defined",
	      hash_defined_p(name_to_switch_function, name));

  return (switch_name_function) hash_get(name_to_switch_function, name);
}

/* This function creates an entity that represents an intrinsic
   function. Fortran operators and basic statements are intrinsic
   functions.
   
   An intrinsic function has a rom storage, an unknown initial value and a
   functional type whose result and arguments have an overloaded basic
   type. The number of arguments is given by the IntrinsicDescriptorTable
   data structure. */

void 
MakeIntrinsic(name, n, intrinsic_type)
     string name;
     int n;
     type (*intrinsic_type)(int);
{
  entity e;
  
  e = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME, name),
		  intrinsic_type(n),
		  make_storage(is_storage_rom, UU),
		  make_value(is_value_intrinsic, NIL));
  
}

/* This function is called one time (at the very beginning) to create
   all intrinsic functions. */

void 
CreateIntrinsics()
{
  IntrinsicDescriptor *pid;
  
  for (pid = IntrinsicDescriptorTable; pid->name != NULL; pid++) {
    MakeIntrinsic(pid->name, pid->nbargs, pid->intrinsic_type);
  }
}

bool 
bootstrap(string workspace)
{
  pips_debug(1, "bootstraping in workspace %s\n", workspace);

  if (db_resource_p(DBR_ENTITIES, "")) 
    pips_internal_error("entities already initialized");
  
  CreateIntrinsics();
  
  /* Creates the dynamic and static areas for the super global
   * arrays such as the logical unit array (see below).
   */
  CreateAreas();
  
  /* The current entity is unknown, but for a TOP-LEVEL:TOP-LEVEL
   * which is used to create the logical unit array for IO effects
   */
  CreateArrays();
  
  /* Create the empty label */
  (void) make_entity(strdup(concatenate(TOP_LEVEL_MODULE_NAME,
					MODULE_SEP_STRING, 
					LABEL_PREFIX,
					NULL)),
		     MakeTypeStatement(),
		     MakeStorageRom(),
		     make_value(is_value_constant,
				MakeConstantLitteral()));
  
  /* FI: I suppress the owner filed to make the database moveable */
  /* FC: the content must be consistent with pipsdbm/methods.h */
  DB_PUT_MEMORY_RESOURCE(DBR_ENTITIES, "", (char*) entity_domain);

  pips_debug(1, "bootstraping done\n");

  return TRUE;
}

value 
MakeValueLitteral()
{
  return(make_value(is_value_constant, 
		    make_constant(is_constant_litteral, UU)));
}

string 
MakeFileName(prefix, base, suffix)
     char *prefix, *base, *suffix;
{
  char *s;
  
  s = (char*) malloc(strlen(prefix)+strlen(base)+strlen(suffix)+1);
  
  strcpy(s, prefix);
  strcat(s, base);
  strcat(s, suffix);
  
  return(s);
}

/* This function creates a fortran operator parameter, i.e. a zero
   dimension variable with an overloaded basic type. */

char *
AddPackageToName(p, n)
     string p, n;
{
  string ps;
  int l;
  
  l = strlen(p);
  ps = strndup(l + 1 + strlen(n) +1, p);
  
  *(ps+l) = MODULE_SEP;
  *(ps+l+1) = '\0';
  strcat(ps, n);
  
  return(ps);
}
