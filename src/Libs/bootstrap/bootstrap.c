/* Symbol table initialization with Fortran operators, commands and intrinsics
   
   More information is provided in effects/effects.c
   
   Remi Triolet
   
   Modifications:
   - add intrinsics according to Fortran standard Table 5, pp. 15.22-15-25,
   Francois Irigoin, 02/06/90
   - add .SEQ. to handle ranges outside of arrays [pj]
   - add intrinsic DFLOAT. bc. 13/1/96.
   - add pseudo-intrinsics SUBSTR and ASSIGN_SUBSTR to handle strings,
   fi, 25/12/96
   Bugs:
   - intrinsics are not properly typed
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

#define LOCAL static

/* CLASSIFICATION OF BASIC */
#define basic_numeric_simple_p(b) (basic_int_p(b) || basic_float_p(b))
#define basic_numeric_p(b) (basic_numeric_simple_p(b) || basic_complex_p(b))
#define basic_compatible_simple_p(b1, b2) (\
                          (basic_numeric_simple_p(b1) && basic_numeric_simple_p(b2)) ||\
                          (basic_string_p(b1) && basic_string_p(b2)) ||\
                          (basic_logical_p(b1) && basic_logical_p(b2)) ||\
                          (basic_overloaded_p(b1) && basic_overloaded_p(b2)) ||\
                          (basic_undefined_p(b1) && basic_undefined_p(b2)))
#define basic_compatible_p(b1, b2) (\
                          (basic_numeric_p(b1) && basic_numeric_p(b2)) ||\
                          (basic_string_p(b1) && basic_string_p(b2)) ||\
                          (basic_logical_p(b1) && basic_logical_p(b2)) ||\
                          (basic_overloaded_p(b1) && basic_overloaded_p(b2)) ||\
                          (basic_undefined_p(b1) && basic_undefined_p(b2)))

/* Working with hash_table of basic
 */
#define GET_TYPE(h, e) ((basic)hash_get(h, (char*)(e)))
#define PUT_TYPE(h, e, b) hash_put(h, (char*)(e), (char*)(b))

/* This declaration is temporal. It will be moved to ri-util.h */
extern expression insert_cast(basic cast, basic from, expression exp);

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

    ent = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,IO_EFFECTS_PACKAGE_NAME),
		      make_type(is_type_functional,
				make_functional(NIL,make_type(is_type_void,NIL))),
		      make_storage(is_storage_rom, UU),
		      make_value(is_value_code,make_code(NIL, strdup(""))));

    set_current_module_entity(ent);

    make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME, 
				 STATIC_AREA_LOCAL_NAME),
		make_type(is_type_area, make_area(0, NIL)),
		make_storage(is_storage_rom, UU),
		make_value(is_value_unknown, UU));

    /* GO: entity for io logical units: It is an array which*/
    make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
				 IO_EFFECTS_ARRAY_NAME),
		MakeTypeArray(make_basic(is_basic_int,
				  UUINT(IO_EFFECTS_UNIT_SPECIFIER_LENGTH)),
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
		MakeTypeArray(make_basic(is_basic_logical,
				 UUINT(IO_EFFECTS_UNIT_SPECIFIER_LENGTH)),
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
		MakeTypeArray(make_basic(is_basic_logical,
				UUINT(IO_EFFECTS_UNIT_SPECIFIER_LENGTH)),
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

    if (n < (INT_MAX)) {
	int i = n;
	while (i-- > 0) {
	    l = CONS(PARAMETER, mkprm(), l);
	}
    }
    else {
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
    functional_parameters(ft) = make_parameter_list(n, MakeDoublecomplexParameter);
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
    functional_parameters(ft) = make_parameter_list(n, MakeDoublecomplexParameter);
    t = make_type(is_type_functional, ft);

    return t;
}

static type 
character_to_integer_type(int n)
{
    type t = type_undefined;
    functional ft = functional_undefined;

    ft = make_functional(NIL, MakeIntegerResult());
    functional_parameters(ft) = make_parameter_list(n, MakeCharacterParameter);
    t = make_type(is_type_functional, ft);

    return t;
}

static type 
character_to_logical_type(int n)
{
    type t = type_undefined;
    functional ft = functional_undefined;

    ft = make_functional(NIL, MakeLogicalResult());
    functional_parameters(ft) = make_parameter_list(n, MakeCharacterParameter);
    t = make_type(is_type_functional, ft);

    return t;
}

static type 
character_to_character_type(int n)
{
    type t = type_undefined;
    functional ft = functional_undefined;

    ft = make_functional(NIL, MakeCharacterResult());
    functional_parameters(ft) = make_parameter_list(n, MakeCharacterParameter);
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

/**************************************************************** TYPE A CALL FUNCTIONS */
/* I have to cast b to REAL or DOUBLE. Between REAL et DOUBLE, 
 * I have to choose the type that is the nearest to b!
 * e.g: type_nearest_RealDouble(INT(x)) --> REAL
 * e.g: type_nearest_RealDouble(CMPLX(x)) --> DOUBLE
 *
 * WARNING: b must be a type numeric, otherwise, I return basic_undefined
 */
basic type_nearest_RealDouble(basic b)
{
    if(basic_int_p(b) || (basic_float_p(b) && basic_float(b)==4))
    {
        return make_basic_float(4);
    }
    if(basic_complex_p(b) || (basic_float_p(b) && basic_float(b)==8))
    {
        return make_basic_float(8);
    }
    return basic_undefined;
}
basic type_nearest_IntegerRealDouble(basic b)
{
    if(basic_int_p(b))
    {
        return copy_basic(b);
    }
    return type_nearest_RealDouble(b);
}
basic type_nearest_RealDoubleComplex(basic b)
{
    if(basic_complex_p(b))
    {
        return copy_basic(b);
    }
    return type_nearest_RealDouble(b);
}
basic type_nearest_IntegerRealDoubleComplex(basic b)
{
    if(basic_complex_p(b))
    {
        return copy_basic(b);
    }
    return type_nearest_IntegerRealDouble(b);
}
/***************************************************************************************** 
 * Determine the longest basic among the arguments of c
 */
basic basic_union_arguments(call c, hash_table types)
{
    basic b1, b2;
    list args = call_arguments(c);

    // #arguments = 0
    if (args == NIL)
    {
        return basic_undefined;
    }

    // #arguments >= 1
    b1 = GET_TYPE(types, EXPRESSION(CAR(args)));
    args = CDR(args);
    while (args != NIL)
    {
	b2 = GET_TYPE(types, EXPRESSION(CAR(args)));
	if (is_inferior_basic(b1, b2))
	{
	    b1 = b2;
	}
	args = CDR(args);
    }
    return copy_basic(b1);
}
/********************** CHECK THE VALIDE OF ARGUMENTS BASIC OF FUNCTION ******************/
/* Verify if all the arguments basic of function C are INTEGER
 * If there is no argument, I return TRUE
 */
bool arguments_are_integer(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if (!basic_int_p(b))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}
bool arguments_are_real(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if ( !(basic_float_p(b) && (basic_float(b) == 4)))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}
bool arguments_are_double(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if ( !(basic_float_p(b) && (basic_float(b) == 8)))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}
bool arguments_are_complex(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if ( !(basic_complex_p(b) && basic_complex(b) == 8))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}
bool arguments_are_dcomplex(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if ( !(basic_complex_p(b) && basic_complex(b) == 16))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}

/***************************************************************************************** 
 * Verify if all the arguments basic of function C are REAL and DOUBLE
 * If there is no argument, I return TRUE
 *
 * Note: I - Integer; R - Real; D - Double; C - Complex
 */
bool arguments_are_RD(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if ( !basic_float_p(b) )
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}

bool arguments_are_IRD(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if ( !basic_int_p(b) && !basic_float_p(b))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}
bool arguments_are_RDC(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if ( !basic_float_p(b) &&
	     !(basic_complex_p(b) && basic_complex(b) == 8))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}
bool arguments_are_IRDC(call c, hash_table types)
{
    basic b;
    expression e;
    list args = call_arguments(c);

    while (args != NIL)
    {      
	b = GET_TYPE(types, EXPRESSION(CAR(args))); 
	e = EXPRESSION(CAR(args));
	if(expression_call_p(e))
	  fprintf(stdout, "e = %s\n", 
		  entity_local_name(call_function(syntax_call(expression_syntax((e))))));
	if(expression_reference_p(e))
	  fprintf(stdout, "e = %s\n", 
		  entity_local_name(reference_variable(syntax_reference(expression_syntax((e))))));

	if ( !basic_int_p(b) && 
	     !basic_float_p(b) &&
	     !(basic_complex_p(b) && basic_complex(b) == 8))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}

bool arguments_are_character(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if (!basic_string_p(b))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}
bool arguments_are_logical(call c, hash_table types)
{
    basic b;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b = GET_TYPE(types, EXPRESSION(CAR(args)));
	if (!basic_logical_p(b))
	{
	  return FALSE;
	}
	args = CDR(args);
    }
    return TRUE;
}

/***************************************************************************************** 
 * Typing all the arguments of c to basic b if their basic <> b
 */
void typing_arguments(call c, hash_table types, basic b)
{
    basic b1;
    list args = call_arguments(c);

    while (args != NIL)
    {
	b1 = GET_TYPE(types, EXPRESSION(CAR(args)));
	if (!basic_equal_p(b, b1))
	{
	    EXPRESSION(CAR(args)) = insert_cast(b, b1, EXPRESSION(CAR(args)));
	    // Update hash table
	    PUT_TYPE(types, EXPRESSION(CAR(args)), b);
	}
	args = CDR(args);
    }
}

/***************************************************************************************** 
 * Typing arithmetic operator (+, -, *, /), except **
 */
static basic
typing_arithmetic_operator(call c, hash_table types)
{
    basic b;

    entity function_called = call_function(c);

    if(!arguments_are_IRDC(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not INT, REAL, DOUBLE or COMPLEX\n", 
		entity_name(function_called));
	// Just for return a result
	return make_basic_float(4); 
    }
    // Find the longest type amongs all arguments
    b = basic_union_arguments(c, types);

    // Typing all arguments to b if necessary
    typing_arguments(c, types, b);

    return copy_basic(b);    
}
/***************************************************************************************** 
 * Typing power operator (**)
 */
static basic
typing_power_operator(call c, hash_table types)
{
    basic b, b1, b2;
    list /* of expression */ args = call_arguments(c);
    entity function_called = call_function(c);
    b = basic_undefined;

    if(!arguments_are_IRDC(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not INT, REAL, DOUBLE or COMPLEX\n", 
		entity_name(function_called));
	// Just for return a result
	return make_basic_float(4); 
    }

    b1 = GET_TYPE(types, EXPRESSION(CAR(args)));
    b2 = GET_TYPE(types, EXPRESSION(CAR(CDR(args))));
    if ((basic_float_p(b1) && (basic_float(b1)==8) && basic_complex_p(b2)) ||
	(basic_float_p(b2) && (basic_float(b2)==8) && basic_complex_p(b1)))
    {
        // ERROR: C**D, D**C: prohibited!
        fprintf(stderr,"Intrinsic [%s]: <COMPLEX>**<DOUBLE> or <DOUBLE>**<COMPLEX> are prohibited\n", 
		entity_name(function_called));
	// Just for return a result
	return make_basic_float(4); 
    }

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
	  insert_cast(b, b1, EXPRESSION(CAR(args)));
    }
    // Fortran prefers: 
    // "var_double = var_double ** var_int" instead of
    // "var_double = var_double ** DBLE(var_int)"
    if (!basic_equal_p(b, b2) && !basic_int_p(b2))
    {
	EXPRESSION(CAR(CDR(args))) = 
	  insert_cast(b, b2, EXPRESSION(CAR(CDR(args))));
    }
    return copy_basic(b);
}
/***************************************************************************************** 
 * Typing relational operator (LT, LE, EQ, GT, GE) 
 */
static basic
typing_relational_operator(call c, hash_table types)
{
    basic b;

    entity function_called = call_function(c);

    if(!arguments_are_IRDC(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not INT, REAL, DOUBLE or COMPLEX\n", 
		entity_name(function_called));
	// Just for return a result
	return make_basic(is_basic_logical, UUINT(4));
    }
    // Find the longest type amongs all arguments
    b = basic_union_arguments(c, types);

    // Typing all arguments to b if necessary
    typing_arguments(c, types, b);

    return make_basic(is_basic_logical, UUINT(4));
}
/***************************************************************************************** 
 * Typing logical operator (NOT, AND, OR, EQV, NEQV)
 */
static basic
typing_logical_operator(call c, hash_table types)
{
    entity function_called = call_function(c);

    if(!arguments_are_logical(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not LOGICAL \n",
		entity_name(function_called));
	// Just for return a result
	return make_basic(is_basic_logical, UUINT(4));
    }
    return make_basic(is_basic_logical, UUINT(4));
}
/***************************************************************************************** 
 * Typing concatenate operator (//)
 */
static basic
typing_concat_operator(call c, hash_table types)
{
    entity function_called = call_function(c);

    if(!arguments_are_character(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not CHARACTER\n", 
		entity_name(function_called));
	// Just for return a result
	return make_basic(is_basic_string, value_undefined);
    }
    return make_basic(is_basic_string, value_undefined);
}

/***************************************************************************************** 
 * Typing function C whose argument type is from_type and whose return type is to_type
 */
static basic 
typing_function_argument_type_to_return_type(call c, hash_table types, 
					     basic from_type, basic to_type)
{
    bool check_arg = FALSE;
    entity function_called = call_function(c);

    // INT
    if(basic_int_p(from_type))
    {
        check_arg = arguments_are_integer(c, types);
    }
    // REAL
    else if(basic_float_p(from_type) && basic_float(from_type) == 4)
    {
        check_arg = arguments_are_real(c, types);
    }
    // DOUBLE
    else if(basic_float_p(from_type) && basic_float(from_type) == 8)
    {
        check_arg = arguments_are_double(c, types);
    }
    // COMPLEX
    else if(basic_complex_p(from_type) && basic_complex(from_type) == 8)
    {
        check_arg = arguments_are_complex(c, types);
    }
    // DOUBLE COMPLEX
    else if(basic_complex_p(from_type) && basic_complex(from_type) == 16)
    {
        check_arg = arguments_are_dcomplex(c, types);
    }
    // CHAR
    else if(basic_string_p(from_type))
    {
        check_arg = arguments_are_character(c, types);
    }
    // LOGICAL
    else if(basic_logical_p(from_type))
    {
        check_arg = arguments_are_logical(c, types);
    }
    // UNEXPECTED
    else
    {
        pips_internal_error("Unexpected basic: %s \n", basic_to_string(from_type));
    }

    // ERROR: Invalide of argument type
    if(check_arg == FALSE)
    {
      //fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not [%s] !\n", 
      //	entity_name(function_called), basic_to_string(from_type));
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not valid !\n", 
		entity_name(function_called));
    }

    return copy_basic(to_type);
}

static basic
typing_function_int_to_int(call c, hash_table types)
{
    basic type_INT = make_basic_int(4);
    return typing_function_argument_type_to_return_type(c, types, type_INT, type_INT);
}
static basic
typing_function_real_to_real(call c, hash_table types)
{
    basic type_REAL = make_basic_float(4);
    return typing_function_argument_type_to_return_type(c, types, type_REAL, type_REAL);
}
static basic
typing_function_double_to_double(call c, hash_table types)
{
    basic type_DBLE = make_basic_float(8);
    return typing_function_argument_type_to_return_type(c, types, type_DBLE, type_DBLE);
}
static basic
typing_function_complex_to_complex(call c, hash_table types)
{
    basic type_CMPLX = make_basic_complex(8);
    return typing_function_argument_type_to_return_type(c, types, type_CMPLX, type_CMPLX);
}
static basic
typing_function_dcomplex_to_dcomplex(call c, hash_table types)
{
    basic type_DCMPLX = make_basic_complex(16);
    return typing_function_argument_type_to_return_type(c, types, type_DCMPLX, type_DCMPLX);
}
static basic
typing_function_char_to_int(call c, hash_table types)
{
    basic type_INT = make_basic_int(4);
    basic type_CHAR = make_basic(is_basic_string, value_undefined);
    return typing_function_argument_type_to_return_type(c, types, type_CHAR, type_INT);
}
static basic
typing_function_int_to_char(call c, hash_table types)
{
    basic type_INT = make_basic_int(4);
    basic type_CHAR = make_basic(is_basic_string, value_undefined);
    return typing_function_argument_type_to_return_type(c, types, type_INT, type_CHAR);
}
static basic
typing_function_real_to_int(call c, hash_table types)
{
    basic type_INT = make_basic_int(4);
    basic type_REAL = make_basic_float(4);
    return typing_function_argument_type_to_return_type(c, types, type_REAL, type_INT);
}
static basic
typing_function_int_to_real(call c, hash_table types)
{
    basic type_INT = make_basic_int(4);
    basic type_REAL = make_basic_float(4);
    return typing_function_argument_type_to_return_type(c, types, type_INT, type_REAL);
}
static basic
typing_function_double_to_int(call c, hash_table types)
{
    basic type_INT = make_basic_int(4);
    basic type_DBLE = make_basic_float(8);
    return typing_function_argument_type_to_return_type(c, types, type_DBLE, type_INT);
}
static basic
typing_function_real_to_double(call c, hash_table types)
{
    basic type_REAL = make_basic_float(4);
    basic type_DBLE = make_basic_float(8);
    return typing_function_argument_type_to_return_type(c, types, type_REAL, type_DBLE);
}
/*
static basic
typing_function_double_to_real(call c, hash_table types)
{
    basic type_REAL = make_basic_float(4);
    basic type_DBLE = make_basic_float(8);
    return typing_function_argument_type_to_return_type(c, types, type_DBLE, type_REAL);
}
static basic
typing_function_complex_to_int(call c, hash_table types)
{
    basic type_INT = make_basic_int(4);
    basic type_CMPLX = make_basic_complex(8);
    return typing_function_argument_type_to_return_type(c, types, type_CMPLX, type_INT);
}*/
static basic
typing_function_complex_to_real(call c, hash_table types)
{
    basic type_REAL = make_basic_float(4);
    basic type_CMPLX = make_basic_complex(8);
    return typing_function_argument_type_to_return_type(c, types, type_CMPLX, type_REAL);
}
static basic
typing_function_dcomplex_to_double(call c, hash_table types)
{
    basic type_DBLE = make_basic_float(8);
    basic type_DCMPLX = make_basic_complex(16);
    return typing_function_argument_type_to_return_type(c, types, type_DCMPLX, type_DBLE);
}
static basic
typing_function_char_to_logical(call c, hash_table types)
{
    basic type_LOGICAL = make_basic_logical(4);
    basic type_CHAR = make_basic(is_basic_string, value_undefined);
    return typing_function_argument_type_to_return_type(c, types, type_CHAR, type_LOGICAL);
}

/***************************************************************************************** 
 * Arguments are REAL (or DOUBLE); and the return is the same with argument
 */
static basic
typing_function_RealDouble_to_RealDouble(call c, hash_table types)
{
    basic b;

    entity function_called = call_function(c);

    if(!arguments_are_RD(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not REAL ou DOUBLE\n", 
		entity_name(function_called));
	return make_basic_float(4); // Just for return a result
    }
    // Find the longest type amongs all arguments
    b = basic_union_arguments(c, types);

    // Find the nearest type between REAL and DOUBLE
    //b = type_nearest_RealDouble(b); 

    // Typing all arguments to b if necessary
    typing_arguments(c, types, b);

    return copy_basic(b);    
}
static basic
typing_function_RealDoubleComplex_to_RealDoubleComplex(call c, hash_table types)
{
    basic b;

    entity function_called = call_function(c);

    if(!arguments_are_RDC(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not REAL, DOUBLE or COMPLEX\n", 
		entity_name(function_called));
	return make_basic_float(4); // Just for return a result
    }
    // Find the longest type amongs all arguments
    b = basic_union_arguments(c, types);

    // Find the nearest type between REAL, DOUBLE and COMPLEX
    //b = type_nearest_RealDoubleComplex(b); 

    // Typing all arguments to b if necessary
    typing_arguments(c, types, b);

    return copy_basic(b);
}
static basic
typing_function_IntegerRealDouble_to_IntegerRealDouble(call c, hash_table types)
{
    basic b;

    entity function_called = call_function(c);

    if(!arguments_are_IRD(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not INT, REAL or DOUBLE\n", 
		entity_name(function_called));
	return make_basic_float(4); // Just for return a result
    }
    // Find the longest type amongs all arguments
    b = basic_union_arguments(c, types);
    // Find the nearest type between REAL and DOUBLE
    b = type_nearest_IntegerRealDouble(b); 
    // Typing all arguments to b if necessary
    typing_arguments(c, types, b);

    return copy_basic(b);    
}
/***************************************************************************************** 
 * The arguments are INT, REAL, DOUBLE or COMPLEX. The return is the same with the argument
 * except case argument are COMPLEX, return is REAL
 *
 * Note: Only for Intrinsic ABS(): ABS(CMPLX(x)) --> REAL
 */
static basic
typing_function_IntegerRealDoubleComplex_to_IntegerRealDoubleReal(call c, hash_table types)
{
    basic b;

    entity function_called = call_function(c);

    if(!arguments_are_IRDC(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not INT, REAL, DOUBLE or COMPLEX\n", 
		entity_name(function_called));
	return make_basic_float(4); // Just for return result
    }
    // Find the longest type amongs all arguments
    b = basic_union_arguments(c, types);

    // Find the nearest type between REAL and DOUBLE
    //b = type_nearest_IntegerRealDoubleComplex(b); 

    // Typing all arguments to b if necessary
    typing_arguments(c, types, b);

    if (basic_complex_p(b))
    {
        b = make_basic_float(4); // CMPLX --> REAL
    }
    return copy_basic(b);
}

/***************************************************************************************** 
 * Intrinsic conversion to a numeric
 *
 * Note: argument must be numeric
 */
static basic
typing_function_conversion_to_numeric(call c, hash_table types, basic to_type)
{
    if(!arguments_are_IRDC(c, types))
    {
        // ERROR: Invalide of type
        fprintf(stderr,"Intrinsic [%s]: Argument(s) is (are) not INT, REAL, DOUBLE or COMPLEX\n", 
		entity_name(call_function(c)));
    }
    return copy_basic(to_type);
}
static basic
typing_function_conversion_to_integer(call c, hash_table types)
{
    return typing_function_conversion_to_numeric(c, types, make_basic_int(4));
}
static basic
typing_function_conversion_to_real(call c, hash_table types)
{
    return typing_function_conversion_to_numeric(c, types, make_basic_float(4));
}
static basic
typing_function_conversion_to_double(call c, hash_table types)
{
    return typing_function_conversion_to_numeric(c, types, make_basic_float(8));
}
static basic
typing_function_conversion_to_complex(call c, hash_table types)
{
    return typing_function_conversion_to_numeric(c, types, make_basic_complex(8));
}
static basic
typing_function_conversion_to_dcomplex(call c, hash_table types)
{
    return typing_function_conversion_to_numeric(c, types, make_basic_complex(16));
}

/*********************************************************************** INTRINSICS LIST */

/* the following data structure describes an intrinsic function: its
name and its arity and its type. */

typedef basic (*typing_function_t)(call, hash_table);

typedef struct IntrinsicDescriptor {
  string name;
  int nbargs;
  type (*intrinsic_type)(int);
  typing_function_t type_function;
} IntrinsicDescriptor;

/* the table of intrinsic functions. this table is used at the begining
of linking to create Fortran operators, commands and intrinsic functions. 

Functions with a variable number of arguments are declared with INT_MAX
arguments.
*/

static IntrinsicDescriptor IntrinsicDescriptorTable[] = {
    {"+", 2, default_intrinsic_type, typing_arithmetic_operator},
    {"-", 2, default_intrinsic_type, typing_arithmetic_operator},
    {"/", 2, default_intrinsic_type, typing_arithmetic_operator},
    {"INV", 1, real_to_real_type, 0},
    {"*", 2, default_intrinsic_type, typing_arithmetic_operator},
    {"--", 1, default_intrinsic_type, 0},
    {"**", 2, default_intrinsic_type, typing_power_operator},

    {"=", 2, default_intrinsic_type, 0},

    {".EQV.", 2, overloaded_to_logical_type, typing_logical_operator},
    {".NEQV.", 2, overloaded_to_logical_type, typing_logical_operator},

    {".OR.", 2, logical_to_logical_type, typing_logical_operator},
    {".AND.", 2, logical_to_logical_type, typing_logical_operator},
    {".NOT.", 1, logical_to_logical_type, typing_logical_operator},

    {".LT.", 2, overloaded_to_logical_type, typing_relational_operator},
    {".GT.", 2, overloaded_to_logical_type, typing_relational_operator},
    {".LE.", 2, overloaded_to_logical_type, typing_relational_operator},
    {".GE.", 2, overloaded_to_logical_type, typing_relational_operator},
    {".EQ.", 2, overloaded_to_logical_type, typing_relational_operator},
    {".NE.", 2, overloaded_to_logical_type, typing_relational_operator},

    {"//", 2, character_to_character_type, typing_concat_operator},

    {"WRITE", (INT_MAX), default_intrinsic_type, 0},
    {"REWIND", (INT_MAX), default_intrinsic_type, 0},
    {"BACKSPACE", (INT_MAX), default_intrinsic_type, 0},
    {"OPEN", (INT_MAX), default_intrinsic_type, 0},
    {"CLOSE", (INT_MAX), default_intrinsic_type, 0},
    {"READ", (INT_MAX), default_intrinsic_type, 0},
    {"BUFFERIN", (INT_MAX), default_intrinsic_type, 0},
    {"BUFFEROUT", (INT_MAX), default_intrinsic_type, 0},
    {"ENDFILE", (INT_MAX), default_intrinsic_type, 0},
    {"IMPLIED-DO", (INT_MAX), default_intrinsic_type, 0},
    {FORMAT_FUNCTION_NAME, 1, default_intrinsic_type, 0},
    {"INQUIRE", (INT_MAX), default_intrinsic_type, 0},

    {SUBSTRING_FUNCTION_NAME, 3, substring_type, 0},
    {ASSIGN_SUBSTRING_FUNCTION_NAME, 4, assign_substring_type, 0},

    {"CONTINUE", 0, default_intrinsic_type, 0},
    {"ENDDO", 0, default_intrinsic_type, 0},
    {"PAUSE", 1, default_intrinsic_type, 0},
    {"RETURN", 0, default_intrinsic_type, 0},
    {"STOP", 0, default_intrinsic_type, 0},
    {"END", 0, default_intrinsic_type, 0},

    {"INT", 1, overloaded_to_integer_type, typing_function_conversion_to_integer},
    {"IFIX", 1, real_to_integer_type, typing_function_real_to_int},
    {"IDINT", 1, double_to_integer_type, typing_function_double_to_int},
    {"REAL", 1, overloaded_to_real_type, typing_function_conversion_to_real},
    {"FLOAT", 1, overloaded_to_real_type, typing_function_conversion_to_real},
    {"DFLOAT", 1, overloaded_to_double_type, typing_function_conversion_to_real},
    {"SNGL", 1, overloaded_to_real_type, typing_function_conversion_to_real},
    {"DBLE", 1, overloaded_to_double_type, typing_function_conversion_to_double},
    {"DREAL", 1, overloaded_to_double_type, 
     typing_function_conversion_to_double}, /* Arnauld Leservot, code CEA */
    {"CMPLX", (INT_MAX), overloaded_to_complex_type, 
     typing_function_conversion_to_complex},
    {"DCMPLX", (INT_MAX), overloaded_to_doublecomplex_type, 
     typing_function_conversion_to_dcomplex},

    /* (0.,1.) -> switched to a function call...
     */
    { IMPLIED_COMPLEX_NAME, 2, overloaded_to_complex_type, 0},
    { IMPLIED_DCOMPLEX_NAME, 2, overloaded_to_doublecomplex_type, 0},

    {"ICHAR", 1, default_intrinsic_type, typing_function_char_to_int},
    {"CHAR", 1, default_intrinsic_type, typing_function_int_to_char},
    {"AINT", 1, real_to_real_type, typing_function_real_to_real},
    {"DINT", 1, double_to_double_type, typing_function_double_to_double},
    {"ANINT", 1, real_to_real_type, typing_function_real_to_real},
    {"DNINT", 1, double_to_double_type, typing_function_double_to_double},
    {"NINT", 1, real_to_integer_type, typing_function_real_to_int},
    {"IDNINT", 1, double_to_integer_type, typing_function_double_to_int},
    {"IABS", 1, integer_to_integer_type, typing_function_int_to_int},
    {"ABS", 1, real_to_real_type, 
     typing_function_IntegerRealDoubleComplex_to_IntegerRealDoubleReal},
    {"DABS", 1, double_to_double_type, typing_function_double_to_double},
    {"CABS", 1, complex_to_real_type, typing_function_complex_to_real},
    {"CDABS", 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double},

    {"MOD", 2, default_intrinsic_type, 
     typing_function_IntegerRealDouble_to_IntegerRealDouble},
    {"AMOD", 2, real_to_real_type, typing_function_real_to_real},
    {"DMOD", 2, double_to_double_type, typing_function_double_to_double},
    {"ISIGN", 2, integer_to_integer_type, typing_function_int_to_int},
    {"SIGN", 2, default_intrinsic_type, 
     typing_function_IntegerRealDouble_to_IntegerRealDouble},
    {"DSIGN", 2, double_to_double_type, typing_function_double_to_double},
    {"IDIM", 2, integer_to_integer_type, typing_function_int_to_int},
    {"DIM", 2, default_intrinsic_type, 
     typing_function_IntegerRealDouble_to_IntegerRealDouble},
    {"DDIM", 2, double_to_double_type, typing_function_double_to_double},
    {"DPROD", 2, real_to_double_type, typing_function_real_to_double},
    {"MAX", (INT_MAX), default_intrinsic_type, 
     typing_function_IntegerRealDouble_to_IntegerRealDouble},
    {"MAX0", (INT_MAX), integer_to_integer_type, typing_function_int_to_int},
    {"AMAX1", (INT_MAX), real_to_real_type, typing_function_real_to_real},
    {"DMAX1", (INT_MAX), double_to_double_type, typing_function_double_to_double},
    {"AMAX0", (INT_MAX), integer_to_real_type, typing_function_int_to_real},
    {"MAX1", (INT_MAX), real_to_integer_type, typing_function_real_to_int},
    {"MIN", (INT_MAX), default_intrinsic_type, 
     typing_function_IntegerRealDouble_to_IntegerRealDouble},
    {"MIN0", (INT_MAX), integer_to_integer_type, typing_function_int_to_int},
    {"AMIN1", (INT_MAX), real_to_real_type, typing_function_real_to_real},
    {"DMIN1", (INT_MAX), double_to_double_type, typing_function_double_to_double},
    {"AMIN0", (INT_MAX), integer_to_real_type, typing_function_int_to_real},
    {"MIN1", (INT_MAX), real_to_integer_type, typing_function_real_to_int},
    {"LEN", 1, character_to_integer_type, typing_function_char_to_int},
    {"INDEX", 2, character_to_integer_type, typing_function_char_to_int},
    {"AIMAG", 1, complex_to_real_type, typing_function_complex_to_real},
    {"DIMAG", 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double},
    {"CONJG", 1, complex_to_complex_type, typing_function_complex_to_complex},
    {"DCONJG", 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex},
    {"SQRT", 1, default_intrinsic_type, 
     typing_function_RealDoubleComplex_to_RealDoubleComplex},
    {"DSQRT", 1, double_to_double_type, typing_function_double_to_double},
    {"CSQRT", 1, complex_to_complex_type, typing_function_complex_to_complex},

    {"EXP", 1, default_intrinsic_type, 
     typing_function_RealDoubleComplex_to_RealDoubleComplex},
    {"DEXP", 1, double_to_double_type, typing_function_double_to_double},
    {"CEXP", 1, complex_to_complex_type, typing_function_complex_to_complex},
    {"LOG", 1, default_intrinsic_type, 
     typing_function_RealDoubleComplex_to_RealDoubleComplex},
    {"ALOG", 1, real_to_real_type, typing_function_real_to_real},
    {"DLOG", 1, double_to_double_type, typing_function_double_to_double},
    {"CLOG", 1, complex_to_complex_type, typing_function_complex_to_complex},
    {"LOG10", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"ALOG10", 1, real_to_real_type, typing_function_real_to_real},
    {"DLOG10", 1, double_to_double_type, typing_function_double_to_double},
    {"SIN", 1, default_intrinsic_type, 
     typing_function_RealDoubleComplex_to_RealDoubleComplex},
    {"DSIN", 1, double_to_double_type, typing_function_double_to_double},
    {"CSIN", 1, complex_to_complex_type, typing_function_complex_to_complex},
    {"COS", 1, default_intrinsic_type, 
     typing_function_RealDoubleComplex_to_RealDoubleComplex},
    {"DCOS", 1, double_to_double_type, typing_function_double_to_double},
    {"CCOS", 1, complex_to_complex_type, typing_function_complex_to_complex},
    {"TAN", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DTAN", 1, double_to_double_type, typing_function_double_to_double},
    {"ASIN", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DASIN", 1, double_to_double_type, typing_function_double_to_double},
    {"ACOS", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DACOS", 1, double_to_double_type, typing_function_double_to_double},
    {"ATAN", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DATAN", 1, double_to_double_type, typing_function_double_to_double},
    {"ATAN2", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DATAN2", 1, double_to_double_type, typing_function_double_to_double},
    {"SINH", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DSINH", 1, double_to_double_type, typing_function_double_to_double},
    {"COSH", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DCOSH", 1, double_to_double_type, typing_function_double_to_double},
    {"TANH", 1, default_intrinsic_type, typing_function_RealDouble_to_RealDouble},
    {"DTANH", 1, double_to_double_type, typing_function_double_to_double},

    {"LGE", 2, character_to_logical_type, typing_function_char_to_logical},
    {"LGT", 2, character_to_logical_type, typing_function_char_to_logical},
    {"LLE", 2, character_to_logical_type, typing_function_char_to_logical},
    {"LLT", 2, character_to_logical_type, typing_function_char_to_logical},

    {LIST_DIRECTED_FORMAT_NAME, 0, default_intrinsic_type, 0},
    {UNBOUNDED_DIMENSION_NAME, 0, default_intrinsic_type, 0},

    /* These operators are used within the OPTIMIZE transformation in
order to manipulate operators such as n-ary add and multiply or
multiply-add operators ( JZ - sept 98) */
    {EOLE_SUM_OPERATOR_NAME, (INT_MAX), default_intrinsic_type , 0},
    {EOLE_PROD_OPERATOR_NAME, (INT_MAX), default_intrinsic_type , 0},
    {EOLE_FMA_OPERATOR_NAME, 3, default_intrinsic_type , 0},

    {NULL, 0, 0, 0}
};

typing_function_t get_typing_function_for_intrinsic(string name)
{
  static hash_table name_to_type_function = NULL;

  /* initialize first time */
  if (!name_to_type_function) 
  {
    IntrinsicDescriptor * pdt = IntrinsicDescriptorTable;

    name_to_type_function = hash_table_make(hash_string, 0);
    
    for(; pdt->name; pdt++)
    {
        hash_put(name_to_type_function, (char*)pdt->name, (char*)pdt->type_function);
    }
  }

  if (!hash_defined_p(name_to_type_function, name))
    {
      pips_internal_error("No type function for intrinsics %s\n", name);
    }

  return (typing_function_t) hash_get(name_to_type_function, name);
}

/* this function creates an entity that represents an intrinsic
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

/* this function is called one time (at the very beginning) to create
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

/* this function creates a fortran operator parameter, i.e. a zero
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

