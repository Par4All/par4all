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

/* the following data structure describes an intrinsic function: its
name and its arity and its type. */

typedef struct IntrinsicDescriptor {
    string name;
    int nbargs;
  type (*intrinsic_type)(int);
} IntrinsicDescriptor;



/* the table of intrinsic functions. this table is used at the begining
of linking to create Fortran operators, commands and intrinsic functions. 

Functions with a variable number of arguments are declared with INT_MAX
arguments.
*/

static IntrinsicDescriptor IntrinsicDescriptorTable[] = {
    {"+", 2, default_intrinsic_type},
    {"-", 2, default_intrinsic_type},
    {"/", 2, default_intrinsic_type},
    {"INV", 1, real_to_real_type},
    {"*", 2, default_intrinsic_type},
    {"--", 1, default_intrinsic_type},
    {"**", 2, default_intrinsic_type},

    {"=", 2, default_intrinsic_type},

    {".EQV.", 2, overloaded_to_logical_type},
    {".NEQV.", 2, overloaded_to_logical_type},

    {".OR.", 2, logical_to_logical_type},
    {".AND.", 2, logical_to_logical_type},

    {".LT.", 2, overloaded_to_logical_type},
    {".GT.", 2, overloaded_to_logical_type},
    {".LE.", 2, overloaded_to_logical_type},
    {".GE.", 2, overloaded_to_logical_type},
    {".EQ.", 2, overloaded_to_logical_type},
    {".NE.", 2, overloaded_to_logical_type},

    {"//", 2, character_to_character_type},

    {".NOT.", 1, logical_to_logical_type},

    {"WRITE", (INT_MAX), default_intrinsic_type},
    {"REWIND", (INT_MAX), default_intrinsic_type},
    {"BACKSPACE", (INT_MAX), default_intrinsic_type},
    {"OPEN", (INT_MAX), default_intrinsic_type},
    {"CLOSE", (INT_MAX), default_intrinsic_type},
    {"READ", (INT_MAX), default_intrinsic_type},
    {"BUFFERIN", (INT_MAX), default_intrinsic_type},
    {"BUFFEROUT", (INT_MAX), default_intrinsic_type},
    {"ENDFILE", (INT_MAX), default_intrinsic_type},
    {"IMPLIED-DO", (INT_MAX), default_intrinsic_type},
    {FORMAT_FUNCTION_NAME, 1, default_intrinsic_type},
    {"INQUIRE", (INT_MAX), default_intrinsic_type},

    {SUBSTRING_FUNCTION_NAME, 3, substring_type},
    {ASSIGN_SUBSTRING_FUNCTION_NAME, 4, assign_substring_type},

    {"CONTINUE", 0, default_intrinsic_type},
    {"ENDDO", 0, default_intrinsic_type},
    {"PAUSE", 1, default_intrinsic_type},
    {"RETURN", 0, default_intrinsic_type},
    {"STOP", 0, default_intrinsic_type},
    {"END", 0, default_intrinsic_type},

    {"INT", 1, overloaded_to_integer_type},
    {"IFIX", 1, real_to_integer_type},
    {"IDINT", 1, double_to_integer_type},
    {"REAL", 1, overloaded_to_real_type},
    {"FLOAT", 1, overloaded_to_real_type},
    {"DFLOAT", 1, overloaded_to_double_type},
    {"SNGL", 1, overloaded_to_real_type},
    {"DBLE", 1, overloaded_to_double_type},
    {"DREAL", 1, overloaded_to_double_type}, /* Arnauld Leservot, code CEA */
    {"CMPLX", (INT_MAX), overloaded_to_complex_type},
    {"DCMPLX", (INT_MAX), overloaded_to_doublecomplex_type},

    /* (0.,1.) -> switched to a function call...
     */
    { IMPLIED_COMPLEX_NAME, 2, overloaded_to_complex_type},
    { IMPLIED_DCOMPLEX_NAME, 2, overloaded_to_doublecomplex_type},

    {"ICHAR", 1, default_intrinsic_type},
    {"CHAR", 1, default_intrinsic_type},
    {"AINT", 1, real_to_real_type},
    {"DINT", 1, double_to_double_type},
    {"ANINT", 1, real_to_real_type},
    {"DNINT", 1, double_to_double_type},
    {"NINT", 1, real_to_integer_type},
    {"IDNINT", 1, double_to_integer_type},
    {"IABS", 1, integer_to_integer_type},
    {"ABS", 1, real_to_real_type},
    {"DABS", 1, double_to_double_type},
    {"CABS", 1, complex_to_real_type},
    {"CDABS", 1, doublecomplex_to_double_type},

    {"MOD", 2, default_intrinsic_type},
    {"AMOD", 2, real_to_real_type},
    {"DMOD", 2, double_to_double_type},
    {"ISIGN", 2, integer_to_integer_type},
    {"SIGN", 2, default_intrinsic_type},
    {"DSIGN", 2, double_to_double_type},
    {"IDIM", 2, integer_to_integer_type},
    {"DIM", 2, default_intrinsic_type},
    {"DDIM", 2, double_to_double_type},
    {"DPROD", 2, real_to_double_type},
    {"MAX", (INT_MAX), default_intrinsic_type},
    {"MAX0", (INT_MAX), integer_to_integer_type},
    {"AMAX1", (INT_MAX), real_to_real_type},
    {"DMAX1", (INT_MAX), double_to_double_type},
    {"AMAX0", (INT_MAX), integer_to_real_type},
    {"MAX1", (INT_MAX), real_to_integer_type},
    {"MIN", (INT_MAX), default_intrinsic_type},
    {"MIN0", (INT_MAX), integer_to_integer_type},
    {"AMIN1", (INT_MAX), real_to_real_type},
    {"DMIN1", (INT_MAX), double_to_double_type},
    {"AMIN0", (INT_MAX), integer_to_real_type},
    {"MIN1", (INT_MAX), real_to_integer_type},
    {"LEN", 1, character_to_integer_type},
    {"INDEX", 2, character_to_integer_type},
    {"AIMAG", 1, complex_to_real_type},
    {"DIMAG", 1, doublecomplex_to_double_type},
    {"CONJG", 1, complex_to_complex_type},
    {"DCONJG", 1, doublecomplex_to_doublecomplex_type},
    {"SQRT", 1, default_intrinsic_type},
    {"DSQRT", 1, double_to_double_type},
    {"CSQRT", 1, complex_to_complex_type},

    {"EXP", 1, default_intrinsic_type},
    {"DEXP", 1, double_to_double_type},
    {"CEXP", 1, complex_to_complex_type},
    {"LOG", 1, default_intrinsic_type},
    {"ALOG", 1, real_to_real_type},
    {"DLOG", 1, double_to_double_type},
    {"CLOG", 1, complex_to_complex_type},
    {"LOG10", 1, default_intrinsic_type},
    {"ALOG10", 1, real_to_real_type},
    {"DLOG10", 1, double_to_double_type},
    {"SIN", 1, default_intrinsic_type},
    {"DSIN", 1, double_to_double_type},
    {"CSIN", 1, complex_to_complex_type},
    {"COS", 1, default_intrinsic_type},
    {"DCOS", 1, double_to_double_type},
    {"CCOS", 1, complex_to_complex_type},
    {"TAN", 1, default_intrinsic_type},
    {"DTAN", 1, double_to_double_type},
    {"ASIN", 1, default_intrinsic_type},
    {"DASIN", 1, double_to_double_type},
    {"ACOS", 1, default_intrinsic_type},
    {"DACOS", 1, double_to_double_type},
    {"ATAN", 1, default_intrinsic_type},
    {"DATAN", 1, double_to_double_type},
    {"ATAN2", 1, default_intrinsic_type},
    {"DATAN2", 1, double_to_double_type},
    {"SINH", 1, default_intrinsic_type},
    {"DSINH", 1, double_to_double_type},
    {"COSH", 1, default_intrinsic_type},
    {"DCOSH", 1, double_to_double_type},
    {"TANH", 1, default_intrinsic_type},
    {"DTANH", 1, double_to_double_type},

    {"LGE", 2, character_to_logical_type},
    {"LGT", 2, character_to_logical_type},
    {"LLE", 2, character_to_logical_type},
    {"LLT", 2, character_to_logical_type},

    {LIST_DIRECTED_FORMAT_NAME, 0, default_intrinsic_type},
    {UNBOUNDED_DIMENSION_NAME, 0, default_intrinsic_type},

    /* These operators are used within the OPTIMIZE transformation in
order to manipulate operators such as n-ary add and multiply or
multiply-add operators ( JZ - sept 98) */
    {EOLE_SUM_OPERATOR_NAME, (INT_MAX), default_intrinsic_type },
    {EOLE_PROD_OPERATOR_NAME, (INT_MAX), default_intrinsic_type },
    {EOLE_FMA_OPERATOR_NAME, 3, default_intrinsic_type },

    {NULL, 0, 0}
};


  
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

