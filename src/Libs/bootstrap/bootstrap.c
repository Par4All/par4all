/* Symbol table initialization with Fortran operators, commands and intrinsics
   
   More information is provided in effects/effects.c
   
   Remi Triolet
   
   Modifications:
   - add intrinsics according to Fortran standard Table 5, pp. 15.22-15-25,
   Francois Irigoin, 02/06/90
   - add .SEQ. to handle ranges outside of arrays [pj]
   
   Bugs:
   - intrinsics are not properly typed
   */

#include <stdio.h>
#include <string.h>
/* #include <values.h> */
#include <limits.h>
#include <stdlib.h>

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

void CreateAreas()
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

void CreateArrays()
{
    /* First a dummy function - close to C one "crt0()" - in order to
       - link the next entity to its ram
       - make an unbounded dimension for this entity
       */

    entity ent;

    ent = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,TOP_LEVEL_MODULE_NAME),
		      make_type(is_type_functional,
				make_functional(NIL,make_type(is_type_void,NIL))),
		      make_storage(is_storage_rom, UU),
		      make_value(is_value_code,make_code(NIL, "")));

    set_current_module_entity(ent);

    /* GO: entity for io effects : It is an array which*/
    make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,
				 IO_EFFECTS_ARRAY_NAME),
		MakeTypeArray(make_basic(is_basic_int,
					 IO_EFFECTS_UNIT_SPECIFIER_LENGTH),
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
				      global_name_to_entity(TOP_LEVEL_MODULE_NAME, 
							    STATIC_AREA_LOCAL_NAME),
				      0, NIL)),
		make_value(is_value_unknown, UU));

    reset_current_module_entity();
}

/* the following data structure describes an intrinsic function: its
name and its arity. */

typedef struct IntrinsicDescriptor {
    string name;
    int nbargs;
} IntrinsicDescriptor;



/* the table of intrinsic functions. this table is used at the begining
of linking to create Fortran operators, commands and intrinsic functions. 

Functions with a variable number of arguments are declared with INT_MAX
arguments.
*/

LOCAL IntrinsicDescriptor IntrinsicDescriptorTable[] = {
    {"+", 2},
    {"-", 2},
    {"/", 2},
    {"*", 2},
    {"--", 1},
    {"**", 2},

    {"=", 2},

    {".EQV.", 2},
    {".NEQV.", 2},

    {".OR.", 2},
    {".AND.", 2},

    {".LT.", 2},
    {".GT.", 2},
    {".LE.", 2},
    {".GE.", 2},
    {".EQ.", 2},
    {".NE.", 2},

    {"//", 2},

    {".NOT.", 1},

    {"WRITE", (INT_MAX)},
    {"REWIND", (INT_MAX)},
    {"BACKSPACE", (INT_MAX)},
    {"OPEN", (INT_MAX)},
    {"CLOSE", (INT_MAX)},
    {"READ", (INT_MAX)},
    {"BUFFERIN", (INT_MAX)},
    {"BUFFEROUT", (INT_MAX)},
    {"ENDFILE", (INT_MAX)},
    {"IMPLIED-DO", (INT_MAX)},
    {"FORMAT", 1},
    {"INQUIRE", (INT_MAX)},

    {"CONTINUE", 0},
    {"ENDDO", 0},
    {"PAUSE", 1},
    {"RETURN", 0},
    {"STOP", 0},
    {"END", 0},

    {"INT", 1},
    {"IFIX", 1},
    {"IDINT", 1},
    {"REAL", 1},
    {"FLOAT", 1},
    {"SNGL", 1},
    {"DBLE", 1},
    {"CMPLX", 1},
    {"ICHAR", 1},
    {"CHAR", 1},
    {"AINT", 1},
    {"DINT", 1},
    {"ANINT", 1},
    {"DNINT", 1},
    {"NINT", 1},
    {"IDNINT", 1},
    {"IABS", 1},
    {"ABS", 1},
    {"DABS", 1},
    {"CABS", 1},

    {"MOD", 2},
    {"AMOD", 2},
    {"DMOD", 2},
    {"ISIGN", 2},
    {"SIGN", 2},
    {"DSIGN", 2},
    {"IDIM", 2},
    {"DIM", 2},
    {"DDIM", 2},
    {"DPROD", 2},
    {"MAX", (INT_MAX)},
    {"MAX0", (INT_MAX)},
    {"AMAX1", (INT_MAX)},
    {"DMAX1", (INT_MAX)},
    {"AMAX0", (INT_MAX)},
    {"MAX1", (INT_MAX)},
    {"MIN", (INT_MAX)},
    {"MIN0", (INT_MAX)},
    {"AMIN1", (INT_MAX)},
    {"DMIN1", (INT_MAX)},
    {"AMIN0", (INT_MAX)},
    {"MIN1", (INT_MAX)},
    {"LEN", 1},
    {"INDEX", 2},
    {"AIMAG", 1},
    {"CONJG", 1},
    {"SQRT", 1},
    {"DSQRT", 1},
    {"CSQRT", 1},

    {"EXP", 1},
    {"DEXP", 1},
    {"CEXP", 1},
    {"LOG", 1},
    {"ALOG", 1},
    {"DLOG", 1},
    {"CLOG", 1},
    {"LOG10", 1},
    {"ALOG10", 1},
    {"DLOG10", 1},
    {"SIN", 1},
    {"DSIN", 1},
    {"CSIN", 1},
    {"COS", 1},
    {"DCOS", 1},
    {"CCOS", 1},
    {"TAN", 1},
    {"DTAN", 1},
    {"ASIN", 1},
    {"DASIN", 1},
    {"ACOS", 1},
    {"DACOS", 1},
    {"ATAN", 1},
    {"DATAN", 1},
    {"ATAN2", 1},
    {"DATAN2", 1},
    {"SINH", 1},
    {"DSINH", 1},
    {"COSH", 1},
    {"DCOSH", 1},
    {"TANH", 1},
    {"DTANH", 1},

    {"LGE", 2},
    {"LGT", 2},
    {"LLE", 2},
    {"LLT", 2},

    {LIST_DIRECTED_FORMAT_NAME, 0},
    {UNBOUNDED_DIMENSION_NAME, 0},

    {NULL, 0}
};


  
/* this function creates an entity that represents an intrinsic
function. Fortran operators and basic statements are intrinsic
functions.

An intrinsic function has a rom storage, an unknown initial value and a
functional type whose result and arguments have an overloaded basic
type. The number of arguments is given by the IntrinsicDescriptorTable
data structure. */

void MakeIntrinsic(name, n)
string name;
int n;
{
    entity e;
    functional ft;

    e = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME, name),
		    make_type(is_type_functional,  
			      (ft = make_functional(NIL, 
						    MakeOverloadedResult()))),
		    make_storage(is_storage_rom, UU),
		    make_value(is_value_intrinsic, NIL));
    
    if (n < (INT_MAX)) {
	while (n-- > 0) {
	    functional_parameters(ft) = 
		    CONS(PARAMETER, MakeOverloadedParameter(),
			 functional_parameters(ft));
	}
    }
}



/* this function is called one time (at the very beginning) to create
all intrinsic functions. */

void CreateIntrinsics()
{
    IntrinsicDescriptor *pid;

    for (pid = IntrinsicDescriptorTable; pid->name != NULL; pid++) {
	MakeIntrinsic(pid->name, pid->nbargs);
    }
}

bool bootstrap(string workspace)
{
    CreateIntrinsics();

    /* Creates the dynamic and static areas for the super global
     * arrays such as the logical unit array (see below).
     */
    CreateAreas();

    /* The current entity is unknown, but for a TOP-LEVEL:TOP-LEVEL
     * which is used to create the logical unit array for IO effects
     */
    CreateArrays();

    /* FI: I suppress the owner filed to make the database moveable */
    /* FI: I guess no pointers to the resource is passed because it
       is a tabulated NewGen field. */
    /* FC: switched to string_undefined to avoid free coredump */
    DB_PUT_MEMORY_RESOURCE(DBR_ENTITIES, "", string_undefined);

    return TRUE;
}

value MakeValueLitteral()
{
    return(make_value(is_value_constant, 
		      make_constant(is_constant_litteral, UU)));
}

string MakeFileName(prefix, base, suffix)
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

char *AddPackageToName(p, n)
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

