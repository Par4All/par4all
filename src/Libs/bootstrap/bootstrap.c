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
  Symbol table initialization with Fortran operators, commands and
  intrinsics

  More information is provided in effects/effects.c

  Remi Triolet

  Modifications :
  ---------------
  - add intrinsics according to Fortran standard Table 5, pp. 15.22-15-25,
  Francois Irigoin, 02/06/90
  - add .SEQ. to handle ranges outside of arrays [pj]
  - add intrinsic DFLOAT. bc. 13/1/96.
  - add pseudo-intrinsics SUBSTR and ASSIGN_SUBSTR to handle strings,
    FI, 25/12/96
  - Fortran specification conformant typing of expressions...

  Molka Becher (MB), June 2010
  - Check of C intrinsics already added
  - Add of missing C intrinsics according to ISO/IEC 9899:TC2
  - Add of functions handling long double type and long double complex type

  Molka Becher (MB), June 2010
  - Check of C intrinsics already added
  - Add of missing C intrinsics according to ISO/IEC 9899:TC2
  - Add of functions handling long double type and long double complex type


  Bugs:
  -----
  - intrinsics are not properly typed

*/

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"

#include "bootstrap.h"

#include "misc.h"
#include "pipsdbm.h"
#include "parser_private.h"
#include "constants.h"
#include "resources.h"

#include "properties.h"

#define LOCAL static
#undef make_entity

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


void CreateAreas()
{
  make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,
			       DYNAMIC_AREA_LOCAL_NAME),
	      make_type(is_type_area, make_area(0, NIL)),
	      make_storage(is_storage_rom, UU),
	      make_value(is_value_unknown, UU),
          ABSTRACT_LOCATION|ENTITY_DYNAMIC_AREA);


  make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,
			       STATIC_AREA_LOCAL_NAME),
	      make_type(is_type_area, make_area(0, NIL)),
	      make_storage(is_storage_rom, UU),
	      make_value(is_value_unknown, UU),
          ABSTRACT_LOCATION|ENTITY_STATIC_AREA);
}

static void CreateLogicalUnits()
{
  /* First a dummy function - close to C one "crt0()" - in order to
     - link the next entity to its ram
     - make an unbounded dimension for this entity
  */

  entity ent = entity_undefined;
  entity luns = entity_undefined;
  sequence s = make_sequence(NIL);
  code c = make_code(NIL, strdup(""), s, NIL, make_language_unknown());

  code_initializations(c) = s;

  ent = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,
                                     IO_EFFECTS_PACKAGE_NAME),
                    make_type(is_type_functional,
                              make_functional(NIL,make_type(is_type_void,
                                                            NIL))),
                    make_storage(is_storage_rom, UU),
                    make_value(is_value_code, c),
                    DEFAULT_ENTITY_KIND);

  set_current_module_entity(ent);

  make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
                               STATIC_AREA_LOCAL_NAME),
              make_type(is_type_area, make_area(0, NIL)),
              make_storage(is_storage_rom, UU),
              make_value(is_value_unknown, UU),
              EFFECTS_PACKAGE|ABSTRACT_LOCATION|ENTITY_STATIC_AREA);

  /* GO: entity for io logical units: It is an array which*/
  make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
                               IO_EFFECTS_ARRAY_NAME),
          MakeTypeArray(make_basic_int(IO_EFFECTS_UNIT_SPECIFIER_LENGTH),
                        CONS(DIMENSION,
                             make_dimension
                             (int_to_expression(0),
                                /*
                                  MakeNullaryCall
                                  (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
                                */
                              int_to_expression(2000)
                              ),
                             NIL)),
              /* make_storage(is_storage_ram,
                 make_ram(entity_undefined, DynamicArea, 0, NIL))
              */
              make_storage(is_storage_ram,
                           make_ram(ent,
                           FindEntity(IO_EFFECTS_PACKAGE_NAME,
                                                 STATIC_AREA_LOCAL_NAME),
                                    0, NIL)),
              make_value(is_value_unknown, UU),
              EFFECTS_PACKAGE);

  /* GO: entity for io logical units: It is an array which*/
  make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
                               IO_EOF_ARRAY_NAME),
        MakeTypeArray(make_basic_logical(IO_EFFECTS_UNIT_SPECIFIER_LENGTH),
                      CONS(DIMENSION,
                           make_dimension
                           (int_to_expression(0),
                            /*
                              MakeNullaryCall
                              (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
                            */
                            int_to_expression(2000)
                            ),
                           NIL)),
              /* make_storage(is_storage_ram,
                 make_ram(entity_undefined, DynamicArea, 0, NIL))
              */
              make_storage(is_storage_ram,
                    make_ram(ent,
                             FindEntity(IO_EFFECTS_PACKAGE_NAME,
                                                   STATIC_AREA_LOCAL_NAME),
                             0, NIL)),
              make_value(is_value_unknown, UU),
              EFFECTS_PACKAGE);

  /* GO: entity for io logical units: It is an array which*/
  luns = make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
				      IO_ERROR_ARRAY_NAME),
		     MakeTypeArray(make_basic_logical(IO_EFFECTS_UNIT_SPECIFIER_LENGTH),
				   CONS(DIMENSION,
					make_dimension
					(int_to_expression(0),
					 /*
					   MakeNullaryCall
					   (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
					 */
					 int_to_expression(2000)
					 ),
					NIL)),
		     /* make_storage(is_storage_ram,
			make_ram(entity_undefined, DynamicArea, 0, NIL))
		     */
		     make_storage(is_storage_ram,
				  make_ram(ent,
					   FindEntity(IO_EFFECTS_PACKAGE_NAME,
								 STATIC_AREA_LOCAL_NAME),
					   0, NIL)),
		     make_value(is_value_unknown, UU),
             EFFECTS_PACKAGE);

  reset_current_module_entity();
  add_abstract_state_variable(luns);
}

/* added to handle xxxrandxxx functions.Amira Mensi and then
   generalized for other hidden libc variables
*/
static entity CreateAbstractStateVariable(string pn, string vn)
{
  /* First a dummy function - close to C one "crt0()" - in order to
     - link the next entity to its ram
     - make an unbounded dimension for this entity
  */

  entity ent = entity_undefined;
  entity as = entity_undefined;
  sequence s = make_sequence(NIL);
  code c = make_code(NIL, strdup(""), s, NIL, make_language_unknown());

  code_initializations(c) = s;

  ent = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME,
				     pn),
		    make_type(is_type_functional,
			      make_functional(NIL,make_type(is_type_void,
							    NIL))),
		    make_storage(is_storage_rom, UU),
		    make_value(is_value_code, c),
            DEFAULT_ENTITY_KIND);

  set_current_module_entity(ent);

  make_entity(AddPackageToName(pn,
			       STATIC_AREA_LOCAL_NAME),
	      make_type(is_type_area, make_area(0, NIL)),
	      make_storage(is_storage_rom, UU),
	      make_value(is_value_unknown, UU),
          EFFECTS_PACKAGE|ABSTRACT_LOCATION|ENTITY_STATIC_AREA);

  /* entity for random seed or other abstract states like heap: It is
     an unsigned int. */
  as = make_entity(AddPackageToName(pn, vn),
		   make_scalar_integer_type(DEFAULT_INTEGER_TYPE_SIZE),
		   /* make_storage(is_storage_ram,
		      make_ram(entity_undefined, DynamicArea, 0, NIL))
		   */
		   make_storage(is_storage_ram,
				make_ram(ent,
					 FindEntity(pn,
							       STATIC_AREA_LOCAL_NAME),
					 0, NIL)),
		   make_value(is_value_unknown, UU),
           EFFECTS_PACKAGE);

  reset_current_module_entity();
  return as;
}

// added to handle xxxrandxxx functions.Amira Mensi
static void CreateRandomSeed()
{
  entity as =
    CreateAbstractStateVariable(RAND_EFFECTS_PACKAGE_NAME, RAND_GEN_EFFECTS_NAME);
  //add_thread_safe_variable(as);
  add_abstract_state_variable(as);
}
// added to handle time functions
static void CreateTimeSeed()
{
  entity as =
    CreateAbstractStateVariable(TIME_EFFECTS_PACKAGE_NAME, TIME_EFFECTS_VARIABLE_NAME);
  //add_thread_safe_variable(as);
  add_abstract_state_variable(as);
}

static void CreateHeapAbstractState()
{
  entity as =
    CreateAbstractStateVariable(MALLOC_EFFECTS_PACKAGE_NAME, MALLOC_EFFECTS_NAME);

  add_thread_safe_variable(as);
  add_abstract_state_variable(as);
}
//Molka Becher : Added to handle Memmove function.
static void CreateMemmoveAbstractState()
{
  entity as =
    CreateAbstractStateVariable(MEMMOVE_EFFECTS_PACKAGE_NAME, MEMMOVE_EFFECTS_NAME);

  add_thread_safe_variable(as);
  add_abstract_state_variable(as);
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
    parameter vp = make_parameter(v, make_mode(is_mode_reference, UU), make_dummy_unknown());

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
overloaded_to_longdouble_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeQuadprecisionResult());
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

//static type
//overloaded_to_longdoublecomplex_type(int n)  /* MB */
/*{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongDoublecomplexResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);

  return t;
}*/

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

/* to handle BTEST function which takes integer as parameter and
   returns logical. Amira Mensi */
static type
integer_to_logical_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLogicalResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeIntegerParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

/* Why do we make these functions static and keep them here instead of
   populating ri-util/type.c? */
static type integer_to_integer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeIntegerParameter);
  t = make_type(is_type_functional, ft);

  return t;
}
static type unsigned_integer_to_void_pointer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeVoidPointerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeUnsignedIntegerParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

/* Can be used for C or Fortran functions. E.g. abort() */
static type void_to_void_type(int n __attribute__ ((unused)))
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, make_type_void(NIL));
  functional_parameters(ft) =
    CONS(PARAMETER, make_parameter(make_type_void(NIL),
				   make_mode_value(), // not
						      // significant
						      // for void...
				   make_dummy_unknown()), NIL);
  t = make_type(is_type_functional, ft);

  pips_assert("t is consistent", type_consistent_p(t));

  return t;
}

/* C only because of pointer. e.g. atexit() */
static type void_to_void_to_int_pointer_type(int n __attribute__ ((unused)))
{
  type t = type_undefined;
  functional ft = functional_undefined;
  type vtv = void_to_void_type(0);
  type vtvp = type_to_pointer_type(vtv);

  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) =
    CONS(PARAMETER, make_parameter(vtvp,
				   make_mode_value(), // not
						      // significant
						      // for void...
				   make_dummy_unknown()), NIL);
  t = make_type(is_type_functional, ft);

  pips_assert("t is consistent", type_consistent_p(t));

  return t;
}

/* C only because of pointer. e.g. atof() */
static type char_pointer_to_double_type(int n __attribute__ ((unused)))
{
  type t = type_undefined;
  functional ft = functional_undefined;
  type cp = MakeCharacterResult();
  //type cp = type_to_pointer_type(c);
  type d = MakeDoubleprecisionResult();

  variable_qualifiers(type_variable(cp))
    = CONS(QUALIFIER, make_qualifier_const(), NIL);

  ft = make_functional(NIL, d);
  functional_parameters(ft) =
    CONS(PARAMETER, make_parameter(cp,
				   make_mode_value(), // not
						      // significant
						      // for void...
				   make_dummy_unknown()), NIL);
  t = make_type(is_type_functional, ft);

  pips_assert("t is consistent", type_consistent_p(t));

  return t;
}

static type integer_to_real_type(int n)
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
real_to_longinteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongIntegerResult());
  functional_parameters(ft) = make_parameter_list(n, MakeRealParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

static type
real_to_longlonginteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongLongIntegerResult());
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
double_to_longinteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeDoubleprecisionParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

static type
double_to_longlonginteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongLongIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeDoubleprecisionParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

static type
longinteger_to_longinteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeDoubleprecisionParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

static type
longlonginteger_to_longlonginteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongLongIntegerResult());
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
 // t = make_type(is_type_functional, ft);

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
longdouble_to_integer_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeQuadprecisionParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

static type
longdouble_to_longinteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeQuadprecisionParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

static type
longdouble_to_longlonginteger_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongLongIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeQuadprecisionParameter);
  t = make_type(is_type_functional, ft);

  return t;
}


static type
longdouble_to_longdouble_type(int n)  /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeQuadprecisionResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeQuadprecisionParameter);
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
longdoublecomplex_to_longdoublecomplex_type(int n) /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeLongDoublecomplexResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeLongDoublecomplexParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

static type
longdoublecomplex_to_longdouble_type(int n) /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL,  MakeQuadprecisionResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeLongDoublecomplexParameter);
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

  pips_assert("valid arity", (int)gen_length(functional_parameters(ft))==n);

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

  pips_assert("valid arity", (int)gen_length(functional_parameters(ft))==n);

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
 * Return true if OK
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
        pips_internal_error("Unexpected integer size %d", basic_int(cast));
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
      pips_internal_error("Unexpected float size %d", basic_float(cast));
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
      pips_internal_error("Unexpected logical size %d", basic_logical(cast));
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
      pips_internal_error("Unexpected complex size %d", basic_complex(cast));
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
    pips_internal_error("Cast function is not verified!");
  }
  if (cast_function == entity_undefined)
  {
    pips_internal_error("Can not convert to LOGICAL!");
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

/* MB : Type check long double complex */
#define TC_LONGDCOMPLEX \
get_bool_property("TYPE_CHECKER_LONG_DOUBLE_COMPLEX_EXTENSION")

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
      return false;
    }
  }
      , le);

  return true;
}

static bool
is_basic_int_p(basic b)
{
  return basic_int_p(b) && basic_int(b)==4;
}
static bool 
is_basic_longint_p(basic b) /* MB */
{
  return basic_int_p(b) && basic_int(b)==6;
}
static bool 
is_basic_longlongint_p(basic b) /* MB */
{
  return basic_int_p(b) && basic_int(b)==8;
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
is_basic_longdouble_p(basic b)  /* MB */
{
  return basic_float_p(b) && basic_float(b)==16;
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
is_basic_longdcomplex_p(basic b)  /* MB */
{
  return basic_complex_p(b) && basic_complex(b)==32;
}

static bool
arguments_are_integer(call c, hash_table types)
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_int_p);
}
static bool
arguments_are_longinteger(call c, hash_table types) /* MB */
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_longint_p);
}
static bool
arguments_are_longlonginteger(call c, hash_table types) /* MB */
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_longlongint_p);
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
arguments_are_longdouble(call c, hash_table types) /* MB */
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_longdouble_p);
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
static bool
arguments_are_longdcomplex(call c, hash_table types)  /* MB */
{
  return check_if_basics_ok(call_arguments(c), types, is_basic_longdcomplex_p);
}


/**************************************************************************
 * Verify if all the arguments basic of function C
 * If there is no argument, I return TRUE
 *
 * Note: I - Integer; R - Real; D - Double; C - Complex;
 */

/* Molka Becher: add of long int, long long int, long double and long
   double complex types */
static bool
arguments_are_something(
    call c,
    type_context_p context,
    bool integer_ok,
    bool longinteger_ok,
    bool longlonginteger_ok,
    bool real_ok,
    bool double_ok,
    bool longdouble_ok,
    bool complex_ok,
    bool dcomplex_ok,
    bool longdcomplex_ok,
    bool logical_ok,
    bool character_ok)
{
  basic b;
  int argnumber = 0;
  bool
    okay = true,
    arg_double = false,
    arg_cmplx = false;

  list args = call_arguments(c);

  FOREACH(EXPRESSION, e, args)
    {
      argnumber++;

      pips_assert("type is defined", hash_defined_p(context->types, e));

      b = GET_TYPE(context->types, e);

      /* Subroutine maybe be used as a function */
      if (basic_overloaded_p(b))
	{
	  syntax s = expression_syntax(e);
	  const char* what  ;
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
	  okay = false;
	}
      else if (!((integer_ok && basic_int_p(b) && basic_int(b)==4) ||
		 (longinteger_ok && basic_int_p(b) && basic_int(b)==6) ||
		 (longlonginteger_ok && basic_int_p(b) && basic_int(b)==8) ||
		 (real_ok && basic_float_p(b) && basic_float(b)==4) ||
		 (double_ok &&  basic_float_p(b) && basic_float(b)==8) ||
		 (longdouble_ok &&  basic_float_p(b) && basic_float(b)==16) ||
		 (complex_ok && basic_complex_p(b) && basic_complex(b)==8) ||
		 (dcomplex_ok && basic_complex_p(b) && basic_complex(b)==16) ||
		 (longdcomplex_ok && basic_complex_p(b) && basic_complex(b)==32) ||
		 (logical_ok && basic_logical_p(b)) ||
		 (character_ok && basic_string_p(b))))
	{
	  /* The message should be language dependent, C or Fortran */
	  if(fortran_language_module_p(get_current_module_entity())) {
	  add_one_line_of_comment((statement) stack_head(context->stats),
				  "#%d argument of '%s' must be "
				  "%s%s%s%s%s%s%s%s%s%s%s but not %s",
				  argnumber,
				  entity_local_name(call_function(c)),
				  integer_ok? "INT, ": "",
				  // The next two types are not used
				  // for Fortran
				  longinteger_ok? "": "",
				  longlonginteger_ok? "": "",
				  real_ok? "REAL, ": "",
				  double_ok? "DOUBLE, ": "",
				  // FI: Used to be QUAD. Exists or not?
				  //longdouble_ok? "DOUBLE*16, ": "",
				  longdouble_ok? "": "",
				  complex_ok? "COMPLEX, ": "",
				  dcomplex_ok? "DCOMPLEX, ": "",
				  // FI: what should it be in Fortran?
				  longdcomplex_ok? "": "",
				  logical_ok? "LOGICAL, ": "",
				  character_ok? "CHARACTER, ": "",
				  basic_to_string(b));
	  } else { /* Assume C */
	    /* FI: assumes no pointers ever; still pretty much
	       Fortran stuff */
	  add_one_line_of_comment((statement) stack_head(context->stats),
				  "#%d argument of '%s' must be "
				  "%s%s%s%s%s%s%s%s%s%s%s but not %s",
				  argnumber,
				  entity_local_name(call_function(c)),
				  integer_ok? "int, ": "",
				  longinteger_ok? "long int, ": "",
				  longlonginteger_ok? "long long int, ": "",
				  real_ok? "float, ": "",
				  double_ok? "double, ": "",
				  longdouble_ok? "long double, ": "",
				  complex_ok? "complex, ": "",
				  dcomplex_ok? "double complex, ": "",
				  longdcomplex_ok? "long double complex, ": "",
				  logical_ok? "bool, ": "",
				  /* FI: nothing about strings? */
				  character_ok? "char, ": "",
				  basic_to_string(b));
	  }
	  context->number_of_error++;
	  okay = false;
	}

      /* if TC_DCOMPLEX, maybe they should not be incompatible? */
      arg_cmplx = arg_cmplx ||
	(complex_ok && basic_complex_p(b) && basic_complex(b)==8);

      arg_double = arg_double ||
	(double_ok &&  basic_float_p(b) && basic_float(b)==8);
    }

  if (arg_cmplx && arg_double)
    {
      add_one_line_of_comment((statement) stack_head(context->stats),
			      "mixed complex and double arguments of '%s' forbidden",
			      entity_local_name(call_function(c)));
      context->number_of_error++;
      okay = false;
    }

  return okay;
}

static bool
arguments_are_IRDCS(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, true, true, true, true, true, true, true, TC_DCOMPLEX, TC_LONGDCOMPLEX, false, true);
}

static bool
arguments_are_IRDC(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, true, true, true, true, true, true, true, TC_DCOMPLEX, TC_LONGDCOMPLEX, false, false);
}
static bool
arguments_are_character(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, false, false, false, false, false, false, false, false, false, false, true);
}
static bool
arguments_are_logical(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, false, false, false, false, false, false, false, false, false, true, false);
}
static bool
arguments_are_RD(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, false, false, false, true, true, true, false, false, false, false, false);
}

static bool
arguments_are_IR(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, true, true, true, true, false, false, false, false, false, false, false);
}
static bool
arguments_are_IRD(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, true, true, true, true, true, true, false, false, false, false, false);
}
/**************************************************************************
 * Verify if all the arguments basic of function C are REAL, DOUBLE
 * and COMPLEX
 * According to (ANSI X3.9-1978 FORTRAN 77, Table 2 & 3, Page 6-5 & 6-6),
 * it is prohibited an arithetic operator operaters on
 * a pair of DOUBLE and COMPLEX, so that I return false in that case.
 *
 * PDSon: If there is no argument, I return TRUE
 */
static bool
arguments_are_RDC(call c, type_context_p context)
{
  return arguments_are_something
    (c, context, false, false, false, true, true, true, true, TC_DCOMPLEX, TC_LONGDCOMPLEX, false, false);
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
        return false;
      }
    }
  }
      , call_arguments(c));

  return true;
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
      EXPRESSION_(CAR(args)) =
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
    EXPRESSION_(CAR(args)) =
      insert_cast(b, b1, EXPRESSION(CAR(args)), context);
  }
  /* Fortran prefers: (ANSI X3.9-1978, FORTRAN 77, PAGE 6-6, TABLE 3)
   * "var_double = var_double ** var_int" instead of
   * "var_double = var_double ** DBLE(var_int)"
   */
  if (!basic_equal_p(b, b2) && !basic_int_p(b2))
  {
    EXPRESSION_(CAR(CDR(args))) =
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
  bool check_arg = false;

  /* INT */
  if(basic_int_p(from_type) && basic_int(from_type) == 4)
  {
    check_arg = arguments_are_integer(c, context->types);
  }
  /* LONG INT : added by MB */
  else if(basic_int_p(from_type) && basic_int(from_type) == 6)
  {
    check_arg = arguments_are_longinteger(c, context->types);
  }
  /* LONG LONG INT */
  else if(basic_int_p(from_type) && basic_int(from_type) == 8)
  {
    check_arg = arguments_are_longlonginteger(c, context->types);
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
  /* LONG DOUBLE : added by MB */
  else if(basic_float_p(from_type) && basic_float(from_type) == 16)
  {
    check_arg = arguments_are_longdouble(c, context->types);
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
  /* LONG DOUBLE COMPLEX : added by MB*/
  else if(basic_complex_p(from_type) && basic_complex(from_type) == 32)
  {
    if (TC_LONGDCOMPLEX)
      check_arg = arguments_are_longdcomplex(c, context->types);
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
    pips_internal_error("Unexpected basic: %s ",
                        basic_to_string(from_type));
  }

  /* ERROR: Invalide of argument type */
  if(check_arg == false)
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
typing_function_longint_to_longint(call c, type_context_p context) /* MB */
{
  basic result, type_LINT = make_basic_int(6);
  result = typing_function_argument_type_to_return_type(c, context,
                                                     type_LINT, type_LINT);
  free_basic(type_LINT);
  return result;
}
static basic
typing_function_longlongint_to_longlongint(call c, type_context_p context) /* MB */
{
  basic result, type_LLINT = make_basic_int(8);
  result = typing_function_argument_type_to_return_type(c, context,
                                                     type_LLINT, type_LLINT);
  free_basic(type_LLINT);
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
typing_function_longdouble_to_longdouble(call c, type_context_p context)  /* MB */
{
  basic result, type_LDBLE = make_basic_float(16);
  result = typing_function_argument_type_to_return_type(c, context,
                                                      type_LDBLE, type_LDBLE);
  free_basic(type_LDBLE);
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
typing_function_longdcomplex_to_longdcomplex(call c, type_context_p context)  /*MB: added for long double complex type*/
{
  basic result, type_LDCMPLX = make_basic_complex(32);
  result = typing_function_argument_type_to_return_type(c, context,
                                                    type_LDCMPLX, type_LDCMPLX);
  free_basic(type_LDCMPLX);
  return result;
}
static basic
typing_function_longdcomplex_to_longdouble(call c, type_context_p context)  /* MB */
{
  basic result, type_LDBLE = make_basic_float(16);
  basic type_LDCMPLX = make_basic_complex(32);
  result = typing_function_argument_type_to_return_type(c, context,
                                                    type_LDCMPLX, type_LDBLE);
  free_basic(type_LDBLE);
  free_basic(type_LDCMPLX);
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
typing_function_real_to_longint(call c, type_context_p context) /* MB */
{
  basic result, type_LINT = make_basic_int(6);
  basic type_REAL = make_basic_float(4);
  result = typing_function_argument_type_to_return_type(c, context,
                                                        type_REAL,
                                                        type_LINT);
  free_basic(type_LINT);
  free_basic(type_REAL);
  return result;
}
static basic
typing_function_real_to_longlongint(call c, type_context_p context) /* MB */
{
  basic result, type_LLINT = make_basic_int(8);
  basic type_REAL = make_basic_float(4);
  result = typing_function_argument_type_to_return_type(c, context,
                                                        type_REAL,
                                                        type_LLINT);
  free_basic(type_LLINT);
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

/* function added to handle one of the bit manipulation functions :
   BTEST. Amira Mensi */
static basic
typing_function_int_to_logical(call c, type_context_p context)
{
  basic result, type_INT = make_basic_int(4);
  basic type_LOGICAL = make_basic_float(4);
  result = typing_function_argument_type_to_return_type(c, context,
                                                        type_INT,
                                                        type_LOGICAL);
  free_basic(type_INT);
  free_basic(type_LOGICAL);
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
typing_function_double_to_longint(call c, type_context_p context) /* MB */
{
  basic result, type_LINT = make_basic_int(6);
  basic type_DBLE = make_basic_float(8);
  result = typing_function_argument_type_to_return_type(c, context,
                                                      type_DBLE, type_LINT);
  free_basic(type_LINT);
  free_basic(type_DBLE);
  return result;
}

static basic
typing_function_double_to_longlongint(call c, type_context_p context) /* MB */
{
  basic result, type_LLINT = make_basic_int(8);
  basic type_DBLE = make_basic_float(8);
  result = typing_function_argument_type_to_return_type(c, context,
                                                      type_DBLE, type_LLINT);
  free_basic(type_LLINT);
  free_basic(type_DBLE);
  return result;
}
static basic
typing_function_longdouble_to_int(call c, type_context_p context)  /* MB */
{
  basic result, type_INT = make_basic_int(4);
  basic type_LDBLE = make_basic_float(16);
  result = typing_function_argument_type_to_return_type(c, context,
                                                        type_LDBLE,
                                                        type_INT);
  free_basic(type_INT);
  free_basic(type_LDBLE);
  return result;
}

static basic
typing_function_longdouble_to_longint(call c, type_context_p context)  /* MB */
{
  basic result, type_LINT = make_basic_int(6);
  basic type_LDBLE = make_basic_float(16);
  result = typing_function_argument_type_to_return_type(c, context,
                                                        type_LDBLE,
                                                        type_LINT);
  free_basic(type_LINT);
  free_basic(type_LDBLE);
  return result;
}

static basic
typing_function_longdouble_to_longlongint(call c, type_context_p context)  /* MB */
{
  basic result, type_LLINT = make_basic_int(8);
  basic type_LDBLE = make_basic_float(16);
  result = typing_function_argument_type_to_return_type(c, context,
                                                        type_LDBLE,
                                                        type_LLINT);
  free_basic(type_LLINT);
  free_basic(type_LDBLE);
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
typing_function_overloaded(call __attribute__ ((unused)) c,
                           type_context_p __attribute__ ((unused)) context)
{
  return make_basic_overloaded();
}

static basic
typing_function_format_name(call __attribute__ ((unused)) c,
                            type_context_p __attribute__ ((unused)) context)
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
      EXPRESSION_(CAR(CDR(args))) =
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

static basic no_typing(call __attribute__ ((unused)) c,
                       type_context_p __attribute__ ((unused)) context)
{
  basic bt = basic_undefined;
  pips_internal_error("This should not be type-checked because it is not Fortran function");
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
 * YES : return true; otherwise, return FALSE
 */
static bool
is_constant(expression exp)
{
  syntax s = expression_syntax(exp);
  if (!syntax_call_p(s))
  {
    return false;
  }
  return (entity_constant_p(call_function(syntax_call(s))));
}

/* Verify if an expression is a constant of basic b:
 * YES : return true; otherwise, return FALSE
 */
static bool
is_constant_of_basic(expression exp, basic b)
{
  type call_type, return_type;
  basic bb;
  if (!is_constant(exp))
  {
    return false;
  }
  call_type = entity_type(call_function(syntax_call(expression_syntax(exp))));
  return_type = functional_result(type_functional(call_type));
  bb = variable_basic(type_variable(return_type));
  if (basic_undefined_p(bb) || !basic_equal_p(b, bb))
  {
    return false;
  }
  return true;
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
    return false;
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
    return false;
  }
  return true;
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
    return false;
  }
  return true;
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
    return false;
  }
  return true;
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
    return false;
  }

  return true;
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
    return false;
  }
  return true;

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
    return false;
  }

  return true;
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
            const char* specifier,
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
        return true;
      }
      /* else ok */
    }
    else /* not allowed */
    {
      add_one_line_of_comment((statement) stack_head(context->stats),
                              "Specifier '%s' is not allowed", name);
      context->number_of_error++;
    }
    return true; /* checked! */
  }

  return false;
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
  const char* spec;
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
      return false;
    }
  }

  return true;
}

static basic
check_read_write(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
                /* UNIT FMT REC IOSTAT ERR END IOLIST */
                true, true, true, true, true, true, true,
                /* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
                false, false, false, false, false, false, false, false,
                /* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
                false, false, false, false, false, false,
                /* UNFORMATTED NEXTREC */
                false, false);

  return basic_undefined;
}

static basic
check_open(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
                /* UNIT FMT REC IOSTAT ERR END IOLIST */
                true, false, false, true, true, false, false,
                /* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
                true, true, true, true, true, true, false, false,
                /* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
                false, false, false, false, false, false,
                /* UNFORMATTED NEXTREC */
                false, false);
  return basic_undefined;
}

static basic
check_close(call c, type_context_p context)
{
  list args = call_arguments(c);

  check_io_list(args, context,
                /* UNIT FMT REC IOSTAT ERR END IOLIST */
                true, false, false, true, true, false, false,
                /* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
                false, true, false, false, false, false, false, false,
                /* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
                false, false, false, false, false, false,
                /* UNFORMATTED NEXTREC */
                false, false);
  return basic_undefined;
}

static basic
check_inquire(call c, type_context_p context)
{
  list args = call_arguments(c);

  check_io_list(args, context,
                /* UNIT FMT REC IOSTAT ERR END IOLIST */
                true, false, false, true, true, false, false,
                /* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
                true, false, true, true, true, true, true, true,
                /* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
                true, true, true, true, true, true,
                /* UNFORMATTED NEXTREC */
                true, true);

  return basic_undefined;
}

static basic
check_backspace(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
                /* UNIT FMT REC IOSTAT ERR END IOLIST */
                true, false, false, true, true, false, false,
                /* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
                false, false, false, false, false, false, false, false,
                /* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
                false, false, false, false, false, false,
                /* UNFORMATTED NEXTREC */
                false, false);
  return basic_undefined;
}

static basic
check_endfile(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
                /* UNIT FMT REC IOSTAT ERR END IOLIST */
                true, false, false, true, true, false, false,
                /* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
                false, false, false, false, false, false, false, false,
                /* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
                false, false, false, false, false, false,
                /* UNFORMATTED NEXTREC */
                false, false);
  return basic_undefined;
}

static basic
check_rewind(call c, type_context_p context)
{
  list args = call_arguments(c);
  check_io_list(args, context,
                /* UNIT FMT REC IOSTAT ERR END IOLIST */
                true, false, false, true, true, false, false,
                /* FILE STATUS ACCESS FORM BLANK RECL EXIST OPENED */
                false, false, false, false, false, false, false, false,
                /* NUMBER NAMED NAME SEQUENTIAL DIRECT FORMATTED */
                false, false, false, false, false, false,
                /* UNFORMATTED NEXTREC */
                false, false);
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

/* Move the following functions to ri-util/type.c */

type
MakeVoidResult()
{
    return make_type(is_type_void, UU);
}

parameter
MakeVoidParameter()
{
  return make_parameter(make_type(is_type_void, UU),
                        make_mode(is_mode_reference, UU),
                        make_dummy_unknown());
}

type
integer_to_overloaded_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeOverloadedResult());
  t = make_type(is_type_functional, ft);

  functional_parameters(ft) =
    make_parameter_list(n, MakeIntegerParameter);
  return t;
}

type
longinteger_to_overloaded_type(int n) /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeOverloadedResult());
  t = make_type(is_type_functional, ft);

  functional_parameters(ft) =
    make_parameter_list(n, MakeLongIntegerParameter);
  return t;
}

type
longlonginteger_to_overloaded_type(int n) /* MB */
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeOverloadedResult());
  t = make_type(is_type_functional, ft);

  functional_parameters(ft) =
    make_parameter_list(n, MakeLongLongIntegerParameter);
  return t;
}

type
integer_to_void_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, make_type_void(NIL));
  t = make_type(is_type_functional, ft);

  functional_parameters(ft) =
    make_parameter_list(n, MakeIntegerParameter);
  return t;
}

type __attribute__ ((unused))
void_to_overloaded_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeOverloadedResult());
  t = make_type(is_type_functional, ft);

  functional_parameters(ft) =
    make_parameter_list(n, MakeVoidParameter);
  return t;
}

type
overloaded_to_void_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeVoidResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeOverloadedParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

type
void_to_integer_type(int n)
{
  type t = type_undefined;
  functional ft = functional_undefined;

  ft = make_functional(NIL, MakeIntegerResult());
  functional_parameters(ft) =
    make_parameter_list(n, MakeVoidParameter);
  t = make_type(is_type_functional, ft);

  return t;
}

/******************************************************** INTRINSICS LIST */




static hash_table intrinsic_type_descriptor_mapping = hash_table_undefined;

/**************************************************************************
 * Get the function for typing the specified intrinsic
 *
 */
typing_function_t get_typing_function_for_intrinsic(const char* name)
{
    return ((IntrinsicDescriptor *) hash_get(intrinsic_type_descriptor_mapping, name))->type_function;
}
/**************************************************************************
 * Get the function for switching to specific name from generic name
 *
 */
switch_name_function get_switch_name_function_for_intrinsic(const char* name)
{
    return ((IntrinsicDescriptor *) hash_get(intrinsic_type_descriptor_mapping, name))->name_function;
}

/* This function creates an entity that represents an intrinsic
   function. Fortran operators and basic statements are intrinsic
   functions.

   An intrinsic function has a rom storage, an unknown initial value and a
   functional type whose result and arguments have an overloaded basic
   type. The number of arguments is given by the IntrinsicTypeDescriptorTable
   data structure. */
static entity MakeIntrinsic(string name, int arity, type (*intrinsic_type)(int)) {
  entity e;

  e = make_entity(AddPackageToName(TOP_LEVEL_MODULE_NAME, name),
                  intrinsic_type(arity),
                  make_storage(is_storage_rom, UU),
                  make_value(is_value_intrinsic, NIL),
                  DEFAULT_ENTITY_KIND);

  return e;
}

/* This function creates an entity that represents an intrinsic
   function, if the entity does not already exist. Fortran operators
   and basic statements are intrinsic functions.

   An intrinsic function has a rom storage, an unknown initial value and a
   functional type whose result and arguments have an overloaded basic
   type. The number of arguments is given by the IntrinsicTypeDescriptorTable
   data structure. */
entity FindOrMakeIntrinsic(string name, int arity, type (*intrinsic_type)(int))
{
    entity e = FindEntity(TOP_LEVEL_MODULE_NAME, name);

    if (entity_undefined_p(e)) {
        e = MakeIntrinsic(name, arity, intrinsic_type);
    }

    return e;
}


/** Create a default intrinsic

    Useful to create on-the-fly intrinsics.

    It creates an intrinsic with a default type, that is with overload
    parameter and return types.

    @param name is the name of the intrinsic

    @param n is the number of argument

    @return the entity of the intrinsic
*/
entity
FindOrMakeDefaultIntrinsic(string name, int arity)
{
  entity e = FindEntity(TOP_LEVEL_MODULE_NAME, name);
  if (!entity_undefined_p(e))
    /* It seems it has been previously created: */
    return e;

  return MakeIntrinsic(name, arity, default_intrinsic_type);
}


/* This function is called one time (at the very beginning) to create
   all intrinsic functions. */

void register_intrinsic_type_descriptor(IntrinsicDescriptor *p) {
    FindOrMakeIntrinsic(p->name, p->nbargs, p->intrinsic_type);
    hash_put(intrinsic_type_descriptor_mapping,p->name,p);
}

void
CreateIntrinsics( set module_list )
{
    /* The table of intrinsic functions. this table is used at the begining
       of linking to create Fortran operators, commands and intrinsic functions.

       Functions with a variable number of arguments are declared with INT_MAX
       arguments.
       */

    /* Nga Nguyen 27/06/2003 Fuse the tables of intrinsics for C and Fortran.
       Since there are differences between some kind of operators, such as in
       Fortran, "+" is only applied to arithmetic numbers, in C, "+" is also applied
       to pointer, the typing functions are different. So in some cases, we have to
       rename the operators */
    /* Pragma can be represented in the pips IR as a list of expression so new
     * functions/intrinsics are needed. For exmaple to represent OMP pragmas,
     * following intrinscs are needed:
     *    1 - omp, parallel and for which are constant so with 0 argument,
     *    2 - the colom poperator (for reduction) that takes two arguments
     *    3 - private and reduction that takes a variable number of arguments.
     */

    static IntrinsicDescriptor IntrinsicTypeDescriptorTable[] =
    {
        {PLUS_OPERATOR_NAME, 2, default_intrinsic_type, typing_arithmetic_operator, 0},
        {MINUS_OPERATOR_NAME, 2, default_intrinsic_type, typing_arithmetic_operator, 0},
        {DIVIDE_OPERATOR_NAME, 2, default_intrinsic_type, typing_arithmetic_operator, 0},
        {MULTIPLY_OPERATOR_NAME, 2, default_intrinsic_type, typing_arithmetic_operator, 0},
        {UNARY_MINUS_OPERATOR_NAME, 1, default_intrinsic_type, typing_arithmetic_operator, 0}, // unary minus
        {POWER_OPERATOR_NAME, 2, default_intrinsic_type, typing_power_operator, 0},

        /* internal inverse operator... */
        { INVERSE_OPERATOR_NAME, 1, real_to_real_type,
            typing_function_RealDoubleComplex_to_RealDoubleComplex, 0},

        {ASSIGN_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},

        {EQUIV_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_logical_operator, 0},
        {NON_EQUIV_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_logical_operator, 0},

        {OR_OPERATOR_NAME, 2, logical_to_logical_type, typing_logical_operator, 0},
        {AND_OPERATOR_NAME, 2, logical_to_logical_type, typing_logical_operator, 0},
        {NOT_OPERATOR_NAME, 1, logical_to_logical_type, typing_logical_operator, 0},

        {LESS_THAN_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_relational_operator, 0},
        {GREATER_THAN_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_relational_operator, 0},
        {LESS_OR_EQUAL_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_relational_operator, 0},
        {GREATER_OR_EQUAL_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_relational_operator, 0},
        {EQUAL_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_relational_operator, 0},
        {NON_EQUAL_OPERATOR_NAME, 2, overloaded_to_logical_type, typing_relational_operator, 0},

        {CONCATENATION_FUNCTION_NAME, 2, character_to_character_type, typing_concat_operator, 0},

        /* FORTRAN IO statement */
        {WRITE_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_read_write, 0},
        {REWIND_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_rewind, 0},
        {BACKSPACE_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_backspace, 0},
        {OPEN_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_open, 0},
        {CLOSE_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_close, 0},
        {READ_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_read_write, 0},
        {BUFFERIN_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, typing_buffer_inout, 0},
        {BUFFEROUT_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, typing_buffer_inout, 0},
        {ENDFILE_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_endfile, 0},
        {IMPLIED_DO_NAME, (INT_MAX), default_intrinsic_type, typing_implied_do, 0},
        {REPEAT_VALUE_NAME, 2, default_intrinsic_type, no_typing, 0},
        {STATIC_INITIALIZATION_FUNCTION_NAME, (INT_MAX) , default_intrinsic_type, no_typing, 0},
        {DATA_LIST_FUNCTION_NAME, (INT_MAX) , default_intrinsic_type, no_typing, 0},
        {FORMAT_FUNCTION_NAME, 1, default_intrinsic_type, check_format, 0},
        {INQUIRE_FUNCTION_NAME, (INT_MAX), default_intrinsic_type, check_inquire, 0},

        {SUBSTRING_FUNCTION_NAME, 3, substring_type, typing_substring, 0},
        {ASSIGN_SUBSTRING_FUNCTION_NAME, 4, assign_substring_type, typing_assign_substring, 0},

        /* Control statement */
        {CONTINUE_FUNCTION_NAME, 0, default_intrinsic_type, statement_without_argument, 0},
        {"ENDDO", 0, default_intrinsic_type, 0, 0}, // Why do we need this one?
        {PAUSE_FUNCTION_NAME, 1, default_intrinsic_type,
            statement_with_at_most_one_integer_or_character, 0},
        {RETURN_FUNCTION_NAME, 0, default_intrinsic_type,
            statement_with_at_most_one_expression_integer, 0},
        {STOP_FUNCTION_NAME, 0, default_intrinsic_type,
            statement_with_at_most_one_integer_or_character, 0},
        {END_FUNCTION_NAME, 0, default_intrinsic_type, statement_without_argument, 0}, // Is it useful?


        {INT_GENERIC_CONVERSION_NAME, 1, overloaded_to_integer_type,
            typing_function_conversion_to_integer, simplification_int},
        {IFIX_GENERIC_CONVERSION_NAME, 1, real_to_integer_type, typing_function_real_to_int,
            simplification_int},
        {IDINT_GENERIC_CONVERSION_NAME, 1, double_to_integer_type, typing_function_double_to_int,
            simplification_int},
        {REAL_GENERIC_CONVERSION_NAME, 1, overloaded_to_real_type, typing_function_conversion_to_real,
            simplification_real},
        {FLOAT_GENERIC_CONVERSION_NAME, 1, overloaded_to_real_type, typing_function_conversion_to_real,
            simplification_real},
        {DFLOAT_GENERIC_CONVERSION_NAME, 1, overloaded_to_double_type,
            typing_function_conversion_to_double, simplification_double},
        {SNGL_GENERIC_CONVERSION_NAME, 1, overloaded_to_real_type, typing_function_conversion_to_real,
            simplification_real},
        {DBLE_GENERIC_CONVERSION_NAME, 1, overloaded_to_double_type,
            typing_function_conversion_to_double, simplification_double},
        {DREAL_GENERIC_CONVERSION_NAME, 1, overloaded_to_double_type, /* Arnauld Leservot, code CEA */
            typing_function_conversion_to_double, simplification_double},
        {CMPLX_GENERIC_CONVERSION_NAME, (INT_MAX), overloaded_to_complex_type,
            typing_function_conversion_to_complex, simplification_complex},

        {DCMPLX_GENERIC_CONVERSION_NAME, (INT_MAX), overloaded_to_doublecomplex_type,
            typing_function_conversion_to_dcomplex, simplification_dcomplex},

        /* (0.,1.) -> switched to a function call... */
        { IMPLIED_COMPLEX_NAME, 2, overloaded_to_complex_type,
            typing_function_constant_complex, switch_specific_cmplx },
        { IMPLIED_DCOMPLEX_NAME, 2, overloaded_to_doublecomplex_type,
            typing_function_constant_dcomplex, switch_specific_dcmplx },

        {CHAR_TO_INT_CONVERSION_NAME, 1, default_intrinsic_type, typing_function_char_to_int, 0},
        {INT_TO_CHAR_CONVERSION_NAME, 1, default_intrinsic_type, typing_function_int_to_char, 0},

        {AINT_CONVERSION_NAME, 1, real_to_real_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_aint},
        {DINT_CONVERSION_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ANINT_CONVERSION_NAME, 1, real_to_real_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_anint},
        {DNINT_CONVERSION_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {NINT_CONVERSION_NAME, 1, real_to_integer_type,
            typing_function_RealDouble_to_Integer, switch_specific_nint},
        {IDNINT_CONVERSION_NAME, 1, double_to_integer_type, typing_function_double_to_int, 0},

        //Fortran
        {IABS_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ABS_OPERATOR_NAME, 1, real_to_real_type,
            typing_function_IntegerRealDoubleComplex_to_IntegerRealDoubleReal,
            switch_specific_abs},
        {DABS_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CABS_OPERATOR_NAME, 1, complex_to_real_type, typing_function_complex_to_real, 0},
        {CDABS_OPERATOR_NAME, 1, doublecomplex_to_double_type,
            typing_function_dcomplex_to_double, 0},

        {MODULO_OPERATOR_NAME, 2, default_intrinsic_type,
            typing_function_IntegerRealDouble_to_IntegerRealDouble,
            switch_specific_mod},
        {REAL_MODULO_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {DOUBLE_MODULO_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},

        {ISIGN_OPERATOR_NAME, 2, integer_to_integer_type, typing_function_int_to_int, 0},
        {SIGN_OPERATOR_NAME, 2, default_intrinsic_type,
            typing_function_IntegerRealDouble_to_IntegerRealDouble,
            switch_specific_sign},
        {DSIGN_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},

        {IDIM_OPERATOR_NAME, 2, integer_to_integer_type, typing_function_int_to_int, 0},
        {DIM_OPERATOR_NAME, 2, default_intrinsic_type,
            typing_function_IntegerRealDouble_to_IntegerRealDouble,
            switch_specific_dim},
        {DDIM_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},

        {DPROD_OPERATOR_NAME, 2, real_to_double_type, typing_function_real_to_double, 0},

        {MAX_OPERATOR_NAME, (INT_MAX), default_intrinsic_type,
            typing_function_IntegerRealDouble_to_IntegerRealDouble,
            switch_specific_max},
        {MAX0_OPERATOR_NAME, (INT_MAX), integer_to_integer_type,
            typing_function_int_to_int, 0},
        {AMAX1_OPERATOR_NAME, (INT_MAX), real_to_real_type, typing_function_real_to_real, 0},
        {DMAX1_OPERATOR_NAME, (INT_MAX), double_to_double_type,
            typing_function_double_to_double, 0},
        {AMAX0_OPERATOR_NAME, (INT_MAX), integer_to_real_type,
            typing_function_int_to_real, 0},
        {MAX1_OPERATOR_NAME, (INT_MAX), real_to_integer_type, typing_function_real_to_int, 0},

        {MIN_OPERATOR_NAME, (INT_MAX), default_intrinsic_type,
            typing_function_IntegerRealDouble_to_IntegerRealDouble,
            switch_specific_min},
        {MIN0_OPERATOR_NAME, (INT_MAX), integer_to_integer_type,
            typing_function_int_to_int, 0},
        {AMIN1_OPERATOR_NAME, (INT_MAX), real_to_real_type, typing_function_real_to_real, 0},
        {DMIN1_OPERATOR_NAME, (INT_MAX), double_to_double_type,
            typing_function_double_to_double, 0},
        {AMIN0_OPERATOR_NAME, (INT_MAX), integer_to_real_type,
            typing_function_int_to_real, 0},
        {MIN1_OPERATOR_NAME, (INT_MAX), real_to_integer_type, typing_function_real_to_int, 0},

        {LENGTH_OPERATOR_NAME, 1, character_to_integer_type, typing_function_char_to_int, 0},
        {INDEX_OPERATOR_NAME, 2, character_to_integer_type, typing_function_char_to_int, 0},

        {AIMAG_CONVERSION_NAME, 1, complex_to_real_type, typing_function_complex_to_real, 0},
        {DIMAG_CONVERSION_NAME, 1, doublecomplex_to_double_type,
            typing_function_dcomplex_to_double, 0},

        {CONJG_OPERATOR_NAME, 1, complex_to_complex_type,
            typing_function_complex_to_complex, 0},
        {DCONJG_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type,
            typing_function_dcomplex_to_dcomplex, 0},

        {SQRT_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDoubleComplex_to_RealDoubleComplex,
            switch_specific_sqrt},
        {DSQRT_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CSQRT_OPERATOR_NAME, 1, complex_to_complex_type,
            typing_function_complex_to_complex, 0},
        {CDSQRT_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type,
            typing_function_dcomplex_to_dcomplex, 0},

        {EXP_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDoubleComplex_to_RealDoubleComplex,
            switch_specific_exp},
        {DEXP_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CEXP_OPERATOR_NAME, 1, complex_to_complex_type,
            typing_function_complex_to_complex, 0},
        {CDEXP_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type,
            typing_function_dcomplex_to_dcomplex, 0},

        {LOG_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDoubleComplex_to_RealDoubleComplex,
            switch_specific_log},
        {ALOG_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {DLOG_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CLOG_OPERATOR_NAME, 1, complex_to_complex_type,
            typing_function_complex_to_complex, 0},
        {CDLOG_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type,
            typing_function_dcomplex_to_dcomplex, 0},

        {LOG10_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_log10},
        {ALOG10_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {DLOG10_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {SIN_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDoubleComplex_to_RealDoubleComplex,
            switch_specific_sin},
        {DSIN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CSIN_OPERATOR_NAME, 1, complex_to_complex_type,
            typing_function_complex_to_complex, 0},
        {CDSIN_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type,
            typing_function_dcomplex_to_dcomplex, 0},

        {COS_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDoubleComplex_to_RealDoubleComplex,
            switch_specific_cos},
        {DCOS_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CCOS_OPERATOR_NAME, 1, complex_to_complex_type,
            typing_function_complex_to_complex, 0},
        {CDCOS_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type,
            typing_function_dcomplex_to_dcomplex, 0},

        {TAN_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_tan},
        {DTAN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {ASIN_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_asin},
        {DASIN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {ACOS_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_acos},
        {DACOS_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {ATAN_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_atan},
        {DATAN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ATAN2_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_atan2},
        {DATAN2_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {SINH_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_sinh},
        {DSINH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {COSH_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_cosh},
        {DCOSH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {TANH_OPERATOR_NAME, 1, default_intrinsic_type,
            typing_function_RealDouble_to_RealDouble, switch_specific_tanh},
        {DTANH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {LGE_OPERATOR_NAME, 2, character_to_logical_type, typing_function_char_to_logical, 0},
        {LGT_OPERATOR_NAME, 2, character_to_logical_type, typing_function_char_to_logical, 0},
        {LLE_OPERATOR_NAME, 2, character_to_logical_type, typing_function_char_to_logical, 0},
        {LLT_OPERATOR_NAME, 2, character_to_logical_type, typing_function_char_to_logical, 0},

        {LIST_DIRECTED_FORMAT_NAME, 0, default_intrinsic_type,
            typing_function_format_name, 0},
        {UNBOUNDED_DIMENSION_NAME, 0, default_intrinsic_type,
            typing_function_overloaded, 0},

        /* Bit manipulation functions : ISO/IEC 1539 */
        {ISHFT_OPERATOR_NAME, 2, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISHFTC_OPERATOR_NAME, 3,integer_to_integer_type, typing_function_int_to_int, 0},
        {IBITS_OPERATOR_NAME, 3, integer_to_integer_type, typing_function_int_to_int, 0},
        {MVBITS_OPERATOR_NAME, 5,integer_to_integer_type, typing_function_int_to_int, 0},
        {BTEST_OPERATOR_NAME, 2,integer_to_logical_type, typing_function_int_to_logical, 0},
        {IBSET_OPERATOR_NAME, 2,integer_to_integer_type, typing_function_int_to_int, 0},
        {IBCLR_OPERATOR_NAME, 2,integer_to_integer_type, typing_function_int_to_int, 0},
        {BIT_SIZE_OPERATOR_NAME, 2,integer_to_integer_type, typing_function_int_to_int, 0},
        {IAND_OPERATOR_NAME, 2,integer_to_integer_type, typing_function_int_to_int, 0},
        {IEOR_OPERATOR_NAME, 2,integer_to_integer_type, typing_function_int_to_int, 0},
        {IOR_OPERATOR_NAME, 2,integer_to_integer_type, typing_function_int_to_int, 0},

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

        /* integer combined multiply add or sub - FC oct 2005 */
        { IMA_OPERATOR_NAME, 3, integer_to_integer_type,
            typing_function_int_to_int, 0 },
        { IMS_OPERATOR_NAME, 3, integer_to_integer_type,
            typing_function_int_to_int, 0 },

        /* Here are C intrinsics arranged in the order of the standard ISO/IEC 9899:TC2. MB */

        /* ISO 6.5.2.3 structure and union members */
        {FIELD_OPERATOR_NAME, 2, default_intrinsic_type, 0, 0},
        {POINT_TO_OPERATOR_NAME, 2, default_intrinsic_type, 0, 0},
        /* ISO 6.5.2.4 postfix increment and decrement operators, real or pointer type operand */
        {POST_INCREMENT_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        {POST_DECREMENT_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        /* ISO 6.5.3.1 prefix increment and decrement operators, real or pointer type operand */
        {PRE_INCREMENT_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        {PRE_DECREMENT_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        /* ISO 6.5.3.2 address and indirection operators, add pointer type */
        {ADDRESS_OF_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        {DEREFERENCING_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        /* ISO 6.5.3.3 unary arithmetic operators */
        {UNARY_PLUS_OPERATOR_NAME, 1, default_intrinsic_type, typing_arithmetic_operator, 0},
        /* Unuary minus : ALREADY EXIST (FORTRAN)
           {UNARY_MINUS_OPERATOR_NAME, 1, default_intrinsic_type, typing_arithmetic_operator, 0},*/
        {BITWISE_NOT_OPERATOR_NAME, 1, integer_to_overloaded_type, typing_arithmetic_operator, 0},
        {C_NOT_OPERATOR_NAME, 1, overloaded_to_integer_type, 0, 0},
        /* ISO 6.5.5 multiplicative operators : ALREADY EXIST (FORTRAN)
           {MULTIPLY_OPERATOR_NAME, 2, default_intrinsic_type, typing_arithmetic_operator, 0},
           {DIVIDE_OPERATOR_NAME, 2, default_intrinsic_type, typing_arithmetic_operator, 0},*/
        {C_MODULO_OPERATOR_NAME, 2, integer_to_overloaded_type, typing_arithmetic_operator, 0},
        /* ISO 6.5.6 additive operators, arithmetic types or pointer + integer type*/
        {PLUS_C_OPERATOR_NAME, 2, default_intrinsic_type, 0, 0},
        {MINUS_C_OPERATOR_NAME, 2, default_intrinsic_type, 0, 0},
        /* ISO 6.5.7 bitwise shift operators*/
        {LEFT_SHIFT_OPERATOR_NAME, 2, integer_to_overloaded_type, 0, 0},
        {RIGHT_SHIFT_OPERATOR_NAME, 2, integer_to_overloaded_type, 0, 0},
        /* ISO 6.5.8 relational operators,arithmetic or pointer types */
        {C_LESS_THAN_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        {C_GREATER_THAN_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        {C_LESS_OR_EQUAL_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        {C_GREATER_OR_EQUAL_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        /* ISO 6.5.9 equality operators, return 0 or 1*/
        {C_EQUAL_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        {C_NON_EQUAL_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        /* ISO 6.5.10 bitwise AND operator */
        {BITWISE_AND_OPERATOR_NAME, 2, integer_to_integer_type, typing_arithmetic_operator, 0},
        /* ISO 6.5.11 bitwise exclusive OR operator */
        {BITWISE_XOR_OPERATOR_NAME, 2, integer_to_integer_type, typing_arithmetic_operator, 0},
        /* ISO 6.5.12 bitwise inclusive OR operator */
        {BITWISE_OR_OPERATOR_NAME, 2, integer_to_integer_type, typing_arithmetic_operator, 0},
        /* ISO 6.5.13 logical AND operator */
        {C_AND_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        /* ISO 6.5.14 logical OR operator */
        {C_OR_OPERATOR_NAME, 2, overloaded_to_integer_type, 0, 0},
        /* ISO 6.5.15 conditional operator */
        {CONDITIONAL_OPERATOR_NAME, 3, default_intrinsic_type, 0, 0},
        /* ISO 6.5.16.1 simple assignment : ALREADY EXIST (FORTRAN)
           {ASSIGN_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0}, */
        /* ISO 6.5.16.2 compound assignments*/
        {MULTIPLY_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {DIVIDE_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {MODULO_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {PLUS_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {MINUS_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {LEFT_SHIFT_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {RIGHT_SHIFT_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {BITWISE_AND_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {BITWISE_XOR_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        {BITWISE_OR_UPDATE_OPERATOR_NAME, 2, default_intrinsic_type, typing_of_assign, 0},
        /* ISO 6.5.17 comma operator */
        {COMMA_OPERATOR_NAME, (INT_MAX), default_intrinsic_type, 0, 0},

        {BREAK_FUNCTION_NAME, 0, default_intrinsic_type, 0, 0},
        {CASE_FUNCTION_NAME, 0, default_intrinsic_type, 0, 0},
        {DEFAULT_FUNCTION_NAME, 0, default_intrinsic_type, 0, 0},
        {C_RETURN_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},

        /* intrinsic to handle C initialization */
        {BRACE_INTRINSIC, (INT_MAX) , default_intrinsic_type, no_typing, 0},

        /* #include <assert.h> */
        {ASSERT_FUNCTION_NAME, 3, overloaded_to_void_type,0,0},
        {ASSERT_FAIL_FUNCTION_NAME, 4, overloaded_to_void_type,0,0}, /* does not return */

        /* #include <complex.h> */

        {CACOS_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CACOSF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CACOSL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CASIN_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CASINF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CASINL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CATAN_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CATANF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CATANL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {C_CCOS_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CCOSF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CCOSL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {C_CSIN_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CSINF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CSINL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CTAN_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CTANF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CTANL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CACOSH_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CACOSHF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CACOSHL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CASINH_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CASINHF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CASINHL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CATANH_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CATANHF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CATANHL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CCOSH_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CCOSHF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CCOSHL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CSINH_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CSINHF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CSINHL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CTANH_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CTANHF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CTANHL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {C_CEXP_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CEXPF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CEXPL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {C_CLOG_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CLOGF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CLOGL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {C_CABS_OPERATOR_NAME, 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double, 0},
        {CABSF_OPERATOR_NAME, 1, complex_to_real_type, typing_function_complex_to_real, 0},
        {CABSL_OPERATOR_NAME, 1, longdoublecomplex_to_longdouble_type, typing_function_longdcomplex_to_longdouble, 0},
        {CPOW_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CPOWF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CPOWL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {C_CSQRT_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CSQRTF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CSQRTL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CARG_OPERATOR_NAME, 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double, 0},
        {CARGF_OPERATOR_NAME, 1, complex_to_real_type, typing_function_complex_to_real, 0},
        {CARGL_OPERATOR_NAME, 1, longdoublecomplex_to_longdouble_type, typing_function_longdcomplex_to_longdouble, 0},
        {CIMAG_OPERATOR_NAME, 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double, 0},
        {GCC_CIMAG_OPERATOR_NAME, 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double, 0},
        {CIMAGF_OPERATOR_NAME, 1, complex_to_real_type, typing_function_complex_to_real, 0},
        {CIMAGL_OPERATOR_NAME, 1, longdoublecomplex_to_longdouble_type, typing_function_longdcomplex_to_longdouble, 0},
        {CONJ_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CONJF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CONJL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CPROJ_OPERATOR_NAME, 1, doublecomplex_to_doublecomplex_type, typing_function_dcomplex_to_dcomplex, 0},
        {CPROJF_OPERATOR_NAME, 1, complex_to_complex_type, typing_function_complex_to_complex, 0},
        {CPROJL_OPERATOR_NAME, 1, longdoublecomplex_to_longdoublecomplex_type, typing_function_longdcomplex_to_longdcomplex, 0},
        {CREAL_OPERATOR_NAME, 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double, 0},
        {GCC_CREAL_OPERATOR_NAME, 1, doublecomplex_to_double_type, typing_function_dcomplex_to_double, 0},
        {CREALF_OPERATOR_NAME, 1,  complex_to_real_type, typing_function_complex_to_real, 0},
        {CREALL_OPERATOR_NAME, 1, longdoublecomplex_to_longdouble_type, typing_function_longdcomplex_to_longdouble, 0},

        /* #include <ctype.h> */

        {ISALNUM_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISALPHA_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISBLANK_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISCNTRL_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISDIGIT_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISGRAPH_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISLOWER_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISPRINT_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISPUNCT_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISSPACE_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISUPPER_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {ISXDIGIT_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {TOLOWER_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {TOUPPER_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        /* End ctype.h */
        //not found in standard C99 (in GNU C Library)
        {ISASCII_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {TOASCII_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {_TOLOWER_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {_TOUPPER_OPERATOR_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},

        /* Real type is void -> unsigned short int ** */
        {CTYPE_B_LOC_OPERATOR_NAME, 0, integer_to_integer_type, 0, 0},

        /* #include <errno.h> */
        /*  {"errno", 0, overloaded_to_integer_type, 0, 0}, */
        /* bits/errno.h */
        {__ERRNO_LOCATION_OPERATOR_NAME, 0, default_intrinsic_type, 0, 0},


        /* #include <fenv.h> */
        {FECLEAREXCEPT_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {FERAISEEXCEPT_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {FESETEXCEPTFLAG_FUNCTION_NAME, 2,  overloaded_to_integer_type, 0, 0},
        {FETESTEXCEPT_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {FEGETROUND_FUNCTION_NAME, 1, void_to_integer_type, 0, 0},
        {FESETROUND_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        // fenv_t *
        //{FESETENV_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        //{FEUPDATEENV_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},


        /* #include <float.h> */
        /* {"__flt_rounds", 1, void_to_integer_type, 0, 0}, */

        /* #include <inttypes.h> */
        {IMAXABS_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {IMAXDIV_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},

        /* #include <iso646.h> */

        /* {"_sysconf", 1, integer_to_integer_type, 0, 0},

           {"localeconv", 1, default_intrinsic_type, 0, 0},
           {"dcgettext", 3, default_intrinsic_type, 0, 0},
           {"dgettext", 2, default_intrinsic_type, 0, 0},
           {"gettext", 1, default_intrinsic_type, 0, 0},
           {"textdomain", 1, default_intrinsic_type, 0, 0},
           {"bindtextdomain", 2, default_intrinsic_type, 0, 0},
           {"wdinit", 1, void_to_integer_type, 0 ,0},
           {"wdchkind", 1, overloaded_to_integer_type, 0 ,0},
           {"wdbindf", 3, overloaded_to_integer_type, 0 ,0},
           {"wddelim", 3, default_intrinsic_type, 0, 0},
           {"mcfiller", 1, void_to_overloaded_type, 0, 0},
           {"mcwrap", 1, void_to_integer_type, 0 ,0},*/

        /* #include <limits.h> */

        /* #include <locale.h> */
        {SETLOCALE_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},

        /* #include <math.h> */

        {FPCLASSIFY_OPERATOR_NAME, 1, double_to_integer_type,typing_function_double_to_int, 0},
        {ISFINITE_OPERATOR_NAME, 1, double_to_integer_type, typing_function_double_to_int, 0},
        {ISINF_OPERATOR_NAME, 1, double_to_integer_type, typing_function_double_to_int, 0},
        {ISNAN_OPERATOR_NAME, 1, double_to_integer_type, typing_function_double_to_int, 0},
        {ISNANL_OPERATOR_NAME, 1, double_to_integer_type, typing_function_real_to_int, 0},
        {ISNANF_OPERATOR_NAME, 1, real_to_integer_type, typing_function_real_to_int, 0},
        {ISNORMAL_OPERATOR_NAME, 1, double_to_integer_type, typing_function_double_to_int, 0},
        {SIGNBIT_OPERATOR_NAME, 1, double_to_integer_type,typing_function_double_to_int, 0},
        {C_ACOS_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double , 0},
        {ACOSF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ACOSL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_ASIN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ASINF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ASINL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {C_ATAN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ATANF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ATANL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {C_ATAN2_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {ATAN2F_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {ATAN2L_OPERATOR_NAME, 2, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {C_COS_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {COSF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {COSL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {C_SIN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {SINF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {SINL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {C_TAN_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {TANF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {TANL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_ACOSH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ACOSHF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ACOSHL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_ASINH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ASINHF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ASINHL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_ATANH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ATANHF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ATANHL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_COSH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {COSHF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {COSHL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_SINH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {SINHF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {SINHL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_TANH_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {TANHF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {TANHL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_EXP_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {EXPF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {EXPL_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {EXP2_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {EXP2F_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {EXP2L_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {EXPM1_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {EXPM1F_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {EXPM1L_OPERATOR_NAME, 1, longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},    
        {FREXP_OPERATOR_NAME, 2, overloaded_to_double_type, 0, 0},/*?*/
        {ILOGB_OPERATOR_NAME, 1, double_to_integer_type, typing_function_double_to_int, 0},
        {ILOGBF_OPERATOR_NAME, 1, double_to_integer_type, typing_function_double_to_int, 0},
        {ILOGBL_OPERATOR_NAME, 1, longdouble_to_integer_type, typing_function_longdouble_to_int, 0},    
        {LDEXP_OPERATOR_NAME, 2, overloaded_to_double_type, 0, 0},
        {LDEXPF_OPERATOR_NAME, 2, overloaded_to_real_type, 0, 0},
        {LDEXPL_OPERATOR_NAME, 2, overloaded_to_longdouble_type, 0, 0},
        {C_LOG_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {LOGF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {LOGL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {C_LOG10_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {LOG10F_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {LOG10L_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {LOG1P_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {LOG1PF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {LOG1PL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {LOG2_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {LOG2F_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {LOG2L_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {LOGB_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {LOGBF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {LOGBL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {MODF_OPERATOR_NAME, 2, overloaded_to_double_type, 0, 0},
        {MODFF_OPERATOR_NAME, 2, overloaded_to_real_type, 0, 0},
        {SCALBN_OPERATOR_NAME, 2, overloaded_to_double_type, 0, 0},
        {SCALBNF_OPERATOR_NAME, 2, overloaded_to_real_type, 0, 0},
        {SCALBNL_OPERATOR_NAME, 2, overloaded_to_longdouble_type, 0, 0}, 
        {SCALB_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0}, //POSIX.1-2001
        {SCALBLN_OPERATOR_NAME, 2, overloaded_to_double_type, 0, 0},
        {SCALBLNF_OPERATOR_NAME, 2, overloaded_to_real_type, 0, 0},
        {SCALBLNL_OPERATOR_NAME, 2, overloaded_to_longdouble_type, 0, 0},
        {CBRT_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CBRTF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {CBRTL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0}, 
        {FABS_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {FABSF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {FABSL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {HYPOT_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {HYPOTF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {HYPOTL_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {POW_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {POWF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {POWL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {C_SQRT_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {SQRTF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {SQRTL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {ERF_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ERFF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ERFL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {ERFC_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ERFCF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ERFCL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {GAMMA_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0}, /* GNU C Library */
        {LGAMMA_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {LGAMMAF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {LGAMMAL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {TGAMMA_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {TGAMMAF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {CEIL_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {CEILF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {CEILL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {FLOOR_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {FLOORF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {FLOORL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},   
        {NEARBYINT_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {NEARBYINTF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {NEARBYINTL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},  
        {RINT_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {RINTF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {RINTL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},   
        {LRINT_OPERATOR_NAME, 1,  double_to_longinteger_type, typing_function_double_to_longint, 0}, 
        {LRINTF_OPERATOR_NAME, 1,  real_to_longinteger_type, typing_function_real_to_longint, 0},         
        {LRINTL_OPERATOR_NAME, 1,  longdouble_to_longinteger_type, typing_function_longdouble_to_longint, 0},
        {LLRINT_OPERATOR_NAME, 1,  double_to_longlonginteger_type, typing_function_double_to_longlongint, 0},
        {LLRINTF_OPERATOR_NAME, 1,  real_to_longlonginteger_type, typing_function_real_to_longlongint, 0},
        {LLRINTL_OPERATOR_NAME, 1,  longdouble_to_longlonginteger_type, typing_function_longdouble_to_longlongint, 0},
        {ROUND_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {ROUNDF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {ROUNDL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},   
        {LROUND_OPERATOR_NAME, 1,  double_to_longinteger_type, typing_function_double_to_longint, 0},   
        {LROUNDF_OPERATOR_NAME, 1,  real_to_longinteger_type, typing_function_real_to_longint, 0},
        {LROUNDL_OPERATOR_NAME, 1,  longdouble_to_longinteger_type, typing_function_longdouble_to_longint, 0},
        {LLROUND_OPERATOR_NAME, 1,  double_to_longlonginteger_type, typing_function_double_to_longlongint, 0},   
        {LLROUNDF_OPERATOR_NAME, 1,  real_to_longlonginteger_type, typing_function_real_to_longlongint, 0},
        {LLROUNDL_OPERATOR_NAME, 1,  longdouble_to_longlonginteger_type, typing_function_longdouble_to_longlongint, 0},
        {TRUNC_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {TRUNCF_OPERATOR_NAME, 1, real_to_real_type, typing_function_real_to_real, 0},
        {TRUNCL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},   
        {FMOD_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {FMODF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {FMODL_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},   
        {REMAINDER_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {REMAINDERF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {REMAINDERL_OPERATOR_NAME, 1,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},   
        {COPYSIGN_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {COPYSIGNF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {COPYSIGNL_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},
        {NAN_OPERATOR_NAME, 1, char_pointer_to_double_type, 0, 0},
        {NANF_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        {NANL_OPERATOR_NAME, 1, default_intrinsic_type, 0, 0},
        {NEXTAFTER_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {NEXTAFTERF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {NEXTAFTERL_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},
        {NEXTTOWARD_OPERATOR_NAME, 2,  overloaded_to_double_type,0, 0},
        {NEXTTOWARDF_OPERATOR_NAME, 2,  overloaded_to_real_type, 0, 0},
        {NEXTTOWARDL_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},
        {FDIM_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {FDIMF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {FDIML_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},
        {FMAX_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {FMAXF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {FMAXL_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},
        {FMIN_OPERATOR_NAME, 2, double_to_double_type, typing_function_double_to_double, 0},
        {FMINF_OPERATOR_NAME, 2, real_to_real_type, typing_function_real_to_real, 0},
        {FMINL_OPERATOR_NAME, 2,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},
        {FMA_OPERATOR_NAME, 3, double_to_double_type, typing_function_double_to_double, 0},
        {FMAF_OPERATOR_NAME, 3, real_to_real_type, typing_function_real_to_real, 0},
        {FMAL_OPERATOR_NAME, 3,  longdouble_to_longdouble_type, typing_function_longdouble_to_longdouble, 0},
        {ISGREATER_OPERATOR_NAME, 2, double_to_integer_type, typing_function_double_to_int, 0},
        {ISGREATEREQUAL_OPERATOR_NAME, 2, double_to_integer_type, typing_function_double_to_int, 0},
        {ISLESS_OPERATOR_NAME, 2, double_to_integer_type, typing_function_double_to_int, 0},
        {ISLESSEQUAL_OPERATOR_NAME, 2, double_to_integer_type, typing_function_double_to_int, 0},
        {ISLESSGREATER_OPERATOR_NAME, 2, double_to_integer_type, typing_function_double_to_int, 0},
        {ISUNORDERED_OPERATOR_NAME, 2, double_to_integer_type, typing_function_double_to_int, 0},
        /* End math.h */

        {J0_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {J1_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {JN_OPERATOR_NAME, 2, overloaded_to_double_type, 0, 0},

        {Y0_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {Y1_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},
        {YN_OPERATOR_NAME, 2, overloaded_to_double_type, 0, 0},

        {MATHERR_OPERATOR_NAME, 1, overloaded_to_integer_type, 0, 0},
        {SIGNIFICAND_OPERATOR_NAME, 1, double_to_double_type, typing_function_double_to_double, 0},

        {SIGFPE_OPERATOR_NAME, 2, default_intrinsic_type, 0, 0},
        {SINGLE_TO_DECIMAL_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {DOUBLE_TO_DECIMAL_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {EXTENDED_TO_DECIMAL_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {QUADRUPLE_TO_DECIMAL_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {DECIMAL_TO_SINGLE_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {DECIMAL_TO_DOUBLE_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {DECIMAL_TO_EXTENDED_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {DECIMAL_TO_QUADRUPLE_OPERATOR_NAME, 4, overloaded_to_void_type, 0, 0},
        {STRING_TO_DECIMAL_OPERATOR_NAME, 6, overloaded_to_void_type, 0, 0},
        {FUNC_TO_DECIMAL_OPERATOR_NAME, 9, overloaded_to_void_type, 0, 0},
        {FILE_TO_DECIMAL_OPERATOR_NAME, 8, overloaded_to_void_type, 0, 0},
        {SECONVERT_OPERATOR_NAME, 5, default_intrinsic_type, 0, 0},
        {SFCONVERT_OPERATOR_NAME, 5, default_intrinsic_type, 0, 0},
        {SGCONVERT_OPERATOR_NAME, 4, default_intrinsic_type, 0, 0},
        {ECONVERT_OPERATOR_NAME, 5, default_intrinsic_type, 0, 0},
        {FCONVERT_OPERATOR_NAME, 5, default_intrinsic_type, 0, 0},
        {GCONVERT_OPERATOR_NAME, 4, default_intrinsic_type, 0, 0},
        {QECONVERT_OPERATOR_NAME, 5, default_intrinsic_type, 0, 0},
        {QFCONVERT_OPERATOR_NAME, 5, default_intrinsic_type, 0, 0},
        {QGCONVERT_OPERATOR_NAME, 4, default_intrinsic_type, 0, 0},


        /* same name in stdlib
           {"ecvt", 4, default_intrinsic_type, 0, 0},
           {"fcvt", 4, default_intrinsic_type, 0, 0},
           {"gcvt", 3, default_intrinsic_type, 0, 0},
           {"strtod", 2, overloaded_to_double_type, 0, 0}, */

        /*#include <setjmp.h>*/

        {"setjmp", 1, overloaded_to_integer_type, 0, 0},
        {"__setjmp", 1, overloaded_to_integer_type, 0, 0},
        {"longjmp", 2, overloaded_to_void_type, 0, 0},
        {"__longjmp", 2, overloaded_to_void_type, 0, 0},
        {"sigsetjmp", 2, overloaded_to_integer_type, 0, 0},
        {"siglongjmp", 2, overloaded_to_void_type, 0, 0},


        /*#include <signal.h>*/
        {SIGNAL_OPERATOR_NAME, 2, default_intrinsic_type, 0, 0},
        {RAISE_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},

        /*#include <stdarg.h>*/
        /*#include <stdbool.h>*/
        /*#include <stddef.h>*/
        /*#include <stdint.h>*/
        /*#include <stdio.h>*/

        {REMOVE_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {RENAME_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {TMPFILE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {TMPNAM_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {FCLOSE_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {FFLUSH_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {FOPEN_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {FREOPEN_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {SETBUF_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {SETVBUF_FUNCTION_NAME, 4, overloaded_to_integer_type, 0, 0},
        {FPRINTF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {FSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {ISOC99_FSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {PRINTF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {SCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {ISOC99_SCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {SPRINTF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {SSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {ISOC99_SSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {VSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {ISOC99_VSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {VSSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {ISOC99_VSSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {VFSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {ISOC99_VFSCANF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {VFPRINTF_FUNCTION_NAME, 3, overloaded_to_integer_type, 0, 0},
        {VPRINTF_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {VSPRINTF_FUNCTION_NAME, 3, overloaded_to_integer_type, 0, 0},
        {FGETC_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {FGETS_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {FPUTC_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {FPUTS_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {GETC_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {_IO_GETC_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {PUTC_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {_IO_PUTC_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {GETCHAR_FUNCTION_NAME, 1, void_to_integer_type, 0, 0},
        {PUTCHAR_FUNCTION_NAME, 1, integer_to_integer_type, 0, 0},
        {GETS_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {PUTS_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {UNGETC_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {FREAD_FUNCTION_NAME, 4, default_intrinsic_type, 0, 0},
        {FWRITE_FUNCTION_NAME, 4, default_intrinsic_type, 0, 0},
        {FGETPOS_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {FSEEK_FUNCTION_NAME, 3, overloaded_to_integer_type, 0, 0},
        {FSETPOS_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {FTELL_FUNCTION_NAME,1, default_intrinsic_type, 0, 0},
        {C_REWIND_FUNCTION_NAME,1, default_intrinsic_type, 0, 0},
        {CLEARERR_FUNCTION_NAME,1, default_intrinsic_type, 0, 0},
        {FEOF_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {FERROR_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {PERROR_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {__FILBUF_FUNCTION_NAME, 1, overloaded_to_integer_type, 0, 0},
        {__FILSBUF_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {SETBUFFER_FUNCTION_NAME, 3, overloaded_to_void_type, 0, 0},
        {SETLINEBUF_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {SNPRINTF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {VSNPRINTF_FUNCTION_NAME, 4, overloaded_to_integer_type, 0, 0},
        {FDOPEN_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {CTERMID_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {FILENO_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {POPEN_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {CUSERID_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {TEMPNAM_FUNCTION_NAME, 2,default_intrinsic_type, 0, 0},
        /* same name in stdlib
           {"getopt", 3, overloaded_to_integer_type, 0, 0},
           {"getsubopt", 3, default_intrinsic_type, 0, 0},*/
        {GETW_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {PUTW_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {PCLOSE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {FSEEKO_FUNCTION_NAME, 3, overloaded_to_integer_type, 0, 0},
        {FTELLO_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {FOPEN64_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {FREOPEN64_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {TMPFILE64_FUNCTION_NAME, 1,default_intrinsic_type, 0, 0},
        {FGETPOS64_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {FSETPOS64_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {FSEEKO64_FUNCTION_NAME, 3, overloaded_to_integer_type, 0, 0},
        {FTELLO64_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},

        /* C IO system functions in man -S 2. The typing could be refined. See unistd.h */

        {C_OPEN_FUNCTION_NAME,    (INT_MAX), overloaded_to_integer_type, 0, 0}, /* 2 or 3 arguments */
        {CREAT_FUNCTION_NAME,     2,         overloaded_to_integer_type, 0, 0},
        {C_CLOSE_FUNCTION_NAME,   1,         integer_to_integer_type,    0, 0},
        {C_WRITE_FUNCTION_NAME,   2,         default_intrinsic_type, 0, 0}, /* returns ssize_t */
        {C_READ_FUNCTION_NAME,    2,         default_intrinsic_type, 0, 0},
        {USLEEP_FUNCTION_NAME,    1,         default_intrinsic_type, 0, 0},
	{LINK_FUNCTION_NAME,      2,         overloaded_to_integer_type, 0, 0},
	{SYMLINK_FUNCTION_NAME,   2,         overloaded_to_integer_type, 0, 0},
	{UNLINK_FUNCTION_NAME,    1,         overloaded_to_integer_type, 0, 0},

        /* {FCNTL_FUNCTION_NAME,     (INT_MAX), overloaded_to_integer_type, 0, 0},*/ /* 2 or 3 arguments of various types*/ /* located with fcntl.h */
        {FSYNC_FUNCTION_NAME,     2,         integer_to_integer_type, 0, 0},
        {FDATASYNC_FUNCTION_NAME, 2,         integer_to_integer_type, 0, 0},
        {IOCTL_FUNCTION_NAME,     (INT_MAX), overloaded_to_integer_type, 0, 0},
        {SELECT_FUNCTION_NAME,    5,         overloaded_to_integer_type, 0, 0},
        {PSELECT_FUNCTION_NAME,   6,         overloaded_to_integer_type, 0, 0},
        {STAT_FUNCTION_NAME,      2,         overloaded_to_integer_type, 0, 0},
        {FSTAT_FUNCTION_NAME,     2,         overloaded_to_integer_type, 0, 0},
        {LSTAT_FUNCTION_NAME,     2,         overloaded_to_integer_type, 0, 0},
        /*#include <stdlib.h>*/

        {ATOF_FUNCTION_NAME, 1, char_pointer_to_double_type, 0, 0},
        {ATOI_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {ATOL_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {ATOLL_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {STRTOD_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {STRTOF_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {STRTOLD_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {STRTOL_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {STRTOLL_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {STRTOUL_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {STRTOULL_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {RAND_FUNCTION_NAME, 1,  void_to_integer_type, 0, 0},
        {SRAND_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {CALLOC_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {FREE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {MALLOC_FUNCTION_NAME, 1, unsigned_integer_to_void_pointer_type, 0, 0},
        {REALLOC_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {ALLOCA_FUNCTION_NAME, 1, unsigned_integer_to_void_pointer_type, 0, 0},
        {ABORT_FUNCTION_NAME, 1, void_to_void_type, 0, 0},
        {ATEXIT_FUNCTION_NAME, 1, void_to_void_to_int_pointer_type, 0, 0},
        {EXIT_FUNCTION_NAME, 1, integer_to_void_type, 0, 0},
        {_EXIT_FUNCTION_NAME, 1, integer_to_void_type, 0, 0},
        {GETENV_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {SYSTEM_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {BSEARCH_FUNCTION_NAME, 5, default_intrinsic_type, 0, 0},
        {QSORT_FUNCTION_NAME, 4, default_intrinsic_type, 0, 0},
        {C_ABS_FUNCTION_NAME, 1, integer_to_integer_type, typing_function_int_to_int, 0},
        {LABS_FUNCTION_NAME, 1, longinteger_to_longinteger_type, typing_function_longint_to_longint, 0},
        {LLABS_FUNCTION_NAME, 1, longlonginteger_to_longlonginteger_type, typing_function_longlongint_to_longlongint, 0},
        {DIV_FUNCTION_NAME, 2, integer_to_overloaded_type, 0, 0},
        {LDIV_FUNCTION_NAME, 2, longinteger_to_overloaded_type, 0, 0},
        {LLDIV_FUNCTION_NAME, 2, longlonginteger_to_overloaded_type, 0, 0},
        {MBLEN_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {MBTOWC_FUNCTION_NAME, 3, overloaded_to_integer_type, 0, 0},
        {WCTOMB_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0},
        {MBSTOWCS_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {WCSTOMBS_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},


        //to check
        {POSIX_MEMALIGN_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {ATOQ_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {EXITHANDLE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {DRAND48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {ERAND48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {JRAND48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {LCONG48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {LRAND48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {MRAND48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {NRAND48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {SEED48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {SRAND48_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {PUTENV_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {SETKEY_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {SWAB_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {MKSTEMP_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {MKSTEMP64_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {A614_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {ECVT_FUNCTION_NAME, 4, default_intrinsic_type, 0, 0},
        {FCVT_FUNCTION_NAME, 4, default_intrinsic_type, 0, 0},
        {GCVT_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {GETSUBOPT_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {GRANTPT_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {INITSTATE_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {C_164A_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {MKTEMP_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {PTSNAME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {RANDOM_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0}, /* void -> long int */
        {REALPATH_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {SETSTATE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {SRANDOM_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {TTYSLOT_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {UNLOCKPT_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {VALLOC_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {DUP2_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {QECVT_FUNCTION_NAME, 4, default_intrinsic_type, 0, 0},
        {QFCVT_FUNCTION_NAME, 4, default_intrinsic_type, 0, 0},
        {QGCVT_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {GETCWD_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {GETEXECNAME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {GETLOGIN_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {GETOPT_FUNCTION_NAME, 3, default_intrinsic_type, 0, 0},
        {GETOPT_LONG_FUNCTION_NAME, 5, default_intrinsic_type, 0, 0},
        {GETOPT_LONG_ONLY_FUNCTION_NAME, 5, default_intrinsic_type, 0, 0},
        {GETPASS_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {GETPASSPHRASE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {GETPW_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {ISATTY_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {MEMALIGN_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {TTYNAME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {LLTOSTR_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {ULLTOSTR_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},

        /*#include <string.h>*/

        {MEMCPY_FUNCTION_NAME,3,default_intrinsic_type, 0, 0},
        {MEMMOVE_FUNCTION_NAME,3,default_intrinsic_type, 0, 0},
        {STRCPY_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRDUP_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {STRNCPY_FUNCTION_NAME,3,default_intrinsic_type, 0, 0},
        {STRCAT_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRNCAT_FUNCTION_NAME,3,default_intrinsic_type, 0, 0},
        {MEMCMP_FUNCTION_NAME,3,overloaded_to_integer_type, 0, 0},
        {STRCMP_FUNCTION_NAME,2,overloaded_to_integer_type, 0, 0},
        {STRCOLL_FUNCTION_NAME,2,overloaded_to_integer_type, 0, 0},
        {STRNCMP_FUNCTION_NAME,3,overloaded_to_integer_type, 0, 0},
        {STRXFRM_FUNCTION_NAME,3,overloaded_to_integer_type, 0, 0},
        {MEMCHR_FUNCTION_NAME,3,default_intrinsic_type, 0, 0},
        {STRCHR_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRCSPN_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRPBRK_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRRCHR_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRSPN_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRSTR_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {STRTOK_FUNCTION_NAME,2,default_intrinsic_type, 0, 0},
        {MEMSET_FUNCTION_NAME,3,default_intrinsic_type, 0, 0},
        {STRERROR_FUNCTION_NAME,1,integer_to_overloaded_type, 0, 0},
        {STRERROR_R_FUNCTION_NAME,3,default_intrinsic_type, 0, 0},
        {STRLEN_FUNCTION_NAME,1,default_intrinsic_type, 0, 0},

        /*#include <tgmath.h>*/
        /*#include <time.h>*/
        {TIME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {LOCALTIME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {DIFFTIME_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {NANOSLEEP_FUNCTION_NAME, 2, default_intrinsic_type, 0, 0},
        {GETTIMEOFDAY_FUNCTION_NAME, 2, overloaded_to_void_type, 0, 0}, // BSD-GNU
        {CLOCK_GETTIME_FUNCTION_NAME, 2, overloaded_to_integer_type, 0, 0}, // BSD-GNU
        {CLOCK_FUNCTION_NAME, 0, void_to_overloaded_type, 0, 0},
        {SECOND_FUNCTION_NAME, 0, void_to_overloaded_type, 0, 0}, //GFORTRAN

        /*#include <wchar.h>*/
        { FWPRINTF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        { FWSCANF_FUNCTION_NAME,  (INT_MAX), overloaded_to_integer_type, 0, 0},
        { SWPRINTF_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        { SWSCANF_FUNCTION_NAME,  (INT_MAX), overloaded_to_integer_type, 0, 0},
        { VFWPRINTF_FUNCTION_NAME,  3,       overloaded_to_integer_type, 0, 0},
        { VFWSCANF_FUNCTION_NAME,   3,       overloaded_to_integer_type, 0, 0},
        { VSWPRINTF_FUNCTION_NAME,  4,       overloaded_to_integer_type, 0, 0},
        { VSWSCANF_FUNCTION_NAME,   3,       overloaded_to_integer_type, 0, 0},
        { VWPRINTF_FUNCTION_NAME,   2,       overloaded_to_integer_type, 0, 0},
        { VWSCANF_FUNCTION_NAME,    2,       overloaded_to_integer_type, 0, 0},
        { WPRINTF_FUNCTION_NAME,  (INT_MAX), overloaded_to_integer_type, 0, 0},
        { WSCANF_FUNCTION_NAME,   (INT_MAX), overloaded_to_integer_type, 0, 0},
        { FGETWC_FUNCTION_NAME,     1,       default_intrinsic_type, 0, 0},
        { FGETWS_FUNCTION_NAME,     3,       default_intrinsic_type, 0, 0},
        { FPUTWC_FUNCTION_NAME,     2,       default_intrinsic_type, 0, 0},
        { FPUTWS_FUNCTION_NAME,     2,       default_intrinsic_type, 0, 0},
        { FWIDE_FUNCTION_NAME,      2,       overloaded_to_integer_type, 0, 0},
        { GETWC_FUNCTION_NAME,      1,       default_intrinsic_type, 0, 0},
        { GETWCHAR_FUNCTION_NAME,   0,       default_intrinsic_type, 0, 0},
        { PUTWC_FUNCTION_NAME,      2,       default_intrinsic_type, 0, 0},
        { PUTWCHAR_FUNCTION_NAME,   1,       default_intrinsic_type, 0, 0},
        { UNGETWC_FUNCTION_NAME,    2,       default_intrinsic_type, 0, 0},
        { WCSTOD_FUNCTION_NAME,     2,       overloaded_to_double_type, 0, 0},
        { WCSTOF_FUNCTION_NAME,     2,       overloaded_to_real_type, 0, 0},
        { WCSTOLD_FUNCTION_NAME,    2,       default_intrinsic_type, 0, 0},
        { WCSTOL_FUNCTION_NAME,     3,       default_intrinsic_type, 0, 0},
        { WCSTOLL_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WCSTOUL_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WCSTOULL_FUNCTION_NAME,   3,       default_intrinsic_type, 0, 0},
        { WCSCPY_FUNCTION_NAME,     2,       default_intrinsic_type, 0, 0},
        { WCSNCPY_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WMEMCPY_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WMEMMOVE_FUNCTION_NAME,   3,       default_intrinsic_type, 0, 0},
        { WCSCAT_FUNCTION_NAME,     2,       default_intrinsic_type, 0, 0},
        { WCSNCAT_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WCSCMP_FUNCTION_NAME,     2,       overloaded_to_integer_type, 0, 0},
        { WCSCOLL_FUNCTION_NAME,    2,       overloaded_to_integer_type, 0, 0},
        { WCSNCMP_FUNCTION_NAME,    3,       overloaded_to_integer_type, 0, 0},
        { WCSXFRM_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WMEMCMP_FUNCTION_NAME,    3,       overloaded_to_integer_type, 0, 0},
        { WCSCHR_FUNCTION_NAME,     2,       default_intrinsic_type, 0, 0},
        { WCSCSPN_FUNCTION_NAME,    2,       default_intrinsic_type, 0, 0},
        { WCSPBRK_FUNCTION_NAME,    2,       default_intrinsic_type, 0, 0},
        { WCSRCHR_FUNCTION_NAME,    2,       default_intrinsic_type, 0, 0},
        { WCSSPN_FUNCTION_NAME,     2,       default_intrinsic_type, 0, 0},
        { WCSSTR_FUNCTION_NAME,     2,       default_intrinsic_type, 0, 0},
        { WCSTOK_FUNCTION_NAME,     3,       default_intrinsic_type, 0, 0},
        { WMEMCHR_FUNCTION_NAME,    2,       default_intrinsic_type, 0, 0},
        { WCSLEN_FUNCTION_NAME,     1,       default_intrinsic_type, 0, 0},
        { WMEMSET_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WCSFTIME_FUNCTION_NAME,   4,       default_intrinsic_type, 0, 0},
        { BTOWC_FUNCTION_NAME,      1,       default_intrinsic_type, 0, 0},
        { WCTOB_FUNCTION_NAME,      1,       overloaded_to_integer_type, 0, 0},
        { MBSINIT_FUNCTION_NAME,    1,       overloaded_to_integer_type, 0, 0},
        { MBRLEN_FUNCTION_NAME,     3,       default_intrinsic_type, 0, 0},
        { MBRTOWC_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { WCRTOMB_FUNCTION_NAME,    3,       default_intrinsic_type, 0, 0},
        { MBSRTOWCS_FUNCTION_NAME,  4,       default_intrinsic_type, 0, 0},
        { WCSRTOMBS_FUNCTION_NAME,  4,       default_intrinsic_type, 0, 0},

        /*#include <wctype.h>*/

        {ISWALNUM_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWALPHA_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWBLANK_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWCNTRL_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWDIGIT_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWGRAPH_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWLOWER_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWPRINT_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWPUNCT_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWSPACE_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWUPPER_OPERATOR_NAME,  1, overloaded_to_integer_type, 0, 0},
        {ISWXDIGIT_OPERATOR_NAME, 1, overloaded_to_integer_type, 0, 0},
        {ISWCTYPE_OPERATOR_NAME,  2, overloaded_to_integer_type, 0, 0},
        {WCTYPE_OPERATOR_NAME,    1, default_intrinsic_type, 0, 0},
        {TOWLOWER_OPERATOR_NAME,  1, default_intrinsic_type, 0, 0},
        {TOWUPPER_OPERATOR_NAME,  1, default_intrinsic_type, 0, 0},
        {TOWCTRANS_OPERATOR_NAME, 2, default_intrinsic_type, 0, 0},
        {WCTRANS_OPERATOR_NAME,   1, default_intrinsic_type, 0, 0},

        /* netdb.h */
        {__H_ERRNO_LOCATION_OPERATOR_NAME, 0, default_intrinsic_type, 0, 0},


        /* #include <fcntl.h>*/

        {FCNTL_FUNCTION_NAME,  (INT_MAX), overloaded_to_integer_type, 0, 0},
        {DIRECTIO_FUNCTION_NAME, 2,       integer_to_integer_type, 0, 0},
        {OPEN64_FUNCTION_NAME, (INT_MAX), overloaded_to_integer_type, 0, 0},
        {CREAT64_FUNCTION_NAME,  2,       overloaded_to_integer_type, 0, 0},

        /* OMP */
        {OMP_IF_FUNCTION_NAME,        1,        default_intrinsic_type, 0, 0},
        {OMP_OMP_FUNCTION_NAME,       0,        default_intrinsic_type, 0, 0},
        {OMP_FOR_FUNCTION_NAME,       0,        default_intrinsic_type, 0, 0},
        {OMP_PRIVATE_FUNCTION_NAME,(INT_MAX),   default_intrinsic_type, 0, 0},
        {OMP_PARALLEL_FUNCTION_NAME,  0,        default_intrinsic_type, 0, 0},
        {OMP_REDUCTION_FUNCTION_NAME,(INT_MAX), default_intrinsic_type, 0, 0},

        /* BSD <err.h> */
        {ERR_FUNCTION_NAME,   (INT_MAX), overloaded_to_void_type, 0, 0},
        {ERRX_FUNCTION_NAME,  (INT_MAX), overloaded_to_void_type, 0, 0},
        {WARN_FUNCTION_NAME,  (INT_MAX), overloaded_to_void_type, 0, 0},
        {WARNX_FUNCTION_NAME, (INT_MAX), overloaded_to_void_type, 0, 0},
        {VERR_FUNCTION_NAME,    3, 	   overloaded_to_void_type, 0, 0},
        {VERRX_FUNCTION_NAME,   3, 	   overloaded_to_void_type, 0, 0},
        {VWARN_FUNCTION_NAME,   2, 	   overloaded_to_void_type, 0, 0},
        {VWARNX_FUNCTION_NAME,  2, 	   overloaded_to_void_type, 0, 0},

        /* F95 */
        {ALLOCATE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {DEALLOCATE_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {ETIME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {DTIME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},
        {CPU_TIME_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},

        /* F2003 */
        {C_LOC_FUNCTION_NAME, 1, default_intrinsic_type, 0, 0},

        /* PIPS run-time support for C code generation
         *
         * Source code located in validation/Hyperplane/run_time.src for the
         * time being.
         */
        {PIPS_C_MIN_OPERATOR_NAME, (INT_MAX), integer_to_integer_type,
            typing_function_int_to_int, 0},
        {PIPS_C_MAX_OPERATOR_NAME, (INT_MAX), integer_to_integer_type,
            typing_function_int_to_int, 0},

        /* assembly function */
        { ASM_FUNCTION_NAME, 1, overloaded_to_void_type },

        /* PIPS intrinsics to simulate various effects */
        {PIPS_MEMORY_BARRIER_OPERATOR_NAME, 0, void_to_void_type, 0, 0},
        {PIPS_IO_BARRIER_OPERATOR_NAME, 0, void_to_void_type, 0, 0},

        {NULL, 0, 0, 0, 0}
    };
    intrinsic_type_descriptor_mapping=hash_table_make(hash_string,sizeof(IntrinsicTypeDescriptorTable));
    for(IntrinsicDescriptor *p = &IntrinsicTypeDescriptorTable[0];p->name;++p) {
        if(!set_belong_p(module_list,p->name))
            register_intrinsic_type_descriptor(p);
    }
}


bool
bootstrap(string workspace)
{
  pips_debug(1, "bootstraping in workspace %s\n", workspace);

  if (db_resource_p(DBR_ENTITIES, ""))
    pips_internal_error("entities already initialized");

  /* Create all intrinsics, skipping user-defined one */
  set module_list = set_make(set_string);
  gen_array_t ml = db_get_module_list();
  for(int i=0; i < (int) gen_array_nitems(ml); i++)
    set_add_element(module_list, module_list, (char*)gen_array_item(ml,i));
  CreateIntrinsics(module_list);
  set_free(module_list);
  gen_array_free(ml);

  /* Creates the dynamic and static areas for the super global
   * arrays such as the logical unit array (see below).
   */
  CreateAreas();

  /* The current entity is unknown, but for a TOP-LEVEL:TOP-LEVEL
   * which is used to create the logical unit array for IO effects
   */
  CreateLogicalUnits();

  /* create hidden variables to modelize the abstract states defined by the libc:

     seed for random function package

     heap abstract state
  */
  CreateRandomSeed();
  CreateTimeSeed();
  CreateHeapAbstractState();
  /* Create hidden variable to modelize the abstract state of :
     temporary arry for memmove function. Molka Becher
  */
  CreateMemmoveAbstractState();

  /* Create the empty label */
  (void) make_entity(strdup(concatenate(TOP_LEVEL_MODULE_NAME,
                                        MODULE_SEP_STRING,
                                        LABEL_PREFIX,
                                        NULL)),
                     MakeTypeStatement(),
                     make_storage_rom(),
                     make_value(is_value_constant,
                                make_constant_litteral()),
                     DEFAULT_ENTITY_KIND);

  /* FI: I suppress the owner filed to make the database moveable */
  /* FC: the content must be consistent with pipsdbm/methods.h */
  DB_PUT_MEMORY_RESOURCE(DBR_ENTITIES, "", (char*) entity_domain);

  pips_debug(1, "bootstraping done\n");

  return true;
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

/* This array is pointed by FILE * pointers returned or used by fopen,
   fclose,... . The argument f must be the intrinsic fopen returning a
   FILE * or another function also returning a FILE *. So we do not
   have to synthesize the type FILE. */
entity MakeIoFileArray(entity f)
{
  entity io_files = FindOrCreateEntity(IO_EFFECTS_PACKAGE_NAME, IO_EFFECTS_IO_FILE_NAME);

  if(type_undefined_p(entity_type(io_files))) {
    /* FI: this initialization is usually performed in
       bootstrap.c, but it is easier to do it here because the
       IO_FILE type does not have to be built from scratch. */
    type rt = functional_result(type_functional(entity_type(f)));
    type ct = copy_type(type_to_pointed_type(rt)); // FI: no risk with typedef
    pips_assert("ct is a scalar type",
		ENDP(variable_dimensions(type_variable(ct))));
    variable_dimensions(type_variable(ct)) =
      CONS(DIMENSION,
	   make_dimension(int_to_expression(0),
			  /*
			    MakeNullaryCall
			    (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
			  */
			  int_to_expression(2000)
			  ),
	   NIL);
    entity_type(io_files) = ct;
    entity ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
				    IO_EFFECTS_PACKAGE_NAME);
    entity_storage(io_files) =
      make_storage(is_storage_ram,
		   make_ram(ent,
			    FindEntity(IO_EFFECTS_PACKAGE_NAME,
				       STATIC_AREA_LOCAL_NAME),
			    0, NIL));
    entity_initial(io_files) = make_value(is_value_unknown, UU);
  }
  return io_files;
}
