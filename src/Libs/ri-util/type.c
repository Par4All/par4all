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
/*
 * Modifications :
 * --------------
 *
 * Molka Becher (MB), June-July 2010
 *
 * Add of functions on Long Int, Long Long Int, Long Double and Long Double Complex types
 *
 */

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "misc.h"

#include "ri-util.h"
#include "text-util.h"

/* generation of types */

basic MakeBasicOverloaded()
{
    return(make_basic(is_basic_overloaded, NIL));
}

mode MakeModeReference()
{
    return(make_mode(is_mode_reference, NIL));
}

mode MakeModeValue()
{
    return(make_mode(is_mode_value, NIL));
}

type MakeTypeStatement()
{
    return(make_type(is_type_statement, NIL));
}

type MakeTypeUnknown()
{
    return(make_type(is_type_unknown, NIL));
}

type MakeTypeVoid()
{
    return(make_type(is_type_void, NIL));
}

type MakeTypeOverloaded()
{
  return MakeTypeVariable(make_basic_overloaded(), NIL);
}

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

type MakeTypeVariable(b, ld)
basic b;
cons * ld;
{
    return(make_type(is_type_variable, make_variable(b, ld,NIL)));
}

/* END_EOLE */

/*
 *
 */
basic MakeBasic(the_tag)
int the_tag;
{
    switch(the_tag)
    {
    case is_basic_int:
	return(make_basic(is_basic_int, UUINT(4)));
	break;
    case is_basic_float:
	return(make_basic(is_basic_float, UUINT(4)));
	break;
    case is_basic_logical:
	return(make_basic(is_basic_logical, UUINT(4)));
	break;
    case is_basic_complex:
	return(make_basic(is_basic_complex, UUINT(8)));
	break;
    case is_basic_overloaded:
	return(make_basic(is_basic_overloaded, UU));
	break;
    case is_basic_string:
	return(make_basic(is_basic_string, string_undefined));
	break;
    default:
	pips_internal_error("unexpected basic tag: %d",
		   the_tag);
	break;
    }

    return(basic_undefined);
}

/* functions on types */

type MakeTypeArray(basic b, cons * ld)
{
    return(make_type(is_type_variable, make_variable(b, ld,NIL)));
}

parameter MakeOverloadedParameter()
{
    return MakeAnyScalarParameter(is_basic_overloaded, 0);
}

parameter MakeIntegerParameter()
{
  return MakeAnyScalarParameter(is_basic_int, DEFAULT_INTEGER_TYPE_SIZE);
}

parameter MakeUnsignedIntegerParameter()
{
  return MakeAnyScalarParameter(is_basic_int, DEFAULT_INTEGER_TYPE_SIZE+ DEFAULT_UNSIGNED_TYPE_SIZE*10);
}


parameter MakeLongIntegerParameter() /* MB */
{
  return MakeAnyScalarParameter(is_basic_int, DEFAULT_LONG_INTEGER_TYPE_SIZE);
}

parameter MakeLongLongIntegerParameter() /* MB */
{
  return MakeAnyScalarParameter(is_basic_int, DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE);
}

parameter MakeRealParameter()
{
  return MakeAnyScalarParameter(is_basic_float, DEFAULT_REAL_TYPE_SIZE);
}

parameter MakeDoubleprecisionParameter()
{
  return MakeAnyScalarParameter(is_basic_float, DEFAULT_DOUBLEPRECISION_TYPE_SIZE);
}

parameter MakeQuadprecisionParameter() /* MB */
{
  return MakeAnyScalarParameter(is_basic_float, DEFAULT_QUADPRECISION_TYPE_SIZE);
}

parameter MakeLogicalParameter()
{
  return MakeAnyScalarParameter(is_basic_logical, DEFAULT_LOGICAL_TYPE_SIZE);
}

parameter MakeComplexParameter()
{
  return MakeAnyScalarParameter(is_basic_complex, DEFAULT_COMPLEX_TYPE_SIZE);
}

parameter MakeDoublecomplexParameter()
{
  return MakeAnyScalarParameter(is_basic_complex, DEFAULT_DOUBLECOMPLEX_TYPE_SIZE);
}

parameter MakeLongDoublecomplexParameter() /* MB */
{
  return MakeAnyScalarParameter(is_basic_complex, DEFAULT_LONGDOUBLECOMPLEX_TYPE_SIZE);
}

parameter MakeCharacterParameter()
{
  return make_parameter(MakeTypeArray(make_basic(is_basic_string,
	 make_value(is_value_constant,
		    make_constant(is_constant_int,
				  UUINT(DEFAULT_CHARACTER_TYPE_SIZE)))),
				      NIL),
			make_mode(is_mode_reference, UU),
			make_dummy_unknown());
}

/* For Fortran */
parameter MakeAnyScalarParameter(tag t, _int size)
{
  return make_parameter(MakeTypeArray(make_basic(t, UUINT(size)), NIL),
			make_mode_reference(),
			make_dummy_unknown());
}

/* this function creates a default fortran operator result, i.e. a zero
 * dimension variable with an overloaded basic type.
 */
type MakeOverloadedResult()
{
    return MakeAnyScalarResult(is_basic_overloaded, 0);
}
type MakeUnsignedIntegerResult()
{
    return MakeAnyScalarResult(is_basic_int, DEFAULT_UNSIGNED_TYPE_SIZE*10+
                                                DEFAULT_INTEGER_TYPE_SIZE);
}

type MakeVoidPointerResult()
{
    return MakeTypeArray(make_basic_pointer(make_type_void(NIL)), NIL);
}

type MakeIntegerResult()
{
    return MakeAnyScalarResult(is_basic_int, DEFAULT_INTEGER_TYPE_SIZE);
}

type MakeLongIntegerResult() /* MB */
{
    return MakeAnyScalarResult(is_basic_int, DEFAULT_LONG_INTEGER_TYPE_SIZE);
}

type MakeLongLongIntegerResult() /* MB */
{
    return MakeAnyScalarResult(is_basic_int, DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE);
}

type MakeRealResult()
{
    return MakeAnyScalarResult(is_basic_float, DEFAULT_REAL_TYPE_SIZE);
}

type MakeDoubleprecisionResult()
{
    return MakeAnyScalarResult(is_basic_float, DEFAULT_DOUBLEPRECISION_TYPE_SIZE);
}

type MakeQuadprecisionResult() /* MB */
{
    return MakeAnyScalarResult(is_basic_float, DEFAULT_QUADPRECISION_TYPE_SIZE);
}

type MakeLogicalResult()
{
    return MakeAnyScalarResult(is_basic_logical, DEFAULT_LOGICAL_TYPE_SIZE);
}

type MakeComplexResult()
{
    return MakeAnyScalarResult(is_basic_complex, DEFAULT_COMPLEX_TYPE_SIZE);
}

type MakeDoublecomplexResult()
{
    return MakeAnyScalarResult(is_basic_complex,
			       DEFAULT_DOUBLECOMPLEX_TYPE_SIZE);
}

type MakeLongDoublecomplexResult() /* MB */
{
    return MakeAnyScalarResult(is_basic_complex,
			       DEFAULT_LONGDOUBLECOMPLEX_TYPE_SIZE);
}

type MakeCharacterResult()
{
  return MakeTypeArray(make_basic(is_basic_string,
	 make_value(is_value_constant,
		    make_constant(is_constant_int,
				  UUINT(DEFAULT_CHARACTER_TYPE_SIZE)))),
		       NIL);
}

type MakeAnyScalarResult(tag t, _int size)
{
    return MakeTypeArray(make_basic(t, UUINT(size)), NIL);
}



/* similar as type_equals but by passes typedefs
 * FI Warning: current version only compares ultimate_types
 * but check the various typedef that follows.
 *
 * typedef int foo;
 *
 * type_equal_p(int, foo)==true, because foo is simply a renaming for
 * int, but note that type_int_p(entity_type(foo)) returns
 * false. Hence, type_int_p(t1) and type_equal_p(t1,t2) does not imply
 * type_int_p(t2), which may lead to funny bugs.
 *
 * typedef struct a {
 *     int x;
 *     int y;
 * } a_t;
 *
 * typedef struct b {
 *     int x;
 *     int y;
 * } b_t;
 *
 * type_equal_p(a_t, b_t)==false, because the underlying structures
 * have different names and because the C type system does not use
 * structural equivalence.
 *
 * typedef struct  {
 *     int x;
 *     int y;
 * } c_t;
 *
 * typedef struct  {
 *     int x;
 *     int y;
 * } d_t;
 *
 * type_EQUAL_P(c_t, d_t)==FALSE because structures (or unions or
 * enums) with implicit names receive each a unique name.
 */
bool same_type_p(type t1, type t2)
{
  t1= ultimate_type(t1);
  t2= ultimate_type(t2);
  return type_equal_p(t1,t2);
}

/* Warning: the lengths of string basics are not checked!!!
 * string_type_size() could be used but it is probably not very robust.
 *
 * typedef int foo[n+n];
 *
 * typedef int fii[2*n];
 *
 * type_equal_p(foo, fii)==undefined, because it depends on their
 * respective stores. FI: I do not know what's implemented, see
 * variable_equal_p().
 */
/* the unknown type is not handled in a satisfactory way: in some
   sense, it should be declared equal to any other type but the
   undefined type; currently unknown is equal to unknown, and
   different from other types.

   undefined types could also be seen in a different way, as really
   undefined values; so if t1 or t2 is undefined, the procedure
   should abort; but type_undefined is considered equal to type_undefined.

   Francois Irigoin, 10 March 1992
*/
/* If strict_p, a typedef type is not equal to its definition (default value).
 *
 * If qualifier_p, qualifiers must be equal (default value).
 *
 * This function should be static and used only indirectly via
 * type_equal_p(), etc.
 */
bool generic_type_equal_p(type t1, type t2, bool strict_p, bool qualifier_p)
{
  bool tequal = false;

  if(t1 == t2)
    return true;
  else if (t1 == type_undefined && t2 != type_undefined)
    return false;
  else if (t1 != type_undefined && t2 == type_undefined)
    return false;
  else if (type_tag(t1) != type_tag(t2))
    return false;

  /* assertion: t1 and t2 are defined and have the same tag */
  switch(type_tag(t1)) {
  case is_type_statement:
    return true;
  case is_type_area:
    return area_equal_p(type_area(t1), type_area(t2));
  case is_type_variable:
    tequal = generic_variable_equal_p(type_variable(t1), type_variable(t2), strict_p, qualifier_p);
    return tequal;
  case is_type_functional:
    return generic_functional_equal_p(type_functional(t1), type_functional(t2), strict_p, qualifier_p);
  case is_type_unknown:
    return true;
  case is_type_void:
    return true;
  default:
    pips_internal_error("unexpected tag %d", type_tag(t1));
  }

  return false; /* just to avoid a warning */
}

bool type_equal_p(type t1, type t2)
{
  return generic_type_equal_p(t1, t2, true, true);
}

bool type_equal_up_to_qualifiers_p(type t1, type t2)
{
  return generic_type_equal_p(t1, t2, true, false);
}

bool ultimate_type_equal_p(type t1, type t2)
{
  type ut1 = ultimate_type(t1);
  type ut2 = ultimate_type(t2);
  return generic_type_equal_p(ut1, ut2, false, true);
}

/* Expand typedefs before the type comparison. */
bool concrete_type_equal_p(type t1, type t2)
{
  type ct1 = compute_basic_concrete_type(t1);
  type ct2 = compute_basic_concrete_type(t2);
  bool equal_p = generic_type_equal_p(ct1, ct2, false, true);
  free_type(ct1), free_type(ct2);
  return equal_p;
}

/* assume that a pointer to type x is equal to a 1-D array of x */
bool array_pointer_type_equal_p(type t1, type t2)
{
  bool equal_p = true;
  if(!type_equal_up_to_qualifiers_p(t1,t2)) {
    if(pointer_type_p(t1) && array_type_p(t2)) {
      type pt = type_to_pointed_type(t1);
      basic pb = variable_basic(type_variable(pt));
      variable etv = type_variable(t2);
      basic eb = variable_basic(etv);
      int d = (int) gen_length(variable_dimensions(etv));
      equal_p = (d==1 && basic_equal_p(pb, eb));
    }
    else if(pointer_type_p(t2) && array_type_p(t1)) {
      type pt = type_to_pointed_type(t2);
      basic pb = variable_basic(type_variable(pt));
      variable etv = type_variable(t1);
      basic eb = variable_basic(etv);
      int d = (int) gen_length(variable_dimensions(etv));
      equal_p = (d==1 && basic_equal_p(pb, eb));
    }
    else
      equal_p = false;
  }
  return equal_p;
}

/* is "et" the type of an element of an array of type "at"? */
bool array_element_type_p(type at, type et)
{
  bool equal_p = false;

  if(array_type_p(at)) {
    if(type_variable_p(et)) {
      basic ab = variable_basic(type_variable(at));
      basic eb = variable_basic(type_variable(et));
      equal_p = basic_equal_p(ab, eb);
    }
  }

  return equal_p;
}

/* Same as above, but resolve typedefs first. */
bool concrete_array_pointer_type_equal_p(type t1, type t2)
{
  type ct1 = compute_basic_concrete_type(t1);
  type ct2 = compute_basic_concrete_type(t2);
  bool equal_p = array_pointer_type_equal_p(ct1, ct2);
  free_type(ct1), free_type(ct2);
  return equal_p;
}

type make_scalar_integer_type(_int n)
{
    type t = make_type(is_type_variable,
		       make_variable(make_basic(is_basic_int, UUINT(n)), NIL,NIL));
    return t;
}

type make_scalar_complex_type(_int n)
{
    type t = make_type(is_type_variable,
		       make_variable(make_basic(is_basic_complex, UUINT(n)), NIL,NIL));
    return t;
}

bool area_equal_p(area a1, area a2)
{
    if(a1 == a2)
	return true;
    else if (a1 == area_undefined && a2 != area_undefined)
	return false;
    else if (a1 != area_undefined && a2 == area_undefined)
	return false;
    else
	/* layouts are independent ? */
	return (area_size(a1) == area_size(a2));
}

bool
dimension_equal_p(dimension d1, dimension d2)
{
    return /* same values */
        expression_equal_p(dimension_lower(d1), dimension_lower(d2)) &&
        expression_equal_p(dimension_upper(d1), dimension_upper(d2)) ;
}

bool dimensions_equal_p(list dims1, list dims2) {
    return gen_equals(dims1,dims2,(gen_eq_func_t)dimension_equal_p);
}

bool qualifiers_equal_p(list dims1, list dims2) {
    return gen_equals(dims1,dims2,(gen_eq_func_t)qualifier_equal_p);
}         

bool generic_variable_equal_p(variable v1, variable v2, bool strict_p, bool qualifier_p)
{
  if(v1 == v2)
      return true;
  else if (v1 == variable_undefined && v2 != variable_undefined)
      return false;
  else if (v1 != variable_undefined && v2 == variable_undefined)
      return false;
  else {
      /* must check basic, dimension and qualifiers */
      return 
	generic_basic_equal_p(variable_basic(v1), variable_basic(v2), strict_p)
	&& dimensions_equal_p(variable_dimensions(v1), variable_dimensions(v2))
	&& (!qualifier_p || qualifiers_equal_p(variable_qualifiers(v1), variable_qualifiers(v2))) ;

      list ld1 = variable_dimensions(v1);
      list ld2 = variable_dimensions(v2);

      if(ld1==NIL && ld2==NIL)
	  return true;
      else
      {
	  /* dimensions should be checked, but it's hard: the only
	     Fortran requirement is that the space allocated in
	     the callers is bigger than the space used in the
	     callee; stars represent any strictly positive integer;
	     we do not know if v1 is the caller type or the callee type;
	  I do not know what should be done; FI */
	  /* FI: I return false because the exact test should never be useful
	     in the parser; 1 February 1994 */
	  /* FC: I need this in the prettyprinter... */
	  int l1 = gen_length(ld1), l2 = gen_length(ld2);
	  if (l1!=l2)
	      return false;
	  for (; ld1; POP(ld1), POP(ld2))
	  {
	      dimension d1 = DIMENSION(CAR(ld1)), d2 = DIMENSION(CAR(ld2));
	      if (!dimension_equal_p(d1, d2))
		  return false;
	  }
      }
  }
  return true;
}

bool variable_equal_p(variable v1, variable v2)
{
  return generic_variable_equal_p(v1, v2, true, true);
}

bool generic_basic_equal_p(basic b1, basic b2, bool strict_p)
{
  if(b1 == b2)
    return true;
  else if (b1 == basic_undefined && b2 != basic_undefined)
    return false;
  else if (b1 != basic_undefined && b2 == basic_undefined)
    return false;
  else if (basic_tag(b1) != basic_tag(b2)) {
    if(strict_p)
      return false;
    else
      return same_basic_p(b1, b2);
  }

  /* assertion: b1 and b2 are defined and have the same tag
     (see previous tests) */

  switch(basic_tag(b1)) {
  case is_basic_int:
    return basic_int(b1) == basic_int(b2);
  case is_basic_float:
    return basic_float(b1) == basic_float(b2);
  case is_basic_logical:
    return basic_logical(b1) == basic_logical(b2);
  case is_basic_overloaded:
    return true;
  case is_basic_complex:
    return basic_complex(b1) == basic_complex(b2);
  case is_basic_bit: {
    symbolic s1 = basic_bit(b1);
    symbolic s2 = basic_bit(b2);

    /* FI: there are two definition of equality, one more strict via
       the expression, and one more practical via the constant. But
       the constant is not always defined. And the Newgen declaration
       of symbolic looks wrong with "float:int"... */

    expression e1 = symbolic_expression(s1);
    expression e2 = symbolic_expression(s2);
    return expression_equal_p(e1, e2);
  }
  case is_basic_pointer:
    {
      type t1 = basic_pointer(b1);
      type t2 = basic_pointer(b2);

      return (type_void_p(t1) && type_void_p(t2)) ||
             (type_variable_p(t1) && type_variable_p(t2) &&
              generic_basic_equal_p(variable_basic(type_variable(t1)),
				    variable_basic(type_variable(t2)), strict_p));
    }
  case is_basic_derived:
    {
      entity e1 = basic_derived(b1);
      entity e2 = basic_derived(b2);

      return e1==e2;
    }
  case is_basic_string:
    /* Do we want string types to be equal only if lengths are equal?
     * I do not think so
     */
    /*
      pips_internal_error("string type comparison not implemented");
    */
    /* could be a star or an expression; a value_equal_p() is needed! */
    return true;
  case is_basic_typedef:
    /* FI->BC (?): suffixes _p should be removed */
    return basic_typedef_p(b2)
      && same_entity_p(basic_typedef(b1),basic_typedef(b2));
  default: pips_internal_error("unexpected tag %d", basic_tag(b1));
  }
  return false; /* just to avoid a warning */
}

bool basic_equal_p(basic b1, basic b2)
{
  return generic_basic_equal_p(b1, b2, true);
}

/* check if two basics are similar. That is if they are equal modulo typedefs */
bool same_basic_p(basic b1, basic b2)
{
  if( basic_typedef_p(b1) )
    {
      type t1 = ultimate_type( entity_type(basic_typedef(b1)) );
      b1 = variable_basic(type_variable(t1));
    }

  if( basic_typedef_p(b2) )
    {
      type t2 = ultimate_type( entity_type(basic_typedef(b2)) );
      b2 = variable_basic(type_variable(t2));
    }
  return basic_equal_p(b1,b2);

}

bool generic_functional_equal_p(functional f1, functional f2, bool strict_p,
				bool qualifier_p)
{
    if(f1 == f2)
	return true;
    else if (f1 == functional_undefined && f2 != functional_undefined)
	return false;
    else if (f1 != functional_undefined && f2 == functional_undefined)
	return false;
    else {
	list lp1 = functional_parameters(f1);
	list lp2 = functional_parameters(f2);

	if(gen_length(lp1) != gen_length(lp2))
	    return false;

	for( ; !ENDP(lp1); POP(lp1), POP(lp2)) {
	    parameter p1 = PARAMETER(CAR(lp1));
	    parameter p2 = PARAMETER(CAR(lp2));

	    if(!generic_parameter_equal_p(p1, p2, strict_p, qualifier_p))
		return false;
	}

	return generic_type_equal_p(functional_result(f1), functional_result(f2), strict_p, qualifier_p);
    }
}

bool functional_equal_p(functional f1, functional f2)
{
  return generic_functional_equal_p(f1, f2, true, true);
}

bool generic_parameter_equal_p(parameter p1, parameter p2, bool strict_p,
			       bool qualifier_p)
{
    if(p1 == p2)
	return true;
    else if (p1 == parameter_undefined && p2 != parameter_undefined)
	return false;
    else if (p1 != parameter_undefined && p2 == parameter_undefined)
	return false;
    else
      return generic_type_equal_p(parameter_type(p1), parameter_type(p2), strict_p, qualifier_p)
	    && mode_equal_p(parameter_mode(p1), parameter_mode(p2));
}

bool parameter_equal_p(parameter p1, parameter p2)
{
  return generic_parameter_equal_p(p1, p2, true, true);
}

bool mode_equal_p(mode m1, mode m2)
{
    if(m1 == m2)
	return true;
    else if (m1 == mode_undefined && m2 != mode_undefined)
	return false;
    else if (m1 != mode_undefined && m2 == mode_undefined)
	return false;
    else
	return mode_tag(m1) == mode_tag(m2);
}

int string_type_size(basic b)
{
    int size = -1;
    value v = basic_string(b);
    constant c = constant_undefined;

    switch(value_tag(v)) {
    case is_value_constant:
      c = value_constant(v);
      if(constant_int_p(c))
	size = constant_int(c);
      else
	pips_internal_error("Non-integer constant to size a string");
      break;
    case is_value_unknown:
      /* The size may be unknown as in CHARACTER*(*) */
      /* No way to check it really was a '*'? */
	size = -1;
      break;
    default:
	pips_internal_error("Non-constant value to size a string");
    }

    return size;
}

/* See also SizeOfElements() */
int basic_type_size(basic b)
{
    int size = -1;

    switch(basic_tag(b)) {
    case is_basic_int: size = basic_int(b);
	break;
    case is_basic_float: size = basic_float(b);
	break;
    case is_basic_logical: size = basic_logical(b);
	break;
    case is_basic_overloaded:
	pips_internal_error("undefined for type overloaded");
	break;
    case is_basic_complex: size = basic_complex(b);
	break;
    case is_basic_string:
      /* pips_internal_error("undefined for type string"); */
      size = string_type_size(b);
	break;
    case is_basic_pointer:
      size = DEFAULT_POINTER_TYPE_SIZE;
      break;
    default: size = basic_int(b);
	pips_internal_error("ill. tag %d", basic_tag(b));
	break;
    }

    return size;
}


/*
 * See basic_of_expression() which is much more comprehensive
 * Intrinsic overloading is not resolved!
 *
 * IO statements contain call to labels of type statement. An
 * undefined_basic is returned for such expressions.
 *
 * WARNING: a pointer to an existing data structure is returned.
 */
basic expression_basic(expression expr)
{
    syntax the_syntax=expression_syntax(expr);
    basic b = basic_undefined;

    switch(syntax_tag(the_syntax))
    {
    case is_syntax_reference:
	b = entity_basic(reference_variable(syntax_reference(the_syntax)));
	break;
    case is_syntax_range:
	/* should be int */
	b = expression_basic(range_lower(syntax_range(the_syntax)));
	break;
    case is_syntax_call:
	/*
	 * here is a little problem with pips...  every intrinsics are
	 * overloaded, what is not exactly what is desired...
	 */
        return(entity_basic(call_function(syntax_call(the_syntax))));
	break;
    case is_syntax_cast:
      {
	cast c = syntax_cast(the_syntax);
	type t = cast_type(c);
	type ut = ultimate_type(t);
	b = variable_basic(type_variable(ut));
	pips_assert("Type is \"variable\"", type_variable_p(ut));
      break;
      }
    case is_syntax_sizeofexpression:
      {
	/* How to void a memory leak? Where can we find a basic int? A static variable? */
	b = make_basic(is_basic_int, (void *) 4);
      break;
      }
    default:
	pips_internal_error("unexpected syntax tag");
	break;
    }

    return b;
}

dimension dimension_dup(dimension d)
{
    return(make_dimension(copy_expression(dimension_lower(d)),
			  copy_expression(dimension_upper(d))));
}

list ldimensions_dup(list l)
{
    list result = NIL ;

    MAPL(cd,
     {
	 result = CONS(DIMENSION, dimension_dup(DIMENSION(CAR(cd))),
		       result);
     },
	 l);

    return(gen_nreverse(result));
}

dimension FindIthDimension(entity e, int i)
{
    cons * pc;

    if (!type_variable_p(entity_type(e)))
	pips_internal_error("not a variable");

    if (i <= 0)
	pips_internal_error("invalid dimension");

    pc = variable_dimensions(type_variable(entity_type(e)));

    while (pc != NULL && --i > 0)
	pc = CDR(pc);

    if (pc == NULL)
	pips_internal_error("not enough dimensions");

    return(DIMENSION(CAR(pc)));
}

/*
 * returns a string defining a type.
 */
string type_to_string(type t)
{
    switch (type_tag(t))
    {
    case is_type_statement:
	return "statement";
    case is_type_area:
	return "area";
    case is_type_variable:
	return "variable";
    case is_type_functional:
	return "functional";
    case is_type_varargs:
	return "varargs";
    case is_type_unknown:
	return "unknown";
    case is_type_void:
	return "void";
    case is_type_struct:
	return "struct";
    case is_type_union:
	return "union";
    case is_type_enum:
	return "enum";
    default: break;
    }

    pips_internal_error("unexpected type: 0x%x (tag=%d)",
			t,
			type_tag(t));

    return(string_undefined); /* just to avoid a gcc warning */
}

string safe_type_to_string(type t)
{
  if(type_undefined_p(t))
    return "undefined type";
  else
    return type_to_string(t);
}

/* Provide a full ASCII description of type "t"
 *
 * FI: I am not sure about the language used.
 */
string type_to_full_string_definition(type t)
{
  debug_on("PRETTYPRINT_DEBUG_LEVEL");
  string s = words_to_string(words_type(t, NIL, false));
  debug_off();
  return s;
}

/* SG: I don't understand the previous functions */
string string_of_type(const type t)
{
  list wl = words_type(t, NIL, false);
  string s = words_to_string(wl);
  FOREACH(STRING,s,wl) free(s);
  gen_free_list(wl);
  return s;
}

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

/*
 * returns the string to declare a basic type.
 */
 string basic_to_string(basic b)
{
  /* Nga Nguyen, 19/09/2003: To not rewrite the same thing, I use the words_basic() function*/
  return list_to_string(words_basic(b, NIL));
}

/* SG: added so that the basic of a dereferenced pointer is the basic of the dereferenced value
BC : modified because it was assumed that each dimension was of pointer type which is not
true when tab is declared int ** tab[10] and the expression is tab[3][4][5].
*/
static basic basic_and_indices_to_basic(basic b, list indices,bool ultimate_p)
{
    bool finished = false;
    for(list l_ind = indices ; !ENDP(l_ind) && !finished ; POP(l_ind) )
    {
        if(basic_pointer_p(b))
        {
            type t = basic_pointer(b);
            if(type_variable_p(t)) {
                basic bt =
                    copy_basic(variable_basic(type_variable(
                                    ultimate_p?ultimate_type(t):t)));
                free_basic(b);
                b=bt;
            }
            else if(type_functional_p(t)) {
                /* The expression can denote a function, because a
                   function is equivalent to a pointer towards a
                   function */
                /* FI: we might as well return a pointer to a
                   function */
                ;
            }
            else
                pips_internal_error("Unexpected basic kind");
        }
        else
        {
            /* the basic type is reached. Should I add pips_assert("basic reached, it should be the last index\n", ENDP(CAR(l_ind)))? */
            finished = true;
        }
    }
    return b;
}


/* basic basic_of_any_expression(expression exp, bool apply_p): Makes
 * a basic of the same basic as the expression "exp" if "apply_p" is
 * FALSE. If "apply_p" is true and if the expression returns a
 * function, then return the resulting type of the function.
 *
 * WARNING: a new basic object is allocated
 *
 *  PREFER (???) expression_basic
 *
 */
basic some_basic_of_any_expression(expression exp, bool apply_p, bool ultimate_p)
{
  syntax sy = expression_syntax(exp);
  basic b = basic_undefined;

  ifdebug(6){
    pips_debug(6, "begins with apply_p=%s and expression ", bool_to_string(apply_p));
    print_expression(exp);
    pips_debug(6, "\n");
  }

  switch(syntax_tag(sy)) {
  case is_syntax_reference:
    {
      b = basic_of_any_reference(syntax_reference(sy),apply_p,ultimate_p);
      break;
    }
  case is_syntax_call:
    b = basic_of_call(syntax_call(sy), apply_p, ultimate_p);
    break;
  case is_syntax_range:
    /* Well, let's assume range are well formed... */
    b = basic_of_expression(range_lower(syntax_range(sy)));
    break;
  case is_syntax_cast:
    {
      type t = cast_type(syntax_cast(sy));

      if (type_tag(t) != is_type_variable) {
	if(type_void_p(t))
	  /* This happens with the assert macro... but could happen
	     anywhere as in (void) printf(...); */
	  /* FI: I cannot think of anything better... */
	  b = basic_undefined;
	else
	  pips_internal_error("Bad reference type tag %d",type_tag(t));
      }
      else
	b = copy_basic(variable_basic(type_variable(t)));
      break;
    }
  case is_syntax_sizeofexpression:
    /* SG: following code fragment seems wrong to me
    {
      sizeofexpression se = syntax_sizeofexpression(sy);
      if (sizeofexpression_type_p(se))
       {
         type t = sizeofexpression_type(se);
         if (type_tag(t) != is_type_variable)
           pips_internal_error("Bad reference type tag %d",type_tag(t));
         b = copy_basic(variable_basic(type_variable(t)));
       }
      else
       {
         b = some_basic_of_any_expression(sizeofexpression_expression(se), apply_p, ultimate_p);
       }
       */

      b= make_basic_int(DEFAULT_INTEGER_TYPE_SIZE);
      break;

  case is_syntax_subscript:
    {
      b = some_basic_of_any_expression(subscript_array(syntax_subscript(sy)), apply_p, ultimate_p);
      /* depending on the depth of the subscript, we should change the basic */
      b = basic_and_indices_to_basic(b,subscript_indices(syntax_subscript(sy)),ultimate_p);
      break;
    }
  case is_syntax_application:
    {
      b = basic_of_any_expression(application_function(syntax_application(sy)), apply_p);
      break;
    }
  case is_syntax_va_arg:
    {
      /* the second argument is the type of the returned object */
      sizeofexpression sofe = SIZEOFEXPRESSION(CAR(CDR(syntax_va_arg(sy))));

      if(sizeofexpression_type_p(sofe)) {
	type t = ultimate_type(sizeofexpression_type(sofe));

	if(type_variable_p(t))
	  b = copy_basic(variable_basic(type_variable(t)));
	else
	  pips_internal_error("Not implemented");
      }
      else {
	expression e = sizeofexpression_expression(sofe);
	b = basic_of_any_expression(e, true);
	pips_internal_error("expression not expected here");
      }
      break;
    }
  default:
    pips_internal_error("Bad syntax tag %d", syntax_tag(sy));
    /* Never go there... */
    b = make_basic(is_basic_overloaded, UUINT(4));
  }

  pips_debug(6, "returns with %s\n", basic_to_string(b));

  return b;
}


basic basic_of_any_expression(expression exp, bool apply_p)
{
  return some_basic_of_any_expression(exp, apply_p, true);
}

/* basic basic_of_expression(expression exp): Makes a basic of the same
 * basic as the expression "exp". Indeed, "exp" will be assigned to
 * a temporary variable, which will have the same declaration as "exp".
 *
 * Does not work if the expression is a reference to a functional
 * entity, as may be the case in a Fortran call or a C functional
 * pointer initialization. For C, a pointer to a functional should
 * be/is returned.
 *
 * WARNING: a new basic object is allocated
 *
 *  PREFER (???) expression_basic
 *
 */
basic basic_of_expression(expression exp)
{
  return basic_of_any_expression(exp, false);
}

/**
 * Retrieves the basic of a reference in a newly allocated basic object
 *
 * @param r reference we want the basic of
 *
 * @return allocated basic of the reference
 */
basic basic_of_any_reference(reference r, bool apply_p, bool ultimate_p) {
  basic b = basic_undefined;
  entity v = reference_variable(r);
  type vt = entity_type(v);

  /* When called from the parser, the entity type may not yet be
   stored in the field type. This happens when
   simplify_C_expression is called for initialization
   expressions which are grouped in one statement. */
  if(!type_undefined_p(vt)) {
    type exp_type = ultimate_p ? ultimate_type(vt) : copy_type(entity_type(v));
    list l_dim = NIL;

    if(apply_p) {
      if(!type_functional_p(exp_type))
        pips_internal_error("Bad reference type tag %d \"%s\"",
            type_tag(exp_type));
      else {
        type rt = functional_result(type_functional(exp_type));
        type urt = ultimate_p ? ultimate_type(rt) : copy_type(rt);

        if(type_variable_p(urt))
          b = copy_basic(variable_basic(type_variable(urt)));
          else
          pips_internal_error("Unexpected type tag %s", type_to_string(urt));
      }
    } else {
      if(type_variable_p(exp_type)) {
        b = copy_basic(variable_basic(type_variable(exp_type)));

        /* BC : if the variable has dimensions, then it's an array name which is converted
         into a pointer itself. And each dimension is converted into a pointer on the next one.
         (except in a few cases which should be handled in basic_of_call)
         to be verified or done
         */

        for (l_dim = variable_dimensions(type_variable(exp_type)); !ENDP(l_dim); POP(l_dim)) {
          b
              = make_basic_pointer(make_type_variable(make_variable(b, NIL, NIL)));
        }
      } else if(type_functional_p(exp_type)) {
        /* A reference to a function returns a pointer to a function of the very same time */
        b = make_basic_pointer(copy_type(exp_type));
      } else {
        pips_internal_error("Bad reference type tag %d \"%s\"",
            type_tag(exp_type), type_to_string(exp_type));
      }
    }
    b = basic_and_indices_to_basic(b,
                                   reference_indices(r),
                                   ultimate_p);
  }
  return b;
}


/**
 * Retrieves the basic of a reference in a newly allocated basic object
 *
 * @param r reference we want the basic of
 *
 * @return allocated basic of the reference
 */
basic basic_of_reference(reference r) {
  return basic_of_any_reference(r,false,true);
}

/* basic basic_of_call(call c): returns the basic of the result given
 * by the call "c". If ultimate_p is true, replaced typdef'ed types by
 * their definitions recursively. If not, preserve typedef'ed types.
 *
 * WARNING: a new basic is allocated
 */
basic basic_of_call(call c, bool apply_p, bool ultimate_p)
{
    entity e = call_function(c);
    tag t = value_tag(entity_initial(e));
    basic b = basic_undefined;

    switch (t)
    {
    case is_value_code:
	b = copy_basic(basic_of_external(c));
	break;
    case is_value_intrinsic:
      b = basic_of_intrinsic(c, apply_p, ultimate_p);
	break;
    case is_value_symbolic:
	/* b = make_basic(is_basic_overloaded, UU); */
	b = copy_basic(basic_of_constant(c));
	break;
    case is_value_constant:
	b = copy_basic(basic_of_constant(c));
	break;
    case is_value_unknown:
	pips_debug(1, "function %s has no initial value.\n"
	      " Maybe it has not been parsed yet.\n",
	      entity_name(e));
	b = copy_basic(basic_of_external(c));
	break;
    default: pips_internal_error("unknown tag %d", t);
	/* Never go there... */
    }
    return b;
}



/* basic basic_of_external(call c): returns the basic of the result given by
 * the call to an external function.
 *
 * WARNING: returns a pointer
 */
basic basic_of_external(call c)
{
    type return_type = type_undefined;
    entity f = call_function(c);
    basic b = basic_undefined;
    type call_type = entity_type(f);


    pips_debug(7, "External call to %s\n", entity_name(f));

    /* support calling a function pointer :) */
    type ut = ultimate_type(call_type);
    if(type_variable_p(ut) &&
        basic_pointer_p(variable_basic(type_variable(ut))))
      ut = basic_pointer(variable_basic(type_variable(ut)));

    if (! type_functional_p(ut) )
      pips_internal_error("Bad call type tag");

    return_type = functional_result(type_functional(ut));

    if (!type_variable_p(return_type)) {
      if(type_void_p(return_type)) {
	pips_user_error("A subroutine or void returning function is used as an expression\n");
      }
      else {
	pips_internal_error("Bad return call type tag \"%s\"", type_to_string(return_type));
      }
    }

    b = (variable_basic(type_variable(return_type)));

    pips_debug(7, "Returned type is %s\n", basic_to_string(b));

    return b;
}

/* basic basic_of_intrinsic(call c): returns the basic of the result
 * given by call to an intrinsic function. This basic must be computed
 * with the basic of the arguments of the intrinsic for overloaded
 * operators.  It should be able to accommodate more than two arguments
 * as for generic min and max operators. ultimate_p controls the
 * behavior when typedef'ed types are encountered: should they be
 * replaced by their definitions or not?
 *
 * WARNING: returns a newly allocated basic object */
basic basic_of_intrinsic(call c, bool apply_p, bool ultimate_p)
{
    entity f = call_function(c);
    type rt = functional_result(type_functional(entity_type(f)));
    pips_assert("not calling basic_of_intrinsic on something returning void",!type_void_p(rt));
    basic rb = copy_basic(variable_basic(type_variable(rt)));

    pips_debug(7, "Intrinsic call to intrinsic \"%s\" with a priori result type \"%s\"\n",
            module_local_name(f),
            basic_to_string(rb));

    if(basic_overloaded_p(rb)) {
        list args = call_arguments(c);

        if (ENDP(args)) {
            /* I don't know the type since there is no arguments !
               Bug encountered with a FMT=* in a PRINT.
               RK, 21/02/1994 : */
            /* leave it overloaded */
            ;
        }
        else if(ENTITY_ADDRESS_OF_P(f)) {
            //string s = entity_user_name(f);
            //bool b = ENTITY_ADDRESS_OF_P(f);
            expression e = EXPRESSION(CAR(args));
            basic eb = some_basic_of_any_expression(e, false, ultimate_p);
            // Forget multidimensional types
            type et = make_type(is_type_variable,
                    make_variable(eb, NIL, NIL));

            //fprintf(stderr, "b=%d, s=%s\n", b, s);
            free_basic(rb);
            rb = make_basic(is_basic_pointer, et);
        }
        else if(ENTITY_DEREFERENCING_P(f)) {
            expression e = EXPRESSION(CAR(args));
            free_basic(rb);
            rb = basic_of_expression(e);
            if(basic_pointer_p(rb)) {
                type pt = type_undefined;

		if(ultimate_p && !type_undefined_p(basic_pointer(rb)))
                    pt = copy_type(ultimate_type(basic_pointer(rb)));
                else
                    pt = copy_type(basic_pointer(rb));

                pips_assert("The pointed type is consistent", type_consistent_p(pt));
		if(type_undefined_p(pt)) {
		  /* Too bad, this may happen in the parser */
		  free_basic(rb);
		  rb = basic_undefined;
		}
		else if(type_variable_p(pt) && !apply_p) {
                    free_basic(rb);
                    variable v = type_variable(pt);
                    if(ENDP(variable_dimensions(v)))
                        rb = copy_basic(variable_basic(v));
                    else {
                        /* consider int a[12][13] is of type int (*)[13]*/
                        rb = make_basic_pointer(
                                make_type_variable(
                                    make_variable(
                                        copy_basic(variable_basic(v)),
                                        gen_full_copy_list(CDR(variable_dimensions(v))),
                                        gen_full_copy_list(variable_qualifiers(v))
                                        )
                                    )
                                );
                    }
                }
                else if(type_functional_p(pt)) {
                    if(apply_p) {
                        free_basic(rb);
                        type rt = ultimate_type(functional_result(type_functional(pt)));
                        if(type_variable_p(rt))
                            rb = copy_basic(variable_basic(type_variable(rt)));
                        else {
                            /* Too bad for "void"...
                             * SG: should not happen because dereferencing a void* is a mistake */
                            pips_internal_error("input code seems to derference a void* pointer ?");
                        }
                    }
                    else {
                        return rb;
                    }
                }
                else {
                    pips_internal_error("unhandled case");
                }
            }
            else {
                type et = call_to_type(c);
                if(ultimate_p) et=ultimate_type(et);
                if( type_variable_p(et) )
                {
		    variable v=type_variable(et);
                    free_basic(rb);
                    if(ENDP(variable_dimensions(v)))
                        rb = copy_basic(variable_basic(v));
                    else {
                        /* consider int a[12][13] is of type int (*)[13]*/
                        rb = make_basic_pointer(
                                make_type_variable(
                                    make_variable(
                                        copy_basic(variable_basic(v)),
                                        gen_full_copy_list(CDR(variable_dimensions(v))),
                                        gen_full_copy_list(variable_qualifiers(v))
                                        )
                                    )
                                );
                    }
                }
                if( !type_variable_p(et) ) {

                    /* This can also be a user error, but if the function is
                       called from the parser, a CParserError() should be called:
                       how to guess what to do? */
                    pips_internal_error("Dereferencing of a non-pointer, non array expression"
                            "Please use gcc to check that your source code is legal\n");
                }
                if(ultimate_p) free_type(et);
            }
        }
        else if(ENTITY_POINT_TO_P(f)) {
            //pips_internal_error("Point to case not implemented yet");
            expression e1 = EXPRESSION(CAR(args));
            expression e2 = EXPRESSION(CAR(CDR(args)));
            free_basic(rb);
            pips_assert("Two arguments for ENTITY_POINT_TO", gen_length(args)==2);
            ifdebug(8) {
                pips_debug(8, "Point to case, e1 = ");
                print_expression(e1);
                pips_debug(8, " and e2 = ");
                print_expression(e1);
                pips_debug(8, "\n");
            }
            rb = basic_of_expression(e2);
        }
        else if(ENTITY_BRACE_INTRINSIC_P(f)) {
            /* We should reconstruct a struct type or an array type... */
            rb = make_basic_overloaded();
        }
        else if(ENTITY_ASSIGN_P(f)) {
            /* returns the type of the left hand side */
            rb = basic_of_expression(EXPRESSION(CAR(args)));
        }
        else if(ENTITY_FIELD_P(f)) {
            free_basic(rb);
            rb = basic_of_expression(EXPRESSION(CAR(CDR(args))));
        }
        else if(ENTITY_COMMA_P(f)) {
            /* The value returned is the value of the last expression in the list. */
            free_basic(rb);
            rb = basic_of_expression(EXPRESSION(CAR(gen_last(args))));
        }
        else if(ENTITY_CONDITIONAL_P(f)) {
            /* The value returned is the value of the first expression in
               the list after the condition. The second expression is
               assumed to have the same value because the code is assumed
               correct. */
            free_basic(rb);
            rb = basic_of_expression(EXPRESSION(CAR(CDR(args))));
        }
        else if(ENTITY_MINUS_C_P(f)) {
            /* This must be a pointer difference. Else, the parser
	       would have used ENTITY_MINUS (see
	       simplify_C_expression()). */
            free_basic(rb);
	    rb = make_basic_int(DEFAULT_INTEGER_TYPE_SIZE);
        }
        else {
            free_basic(rb);
	    // FI: within declaration initializations, rb may be
	    // undefined because the expressions uses a variable that
	    // has not yet been typed by the parser. See C_syntax/simplify01.c
            rb = basic_of_expression(EXPRESSION(CAR(args)));

            FOREACH(EXPRESSION, arg, CDR(args)){
              basic b = basic_of_expression(arg);
              basic new_rb = basic_maximum(rb, b);

              free_basic(rb);
              free_basic(b);
              rb = new_rb;
            }
	    /* logical variables can be used as integer arguments */
	    if(!basic_undefined_p(rb) && basic_logical_p(rb))
	      if(arithmetic_intrinsic_p(f)) {
		free_basic(rb);
		rb = make_basic_int(DEFAULT_INTEGER_TYPE_SIZE);
	      }
        }

    }

    pips_debug(7, "Intrinsic call to intrinsic \"%s\" with a posteriori result type \"%s\"\n",
            module_local_name(f),
            basic_to_string(rb));

    return rb;
}

/* basic basic_of_constant(call c): returns the basic of the call to a
 * constant.
 *
 * WARNING: returns a pointer towards an existing data structure
 */
basic basic_of_constant(call c)
{
    type call_type, return_type;

    debug(7, "basic_of_constant", "Constant call\n");

    call_type = entity_type(call_function(c));

    if (type_tag(call_type) != is_type_functional)
	pips_internal_error("Bad call type tag");

    return_type = functional_result(type_functional(call_type));

    if (type_tag(return_type) != is_type_variable)
	pips_internal_error("Bad return call type tag");

    return(variable_basic(type_variable(return_type)));
}


/* basic basic_union(expression exp1 exp2): returns the basic of the
 * expression which has the most global basic. Then, between "int" and
 * "float", the most global is "float".
 *
 * Note: there are two different "float" : DOUBLE PRECISION and REAL.
 *
 * WARNING: a new basic data structure is allocated (because you cannot
 * always find a proper data structure to return simply a pointer
 */
basic basic_union(expression exp1, expression exp2)
{
  basic b1 = basic_of_expression(exp1);
  basic b2 = basic_of_expression(exp2);
  basic b = basic_maximum(b1, b2);

  free_basic(b1);
  free_basic(b2);
  return b;
}

/* get the ultimate basic from a basic typedef
 */
basic
basic_ultimate(basic b)
{
  if(basic_typedef_p(b)) {
    type t = ultimate_type(entity_type(basic_typedef(b)));
    pips_assert("typedef really has a variable type", type_variable_p(t) );
    b = variable_basic(type_variable(t));
  }
  return b;
}

basic basic_maximum(basic fb1, basic fb2)
{
  basic b = basic_undefined;
  basic b1 = basic_ultimate(fb1);
  basic b2 = basic_ultimate(fb2);


  if(basic_derived_p(fb1)) {
    entity e1 = basic_derived(fb1);

    if(entity_enum_p(e1)) {
      b1 = make_basic(is_basic_int, (void *) 4);
      b = basic_maximum(b1, fb2);
      free_basic(b1);
      return b;
    }
    else
      pips_internal_error("Unanalyzed derived basic b1");
  }

  if(basic_derived_p(fb2)) {
    entity e2 = basic_derived(fb2);

    if(entity_enum_p(e2)) {
      b2 = make_basic(is_basic_int, (void *) 4);
      b = basic_maximum(fb1, b2);
      free_basic(b2);
      return b;
    }
    else
      pips_internal_error("Unanalyzed derived basic b2");
  }

  /* FI: I do not believe this is correct for all intrinsics! */

  pips_debug(7, "Tags: tag exp1 = %d, tag exp2 = %d\n",
	     basic_tag(b1), basic_tag(b2));


  if(basic_overloaded_p(b2)) {
    b = copy_basic(b2);
  }
  else {
    switch(basic_tag(b1)) {

    case is_basic_overloaded:
      b = copy_basic(b1);
      break;

    case is_basic_string:
      if(basic_string_p(b2)) {
	int s1 = SizeOfElements(b1);
	int s2 = SizeOfElements(b2);

	/* Type checking problem for ? : with gcc... */
	if(s1>s2)
	  b = copy_basic(b1);
	else
	  b = copy_basic(b2);
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;

    case is_basic_logical:
      if(basic_logical_p(b2)) {
	_int s1 = basic_logical(b1);
	_int s2 = basic_logical(b2);

	b = make_basic(is_basic_logical,UUINT(s1>s2?s1:s2));
      }
      else if(basic_int_p(b2)) {
	b = copy_basic(b2);
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;

    case is_basic_complex:
      if(basic_complex_p(b2) || basic_float_p(b2) || basic_int_p(b2)) {
	_int s1 = SizeOfElements(b1);
	_int s2 = SizeOfElements(b2);

	b = make_basic(is_basic_complex, UUINT(s1>s2?s1:s2));
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;

    case is_basic_float:
      if(basic_complex_p(b2)) {
	_int s1 = SizeOfElements(b1);
	_int s2 = SizeOfElements(b2);

	b = make_basic(is_basic_complex, UUINT(s1>s2?s1:s2));
      }
      else if(basic_float_p(b2) || basic_int_p(b2)) {
	_int s1 = SizeOfElements(b1);
	_int s2 = SizeOfElements(b2);

	b = make_basic(is_basic_float, UUINT(s1>s2?s1:s2));
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;

    case is_basic_int:
      if(basic_complex_p(b2) || basic_float_p(b2)) {
	_int s1 = SizeOfElements(b1);
	_int s2 = SizeOfElements(b2);

	b = make_basic(basic_tag(b2), UUINT(s1>s2?s1:s2));
      }
      else if(basic_int_p(b2)) {
	_int s1 = SizeOfElements(b1);
	_int s2 = SizeOfElements(b2);

	b = make_basic(is_basic_int, UUINT(s1>s2?s1:s2));
      }
      else if(basic_logical_p(b2)) {
	b = copy_basic(b1);
      }
      else if(basic_pointer_p(b2)) {
          return copy_basic(b2);
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;
      /* NN: More cases are added for C. To be refined  */
    case is_basic_bit:
      if(basic_bit_p(b2)) {
	if(basic_bit(b1)>=basic_bit(b2))
	  b = copy_basic(b1);
	else
	  b = copy_basic(b2);
      }
      else
	/* bit is a lesser type */
	b = copy_basic(b2);
      break;
    case is_basic_pointer:
      {
	if(basic_int_p(b2) || basic_bit_p(b2))
	  b = copy_basic(b1);
	else if(basic_float_p(b2) || basic_logical_p(b2) || basic_complex_p(b2)) {
	  /* Are they really comparable? */
	  b = copy_basic(b1);
	}
	else if(basic_overloaded_p(b2))
	  b = copy_basic(b1);
	else if(basic_pointer_p(b2)) {
	  /* How can we compare two pointer types? Equality? Comparison of the pointed types? */
	  /* pips_internal_error("Comparison of two pointer types not implemented"); */
	  type t1 = basic_pointer(b1);
	  type t2 = basic_pointer(b2);

	  if(type_variable_p(t1) && type_variable_p(t2)) {
        /* SG checks for equality, he doesn't understand the meaning of
         * having a float as the basic maximum of two float* 
         * and cowardly refuses to fix the code */
          if(type_equal_p(t1,t2))
              b = copy_basic(b1);
          else {
              basic nb1 = variable_basic(type_variable(t1));
              basic nb2 = variable_basic(type_variable(t2));


              /* FI: not convincing. As in other places, assuming this
                 is meaningful, it would be better to use a basic
                 comparator, basic_greater_p(), which could return 1, -1
                 or 0 or ??? and deal with non comparable type. */
              b = basic_maximum(nb1, nb2);
          }

	  }
	  else if (type_void_p(t1) && type_void_p(t2) )
          b = copy_basic(b1);
      else
	    pips_internal_error("Comparison of two pointer types not meaningful");
	}
	else if(basic_derived_p(b2))
	  pips_internal_error("Comparison between pointer and struct/union not implemented");
	else if(basic_typedef_p(b2))
	  pips_internal_error("b2 cannot be a typedef basic");
	else
	  pips_internal_error("unknown tag %d for basic b2", basic_tag(b2));
      break;
       }
     case is_basic_derived:
      /* How do you compare a structure or a union to another type?
	 The only case which seems to make sense is equality. */
      pips_internal_error("Derived basic b1 it not comparable to another basic");
      break;
    case is_basic_typedef:
      pips_internal_error("b1 cannot be a typedef basic");
      break;
    default: pips_internal_error("Ill. basic tag %d", basic_tag(b1));
    }
  }

  return b;

  /*
    if( (t1 != is_basic_complex) && (t1 != is_basic_float) &&
    (t1 != is_basic_int) && (t2 != is_basic_complex) &&
    (t2 != is_basic_float) && (t2 != is_basic_int) )
    pips_internal_error("Bad basic tag for expression in numerical function");

    if(t1 == is_basic_complex)
    return(b1);
    if(t2 == is_basic_complex)
    return(b2);
    if(t1 == is_basic_float) {
    if( (t2 != is_basic_float) ||
    (basic_float(b1) == DOUBLE_PRECISION_SIZE) )
    return(b1);
    return(b2);
    }
    if(t2 == is_basic_float)
    return(b2);
    return(b1);
  */
}

basic basic_of_expressions(list expressions,bool skip_overloaded)
{
    if(ENDP(expressions)) return basic_undefined;
    else if(ENDP(CDR(expressions))) return basic_of_expression(EXPRESSION(CAR(expressions)));
    else {
        basic out = basic_of_expression(EXPRESSION(CAR(expressions)));
        FOREACH(EXPRESSION,exp,CDR(expressions)) {
            basic b = basic_of_expression(exp);
            if(skip_overloaded && basic_overloaded_p(out)) {
                free_basic(out);
                out=b;
            }
            else if( skip_overloaded && basic_overloaded_p(b)) {
                free_basic(b);
            }
            else {
                basic tmp =basic_maximum(out,b);
                free_basic(b);
                free_basic(out);
                out=tmp;
            }
        }
        return out;
    }
}


/* END_EOLE */

/**************************************************** expression_to_type */

/**
   @return the (newly allocated) type of the result given by call to
   an intrinsic function.

   This type must be computed with the basic of the arguments of the
   intrinsic for overloaded operators. It should be able to accomodate
   more than two arguments as for generic min and max operators.
*/

type intrinsic_call_to_type(call c)
{

  entity f = call_function(c);
  list args = call_arguments(c);
  type t = type_undefined; /* the result */
  type rt = functional_result(type_functional(entity_type(f)));

  if(type_void_p(rt)) {
    t = copy_type(rt);
  }
  else if(type_variable_p(rt)) {
    basic rb = variable_basic(type_variable(rt));

    pips_debug(9, "Intrinsic call to intrinsic \"%s\" with a priori result type \"%s\"\n",
	       module_local_name(f),
	       words_to_string(words_type(rt, NIL, false)));

    if(basic_overloaded_p(rb))
      {

	if (ENDP(args))
	  {
	    /* I don't know the type since there is no arguments !
	       Bug encountered with a FMT=* in a PRINT.
	       RK, 21/02/1994 : */
	    /* leave it overloaded */
	    t = copy_type(rt);
	  }
	else if(ENTITY_ADDRESS_OF_P(f))
	  {
	    expression e = EXPRESSION(CAR(args));
	    t = expression_to_type(e);
	    t = make_type(is_type_variable,
			  make_variable( make_basic(is_basic_pointer,t),
					 NIL, NIL ));

	  }
	else if(ENTITY_DEREFERENCING_P(f))
	  {
	    expression e = EXPRESSION(CAR(args));
	    type ct = ultimate_type(expression_to_type(e)); /* isn't expression_to_type expected to return a bct?
							       well, there are cases (casts) which are not clear on this point... BC. */

	    if (type_variable_p(ct))
	      {
		variable cv = type_variable(ct);
		basic cb = variable_basic(cv);
		list cd = variable_dimensions(cv);
		if( ENDP(cd)) {
			if(basic_pointer_p(cb))
			{
				t = copy_type(ultimate_type(basic_pointer(cb)));
				pips_assert("The pointed type is consistent",
						type_consistent_p(t));
				free_type(ct); /* isn't it dangerous?, an ultimate_type can be an entity type!  */
			}
			else if(basic_string_p(cb))
			{
				t = make_type_variable(make_variable(make_basic_int(DEFAULT_CHARACTER_TYPE_SIZE), NIL, NIL));
			}
			else
			{
				pips_assert("Dereferencing of a non-pointer expression : it must be an array\n", !ENDP(cd));
			}
		}
		else {
			variable_dimensions(cv) = CDR(cd);
			cd->cdr = NIL;
			gen_full_free_list(cd);
			t = ct;
		}
	      }
	    else
	      {
		pips_internal_error("dereferencing of a non-variable : not handled yet");
	      }
	  }
	else if(ENTITY_POINT_TO_P(f) || ENTITY_FIELD_P(f))
	  {
	    expression e1 = EXPRESSION(CAR(args));
	    expression e2 = EXPRESSION(CAR(CDR(args)));

	    pips_assert("Two arguments for POINT_TO or FIELD \n",
			gen_length(args)==2);

	    ifdebug(9)
	      {
		pips_debug(8, "Point to case, e1 = ");
		print_expression(e1);
		pips_debug(8, " and e2 = ");
		print_expression(e2);
		pips_debug(8, "\n");
	      }
	    t = expression_to_type(e2);
	  }
	else if(ENTITY_BRACE_INTRINSIC_P(f))
	  {
	    /* We should reconstruct a struct type or an array type... */
	    t = make_type(is_type_variable, make_variable(make_basic_overloaded(),
							  NIL,NIL));
	  }
	else if(ENTITY_ASSIGN_P(f))
	  {
	    /* returns the type of the left hand side */
	    t = expression_to_type(EXPRESSION(CAR(args)));
	  }
	else if(ENTITY_COMMA_P(f))
	  {
	    /* The value returned is the value of the last expression in the list. */

	    t = expression_to_type(EXPRESSION(CAR(gen_last(args))));
	  }
	else if( ENTITY_CONDITIONAL_P(f))
	  {
	    /* let us assume that the two last arguments have the same
	       type : basic_maximum does not preserve types enough
	       (see Effects/lhs01.c, expression *(i>2?&i:&j) ). BC.
	    */
	    t = expression_to_type(EXPRESSION(CAR(CDR(args))));
	  }
	else
	  {
	    bool minus_c_pointer_arithmetic = false;

	    // special case for minus operator when first argument is a pointer type
	    if (ENTITY_MINUS_C_P(f) )
	      {
		expression exp1 = EXPRESSION(CAR(args));
		expression exp2 = EXPRESSION(CAR(CDR(args)));
		type t1 = expression_to_type(exp1);
		type t2 = expression_to_type(exp2);

		if (pointer_type_p(t1))
		  {
		    if (pointer_type_p(t2))
		      {
			type pt1 = pointed_type(t1);
			type pt2 = pointed_type(t2);
			if (!type_equal_p(pt1, pt2))
			  {
			    // user application should not pass compilation by a standard compiler
			    // should we also trigger an error here?
			    pips_user_warning("Non matching pointed types in pointer arithmetic expression %s - %s\n",
					      expression_to_string(exp1), expression_to_string(exp2));
			  }
			// result is of type ptrdiff_t (ISO/IEC 9899:TC3)
			t = make_type(is_type_variable, make_variable(make_basic_int(DEFAULT_POINTER_TYPE_SIZE),
								      NIL,NIL));


		      }
		    else
		      {
			t = copy_type(t1);
		      }
		    minus_c_pointer_arithmetic = true;
		    free_type(t1); free_type(t2);
		  }
	      }

	    if (! minus_c_pointer_arithmetic )
	      {
		/* current type of expression is type of first argument */
		type ct = expression_to_type(EXPRESSION(CAR(args)));

		MAP(EXPRESSION, arg, {
		    type nt = expression_to_type(arg);
		    basic nb = variable_basic(type_variable(nt));
		    list  nd = variable_dimensions(type_variable(nt));

		    basic cb = variable_basic(type_variable(ct));
		    list  cd = variable_dimensions(type_variable(ct));

		    /* we need to check the variable dimensions */
		    if (gen_length(nd) == gen_length(cd))
		      {
			/* re-use an existing function. we do not take into
			   account variable dimensions here. It may not be correct.
			   but it's not worse than the previously existing version
			   of expression_to_type
			*/
			pips_debug(9,"same number of dimensions\n");
			basic b = basic_maximum(cb, nb);
			free_type(ct);
			free_type(nt);
			ct = make_type(is_type_variable, make_variable(b, gen_full_copy_list(nd), NIL));
		      }
		    else
		      {
			pips_debug(9,"different number of dimensions\n");
			pips_assert("pointer arithmetic with array name, first element must be the address expression",
				    gen_length(cd) > gen_length(nd));
			/* current type is still valid */
			free_type(nt);
		      }


		  }, CDR(args));
		t = ct;
	      }
	  }
      }
    else {
      t = copy_type(rt);
    }
  }
  else
    pips_internal_error("Unexpected return type.");

  pips_debug(9, "Intrinsic call to intrinsic \"%s\" "
	     "with a posteriori result type \"%s\"\n",
	     module_local_name(f),
	     words_to_string(words_type(t, NIL, false)));

  return t;
}


type call_to_type(call c)
{
    entity e = call_function(c);
    type t = type_undefined;

    switch (value_tag(entity_initial(e)))
    {
    case is_value_code:
      t = make_type(is_type_variable,
		    make_variable(copy_basic(basic_of_external(c)),
				  NIL, NIL));
      break;
    case is_value_intrinsic:
      t = intrinsic_call_to_type(c);
      break;
    case is_value_symbolic:
      /* b = make_basic(is_basic_overloaded, UU); */
      t = make_type(is_type_variable,
		    make_variable(copy_basic(basic_of_constant(c)),
				  NIL, NIL));
      break;
    case is_value_constant:
      t = make_type(is_type_variable,
		    make_variable(copy_basic(basic_of_constant(c)),
				  NIL, NIL));
      break;
    case is_value_unknown:
      pips_debug(1, "function %s has no initial value.\n"
		 " Maybe it has not been parsed yet.\n",
		 entity_name(e));
      t = make_type(is_type_variable,
		    make_variable(copy_basic(basic_of_external(c)),
				  NIL, NIL));
      break;
    default: pips_internal_error("unknown tag %d", t);
      /* Never go there... */
    }

    return t;
}

type reference_to_type(reference ref)
{
  type t = type_undefined;
  pips_debug(9, "input entity type %s\n", words_to_string(words_type(entity_type(reference_variable(ref)), NIL, false)));

  type exp_type = entity_basic_concrete_type(reference_variable(ref));

  pips_debug(9, "reference case \n");
  pips_debug(9, "exp_type %s\n", words_to_string(words_type(exp_type, NIL, false)));

  if(type_variable_p(exp_type))
    {
      type ct = exp_type; /* current type */
      basic cb = variable_basic(type_variable(exp_type)); /* current basic */

      list cd = variable_dimensions(type_variable(exp_type)); /* current dimensions */
      list l_inds = reference_indices(ref);

      pips_debug(9, "reference to a variable, "
		 "we iterate over the indices if any \n");

      while (!ENDP(l_inds))
	{

	  ifdebug(9) {
	    pips_debug(7, "new iteration : current type : %s\n",
		       words_to_string(words_type(ct, NIL, false)));
	    pips_debug(7, "current list of indices: \n");
	    print_expressions(l_inds);
	  }
	  if(!ENDP(cd))
	    {
	      pips_debug(9, "poping one type dimension and one index\n");
	      POP(cd);
	      POP(l_inds);
	    }
	  else
	    {
	      pips_debug(9,"going through pointer dimension. \n");
	      // FI: struct are only possible for constant memory path
	      // the usual internal representation does not use fields
	      // as indices
	      pips_assert("reference has too many indices :"
			  " pointer or struct expected\n",
			  basic_pointer_p(cb) || basic_derived_p(cb));
	      if(basic_pointer_p(cb)) {
		ct = basic_pointer(cb);
		cb = variable_basic(type_variable(ct));
		cd = variable_dimensions(type_variable(ct));
	      }
	      else if(basic_derived_p(cb)) { // must be a struct, see assert
		entity de = basic_derived(cb);
		type st = entity_type(de);
		list fl = type_struct(st);
		// FI: I am not sure about the internal representation...
		// Do we find an integer or a field reference as
		// subscript expression?
		expression ind = EXPRESSION(CAR(l_inds));
		value ind_v = EvalExpression(ind);
		if(value_constant_p(ind_v)) {
		  int n = constant_int(value_constant(ind_v));
		  entity f = ENTITY(gen_nth(n, fl));
		  type ft = entity_type(f);
		  ct = ft;
		  cb = variable_basic(type_variable(ct));
		  cd = variable_dimensions(type_variable(ct));
		}
		else { // FI: assume a reference to a field
		  // pips_internal_error("Unexpected internal representation.\n");
		  pips_assert("The subscript expression is a reference",
			      expression_reference_p(ind));
		  entity f =
		    reference_variable(syntax_reference(expression_syntax(ind)));
		  type ft = entity_type(f);
		  ct = ft;
		  cb = variable_basic(type_variable(ct));
		  cd = variable_dimensions(type_variable(ct));
		}
	      }
	      POP(l_inds);
	    }
	}

      /* Warning : qualifiers are set to NIL, because I do not see
	 the need for something else for the moment. BC.
      */
      t = make_type_variable(
		    make_variable(copy_basic(cb),
				  gen_full_copy_list(cd),
				  NIL));
      pips_debug(9, "t at the end of reference case %s\n", words_to_string(words_type(t, NIL, false)));
    }
  else if(type_functional_p(exp_type))
    {
      pips_debug(9, "functional case \n");
      /* A reference to a function returns a pointer to a function
	 of the very same time */
      t = make_type(is_type_variable,
		    make_variable
		    (make_basic(is_basic_pointer, copy_type(exp_type)),
		     NIL, NIL));
    }
  else
    {
      // The unknown type ends up here
      pips_internal_error("Bad reference type tag %d \"%s\" for reference %s",
			  type_tag(exp_type),
			  type_to_string(exp_type),
			  entity_name(reference_variable(ref)));
    }
  pips_debug(9, "returns with %s\n", words_to_string(words_type(t, NIL, false)));
  return t;
}


/**
  For an array declared as int a[10][20], the type returned for a[i] is
  int [20].

   @param exp is an expression
   @return a new allocated type which is the ntype of the expression in which
           typedef's are replaced by combination of basic types.

*/
type expression_to_type(expression exp)
{
  debug_on("RI-UTIL_DEBUG_LEVEL");
  /* does not cover references to functions ...*/
  /* Could be more elaborated with array types for array expressions */
  type t = type_undefined;

  syntax s_exp = expression_syntax(exp);

  ifdebug(9){
    pips_debug(6, "begins with expression :");
    print_expression(exp);
    fprintf(stderr, "\n");
  }

  switch(syntax_tag(s_exp))
    {
    case is_syntax_reference:
      {
	pips_debug(9, "reference case \n");
	t = reference_to_type(syntax_reference(s_exp));
	break;
      }
    case is_syntax_call:
      {
	pips_debug(9, "call case \n");
	t = call_to_type(syntax_call(s_exp));
	break;
      }
    case is_syntax_range:
      {
	pips_debug(9, "range case \n");
	/* Well, let's assume range are well formed... */
	t = expression_to_type(range_lower(syntax_range(s_exp)));
	break;
      }
    case is_syntax_cast:
      {
	pips_debug(9, "cast case \n");
	t = copy_type(cast_type(syntax_cast(s_exp)));
	if (!type_void_p(t) && type_tag(t) != is_type_variable)
	  pips_internal_error("Bad reference type tag %d",type_tag(t));
	break;
      }
    case is_syntax_sizeofexpression:
      {
          /*
	sizeofexpression se = syntax_sizeofexpression(s_exp);
	pips_debug(9, "size of case \n");
	if (sizeofexpression_type_p(se))
	  {
	    t = copy_type(sizeofexpression_type(se));
	    if (type_tag(t) != is_type_variable)
	      pips_internal_error("Bad reference type tag %d",type_tag(t));
	  }
	else
	  {
	    t = expression_to_type(sizeofexpression_expression(se));
	  }*/
          t = make_type_variable(make_variable(make_basic_int(DEFAULT_POINTER_TYPE_SIZE),NIL,NIL));
	break;
      }
    case is_syntax_subscript:
      {
	/* current type */
	type ct = expression_to_type(subscript_array(syntax_subscript(s_exp)));
	/* current basic */
	basic cb = variable_basic(type_variable(ct));
	/* current dimensions */
	list cd = variable_dimensions(type_variable(ct));
	list l_inds = subscript_indices(syntax_subscript(s_exp));

	pips_debug(9, "subscript case \n");

	while (!ENDP(l_inds))
	  {
	    if(!ENDP(cd))
	      {
		POP(cd);		
	      }
	    else
	      {
		pips_assert("reference has too many indices : pointer expected\n", basic_pointer_p(cb));
		ct= basic_pointer(cb);
		if( type_variable_p(ct) ) {
		  cb = variable_basic(type_variable(ct));
		  cd = variable_dimensions(type_variable(ct));
		}
		else {
		  pips_internal_error("unhandled case");
		}
	      }
	    POP(l_inds);
	  }

	/* Warning : qualifiers are set to NIL, because I do not see
	   the need for something else for the moment. BC.
	*/
    t = make_type(is_type_variable,
            make_variable(copy_basic(cb),
                gen_full_copy_list(cd),
                NIL));
	break;
      }
    case is_syntax_application:
      {
	pips_debug(9, "application case \n");
	t = expression_to_type(application_function(syntax_application(s_exp)));
	break;
      }
    case is_syntax_va_arg:
      {
	pips_debug(9, "va_arg case\n");
	list vararg_list = syntax_va_arg(s_exp);
	sizeofexpression soe = SIZEOFEXPRESSION(CAR(CDR(vararg_list)));

	t = copy_type(sizeofexpression_type(soe));
	break;
      }

    default:
      pips_internal_error("Bad syntax tag %d", syntax_tag(s_exp));
      /* Never go there... */
    }

  pips_debug(9, "returns with %s\n", words_to_string(words_type(t, NIL, false)));
  debug_off();
  return t;
}

/* If the expression is casted, return its type before cast */
type expression_to_uncasted_type(expression exp)
{
  type t = type_undefined;
  syntax s_exp = expression_syntax(exp);

  ifdebug(6){
    pips_debug(6, "begins with expression :");
    print_expression(exp);
    pips_debug(6, "\n");
  }

  if(syntax_cast_p(s_exp)) {
    expression sub_exp = cast_expression(syntax_cast(s_exp));

    t = expression_to_uncasted_type(sub_exp);
  }
  else {
    t = expression_to_type(exp);
  }

  return t;
}

/* Preserve typedef'ed types when possible */
type expression_to_user_type(expression e)
{
  /* does not cover references to functions ...*/
  /* Could be more elaborated with array types for array expressions */
  type t = type_undefined;
  basic b = some_basic_of_any_expression(e, false, false);
  variable v = make_variable(b, NIL, NIL);

  t = make_type(is_type_variable, v);

  return t;
}


/*************************************************************************/



/* Returns true if t is a variable type with a basic overloaded. And
 * false elsewhere. See MakeTypeOverloaded().
 */
bool overloaded_type_p(type t)
{
  //pips_assert("type t is of kind variable", type_variable_p(t));

  if(!type_variable_p(t))
    return false;

  return basic_overloaded_p(variable_basic(type_variable(t)));
}

/* bool is_inferior_basic(basic1, basic2)
 * return true if basic1 is less complex than basic2
 * ex:  int is less complex than float*4,
 *      float*4 is less complex than float*8, ...
 * - overloaded is inferior to any basic.
 * - logical is inferior to any other but overloaded.
 * - string is inferior to any other but overloaded and logical.
 * Used to decide that the sum of an int and a float
 * is a floating-point addition (for ex.)
 */
bool
is_inferior_basic(b1, b2)
basic b1, b2;
{
    if ( b1 == basic_undefined )
	pips_internal_error("first  basic_undefined");
    else if ( b2 == basic_undefined )
	pips_internal_error("second basic_undefined");

    if (basic_overloaded_p(b1))
	return (true);
    else if (basic_overloaded_p(b2))
	return (false);
    else if (basic_logical_p(b1))
	return (true);
    else if (basic_logical_p(b2))
	return (false);
    else if (basic_string_p(b1))
	return (true);
    else if (basic_string_p(b2))
	return (false);
    else if (basic_int_p(b1)) {
	if (basic_int_p(b2))
	    return (basic_int(b1) <= basic_int(b2));
	else
	    return (true);
    }
    else if (basic_float_p(b1)) {
	if (basic_int_p(b2))
	    return (false);
	else if (basic_float_p(b2))
	    return (basic_float(b1) <= basic_float(b2));
	else
	    return (true);
    }
    else if (basic_complex_p(b1)) {
	if (basic_int_p(b2) || basic_float_p(b2))
	    return (false);
	else if (basic_complex_p(b2))
	    return (basic_complex(b1) <= basic_complex(b2));
	else
	    return (true);
    }
    else
	pips_internal_error("Case never occurs.");
    return (true);
}

basic
simple_basic_dup(basic b)
{
    /* basic_int, basic_float, basic_logical, basic_complex are all int's */
    /* so we duplicate them the same manner: with basic_int. */
    if (basic_int_p(b)     || basic_float_p(b) ||
	basic_logical_p(b) || basic_complex_p(b))
	return(make_basic(basic_tag(b), UUINT(basic_int(b))));
    else if (basic_overloaded_p(b))
	return(make_basic(is_basic_overloaded, UU));
    else {
	user_warning("simple_basic_dup",
		     "(tag %td) isn't that simple\n", basic_tag(b));
	if (basic_string_p(b))
	    fprintf(stderr, "string: value tag = %d\n",
		    value_tag(basic_string(b)));
	return make_basic(basic_tag(b), UUINT(basic_int(b)));
    }
}

/* returns the corresponding generic conversion entity, if any.
 * otherwise returns entity_undefined.
 */
entity
basic_to_generic_conversion(basic b)
{
    entity result;

    switch (basic_tag(b))
    {
    case is_basic_int:
	/* what about INTEGER*{2,4,8} ?
	 */
	result = entity_intrinsic(INT_GENERIC_CONVERSION_NAME);
	break;
    case is_basic_float:
    {
	if (basic_float(b)==4)
	    result = entity_intrinsic(REAL_GENERIC_CONVERSION_NAME);
	else if (basic_float(b)==8)
	    result = entity_intrinsic(DBLE_GENERIC_CONVERSION_NAME);
	else
	    result = entity_undefined;
	break;
    }
    case is_basic_complex:
    {
	if (basic_complex(b)==8)
	    result = entity_intrinsic(CMPLX_GENERIC_CONVERSION_NAME);
	else if (basic_complex(b)==16)
	    result = entity_intrinsic(DCMPLX_GENERIC_CONVERSION_NAME);
	else
	    result = entity_undefined;
	break;
    }
    default:
	result = entity_undefined;
    }

    return result;
}

/*
 * A set of predicates to distinguish between types
 */

bool signed_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b)/10 == DEFAULT_SIGNED_TYPE_SIZE)
	  return true;
    }
  return false;
}
bool unsigned_basic_p(basic b) {
    if (basic_int_p(b))
        if (basic_int(b)/10 == DEFAULT_UNSIGNED_TYPE_SIZE)
            return true;
    return false;
}

bool unsigned_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      return unsigned_basic_p(b);
    }
  return false;
}


bool long_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b) == DEFAULT_LONG_INTEGER_TYPE_SIZE)
	  return true;
    }
  return false;
}

bool bit_type_p(type t)
{
  if (!type_undefined_p(t) && type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (!basic_undefined_p(b) && basic_bit_p(b))
	return true;
    }
  return false;
}

bool string_type_p(type t)
{
  if (!type_undefined_p(t) && type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (!basic_undefined_p(b) && basic_string_p(b))
	return true;
    }
  return false;
}

bool logical_type_p(type t)
{
  if (!type_undefined_p(t) && type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (!basic_undefined_p(b) && basic_logical_p(b))
	return true;
    }
  return false;
}

/* return true whether `t' is a char or an unsigned char */
bool char_type_p(type t)
{
  bool is_char = false;

  if (!type_undefined_p(t) && type_variable_p(t)) {
    basic b = variable_basic(type_variable(t));
    if (!basic_undefined_p(b) && basic_int_p(b)) {
      int i = basic_int(b);
    is_char = (i==1)||(i==11); /* see words_basic() */
    }
  }
  return is_char;
}

/* Safer than the other implementation?
bool pointer_type_p(type t)
{
  bool is_pointer = false;

  if (!type_undefined_p(t) && type_variable_p(t)) {
    basic b = variable_basic(type_variable(t));
    if (!basic_undefined_p(b) && basic_pointer_p(b)) {
      is_pointer = true;
    }
  }
  return is_pointer;
}
*/

/* Here is the set of mapping functions, from the RI to C language types*/

/* Returns true if t is one of the following types :
   void, char, short, int, long, float, double, signed, unsigned,
   and there is no array dimensions, of course*/

bool basic_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      return ((variable_dimensions(type_variable(t)) == NIL) &&
	      (basic_int_p(b) || basic_float_p(b) || basic_logical_p(b)
	       || basic_overloaded_p(b) || basic_complex_p(b) || basic_string_p(b)
	       || basic_bit_p(b)));
    }
  return (type_void_p(t) || type_unknown_p(t)) ;
}

/*
 @brief tests whether the basic of the input type is one of the following:
   void, char, short, int, long, float, double, signed, unsigned.
   (even if there are array dimensions
*/
bool type_fundamental_basic_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      return (basic_int_p(b) || basic_float_p(b) || basic_logical_p(b)
	       || basic_overloaded_p(b) || basic_complex_p(b) || basic_string_p(b)
	       || basic_bit_p(b));
    }
  return (type_void_p(t) || type_unknown_p(t)) ; 
}

bool array_type_p(type t)
{
  return (type_variable_p(t) && (variable_dimensions(type_variable(t)) != NIL));
}

bool scalar_type_p(type t)
{
  return (type_variable_p(t) && (variable_dimensions(type_variable(t)) == NIL));
}

bool type_pointer_on_struct_variable_p(type t)
{
        t = ultimate_type(t);
        if(basic_pointer_p(variable_basic(type_variable(t))))
        {
            type pt = basic_pointer(variable_basic(type_variable(t)));
            return type_struct_variable_p(pt);
        }
        return false;
}

/* Is this equivalent to dependent_type_p()? */
bool variable_length_array_type_p(type t)
{
  bool return_val = false;
  if(array_type_p(t)) {
    FOREACH(dimension,dim,variable_dimensions(type_variable(t))) {
      if(!extended_expression_constant_p(dimension_lower(dim)) ||
          !extended_expression_constant_p(dimension_upper(dim))) {
        return_val=true;
        break;
      }
    }
  }
  return return_val;
}

bool fixed_length_array_type_p(type t)
{
    return array_type_p(t) && !variable_length_array_type_p(t);
}

bool pointer_type_p(type t)
{
  return (type_variable_p(t) && basic_pointer_p(variable_basic(type_variable(t)))
	  && (variable_dimensions(type_variable(t)) == NIL));
}

/* Returns OK for "char[]" as well as for "char *".
 * 
 * Does not take care of typedef. Use compute_basic_concrete_type()
 * first is necessary.
 */
bool C_pointer_type_p(type t)
{
  bool pointer_p = false;
  if(type_variable_p(t)) {
    variable v = type_variable(t);
    list dl = variable_dimensions(v);
    basic b = variable_basic(v);
    pointer_p = (ENDP(dl) && basic_pointer_p(b))
      || ((int)gen_length(dl)==1 && unbounded_dimension_p(DIMENSION(CAR(dl))));
  }
  return pointer_p;
}

bool array_of_pointers_type_p(type t)
{
  return (type_variable_p(t) && basic_pointer_p(variable_basic(type_variable(t)))
	  && (variable_dimensions(type_variable(t)) != NIL));
}


/**
   returns the type pointed by the input type if it is a pointer or an array of pointers
 */
type pointed_type(type t)
{
  type res = type_undefined;

  if (type_variable_p(t) && basic_pointer_p(variable_basic(type_variable(t))))
    res = basic_pointer(variable_basic(type_variable(t)));
  return res;
}

// tests if a type is FILE *
// beware: costly because it contains a string operation
bool FILE_star_type_p(type t)
{
  bool res = false;
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_pointer_p(b))
	{
	  t = basic_pointer(b);
	  if (type_variable_p(t))
	    {
	      basic b = variable_basic(type_variable(t));
	      if (basic_derived_p(b))
		{
		  entity te = basic_derived(b);
		  if (strstr(entity_name(te), "_IO_FILE") != NULL)
		    {
		      res = true;
		    }
		}
	    }
	}
    }
  return res;
}


list type_fields(type t)
{
  list l_res = NIL;

  switch (type_tag(t))
    {
    case is_type_struct:
      l_res = type_struct(t);
      break;
    case is_type_union:
      l_res = type_union(t);
      break;
    case is_type_enum:
      l_res = type_enum(t);
      break;
    default:
      pips_internal_error("type_fields improperly called");
    }
  return l_res;

}

/* Returns true if t is of type struct, union or enum. Need to
 * distinguish with the case struct/union/enum in type in RI, these
 * are the definitions of the struct/union/enum themselve, not a
 * variable of this type.
 *
 * Example : struct foo var;
 *
 * Note: arrays of struct are not considered derived types
 */
bool derived_type_p(type t)
{
  return (type_variable_p(t) && basic_derived_p(variable_basic(type_variable(t)))
	  && (variable_dimensions(type_variable(t)) == NIL));
}
bool array_of_derived_type_p(type t)
{
  return (type_variable_p(t) && basic_derived_p(variable_basic(type_variable(t)))
	  && (variable_dimensions(type_variable(t)) != NIL));
}

/* Returns true if t is of type derived and if the derived type is a struct.
 *
 * Example : struct foo var;
 *
 * Note: different trom type_struct_p
 */
bool struct_type_p(type t)
{
  bool struct_p = false;
  if(derived_type_p(t)) {
    basic b = variable_basic(type_variable(t));
    entity dte = basic_derived(b);
    type dt = entity_type(dte);
    struct_p = type_struct_p(dt);
  }
  return struct_p;
}

bool array_of_struct_type_p(type t)
{
  bool struct_p = false;
  if(array_of_derived_type_p(t)) {
    basic b = variable_basic(type_variable(t));
    entity dte = basic_derived(b);
    type dt = entity_type(dte);
    struct_p = type_struct_p(dt);
  }
  return struct_p;
}

/* Returns true if t is of type derived and if the derived type is a union.
 *
 * Example : union foo var;
 *
 * Note: different trom type_union_p
 */
bool union_type_p(type t)
{
  bool union_p = false;
  if(derived_type_p(t)) {
    basic b = variable_basic(type_variable(t));
    entity dte = basic_derived(b);
    type dt = entity_type(dte);
    union_p = type_union_p(dt);
  }
  return union_p;
}

/* Returns true if t is of type derived and if the derived type is a enum.
 *
 * Example : enum foo var;
 *
 * FI: Could be unified with the prevous two functions,
 * struct_type_p() and union_type_p()
 *
 * Note: different from type_enum_p
 */
bool enum_type_p(type t)
{
  bool enum_p = false;
  if(derived_type_p(t)) {
    basic b = variable_basic(type_variable(t));
    entity dte = basic_derived(b);
    type dt = entity_type(dte);
    enum_p = type_enum_p(dt);
  }
  return enum_p;
}

/* Returns true if t is a typedefED type.

   Example : Myint i;
*/

bool typedef_type_p(type t)
{
  return (type_variable_p(t) && basic_typedef_p(variable_basic(type_variable(t)))
	  && (variable_dimensions(type_variable(t)) == NIL));
}


type make_standard_integer_type(type t, int size)
{
  if (t == type_undefined)
    {
      variable v = make_variable(make_basic_int(size),NIL,NIL);
      return make_type_variable(v);
    }
  else
    {
      if (signed_type_p(t) || unsigned_type_p(t))
	{
	  basic b = variable_basic(type_variable(t));
	  int i = basic_int(b);
	  variable v = make_variable(make_basic_int(10*(i/10)+size),NIL,NIL);
	  pips_debug(8,"Old basic size: %d, new size : %d\n",i,10*(i/10)+size);
	  return make_type_variable(v);
	}
      else
	{
	  if (bit_type_p(t))
	    /* If it is int i:5, keep the bit basic type*/
	    return t;
	  else
	    user_warning("Parse error", "Standard integer types\n");
	  return type_undefined;
	}
    }
}

/* Used to encode the long keyword in the parser. Used to detect long
   double type in parser*/
bool standard_long_integer_type_p(type t)
{
  bool long_p = false;
  if(!type_undefined_p(t) && type_variable_p(t)) {
    variable v = type_variable(t);
    basic b = variable_basic(v);
    if(basic_int_p(b)) {
      int s = basic_int(b);

      long_p = ENDP(variable_dimensions(v))
	&& ENDP(variable_qualifiers(v))
	&& (s == DEFAULT_INTEGER_TYPE_SIZE
	    || s == DEFAULT_LONG_INTEGER_TYPE_SIZE
	    || s == DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE);
    }
  }
  return long_p;
}

bool default_complex_type_p(type t)
{
  bool default_complex_p = false;
  if(!type_undefined_p(t) && type_variable_p(t)) {
    variable v = type_variable(t);
    basic b = variable_basic(v);
    if(basic_complex_p(b)) {
      int s = basic_int(b);

      default_complex_p = ENDP(variable_dimensions(v))
	&& ENDP(variable_qualifiers(v))
	&& s == DEFAULT_COMPLEX_TYPE_SIZE;
    }
  }
  return default_complex_p;
}

bool float_type_p(type t)
{
  bool float_p = false;
  if(!type_undefined_p(t) && type_variable_p(t)) {
    variable v = type_variable(t);
    basic b = variable_basic(v);
    if(basic_float_p(b)) {
      float_p = true;
    }
  }
  return float_p;
}

bool scalar_integer_type_p(type t)
{
  bool long_p = false;
  if(!type_undefined_p(t) && type_variable_p(t)) {
    variable v = type_variable(t);
    basic b = variable_basic(v);
    if(basic_int_p(b)) {

      long_p = ENDP(variable_dimensions(v));
      /* The qualifiers do not matter
	&& ENDP(variable_qualifiers(v))
      */
      /* unsigned are as OK as signed */ /*
	&& (s == DEFAULT_INTEGER_TYPE_SIZE
	    || s == DEFAULT_LONG_INTEGER_TYPE_SIZE
	    || s == DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE);
	  */
    }
  }
  return long_p;
}

bool integer_type_p(type t)
{
  bool int_p = false;
  if(!type_undefined_p(t) && type_variable_p(t)) {
    variable v = type_variable(t);
    basic b = variable_basic(v);
    int_p = basic_int_p(b);
  }
  return int_p;
}

type make_standard_long_integer_type(type t)
{
  if (t == type_undefined)
    {
      variable v = make_variable(make_basic_int(DEFAULT_LONG_INTEGER_TYPE_SIZE),NIL,NIL);
      return make_type_variable(v);
    }
  else
    {
      if (signed_type_p(t) || unsigned_type_p(t) || long_type_p(t))
	{
	  basic b = variable_basic(type_variable(t));
	  int i = basic_int(b);
	  variable v;
	  if (i%10 == DEFAULT_INTEGER_TYPE_SIZE)
	    {
	      /* long */
	      v = make_variable(make_basic_int(10*(i/10)+DEFAULT_LONG_INTEGER_TYPE_SIZE),NIL,NIL);
	      pips_debug(8,"Old basic size: %d, new size : %d\n",i,10*(i/10)+DEFAULT_LONG_INTEGER_TYPE_SIZE);
	    }
	  else
	    {
	      /* long long */
	      v = make_variable(make_basic_int(10*(i/10)+DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE),NIL,NIL);
	      pips_debug(8,"Old basic size: %d, new size : %d\n",i,10*(i/10)+DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE);
	    }
	  return make_type_variable(v);
	}
      else
	{
	  if (bit_type_p(t))
	    /* If it is long int i:5, keep the bit basic type*/
	    return t;
	  else
	    user_warning("Parse error", "Standard long integer types\n");
	  return type_undefined;
	}
    }
}

/* FI: there are different notions of "ultimate" types in C.

   We may need to reduce a type to basic concrete types, removing all
   typedefs wherever they are. This is done by type_to_basic_concrete_type, 
   see below.

   We may also need to know if the type is compatible with a function
   call: we need to chase down the pointers as well as the typedefs. See
   call_compatible_type_p().

   Finally, we may need to know how much memory should be allocated to
   hold an object of this type. This is what was needed first, hence the
   semantics of the function below.

   Shoud this function be extended to return a type_undefined whe nthe
   argument is type_undefined or simply core dump to signal an issue
   as soon as possible? The second alternative is chosen.
 */

/* What type should be used to perform memory allocation? No
   allocation of a new type. */
static type private_ultimate_type(type t, bool arrays_only)
{
  type nt;

  // only under debug, because there is a big impact on performance
  ifdebug(1) pips_assert("type consistent",type_consistent_p(t));

  pips_debug(9, "Begins with type \"%s\"\n", type_to_string(t));

  if(type_variable_p(t)) {
    variable vt = type_variable(t);
    basic bt = variable_basic(vt);

    pips_debug(9, "and basic \"%s\"\n", basic_to_string(bt));

    if(basic_typedef_p(bt)) {
      entity e = basic_typedef(bt);
      type st = entity_type(e);

      if (!arrays_only
	  || (type_variable_p(st) &&!ENDP(variable_dimensions(type_variable(st))) ))
	// recursion
	nt = ultimate_type(st);
      else
	nt = t;

      // FC->SG the following stuff requires more comments to be understandable
      // FC->SG why this #if ???
#if 1
      /* without this test, we would erase the dimension ... */
      if( !ENDP(variable_dimensions(vt) ) )
      {
          /* what should we do ? allocate a new type ...
           * but this breaks the semantic of the function
           * we still create a leak for this case, which does not appear to
           * often a warning is printed out, so that we don't forget it
           */
          // ??? FC->SG why this static structure?
          static size_t holder_iter = 0;
          // ??? FC->SG: why 8? why not 314159?
          //
          // this is creazy programming and a time bomb:-(
          //
          // it seems that the returned allocated type is stored there
          // so that it may be freed some time later, with the hope that by
          // the time it is freed it will not be in use anymore.
          //
          // I would prefer a memory leak in place of this kludge.
          // I would rather suggest to memoize the computed types
          // and not to do this kind of hidden garbage collector.
          static type holder[8] = {// SG: this should avoid the leak
              type_undefined,
              type_undefined,
              type_undefined,
              type_undefined,
              type_undefined,
              type_undefined,
              type_undefined,
              type_undefined
          };
          nt=copy_type(nt);
          holder_iter = 7 & ( 1 + holder_iter ); // too much VHDL? :-(
          variable_dimensions(type_variable(nt)) =
            gen_nconc(gen_full_copy_list(variable_dimensions(vt)),
                      variable_dimensions(type_variable(nt)));
          if (!type_undefined_p(holder[holder_iter]))
            free_type(holder[holder_iter]);
          holder[holder_iter]=nt;
      }
#endif
    }
    else
      nt = t;
  }
  else
    nt = t;

  pips_debug(9, "Ends with type \"%s\"\n", type_to_string(nt));
  ifdebug(9) {
    if(type_variable_p(nt)) {
      variable nvt = type_variable(nt);
      basic nbt = variable_basic(nvt);

      pips_debug(9, "and basic \"%s\"\n", basic_to_string(nbt));
    }
  }

  if (!arrays_only)
    pips_assert("nt is not a typedef",
		type_variable_p(nt)? !basic_typedef_p(variable_basic(type_variable(nt))) : true);

  // only under debug, because there is a big impact on performance
  ifdebug(1) pips_assert("type consistent",type_consistent_p(nt));
  return nt;
}

type ultimate_type(type t)
{
  return private_ultimate_type(t, false);
}

type ultimate_array_type(type t)
{
  return private_ultimate_type(t, true);
}

/************************/
/* basic_concrete_types */
/*                      */
/************************/


/*
  basic_concrete_types are types in which all typedefs are removed and
  consecutive array dimensions are gathered at the same level.  For
  entities, these new types are kept in a hash table where the keys
  are the original entity_types. These is done for performance
  reasons.

  At first, I tried to also keep basic concrete types to
  computed types, however, this could lead to memory leaks or to
  segmentation faults depending of whether the original type was freed
  or not during computations. This is not the same for entity types
  because entities survive to the basic concrete types table. BC.
 */
#define BCTYPES_TABLE_INIT_SIZE 10
static hash_table entity_types_to_bctypes = hash_table_undefined;

/*
  Init and reset functions for the basic_concrete_types table.  They
  are currently called by pipsmake (and callgraph) before and after
  performing a phase on a module. This could also be done after
  parsing and just before closing the database, however, I feard that
  it would grow too much to be efficient, and I lack time to check
  this assumtion. BC.
 */

void entity_basic_concrete_types_init()
{
  pips_assert("types_to_bctypes must be undefined", hash_table_undefined_p(entity_types_to_bctypes));
  entity_types_to_bctypes = hash_table_make(hash_pointer, BCTYPES_TABLE_INIT_SIZE);
}

void entity_basic_concrete_types_reset()
{
  /* First, free all basic concrete types */
  HASH_FOREACH(type, t3, type, t4, entity_types_to_bctypes)
    {
      free_type(t4);
    }
  hash_table_free(entity_types_to_bctypes);
  entity_types_to_bctypes = hash_table_undefined;
}


/**
   computes a new type which is the basic concrete type of the input type
   (this new type is *not* stored in the entity_types_to_bctypes table).

 @param t is a type

 @return : a new type in which typedefs have been expanded to reach a basic
           concrete type, except for struct, union, and enum because
	   the inner types of the fields cannot be changed (they are entities).

*/
type compute_basic_concrete_type(type t)
{
  type nt;
  debug_on("RI-UTIL_DEBUG_LEVEL");
  pips_debug(8, "Begin with type \"%s\"\n",
	     words_to_string(words_type(t, NIL, false)));

  switch (type_tag(t))
    {
    case is_type_variable:
      {
	variable vt = type_variable(t);
	basic bt = variable_basic(vt);
	list lt = variable_dimensions(vt);

	pips_debug(9, "of basic \"%s\"and number of dimensions %d.\n",
		   basic_to_string(bt),
		   (int) gen_length(lt));

	if(basic_typedef_p(bt))
	  {
	    entity e = basic_typedef(bt);
	    type st = compute_basic_concrete_type(entity_type(e));

	    pips_debug(9, "typedef  : %s\n", type_to_string(st));
	    if (type_variable_p(st))
	      {
		nt = st;
		variable_dimensions(type_variable(nt)) =
		  gen_nconc(gen_full_copy_list(lt),
			    variable_dimensions(type_variable(nt)));
	      }
	    else if (type_void_p(st))
	      {
		if (ENDP(lt))
		  nt = st;
		else
		  {
		    nt = copy_type(t);
		    free_type(st);
		  }
	      }
	    else if (type_struct_p(st) || type_union_p(st) || type_enum_p(st))
	      {
		nt = make_type_variable(make_variable(make_basic_derived(e),
						      gen_full_copy_list(lt),
						      gen_full_copy_list(variable_qualifiers(vt))));
		free_type(st);
	      }
	    else
	      {
		free_type(st);
		nt = copy_type(t);
	      }
	  }
	else if(basic_pointer_p(bt))
	  {
	    type npt = compute_basic_concrete_type(basic_pointer(bt));

	     pips_debug(9, "pointer \n");
	     nt = make_type_variable
	       (make_variable(make_basic_pointer(npt),
			      gen_full_copy_list(lt),
			      gen_full_copy_list(variable_qualifiers(vt))));
	  }
	else
	  {
	    pips_debug(9, "other  variable case \n");
	     nt = copy_type(t);
	  }
      }
      break;

    default:
      nt = copy_type(t);
    }

  pips_debug(8, "Ends with type \"%s\"\n",
	     words_to_string(words_type(nt, NIL, false)));
  ifdebug(9)
    {
    if(type_variable_p(nt))
      {
	variable nvt = type_variable(nt);
	basic nbt = variable_basic(nvt);
	list nlt = variable_dimensions(nvt);
	pips_debug(9, "of basic \"%s\"and number of dimensions %d.\n",
		 basic_to_string(nbt),
		 (int) gen_length(nlt));
      }
    }

  /* pips_assert("nt is not a typedef",
	      type_variable_p(nt)?
	      !basic_typedef_p(variable_basic(type_variable(nt))) : true); */
  debug_off();
  return nt;
}

/**
   retrieves or computes and then returns the basic concrete type of an entity

   @param e is the input entity
   @return the basic concrete type of e stored in the entity_types_to_bctypes table.

   @beware don't free this type, it will be freed by pipsmake when resetting the table.
 */
type entity_basic_concrete_type(entity e)
{
  debug_on("RI-UTIL_DEBUG_LEVEL");
  pips_assert("types_to_bctypes must be defined", !hash_table_undefined_p(entity_types_to_bctypes));
  type t = entity_type(e);
  type bct = (type) hash_get(entity_types_to_bctypes, (void *) t);

  if (type_undefined_p(bct))
    {
      bct = compute_basic_concrete_type(t);
      hash_put(entity_types_to_bctypes, t, bct);
    }
  debug_off();
  return bct;
}


/**
   returns true when the input type successors may be pointers

   the input type is supposed to be a basic_concrete_type
 */
bool basic_concrete_type_leads_to_pointer_p(type bct)
{
  bool res = false;

  switch (type_tag(bct))
    {
    case is_type_variable :
      {
	variable v = type_variable(bct);
	basic b = variable_basic(v);

	/* If the basic is a pointer type, return true;
	*/
	if(basic_pointer_p(b))
	  {
	    res = true;
	  }
	else if (basic_derived_p(b))
	  {
	    list l_fields = type_fields(entity_type(basic_derived(b)));
	    while(!res && !ENDP(l_fields))
	      {
		type current_type = entity_basic_concrete_type(ENTITY(CAR(l_fields)));
		// we call ourselves recursively
		res = basic_concrete_type_leads_to_pointer_p(current_type);
		POP(l_fields);
	      }
	  }
	else if (!basic_typedef_p(b))
	  {
	    res = false;
	  }
	else
	  {
	    pips_internal_error("unexpected typedef basic");
	  }
	break;
      }
    case is_type_void:
      {
	res = false;
	break;
      }
    default:
      {
	pips_internal_error("case not handled yet");
      }
    } /*switch */
  return res;
}

/*******************************/
/* end of basic_concrete_types */
/*                             */
/*******************************/


/* Is an object of type t compatible with a call? */
bool call_compatible_type_p(type t)
{
  bool compatible_p = true;

  if(!type_functional_p(t)) {
    if(type_variable_p(t)) {
      basic b = variable_basic(type_variable(t));

      if(basic_pointer_p(b))
	compatible_p = call_compatible_type_p(basic_pointer(b));
      else if(basic_typedef_p(b)) {
	entity te = basic_typedef(b);

	compatible_p = call_compatible_type_p(entity_type(te));
      }
      else
	compatible_p = false;
    }
    else
      compatible_p = false;
  }
  return compatible_p;
}

/* returns the type necessary to generate or check a call to an object
   of type t. Does not allocate a new type. Previous function could be
   implemented with this one. */
type call_compatible_type(type t)
{
  type compatible = t;

  pips_assert("t is a consistent type", type_consistent_p(t));

  if(!type_functional_p(t)) {
    if(type_variable_p(t)) {
      basic b = variable_basic(type_variable(t));

      if(basic_pointer_p(b))
	compatible = call_compatible_type(basic_pointer(b));
      else if(basic_typedef_p(b)) {
	entity te = basic_typedef(b);

	compatible = call_compatible_type(entity_type(te));
      }
      else
	compatible = false;
    }
    else
      compatible = false;
  }
  pips_assert("compatible is a functional type", type_functional_p(compatible));
  pips_assert("compatible is a consistent type", type_consistent_p(compatible));
  return compatible;
}

/* The function called can have a functional type, or a typedef type
   or a pointer type to a functional type. FI: I'm not sure this is
   correctly implemented. I do not know if a new type is always
   allocated. I do not understand the semantics if ultimate is turned
   off.  */
type call_to_functional_type(call c, bool ultimate_p)
{
  entity f = call_function(c);
  type ft = entity_type(f);
  type rt = type_undefined;

  if(type_functional_p(ft))
    rt = entity_type(f);
  else if(type_variable_p(ft)) {
    basic ftb = variable_basic(type_variable(ft));
    if(basic_pointer_p(ftb)) {
      type pt = basic_pointer(ftb);
      rt = ultimate_p? ultimate_type(pt) : copy_type(pt);
    }
    else if(basic_typedef_p(ftb)) {
      entity te = basic_typedef(ftb);
      type ut = ultimate_type(entity_type(te));

      if(type_variable_p(ut)) {
	basic utb = variable_basic(type_variable(ut));
	if(basic_pointer_p(utb)) {
	  type pt = basic_pointer(utb);
	  rt = ultimate_p? ultimate_type(pt): copy_type(pt);
	}
	else
	  /* assertion will fail anyway */
	  free_type(ut);
      }
      else /* must be a functional type */
	rt = ut;
    }
    else {
      pips_internal_error("Basic for called function unknown");
    }
  }
  else
    pips_internal_error("Type for called function unknown");

  pips_assert("The typedef type is functional", type_functional_p(rt));

  return rt;
}
bool type_struct_variable_p(type t)
{
    t = ultimate_type(t);
    return basic_derived_p(variable_basic(type_variable(t))) &&
         entity_struct_p(basic_derived(variable_basic(type_variable(t))));
}

bool type_union_variable_p(type t)
{
    t = ultimate_type(t);
    return basic_derived_p(variable_basic(type_variable(t))) &&
         entity_union_p(basic_derived(variable_basic(type_variable(t))));
}

/* Recursive number of fields in a data structure...
 *
 * union and probably enum are not taken into account.
 *
 * FI: I guess enum should be added
 */
int number_of_fields(type t)
{
  int n = 1;
  type ut = ultimate_type(t);

  if(type_variable_p(ut)) {
    basic ub = variable_basic(type_variable(ut));

    if(basic_derived_p(ub)) {
      entity de = basic_derived(ub);
      type dt = entity_type(de);
      n = number_of_fields(dt);
    }
  }
  else if(type_struct_p(t)) {
    list el = type_struct(t);
    list ce = list_undefined;

    n = 0;
    for(ce = el; !ENDP(ce); POP(ce)) {
      entity fe = ENTITY(CAR(ce));
      type ft = entity_type(fe);
      n += number_of_fields(ft);
    }
  }
  else
    pips_internal_error("Illegal type argument");

  return n;
}

/* Same as above, but arrays in struct are taken into account */
int number_of_items(type t)
{
  int n = 1;
  type ut = ultimate_type(t);

  if(type_variable_p(ut)) {
    variable uv = type_variable(ut);
    basic ub = variable_basic(uv);
    int ne;
    bool ok = NumberOfElements(ub, variable_dimensions(uv), &ne);

    if(basic_derived_p(ub)) {
      entity de = basic_derived(ub);
      type dt = entity_type(de);
      n = number_of_fields(dt);
    }

    if(ok)
      n = n*ne;
    else
      pips_internal_error("Unexpected use of this function");
  }
  else if(type_struct_p(t)) {
    list el = type_struct(t);
    list ce = list_undefined;

    n = 0;
    for(ce = el; !ENDP(ce); POP(ce)) {
      entity fe = ENTITY(CAR(ce));
      type ft = entity_type(fe);
      n += number_of_items(ft);
    }
  }
  else
    pips_internal_error("Illegal type argument");

  return n;
}

/* Compute the list of entities implied in the definition of a
   type. This list is empty for basic types such as int or char. But
   it increases rapidly with typedef, struct, union, bit and dimensions
   which can use enum elements in sizing expressions.

   The supporting entities are gathered in an updated list, sel,
   supporting entity list. If entity a depends on entity b, b must
   appear first in the list. Each entity should appear only once but
   first we keep all occurences to make sure the partial order
   between.entities is respected. */

static set supporting_types = set_undefined;

list recursive_functional_type_supporting_entities(list sel, set vt, functional f)
{
  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  set params = set_make(set_string);
  FOREACH(PARAMETER, p,functional_parameters(f))
  {
      sel = recursive_type_supporting_entities(sel, vt, parameter_type(p));
      dummy d = parameter_dummy(p);
      if(dummy_identifier_p(d))
          set_add_element(params,params,entity_user_name(dummy_identifier(d)));
  }

  sel = recursive_type_supporting_entities(sel, vt, functional_result(f));
  list tmp =NIL;

  FOREACH(ENTITY,e,sel)
  {
      if(!set_belong_p(params,entity_user_name(e)))
          tmp=CONS(ENTITY,e,tmp);
  }
  gen_free_list(sel);
  sel=gen_nreverse(tmp);

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  return sel;
}

list functional_type_supporting_entities(list sel, functional f)
{
  set vt = set_make(hash_pointer);

  sel = recursive_functional_type_supporting_entities(sel, vt, f);

  set_free(vt);

  return sel;
}

list enum_supporting_entities(list sel, set vt, entity e)
{
  type t = entity_type(e);
  list ml = type_enum(t);
  list cm = list_undefined;

  pips_assert("type is of enum kind", type_enum_p(t));

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  for(cm = ml; !ENDP(cm); POP(cm)) {
    entity m = ENTITY(CAR(cm));
    value v = entity_initial(m);
    symbolic s = value_symbolic(v);

    pips_assert("m is an enum member", value_symbolic_p(v));

    sel = constant_expression_supporting_entities(sel, vt, symbolic_expression(s));
  }

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  return sel;
}

list generic_constant_expression_supporting_entities(list sel, set vt, expression e, bool language_c_p)
{
  syntax s = expression_syntax(e);

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  if(syntax_call_p(s)) {
    call c = syntax_call(s);
    entity f = call_function(c);

    if(symbolic_constant_entity_p(f)) {
      if(language_c_p) {
	/* In C, f cannot be declared directly, we need its enum */
	entity e_of_f = find_enum_of_member(f);
	//sel = CONS(ENTITY, e_of_f, sel);
	sel = enum_supporting_entities(sel, vt, e_of_f);
	sel = gen_nconc(sel, CONS(ENTITY, e_of_f, NIL));
      }
      else {
	/* In Fortran, symbolic constant are declared directly, but
	   the may depend on other symbolic constants */
	value v = entity_initial(f);
	symbolic s = value_symbolic(v);

	//sel = CONS(ENTITY, f, sel);
	sel = generic_symbolic_supporting_entities(sel, vt, s, language_c_p);
	sel = gen_nconc(sel, CONS(ENTITY, f, NIL));
      }
    }

    MAP(EXPRESSION, se, {
	sel = generic_constant_expression_supporting_entities(sel, vt, se, language_c_p);
    }, call_arguments(c));
  }
  else if(syntax_reference_p(s)) {
    reference r = syntax_reference(s);
    entity v = reference_variable(r);
    list inds = reference_indices(r);
    /* Could be guarded so as not to be added twice. Guard might be
       useless with because types are visited only once. */
    //sel = gen_nconc(sel, CONS(ENTITY, v, NIL));
    MAP(EXPRESSION, se, {
	sel = generic_constant_expression_supporting_entities(sel, vt, se, language_c_p);
      }, inds);
	sel = gen_nconc(sel, CONS(ENTITY, v, NIL));
  }
  else {
    /* do nothing */
    ;
  }

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  return sel;
}

/* C version */
list constant_expression_supporting_entities(list sel, set vt, expression e)
{
  return generic_constant_expression_supporting_entities(sel, vt, e, true);
}

/* Fortran version */
list fortran_constant_expression_supporting_entities(list sel, expression e)
{
  set vt = set_make(hash_pointer);

  sel = generic_constant_expression_supporting_entities(sel, vt, e, false);

  set_free(vt);

  return sel;
}

list generic_symbolic_supporting_entities(list sel, set vt, symbolic s, bool language_c_p)
{
  expression e = symbolic_expression(s);
  sel = generic_constant_expression_supporting_entities(sel, vt, e, language_c_p);
  return sel;
}

/* C version */
list symbolic_supporting_entities(list sel, set vt, symbolic s)
{
  return generic_symbolic_supporting_entities(sel, vt, s, true);
}

list basic_supporting_entities(list sel, set vt, basic b)
{

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  if(basic_int_p(b) ||
     basic_float_p(b) ||
     basic_logical_p(b) ||
     basic_overloaded_p(b) ||
     basic_complex_p(b) ||
     basic_string_p(b))
    ;
  else if(basic_bit_p(b))
    sel = symbolic_supporting_entities(sel, vt, basic_bit(b));
  else if(basic_pointer_p(b))
    sel = recursive_type_supporting_entities(sel, vt, basic_pointer(b));
  else if(basic_derived_p(b)) {
    //sel = CONS(ENTITY, basic_derived(b), sel);
    sel = recursive_type_supporting_entities(sel, vt, entity_type(basic_derived(b)));
    sel = gen_nconc(sel, CONS(ENTITY, basic_derived(b), NIL));
  }
  else if(basic_typedef_p(b)) {
    entity se = basic_typedef(b);
    //sel = CONS(ENTITY, se, sel);
    sel = recursive_type_supporting_entities(sel, vt, entity_type(se));
    sel = gen_nconc(sel, CONS(ENTITY, se, NIL));
  }
  else
    pips_internal_error("Unrecognized basic tag %d", basic_tag(b));

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  return sel;
}

list variable_type_supporting_entities(list sel, set vt, variable v)
{
  basic b = variable_basic(v);
  list dims = variable_dimensions(v);

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(sel);
    fprintf(stderr, "\n");
  }

  FOREACH(DIMENSION, d, dims) {
    expression l = dimension_lower(d);
    expression u = dimension_upper(d);
    sel = constant_expression_supporting_entities(sel, vt, l);
    sel = constant_expression_supporting_entities(sel, vt, u);
  }

  sel = basic_supporting_entities(sel, vt, b);

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  return sel;
}

list recursive_type_supporting_entities(list sel, set vt, type t)
{

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  if(!set_belong_p(vt, t)) {
    vt = set_add_element(vt, vt, t);
    if(type_functional_p(t))
      sel = recursive_functional_type_supporting_entities(sel, vt, type_functional(t));
    else if(type_variable_p(t))
      sel = variable_type_supporting_entities(sel, vt, type_variable(t));
    else if(type_varargs_p(t)) {
      /* varargs do not depend on any other entities */
      //pips_user_warning("varargs case not implemented yet\n"); /* do nothing? */
      type vart = type_varargs(t);
      sel = recursive_type_supporting_entities(sel, vt, vart);
      ;
    }
    else if(type_void_p(t))
      ;
    else if(type_struct_p(t)) {
      list sse = type_struct(t);

      MAP(ENTITY, se, {
	  sel = recursive_type_supporting_entities(sel, vt, entity_type(se));
	}, sse);
    }
    else if(type_union_p(t)) {
      list use = type_union(t);

      MAP(ENTITY, se, {
	  sel = recursive_type_supporting_entities(sel, vt, entity_type(se));
	}, use);
    }
    else if(type_enum_p(t)) {
      list ese = type_enum(t);

      MAP(ENTITY, se, {
	  sel = recursive_type_supporting_entities(sel, vt, entity_type(se));
	}, ese);
    }
    else if(type_unknown_p(t))
      /* This could be considered a pips_internal_error(), at least when
	 the internal representation is built. */
      ;
    else if(type_statement_p(t))
      /* This is weird, but labels also are declared*/
      ;
    else
      pips_internal_error("Unexpected type with tag %d", type_tag(t));
  }
  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(sel);
    fprintf(stderr, "\n\n");
  }

  return sel;
}

list type_supporting_entities(list sel, type t)
{
  /* keep track of already visited types */
  set vt = set_make(hash_pointer);
  sel = recursive_type_supporting_entities(sel, vt, t);
  set_free(vt);
  return sel;
}

/* Compute the list of references implied in the definition of a
   type. This list is empty for basic types such as int or char. But
   it increases rapidly with typedef, struct, union, bit and
   dimensions which can use enum elements in sizing expressions.

   The supporting entities are gathered in an updated list, sel,
   supporting reference list.

   gen_recurse() does not follow thru entities because they are
   tabulated and persistant.
*/
static list recursive_type_supporting_references(list srl, type t);

list functional_type_supporting_references(list srl, functional f)
{
  ifdebug(9) {
    pips_debug(8, "Begin: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  MAP(PARAMETER, p,
      srl = recursive_type_supporting_references(srl, parameter_type(p)),
      functional_parameters(f));

  srl = recursive_type_supporting_references(srl, functional_result(f));

  ifdebug(9) {
    pips_debug(8, "End: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  return srl;
}

list enum_supporting_references(list srl, entity e)
{
  type t = entity_type(e);
  list ml = type_enum(t);
  list cm = list_undefined;

  pips_assert("type is of enum kind", type_enum_p(t));

  ifdebug(9) {
    pips_debug(8, "Begin: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  for(cm = ml; !ENDP(cm); POP(cm)) {
    entity m = ENTITY(CAR(cm));
    value v = entity_initial(m);
    symbolic s = value_symbolic(v);

    pips_assert("m is an enum member", value_symbolic_p(v));

    srl = constant_expression_supporting_references(srl, symbolic_expression(s));
  }

  ifdebug(9) {
    pips_debug(8, "End: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  return srl;
}

/* Only applicable to C expressions */
list constant_expression_supporting_references(list srl, expression e)
{
  syntax s = expression_syntax(e);

  ifdebug(9) {
    pips_debug(8, "Begin: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  if(syntax_call_p(s)) {
    call c = syntax_call(s);
    entity f = call_function(c);

    if(symbolic_constant_entity_p(f)) {
      /* We need to know if we are dealing with C or Fortran code. */
      /* In C, f cannot be declared directly, we need its enum */
      /* But in Fortran, we are done */
      /* FI: suggested kludge: use a Fortran incompatible type for
	 enum member. But currently they are four byte signed integer (c89)
	 and this Fortran INTEGER type :-( */

      entity e_of_f = find_enum_of_member(f);
      //srl = CONS(ENTITY, e_of_f, srl);
      srl = enum_supporting_references(srl, e_of_f);
    }

    MAP(EXPRESSION, se, {
      srl = constant_expression_supporting_references(srl, se);
    }, call_arguments(c));
  }
  else if(syntax_reference_p(s)) {
    reference r = syntax_reference(s);
    list inds = reference_indices(r);
    /* Could be guarded so as not to be added twice. Guard might be
       useless with because types are visited only once. */
    srl = gen_nconc(srl, CONS(REFERENCE, r, NIL));
    MAP(EXPRESSION, se, {
	srl = constant_expression_supporting_references(srl, se);
      }, inds);
  }
  else {
    /* do nothing for the time being... */
    ;
  }

  ifdebug(9) {
    pips_debug(8, "End: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  return srl;
}

list symbolic_supporting_references(list srl, symbolic s)
{
  expression e = symbolic_expression(s);
  srl = constant_expression_supporting_references(srl, e);
  return srl;
}

list basic_supporting_references(list srl, basic b)
{

  ifdebug(9) {
    pips_debug(8, "Begin: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  if(basic_int_p(b) ||
     basic_float_p(b) ||
     basic_logical_p(b) ||
     basic_overloaded_p(b) ||
     basic_complex_p(b) ||
     basic_string_p(b))
    ;
  else if(basic_bit_p(b))
    srl = symbolic_supporting_references(srl, basic_bit(b));
  else if(basic_pointer_p(b))
    srl = recursive_type_supporting_references(srl, basic_pointer(b));
  else if(basic_derived_p(b)) {
    //srl = CONS(ENTITY, basic_derived(b), srl);
    srl = recursive_type_supporting_references(srl, entity_type(basic_derived(b)));
  }
  else if(basic_typedef_p(b)) {
    entity se = basic_typedef(b);
    //srl = CONS(ENTITY, se, srl);
    srl = recursive_type_supporting_references(srl, entity_type(se));
  }
  else
    pips_internal_error("Unrecognized basic tag %d", basic_tag(b));

  ifdebug(9) {
    pips_debug(8, "End: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  return srl;
}

list variable_type_supporting_references(list srl, variable v)
{
  basic b = variable_basic(v);
  list dims = variable_dimensions(v);

  ifdebug(9) {
    pips_debug(8, "Begin: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  MAP(DIMENSION, d, {
    expression l = dimension_lower(d);
    expression u = dimension_upper(d);
    srl = constant_expression_supporting_references(srl, l);
    srl = constant_expression_supporting_references(srl, u);
  }, dims);

  srl = basic_supporting_references(srl, b);

  ifdebug(9) {
    pips_debug(8, "End: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  return srl;
}

list fortran_type_supporting_entities(list srl, type t)
{
  ifdebug(9) {
    pips_debug(8, "Begin: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  if(type_functional_p(t))
    ;
  else if(type_variable_p(t)) {
    /* In Fortran, dependencies are due to the dimension expressions.*/
    variable v = type_variable(t);
    list dims = variable_dimensions(v);

    FOREACH(DIMENSION, d, dims) {
      expression l = dimension_lower(d);
      expression u = dimension_upper(d);
      srl = fortran_constant_expression_supporting_entities(srl, l);
      srl = fortran_constant_expression_supporting_entities(srl, u);
    }
  }
  else if(type_void_p(t))
    ;
  else
    pips_internal_error("Unexpected Fortran type with tag %d", type_tag(t));

  ifdebug(9) {
    pips_debug(8, "End: ");
    print_references(srl);
    fprintf(stderr, "\n");
  }

  return srl;
}

/* This is not Fortran compatible as enum members and symbolic
   constant appear the same but cannot be dealt with in the same
   way.

   What should be done with the tyoe unknown? Return a empty list or
   generate a pips_internal_error()?
 */
static list recursive_type_supporting_references(list srl, type t)
{
  /* Do not recurse if this type has already been visited. */
  if(!set_belong_p(supporting_types, t)) {
    supporting_types = set_add_element(supporting_types, supporting_types, t);
    ifdebug(9) {
      pips_debug(8, "Begin: ");
      print_references(srl);
      fprintf(stderr, "\n");
    }

    if(type_functional_p(t))
      srl = functional_type_supporting_references(srl, type_functional(t));
    else if(type_variable_p(t))
      srl = variable_type_supporting_references(srl, type_variable(t));
    else if(type_varargs_p(t)) {
      /* No references are involved in C... */
      //pips_user_warning("varargs case not implemented yet\n");
      type vt = type_varargs(t);
      srl = recursive_type_supporting_references(srl, vt);
    }
    else if(type_void_p(t))
      ;
    else if(type_struct_p(t)) {
      list sse = type_struct(t);

      MAP(ENTITY, se, {
	  srl = recursive_type_supporting_references(srl, entity_type(se));
	}, sse);
    }
    else if(type_union_p(t)) {
      list use = type_union(t);

      MAP(ENTITY, se, {
	  srl = recursive_type_supporting_references(srl, entity_type(se));
	}, use);
    }
    else if(type_enum_p(t)) {
      list ese = type_enum(t);

      MAP(ENTITY, se, {
	  srl = recursive_type_supporting_references(srl, entity_type(se));
	}, ese);
    }
    else if(type_unknown_p(t)) {
      pips_internal_error("unknown type left in a declaration");
    }
    else
      pips_internal_error("Unexpected type with tag %d", type_tag(t));

    ifdebug(9) {
      pips_debug(8, "End: ");
      print_references(srl);
      fprintf(stderr, "\n");
    }
  }

  return srl;
}

list type_supporting_references(list srl, type t)
{
  /* To avoid multiple recursion through the same type */
  supporting_types = set_make(set_pointer);
  srl = recursive_type_supporting_references(srl, t);
  set_free(supporting_types);
  return srl;
}



/* Check that an effective parameter list is compatible with a
   function type. Or improve the function type when it is not precise
   as with "extern int f()". This is (a bit/partially) redundant with
   undeclared function detection since undeclared functions are
   declared "extern int f()". */
bool check_C_function_type(entity f, list args)
{
  bool ok = true;
  type t = entity_type(f);
  type ct = call_compatible_type(t);
  list parms = functional_parameters(type_functional(ct));

  pips_assert("f can be used to generate a call", call_compatible_type_p(t));

  if(ENDP(parms)) {
    if(ENDP(args))
      ;
    else {
      if(type_functional_p(t)) {
	/* Use parms to improve the type of f, probably declared f()
	   with no type information. */
	/* Should be very similar to call_to_functional_type(). */
	list pl = NIL;
	MAP(EXPRESSION, e, {
	    type et = expression_to_user_type(e);
	    parameter p = make_parameter(et, make_mode_value(), make_dummy_unknown());
	    pl = gen_nconc(pl, CONS(PARAMETER, p, NIL));
	  },
	  args);
	functional_parameters(type_functional(t)) = pl;

	if(type_unknown_p(functional_result(type_functional(t)))) {
	  basic b = make_basic_int(DEFAULT_INTEGER_TYPE_SIZE);
	  functional_result(type_functional(t)) = make_type_variable(make_variable(b, NIL, NIL));
	}

	pips_user_warning("Type updated for function \"%s\"\n", entity_user_name(f));
	ifdebug(8) {
	  text txt = c_text_entity(get_current_module_entity(), f, 0, NIL);
	  print_text(stderr, txt);
	}
      }
      else {
	/* Must be a typedef or a pointer to a function. No need to refine the type*/
	ifdebug(8) {
	  text txt = c_text_entity(get_current_module_entity(), f, 0, NIL);
	  pips_debug(8, "Type not updated for function \"%s\"\n", entity_user_name(f));
	  print_text(stderr, txt);
	}
      }
    }
  }
  else if(gen_length(args)!=gen_length(parms)) {
    /* Take care of the void case */
    if(gen_length(args)==0 && gen_length(parms)==1) {
      parameter p = PARAMETER(CAR(parms));
      type pt = parameter_type(p);
      ok = type_void_p(pt);
    }
    /* Take care of the varargs case*/
    else if(gen_length(parms) >= 2 && gen_length(args) > gen_length(parms)) {
      parameter lp = PARAMETER(CAR(gen_last(parms)));
      type pt = parameter_type(lp);
      ok = type_varargs_p(pt);
    }
    else
      ok = false;
  }
  else {
    /* Check type compatibility: find function in flint?
       type_equal_p() requires lots of extensions to handle C
       types. And you would probably need type conversion to concrete
       type.*/
    ;
  }

  return ok;
}

/* Number of steps to access the lowest leave of type t.  Number of
 dimensions for an array. One for a struct or an union field, plus its own
 dimension(s). This does not take into account longer memory access paths
 due to pointers. Hence, recursive data structures do not end up with
 MAX_INT as type depth. */
size_t type_depth(type t)
{
  size_t d = 0;

  if(type_variable_p(t)) {
    variable v = type_variable(t);

    d = gen_length(variable_dimensions(v))+basic_depth(variable_basic(v));
  }
  else if(type_void_p(t))
    d = 0;
  else if(type_varargs_p(t))
    d = 0;
  else if(type_struct_p(t)) {
    list fl = type_struct(t);
    d = 0;
    MAP(ENTITY, e, {
	size_t i = type_depth(entity_type(e));
	d = d>i?d:i;
      }, fl);
    d++;
  }
  else if(type_union_p(t)) {
    list fl = type_union(t);
    d = 0;
    MAP(ENTITY, e, {
	size_t i = type_depth(entity_type(e));
	d = d>i?d:i;
      }, fl);
    d++;
  }
  else if(type_enum_p(t))
    d = 0;

  return d;
}

int basic_depth(basic b)
{
  int d = 0;

  switch(basic_tag(b)) {
  case is_basic_int:
  case is_basic_float:
  case is_basic_logical:
  case is_basic_overloaded:
  case is_basic_complex:
  case is_basic_string:
  case is_basic_bit:
  case is_basic_pointer:
    break;
  case is_basic_derived:
    {
      entity e = basic_derived(b);
      type t = entity_type(e);
      d = type_depth(t);
      break;
    }
  case is_basic_typedef:
    {
      entity e = basic_typedef(b);
      type t = entity_type(e);

      d = type_depth(t);
      break;
    }
  default:
    pips_internal_error("Unexpected basic tag %d", basic_tag(b));
  }

  return d;
}

/* Number of steps to access the lowest leave of type t.  Number of
 dimensions for an array. One for a struct or an union field, plus its
 dimension. The difference with type_depth is that it does not stop
 recursing when encountering a pointer, and that a pointer is considered
 as having dimension 1. (BC) */
int effect_type_depth(type t)
{
  int d = 0;

  if(type_variable_p(t)) {
    variable v = type_variable(t);

    d = gen_length(variable_dimensions(v))+effect_basic_depth(variable_basic(v));
  }
  else if(type_void_p(t))
    d = 0;
  else if(type_varargs_p(t))
    d = 0;
  else if(type_struct_p(t)) {
    list fl = type_struct(t);
    d = 0;
    MAP(ENTITY, e, {
	int i = type_depth(entity_type(e));
	d = d>i?d:i;
      }, fl);
    d++;
  }
  else if(type_union_p(t)) {
    list fl = type_union(t);
    d = 0;
    MAP(ENTITY, e, {
	int i = type_depth(entity_type(e));
	d = d>i?d:i;
      }, fl);
    d++;
  }
  else if(type_enum_p(t))
    d = 0;

  return d;
}

int effect_basic_depth(basic b)
{
  int d = 0;

  switch(basic_tag(b)) {
  case is_basic_int:
  case is_basic_float:
  case is_basic_logical:
  case is_basic_overloaded:
  case is_basic_complex:
  case is_basic_string:
  case is_basic_bit:
    break;
  case is_basic_pointer:
    {
      d = effect_type_depth(basic_pointer(b))+1;
      break;
    }
  case is_basic_derived:
    {
      entity e = basic_derived(b);
      type t = entity_type(e);
      d = type_depth(t);
      break;
    }
  case is_basic_typedef:
    {
      entity e = basic_typedef(b);
      type t = entity_type(e);

      d = type_depth(t);
      break;
    }
  default:
    pips_internal_error("Unexpected basic tag %d", basic_tag(b));
  }

  return d;
}

/* Compute the list of types implied in the definition of a
   type. This list is empty for basic types such as int or char. But
   it increases rapidly with typedef, struct and union. We assume that
   expressions in enum, bit and dimensions are not relevant.

   The relationship between stl, the supporting type list, and the set
   vt, already visited types, herited from type_supporting_entities()
   is not clear. It might be better to simply return vt...
*/
static list recursive_type_supporting_types(list stl, set vt, type t);

/* Very basic and crude debugging function */
void print_types(list tl)
{
  bool first_p = true;

  fprintf(stderr, "Type list: ");

  FOREACH(TYPE, t, tl) {
    fprintf(stderr, first_p? "%p" : ", %p", t);
    first_p = false;
  }

  fprintf(stderr, "\n");
}

/* For debugging */
void print_type(type t)
{
  debug_on("RI-UTIL_DEBUG_LEVEL");
  if(t==NULL)
    fprintf(stderr, "type is NULL.\n");
  else if(type_undefined_p(t))
    fprintf(stderr, "type is undefined.\n");
  else if(type_domain_number(t)!=type_domain)
    fprintf(stderr, "The argument is not a type.\n");
  else {
  // Might be better to pass true, or even more information, to see
  // what happens with the unknown type
  list wl = words_type(t, NIL, false);
  dump_words(wl);
  }
  debug_off();
}

static list recursive_functional_type_supporting_types(list stl, set vt, functional f)
{
  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  FOREACH(PARAMETER, p, functional_parameters(f))
    stl = recursive_type_supporting_types(stl, vt, parameter_type(p));

  stl = recursive_type_supporting_types(stl, vt, functional_result(f));

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  return stl;
}

/* FI: I'm not sure this function is of any use */
list functional_type_supporting_types(functional f)
{
  set vt = set_make(hash_pointer);
  list stl = NIL;

  stl = recursive_functional_type_supporting_types(stl, vt, f);

  set_free(vt);

  return stl;
}

static list basic_supporting_types(list stl, set vt, basic b)
{

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  if(basic_int_p(b) ||
     basic_float_p(b) ||
     basic_logical_p(b) ||
     basic_overloaded_p(b) ||
     basic_complex_p(b) ||
     basic_string_p(b))
    ;
  else if(basic_bit_p(b))
    ;
  else if(basic_pointer_p(b))
    stl = recursive_type_supporting_types(stl, vt, basic_pointer(b));
  else if(basic_derived_p(b)) {
    entity de = basic_derived(b);
    type dt = entity_type(de);
    stl = CONS(ENTITY, de, stl);
    stl = recursive_type_supporting_types(stl, vt, dt);
  }
  else if(basic_typedef_p(b)) {
    entity se = basic_typedef(b);
    type st = entity_type(se);
    stl = CONS(ENTITY, se, stl);
    stl = recursive_type_supporting_entities(stl, vt, st);
  }
  else
    pips_internal_error("Unrecognized basic tag %d", basic_tag(b));

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  return stl;
}

static list variable_type_supporting_types(list stl, set vt, variable v)
{
  /* The dimensions cannot contain a type declaration */
  basic b = variable_basic(v);

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  stl = basic_supporting_types(stl, vt, b);

  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  return stl;
}

static list recursive_type_supporting_types(list stl, set vt, type t)
{

  ifdebug(8) {
    pips_debug(8, "Begin: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  if(!set_belong_p(vt, t)) {
    vt = set_add_element(vt, vt, t);
    if(type_functional_p(t))
      stl = recursive_functional_type_supporting_types(stl, vt, type_functional(t));
    else if(type_variable_p(t))
      stl = variable_type_supporting_types(stl, vt, type_variable(t));
    else if(type_varargs_p(t)) {
      /* varargs case is self contained: no supporting type is
	 required. */
      //pips_user_warning("varargs case not implemented yet\n"); /* do nothing? */
      type vart = type_varargs(t);
      stl = recursive_type_supporting_types(stl, vt, vart);
      ;
    }
    else if(type_void_p(t))
      ;
    else if(type_struct_p(t)) {
      list sse = type_struct(t);

      FOREACH(ENTITY, se, sse) {
	stl = recursive_type_supporting_types(stl, vt, entity_type(se));
      }
    }
    else if(type_union_p(t)) {
      list use = type_union(t);

      FOREACH(ENTITY, se, use) {
	stl = recursive_type_supporting_types(stl, vt, entity_type(se));
      }
    }
    else if(type_enum_p(t))
      /* Hopefully, a dummy type declaration cannot be put among enum
	 members declarations. */
      ;
    else if(type_unknown_p(t))
      /* This could be considered a pips_internal_error(), at least when
	 the internal representation is built. */
      ;
    else if(type_statement_p(t))
      /* This is weird, but labels also are declared*/
      ;
    else
      pips_internal_error("Unexpected type with tag %d", type_tag(t));
  }
  ifdebug(8) {
    pips_debug(8, "End: ");
    print_entities(stl);
    fprintf(stderr, "\n");
  }

  return stl;
}

/* Return the list of types used to define type t. The goal is to find
   out if a dummy data structure (struct, union or enum) is used
   within another one and hence does not need to be printed out by the
   prettyprinter. */
list type_supporting_types(type t)
{
  /* keep track of already visited types */
  set vt = set_make(hash_pointer);
  list stl = NIL;
  stl = recursive_type_supporting_types(stl, vt, t);
  set_free(vt);
  return stl;
}

type make_char_array_type(int n)
{
  /* Two options: a string of n characters or an array of n char,
     i.e. int. */
  constant c = make_constant_int(n);
  value val = make_value_constant(c);
  basic b = make_basic_string(val);
  variable var = make_variable(b, NIL, NIL);
  type t = make_type_variable(var);

  return t;
}
bool overloaded_parameters_p(list lparams)
{
  bool overloaded_p = true;

  FOREACH(PARAMETER, p, lparams) {
    type pt = parameter_type(p);

    if(!overloaded_type_p(pt)) {
      overloaded_p = false;
      break;
    }
  }

  return overloaded_p;
}

type type_to_pointer_type(type t)
{
  type pt = make_type_variable(make_variable(make_basic_pointer(t), NIL, NIL));

  pips_assert("pt is consistent", type_consistent_p(pt));

  return pt;
}

/* returns t if t is not a pointer type, and the pointed type if t is
   a pointer type. Type definitions are replaced. If t is undefined,
   returns a type_undefined. */
type type_to_pointed_type(type t)
{
  type upt = type_undefined;

  if(!type_undefined_p(t)) {
    type ut = ultimate_type(t);
    type pt = ut;

    if(pointer_type_p(ut))
      pt = basic_pointer(variable_basic(type_variable(ut)));

    if(!type_undefined_p(pt))
      upt = ultimate_type(pt);
  }
  return upt;
}

/* returns t if t is not a functoional type, and the returned type if t is
   a functional type. Type definitions are replaced. If t is undefined,
   returns a type_undefined. */
type type_to_returned_type(type t)
{
  type urt = type_undefined;

  if(!type_undefined_p(t)) {
    type ut = ultimate_type(t);
    type rt = ut;

    if(type_functional_p(ut))
      rt = functional_result(type_functional(ut));

    if(!type_undefined_p(rt))
      urt = ultimate_type(rt);
  }
  return urt;
}

/* returns t if t is not a pointer type, and the first indirectly
   pointed type that is not a pointer if t is
   a pointer type. Type definitions are replaced. */
type type_to_final_pointed_type(type t)
{
  type ut = type_undefined;

  if(!type_undefined_p(t)) {
    ut = ultimate_type(t);

    while(!type_undefined_p(ut) && pointer_type_p(ut)) {
      ut = type_to_pointed_type(ut);
    }
  }
  return ut;
}

/**
 @param t is the entity_type of a basic_derived
 @return the list of fields of the input type
 */
list derived_type_fields(type t)
{
  list l=NIL;

  switch (type_tag(t))
    {
    case is_type_struct:
      l = type_struct(t);
      break;
    case is_type_union:
      l = type_union(t);
      break;
    case is_type_enum:
      l = type_enum(t);
      break;
    default:
      pips_assert("input type is a struct union, or enum\n",
		  type_struct_p(t) || type_union_p(t) || type_enum_p(t) );
    }
  return l;
}

bool qualifier_equal_p(qualifier q1, qualifier q2)
{
  bool equal_p = qualifier_tag(q1)==qualifier_tag(q2);

  return equal_p;
}

string qualifier_to_string(qualifier q)
{
  string s = string_undefined;
  switch (qualifier_tag(q)) {
  case is_qualifier_register:
    s = "register";
    break;
  case is_qualifier_const:
    s = "const";
    break;
  case is_qualifier_restrict:
    s = "restrict";
    break;
  case is_qualifier_volatile:
    s = "volatile";
    break;
  case is_qualifier_auto:
    s = "auto";
    break;
  default :
    pips_internal_error("unexpected tag %d", qualifier_tag(q));
  }
  return s;
}

/* Check that a qualifier list contains the const qualifier */
bool qualifiers_const_p(list ql)
{
  bool const_p = false;
  FOREACH(QUALIFIER, q, ql) {
    if(qualifier_const_p(q)) {
      const_p = true;
      break;
    }
  }
  return const_p;
}

/* Check that a qualifier list contains the restrict qualifier */
bool qualifiers_restrict_p(list ql)
{
  bool restrict_p = false;
  FOREACH(QUALIFIER, q, ql) {
    if(qualifier_restrict_p(q)) {
      restrict_p = true;
      break;
    }
  }
  return restrict_p;
}

/* Is there a const qualifier associated to type t
 *
 * FI: this should be extended in case const can be carried by a
 * typedef, but this first version should be enough for Molka.
 */
bool type_with_const_qualifier_p(type t)
{
  bool qualifier_p = false;

  if(type_variable_p(t)) {
    variable v = type_variable(t);
    list ql = variable_qualifiers(v);
    qualifier_p = qualifiers_const_p(ql);
  }

  return qualifier_p;
}


/* returns the type associated to se. If se can be evaluated as n,
   returns the type of the nth field in the field list. If se is a
   reference to a field f in fl, returns the type of f.

   if se is an not a statically evaluable integer expression and not a
   field reference, returns type_undefined.

   If n is greater then the number of elements in fl, returns
   type_undefined.

   If f is not in fl, returns type_undefined.

   It might be easier for callers to raise a pips_internal_error()
   when the result is undefined...

   Does not allocate a new type.
*/
static type subscripted_field_list_to_type(list fl, expression se)
{
  type ft = type_undefined;

  if(expression_reference_p(se)) {
    /* Must be a reference to a field */
    entity f = reference_variable(expression_reference(se));

    if(gen_in_list_p(f, fl)) {
      ft = entity_type(f);
    }
    else {
      pips_internal_error("Field f is not in field list fl.");
    }
  }
  else {
    /* Must be an integer expression */
    intptr_t n = 0;
    bool ok = expression_integer_value(se, &n);

    if(ok) {
      entity f = ENTITY(CAR(gen_nthcdr(n-1, fl)));
      ft = entity_type(f);
    }
    else {
      pips_internal_error("Unusable subscript expression for a derived type.");
    }
  }

  if(type_undefined_p(ft))
    pips_internal_error("Ill. arguments");

  return ft;
}

/* Returns the type of an object of type t subscripted by expression
   se.

   For instance if t is a struct, the type returned is the type
   corresponding to the se field.

   It is much more difficult for arrays...

   Do we assume that t is a ultimate type?

   A new type is allocated.

*/
type subscripted_type_to_type(type t, expression se)
{
  type st = type_undefined;

  if(type_variable_p(t)) {
    variable v = type_variable(t);
    list dl = variable_dimensions(v);

    if(ENDP(dl)) {
      /* scalar case */
      basic b = variable_basic(v);

      if(basic_pointer_p(b)) {
	st = copy_type(type_to_pointed_type(t));
      }
      else if(basic_derived_p(b)) {
	entity de = basic_derived(b);
	type det = entity_type(de);
	list fl = list_undefined;

	if(type_struct_p(det)) {
	  fl = type_struct(det); // field list
	}
	else if(type_union_p(det)) {
	  fl = type_union(det); // field list
	}
	else if(type_enum_p(det)) {
	  /* enum cannot be subscripted */
	  pips_internal_error("enum type cannot be subscripted.");
	}
	else {
	  pips_internal_error("This type cannot be subscripted.");
	}
	st = copy_type(subscripted_field_list_to_type(fl, se));
      }
      else {
	/* Other basics are incompatible with subscripts */
	pips_internal_error("This type cannot be subscripted");
      }
    }
    else {
      /* array case */
      pips_internal_error("Not implemented yet.");
    }
  }
  else {
    pips_internal_error("Type t is not a variable type.");
  }
  return st;
}
/* This function returns the ith dimension of a list of dimensions */
dimension find_ith_dimension(list dims, int n)
{
  int i;
  pips_assert("find_ith_dimension", n > 0);
  for(i=1; i<n && !ENDP(dims); i++, POP(dims))
    ;
  if(i==n && !ENDP(dims))
    return DIMENSION(CAR(dims));
  return dimension_undefined;
}

int variable_dimension_number(variable v)
{
  int d = 0;

  FOREACH(DIMENSION, cd, variable_dimensions(v))
    d++;

  return d;
}

/* convert a type "t" into a newly allocated array type "at" whose
 * elements are of type "t", unless "t" is void. Then an array of
 * overloaded elements is generated. The new dimension is unbounded.
 *
 * This useful when dealing with pointers that are not precisely
 * typed. In C you have often to assume they point towards an array
 * since pointer arithmetic is OK by default.
 *
 * The behavior of this function is not well defined when typedef are
 * used... t should be a concrete type.
 */
type type_to_array_type(type t)
{
  type at = type_undefined;

  if(type_void_p(t)) {
    at = MakeTypeOverloaded();
  }
  else
    at = copy_type(t);

  variable vt = type_variable(at);
  dimension d = make_dimension(int_to_expression(0),
			       make_unbounded_expression());
  // Add the new dimension as first dimension
  variable_dimensions(vt) = CONS(DIMENSION, d, variable_dimensions(vt));

  return at;
}

/* returns the type of the elements of an array type, as a newly allocated type.
 *
 * It is not clear if it should fail when the argument is not an array
 * type, or if an undefined type should be returned.
 *
 * The qualifiers are dropped.
 */
type array_type_to_element_type(type t)
{
  type et = type_undefined;
  if(type_variable_p(t)) {
    variable v = type_variable(t);
    //list dl = variable_dimensions(v);
    basic b = variable_basic(v);
    et = make_type_variable(make_variable(copy_basic(b), NIL, NIL));
  }
  else
    pips_internal_error("Ill. arg.\n");
  return et; 
}

/* Minimal information to build a d-dimensional array type. */
list make_unbounded_dimensions(int d)
{
  list dl = NIL;
  int i;
  for(i=0; i<d;i++) {
    dimension d = make_dimension(int_to_expression(0),
				 make_unbounded_expression());
    dl = CONS(DIMENSION, d, dl);
  }
  return dl;
}

bool type_void_star_p(type t)
{
  bool void_star_p = false;
  if(pointer_type_p(t))
    void_star_p = type_void_p(type_to_pointed_type(t));
  return void_star_p;
}

/* Beware of typedefs. */
bool char_star_type_p(type t)
{
  bool char_star_p = false;
  if(pointer_type_p(t))
    char_star_p = char_type_p(type_to_pointed_type(t));
  return char_star_p;
}

/*
 *  that is all
 */
