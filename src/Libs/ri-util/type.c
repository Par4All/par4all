/*
 * $Id$
 */
#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "misc.h"

#include "ri-util.h"

/* generation of types */

basic 
MakeBasicOverloaded()
{
    return(make_basic(is_basic_overloaded, NIL));
}

mode 
MakeModeReference()
{
    return(make_mode(is_mode_reference, NIL));
}

mode 
MakeModeValue()
{
    return(make_mode(is_mode_value, NIL));
}

type 
MakeTypeStatement()
{
    return(make_type(is_type_statement, NIL));
}

type 
MakeTypeUnknown()
{
    return(make_type(is_type_unknown, NIL));
}

type 
MakeTypeVoid()
{
    return(make_type(is_type_void, NIL));
}


/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

type 
MakeTypeVariable(b, ld)
basic b;
cons * ld;
{
    return(make_type(is_type_variable, make_variable(b, ld,NIL)));
}

/* END_EOLE */

/*
 *
 */
basic 
MakeBasic(the_tag)
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
	pips_error("MakeBasic", "unexpected basic tag: %d\n",
                   the_tag);
	break;
    }
    
    return(basic_undefined);
}

/* functions on types */

type 
MakeTypeArray(b, ld)
basic b;
cons * ld;
{
    return(make_type(is_type_variable, make_variable(b, ld,NIL)));
}

parameter 
MakeOverloadedParameter()
{
    return MakeAnyScalarParameter(is_basic_overloaded, 0);
}

parameter 
MakeIntegerParameter()
{
  return MakeAnyScalarParameter(is_basic_int, DEFAULT_REAL_TYPE_SIZE);
}

parameter 
MakeRealParameter()
{
  return MakeAnyScalarParameter(is_basic_float, DEFAULT_REAL_TYPE_SIZE);
}

parameter 
MakeDoubleprecisionParameter()
{
  return MakeAnyScalarParameter(is_basic_float, DEFAULT_DOUBLEPRECISION_TYPE_SIZE);
}

parameter 
MakeLogicalParameter()
{
  return MakeAnyScalarParameter(is_basic_logical, DEFAULT_LOGICAL_TYPE_SIZE);
}

parameter 
MakeComplexParameter()
{
  return MakeAnyScalarParameter(is_basic_complex, DEFAULT_COMPLEX_TYPE_SIZE);
}

parameter 
MakeDoublecomplexParameter()
{
  return MakeAnyScalarParameter(is_basic_complex, DEFAULT_DOUBLECOMPLEX_TYPE_SIZE);
}

parameter 
MakeCharacterParameter()
{
  return make_parameter(MakeTypeArray(make_basic(is_basic_string, 
	 make_value(is_value_constant,
		    make_constant(is_constant_int,
				  UUINT(DEFAULT_CHARACTER_TYPE_SIZE)))),
				      NIL),
			make_mode(is_mode_reference, UU));
}

parameter 
MakeAnyScalarParameter(tag t, intptr_t size)
{
    return(make_parameter((MakeTypeArray(make_basic(t, UUINT(size)), NIL)),
			  make_mode(is_mode_reference, UU)));
}

/* this function creates a default fortran operator result, i.e. a zero
 * dimension variable with an overloaded basic type.
 */
type 
MakeOverloadedResult()
{
    return MakeAnyScalarResult(is_basic_overloaded, 0);
}

type 
MakeIntegerResult()
{
    return MakeAnyScalarResult(is_basic_int, DEFAULT_INTEGER_TYPE_SIZE);
}

type 
MakeRealResult()
{
    return MakeAnyScalarResult(is_basic_float, DEFAULT_REAL_TYPE_SIZE);
}

type 
MakeDoubleprecisionResult()
{
    return MakeAnyScalarResult(is_basic_float, DEFAULT_DOUBLEPRECISION_TYPE_SIZE);
}

type 
MakeLogicalResult()
{
    return MakeAnyScalarResult(is_basic_logical, DEFAULT_LOGICAL_TYPE_SIZE);
}

type 
MakeComplexResult()
{
    return MakeAnyScalarResult(is_basic_complex, DEFAULT_COMPLEX_TYPE_SIZE);
}

type 
MakeDoublecomplexResult()
{
    return MakeAnyScalarResult(is_basic_complex, DEFAULT_DOUBLECOMPLEX_TYPE_SIZE);
}

type 
MakeCharacterResult()
{
  return MakeTypeArray(make_basic(is_basic_string, 
	 make_value(is_value_constant,
		    make_constant(is_constant_int,
				  UUINT(DEFAULT_CHARACTER_TYPE_SIZE)))),
		       NIL);
}

type 
MakeAnyScalarResult(tag t, intptr_t size)
{
    return MakeTypeArray(make_basic(t, UUINT(size)), NIL);
}


/* Warning: the lengths of string basics are not checked!!!
 * string_type_size() could be used but it is probably not very robust.
 */
bool 
type_equal_p(t1, t2)
type t1;
type t2;
{
    /* the unknown type is not handled in a satisfactory way: in some
       sense, it should be declared equal to any other type but the
       undefined type;

       undefined types could also be seen in a different way, as really
       undefined values; so if t1 or t2 is undefined, the procedure
       should abort 

       Francois Irigoin, 10 March 1992
       */
  bool tequal = FALSE;

    if(t1 == t2)
	return TRUE;
    else if (t1 == type_undefined && t2 != type_undefined)
	return FALSE;
    else if (t1 != type_undefined && t2 == type_undefined)
	return FALSE;
    else if (type_tag(t1) != type_tag(t2))
	return FALSE;

    /* assertion: t1 and t2 are defined and have the same tag */
    switch(type_tag(t1)) {
    case is_type_statement:
	return TRUE;
    case is_type_area:
	return area_equal_p(type_area(t1), type_area(t2));
    case is_type_variable:
      tequal = variable_equal_p(type_variable(t1), type_variable(t2));
      return tequal;
    case is_type_functional:
	return functional_equal_p(type_functional(t1), type_functional(t2));
    case is_type_unknown:
	return TRUE;
    case is_type_void:
	return TRUE;
    default: 
	pips_error("type_equal_p", "unexpected tag %d\n", type_tag(t1));
    }

    return FALSE; /* just to avoid a warning */
}

type make_scalar_integer_type(intptr_t n)
{
    type t = make_type(is_type_variable,
		       make_variable(make_basic(is_basic_int, UUINT(n)), NIL,NIL));
    return t;
}

bool 
area_equal_p(a1, a2)
area a1;
area a2;
{
    if(a1 == a2)
	return TRUE;
    else if (a1 == area_undefined && a2 != area_undefined)
	return FALSE;
    else if (a1 != area_undefined && a2 == area_undefined)
	return FALSE;
    else
	/* layouts are independent ? */
	return (area_size(a1) == area_size(a2));
}

bool
dimension_equal_p(
    dimension d1, 
    dimension d2)
{
    return /* same values */
	same_expression_p(dimension_lower(d1), dimension_lower(d2)) &&
	same_expression_p(dimension_upper(d1), dimension_upper(d2)) &&
	/* and same names... */
	same_expression_name_p(dimension_lower(d1), dimension_lower(d2)) &&
	same_expression_name_p(dimension_upper(d1), dimension_upper(d2));
}

bool 
variable_equal_p(variable v1, variable v2)
{
  if(v1 == v2)
      return TRUE;
  else if (v1 == variable_undefined && v2 != variable_undefined)
      return FALSE;
  else if (v1 != variable_undefined && v2 == variable_undefined)
      return FALSE;
  else if (!basic_equal_p(variable_basic(v1), variable_basic(v2)))
      return FALSE;
  else {
      list ld1 = variable_dimensions(v1);
      list ld2 = variable_dimensions(v2);
      
      if(ld1==NIL && ld2==NIL)
	  return TRUE;
      else 
      {
	  /* dimensions should be checked, but it's hard: the only
	     Fortran requirement is that the space allocated in
	     the callers is bigger than the space used in the
	     callee; stars represent any strictly positive integer;
	     we do not know if v1 is the caller type or the callee type;
	  I do not know what should be done; FI */
	  /* FI: I return FALSE because the exact test should never be useful
	     in the parser; 1 February 1994 */
	  /* FC: I need this in the prettyprinter... */
	  int l1 = gen_length(ld1), l2 = gen_length(ld2);
	  if (l1!=l2) 
	      return FALSE;
	  for (; ld1; POP(ld1), POP(ld2))
	  {
	      dimension d1 = DIMENSION(CAR(ld1)), d2 = DIMENSION(CAR(ld2));
	      if (!dimension_equal_p(d1, d2))
		  return FALSE;
	  }
      }
  }
  return TRUE;
}

bool 
basic_equal_p(basic b1, basic b2)
{
    if(b1 == b2)
	return TRUE;
    else if (b1 == basic_undefined && b2 != basic_undefined)
	return FALSE;
    else if (b1 != basic_undefined && b2 == basic_undefined)
	return FALSE;
    else if (basic_tag(b1) != basic_tag(b2))
	return FALSE;

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
	return TRUE;
    case is_basic_complex:
	return basic_complex(b1) == basic_complex(b2);
    case is_basic_string:
      /* Do we want string types to be equal only if lengths are equal?
       * I do not think so
       */
      /*
	pips_error("basic_equal_p",
		   "string type comparison not implemented\n");
		   */
	/* could be a star or an expression; a value_equal_p() is needed! */
	return TRUE;
    default: pips_error("basic_equal_p", "unexpected tag %d\n", basic_tag(b1));
    }
    return FALSE; /* just to avoid a warning */
}

bool 
functional_equal_p(functional f1, functional f2)
{
    if(f1 == f2)
	return TRUE;
    else if (f1 == functional_undefined && f2 != functional_undefined)
	return FALSE;
    else if (f1 != functional_undefined && f2 == functional_undefined)
	return FALSE;
    else {
	list lp1 = functional_parameters(f1);
	list lp2 = functional_parameters(f2);

	if(gen_length(lp1) != gen_length(lp2))
	    return FALSE;

	for( ; !ENDP(lp1); POP(lp1), POP(lp2)) {
	    parameter p1 = PARAMETER(CAR(lp1));
	    parameter p2 = PARAMETER(CAR(lp2));

	    if(!parameter_equal_p(p1, p2))
		return FALSE;
	}

	return type_equal_p(functional_result(f1), functional_result(f2));
    }
}

bool 
parameter_equal_p(parameter p1, parameter p2)
{
    if(p1 == p2)
	return TRUE;
    else if (p1 == parameter_undefined && p2 != parameter_undefined)
	return FALSE;
    else if (p1 != parameter_undefined && p2 == parameter_undefined)
	return FALSE;
    else
	return type_equal_p(parameter_type(p1), parameter_type(p2))
	    && mode_equal_p(parameter_mode(p1), parameter_mode(p2));
}

bool 
mode_equal_p(mode m1, mode m2)
{
    if(m1 == m2)
	return TRUE;
    else if (m1 == mode_undefined && m2 != mode_undefined)
	return FALSE;
    else if (m1 != mode_undefined && m2 == mode_undefined)
	return FALSE;
    else 
	return mode_tag(m1) == mode_tag(m2);
}

int 
string_type_size(basic b)
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
	pips_internal_error("Non-integer constant to size a string\n");
      break;
    case is_value_unknown:
      /* The size may be unknown as in CHARACTER*(*) */
      /* No way to check it really was a '*'? */
	size = -1;
      break;
    default:
	pips_internal_error("Non-constant value to size a string\n");
    }

    return size;
}

/* See also SizeOfElements() */
int 
basic_type_size(basic b)
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
	pips_error("basic_type_size", "undefined for type overloaded\n");
	break;
    case is_basic_complex: size = basic_complex(b);
	break;
    case is_basic_string: 
      /* pips_error("basic_type_size", "undefined for type string\n"); */
      size = string_type_size(b);
	break;
    default: size = basic_int(b);
	pips_error("basic_type_size", "ill. tag %d\n", basic_tag(b));
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
basic 
expression_basic(expression expr)
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
	 * here is a little problem with pips...
	 * every intrinsics are overloaded, what is not 
	 * exactly what is desired...
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
	pips_internal_error("unexpected syntax tag\n");
	break;
    }

    return b;
}

/* returns an allocated basic.
 */ 
basic
please_give_me_a_basic_for_an_expression(expression e)
{
  basic r = expression_basic(e);
  if(!basic_undefined_p(r)) {
    if (basic_overloaded_p(r)) 
      r = basic_of_expression(e); /* try something else... */
    else
      r = copy_basic(r);
  }
  return r;
}

dimension 
dimension_dup(dimension d)
{
    return(make_dimension(expression_dup(dimension_lower(d)),
			  expression_dup(dimension_upper(d))));
}

list 
ldimensions_dup(list l)
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

dimension 
FindIthDimension(entity e, int i)
{
    cons * pc;

    if (!type_variable_p(entity_type(e))) 
	pips_error("FindIthDimension", "not a variable\n");

    if (i <= 0)
	pips_error("FindIthDimension", "invalid dimension\n");

    pc = variable_dimensions(type_variable(entity_type(e)));

    while (pc != NULL && --i > 0)
	pc = CDR(pc);

    if (pc == NULL) 
	pips_error("FindIthDimension", "not enough dimensions\n");

    return(DIMENSION(CAR(pc)));
}

/*
 * returns a string defining a type.
 */
string 
type_to_string(type t)
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
	return "unknow";
    case is_type_void:
	return "void";
    default: break;
    }

    pips_error("type_to_string", 
	       "unexpected type: 0x%x (tag=%d)",
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

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

/*
 * returns the string to declare a basic type.
 */
string 
basic_to_string(b)
basic b;
{
  /* Nga Nguyen, 19/09/2003: To not rewrite the same thing, I use the words_basic() function*/
  return list_to_string(words_basic(b));
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
basic 
basic_of_any_expression(expression exp, bool apply_p)
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
      entity v = reference_variable(syntax_reference(sy));
      type exp_type = ultimate_type(entity_type(v));

      if(apply_p) {
	if(!type_functional_p(exp_type))
	  pips_internal_error("Bad reference type tag %d \"%s\"\n",
			      type_tag(exp_type));
	else {
	  type rt = functional_result(type_functional(exp_type));
	  type urt = ultimate_type(rt);

	  if(type_variable_p(urt))
	    b = copy_basic(variable_basic(type_variable(urt)));
	  else
	    pips_internal_error("Unexpected type tag %s\n", type_to_string(urt));
	}
      }
      else {
	if(type_variable_p(exp_type))
	  b = copy_basic(variable_basic(type_variable(exp_type)));
	else if(type_functional_p(exp_type)) {
	  /* A reference to a function returns a pointer to a function of the very same time */
	  b = make_basic(is_basic_pointer, copy_type(exp_type));
	}
	else {
	  pips_internal_error("Bad reference type tag %d \"%s\"\n",
			      type_tag(exp_type), type_to_string(exp_type));
	}
      }
      break;
    }
  case is_syntax_call: 
    b = basic_of_call(syntax_call(sy), apply_p);
    break;
  case is_syntax_range: 
    /* Well, let's assume range are well formed... */
    b = basic_of_expression(range_lower(syntax_range(sy)));
    break;
  case is_syntax_cast: 
    {
      type t = cast_type(syntax_cast(sy));
      if (type_tag(t) != is_type_variable)
	pips_internal_error("Bad reference type tag %d\n",type_tag(t));
      b = copy_basic(variable_basic(type_variable(t)));
      break;
    }
  case is_syntax_sizeofexpression: 
    {
      sizeofexpression se = syntax_sizeofexpression(sy);
      if (sizeofexpression_type_p(se))
	{
	  type t = sizeofexpression_type(se);
	  if (type_tag(t) != is_type_variable)
	    pips_internal_error("Bad reference type tag %d\n",type_tag(t));
	  b = copy_basic(variable_basic(type_variable(t)));
	}
      else
	{
	  b = basic_of_expression(sizeofexpression_expression(se));
	}
      break;
    }
  case is_syntax_subscript: 
    {
      b = basic_of_expression(subscript_array(syntax_subscript(sy)));
      break;
    }
  case is_syntax_application:
    {
      b = basic_of_any_expression(application_function(syntax_application(sy)), TRUE);
      break;
    }
  default:
    pips_internal_error("Bad syntax tag %d\n", syntax_tag(sy));
    /* Never go there... */
    b = make_basic(is_basic_overloaded, UUINT(4));
  }

  pips_debug(6, "returns with %s\n", basic_to_string(b));

  return b;
}

/* basic basic_of_expression(expression exp): Makes a basic of the same
 * basic as the expression "exp". Indeed, "exp" will be assigned to
 * a temporary variable, which will have the same declaration as "exp".
 *
 * Does not work if the expression is a reference to a functional entity,
 * as may be the case in a Fortran call.
 *
 * WARNING: a new basic object is allocated
 *
 *  PREFER (???) expression_basic
 *
 */
basic 
basic_of_expression(expression exp)
{
  return basic_of_any_expression(exp, FALSE);
}

type expression_to_type(expression e)
{
  /* does not cover references to functions ...*/
  /* Could be more elaborated with array types for array expressions */
  type t = type_undefined;
  basic b = basic_of_expression(e);
  variable v = make_variable(b, NIL, NIL);

  t = make_type(is_type_variable, v);

  return t;
}

/* basic basic_of_call(call c): returns the basic of the result given by the
 * call "c".
 *
 * WARNING: a new basic is allocated
 */
basic 
basic_of_call(call c, bool apply_p)
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
      b = basic_of_intrinsic(c, apply_p);
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
    default: pips_internal_error("unknown tag %d\n", t);
	/* Never go there... */
    }
    return b;
}



/* basic basic_of_external(call c): returns the basic of the result given by
 * the call to an external function.
 *
 * WARNING: returns a pointer
 */
basic 
basic_of_external(call c)
{
    type return_type = type_undefined;
    entity f = call_function(c);
    basic b = basic_undefined;
    type call_type = entity_type(f);

    pips_debug(7, "External call to %s\n", entity_name(f));

    if (type_tag(call_type) != is_type_functional)
	pips_error("basic_of_external", "Bad call type tag");

    return_type = functional_result(type_functional(call_type));

    if (!type_variable_p(return_type)) {
      if(type_void_p(return_type)) {
	pips_user_error("A subroutine or void returning function is used as an expression\n");
      }
      else {
	pips_internal_error("Bad return call type tag \"%s\"\n", type_to_string(return_type));
      }
    }

    b = (variable_basic(type_variable(return_type)));

    debug(7, "basic_of_call", "Returned type is %s\n", basic_to_string(b));

    return b;
}

/* basic basic_of_intrinsic(call c): returns the basic of the result given
 * by call to an intrinsic function. This basic must be computed with the
 * basic of the arguments of the intrinsic for overloaded operators.  It
 * should be able to accomodate more than two arguments as for generic min
 * and max operators.
 *
 * WARNING: returns a newly allocated basic object */
basic 
basic_of_intrinsic(call c, bool apply_p)
{
  entity f = call_function(c);
  type rt = functional_result(type_functional(entity_type(f)));
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
      basic eb = basic_of_expression(e);
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
	type pt = ultimate_type(basic_pointer(rb));

	free_basic(rb);
	if(type_variable_p(pt) && !apply_p)
	  rb = copy_basic(variable_basic(type_variable(pt)));
	else if(type_functional_p(pt) && apply_p) {
	  type rt = ultimate_type(functional_result(type_functional(pt)));

	  if(type_variable_p(rt))
	    rb = copy_basic(variable_basic(type_variable(rt)));
	  else {
	    /* Too bad for "void"... */
	    pips_internal_error("result type of a functional type must be a variable type\n");
	  }
	}
      }
      else
	pips_internal_error("Dereferencing of a non-pointer expression\n");
    }
    else if(ENTITY_POINT_TO_P(f)) {
      //pips_internal_error("Point to case not implemented yet\n");
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
    else {
      free_basic(rb);
      rb = basic_of_expression(EXPRESSION(CAR(args)));

      MAP(EXPRESSION, arg, {
	basic b = basic_of_expression(arg);
	basic new_rb = basic_maximum(rb, b);

	free_basic(rb);
	free_basic(b);
	rb = new_rb;
      }, CDR(args));
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
basic 
basic_of_constant(call c)
{
    type call_type, return_type;

    debug(7, "basic_of_constant", "Constant call\n");

    call_type = entity_type(call_function(c));

    if (type_tag(call_type) != is_type_functional)
	pips_error("basic_of_constant", "Bad call type tag");

    return_type = functional_result(type_functional(call_type));

    if (type_tag(return_type) != is_type_variable)
	pips_error("basic_of_constant", "Bad return call type tag");

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
basic 
basic_union(expression exp1, expression exp2)
{
  basic b1 = basic_of_expression(exp1);
  basic b2 = basic_of_expression(exp2);
  basic b = basic_maximum(b1, b2);

  free_basic(b1);
  free_basic(b2);
  return b;
}

basic 
basic_maximum(basic fb1, basic fb2)
{
  basic b = basic_undefined;
  basic b1 = fb1;
  basic b2 = fb2;

  if(basic_typedef_p(fb1)) {
    type t1 = ultimate_type(entity_type(basic_typedef(b1)));

    if(type_variable_p(t1))
      b1 = variable_basic(type_variable(t1));
    else
      pips_internal_error("Incompatible basic b1: not really a variable type\n");
  }

  if(basic_typedef_p(fb2)) {
    type t2 = ultimate_type(entity_type(basic_typedef(b2)));

    if(type_variable_p(t2))
      b2 = variable_basic(type_variable(t2));
    else
      pips_internal_error("Incompatible basic b1: not really a variable type\n");
  }

  if(basic_derived_p(fb1)) {
    entity e1 = basic_derived(fb1);

    if(entity_enum_p(e1)) {
      b1 = make_basic(is_basic_int, (void *) 4);
      b = basic_maximum(b1, fb2);
      free_basic(b1);
    }
    else
      pips_internal_error("Unanalyzed derived basic b1\n");
  }

  if(basic_derived_p(fb2)) {
    entity e2 = basic_derived(fb2);

    if(entity_enum_p(e2)) {
      b2 = make_basic(is_basic_int, (void *) 4);
      b = basic_maximum(fb1, b2);
      free_basic(b2);
    }
    else
      pips_internal_error("Unanalyzed derived basic b2\n");
  }

  /* FI: I do not believe this is correct for all intrinsics! */

  pips_debug(7, "Tags: tag exp1 = %td, tag exp2 = %td\n",
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
	intptr_t s1 = basic_logical(b1);
	intptr_t s2 = basic_logical(b2);

	b = make_basic(is_basic_logical,UUINT(s1>s2?s1:s2));
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;

    case is_basic_complex:
      if(basic_complex_p(b2) || basic_float_p(b2) || basic_int_p(b2)) {
	intptr_t s1 = SizeOfElements(b1);
	intptr_t s2 = SizeOfElements(b2);

	b = make_basic(is_basic_complex, UUINT(s1>s2?s1:s2));
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;

    case is_basic_float:
      if(basic_complex_p(b2)) {
	intptr_t s1 = SizeOfElements(b1);
	intptr_t s2 = SizeOfElements(b2);

	b = make_basic(is_basic_complex, UUINT(s1>s2?s1:s2));
      }
      else if(basic_float_p(b2) || basic_int_p(b2)) {
	intptr_t s1 = SizeOfElements(b1);
	intptr_t s2 = SizeOfElements(b2);

	b = make_basic(is_basic_float, UUINT(s1>s2?s1:s2));
      }
      else
	b = make_basic(is_basic_overloaded, UU);
      break;

    case is_basic_int:
      if(basic_complex_p(b2) || basic_float_p(b2)) {
	intptr_t s1 = SizeOfElements(b1);
	intptr_t s2 = SizeOfElements(b2);
	
	b = make_basic(basic_tag(b2), UUINT(s1>s2?s1:s2));
      }
      else if(basic_int_p(b2)) {
	intptr_t s1 = SizeOfElements(b1);
	intptr_t s2 = SizeOfElements(b2);

	b = make_basic(is_basic_int, UUINT(s1>s2?s1:s2));
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
	else if(basic_pointer_p(b2))
	  /* How can we compare two pointer types? Equality? Comparison of the pointed types? */
	  pips_internal_error("Comparison of two pointer types not implemented\n");
	else if(basic_derived_p(b2))
	  pips_internal_error("Comparison between pointer and struct/union not implemented\n");
	else if(basic_typedef_p(b2))
	  pips_internal_error("b2 cannot be a typedef basic\n");
	else
	  pips_internal_error("unknown tag %d for basic b2\n", basic_tag(b2));
      break;
       }
     case is_basic_derived:
      /* How do you compare a structure or a union to another type?
	 The only case which seems to make sense is equality. */
      pips_internal_error("Derived basic b1 it not comparable to another basic\n");
      break;
    case is_basic_typedef:
      pips_internal_error("b1 cannot be a typedef basic\n");
      break;
    default: pips_internal_error("Ill. basic tag %d\n", basic_tag(b1));
    }
  }

  return b;

  /*
    if( (t1 != is_basic_complex) && (t1 != is_basic_float) &&
    (t1 != is_basic_int) && (t2 != is_basic_complex) &&
    (t2 != is_basic_float) && (t2 != is_basic_int) )
    pips_error("basic_union",
    "Bad basic tag for expression in numerical function");

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

/* END_EOLE */

bool 
overloaded_type_p(type t)
{
    pips_assert("overloaded_type_p", type_variable_p(t));

    return basic_overloaded_p(variable_basic(type_variable(t)));
}

/* bool is_inferior_basic(basic1, basic2)
 * return TRUE if basic1 is less complex than basic2
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
	pips_error("is_inferior_basic", "first  basic_undefined\n");
    else if ( b2 == basic_undefined )
	pips_error("is_inferior_basic", "second basic_undefined\n");

    if (basic_overloaded_p(b1))
	return (TRUE);
    else if (basic_overloaded_p(b2))
	return (FALSE);
    else if (basic_logical_p(b1))
	return (TRUE);
    else if (basic_logical_p(b2))
	return (FALSE);
    else if (basic_string_p(b1))
	return (TRUE);
    else if (basic_string_p(b2))
	return (FALSE);
    else if (basic_int_p(b1)) {
	if (basic_int_p(b2))
	    return (basic_int(b1) <= basic_int(b2));
	else
	    return (TRUE);
    }
    else if (basic_float_p(b1)) {
	if (basic_int_p(b2))
	    return (FALSE);
	else if (basic_float_p(b2))
	    return (basic_float(b1) <= basic_float(b2));
	else
	    return (TRUE);
    }
    else if (basic_complex_p(b1)) {
	if (basic_int_p(b2) || basic_float_p(b2))
	    return (FALSE);
	else if (basic_complex_p(b2))
	    return (basic_complex(b1) <= basic_complex(b2));
	else
	    return (TRUE);
    }
    else
	pips_error("is_inferior_basic", "Case never occurs.\n");
    return (TRUE);
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
	    fprintf(stderr, "string: value tag = %td\n", 
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

bool signed_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b)/10 == DEFAULT_SIGNED_TYPE_SIZE)
	  return TRUE;
    }
  return FALSE;
}

bool unsigned_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b)/10 == DEFAULT_UNSIGNED_TYPE_SIZE)
	  return TRUE;
    }
  return FALSE;
}

bool long_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b) == DEFAULT_LONG_INTEGER_TYPE_SIZE)
	  return TRUE;
    }
  return FALSE;
}

bool bit_type_p(type t)
{
  if (!type_undefined_p(t) && type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (!basic_undefined_p(b) && basic_bit_p(b))
	return TRUE;
    }
  return FALSE;
}

bool char_type_p(type t)
{
  bool is_char = FALSE;

  if (!type_undefined_p(t) && type_variable_p(t)) {
    basic b = variable_basic(type_variable(t));
    if (!basic_undefined_p(b) && basic_int_p(b)) {
      int i = basic_int(b);
    is_char = (i==1); /* see words_basic() */
    }
  }
  return is_char;
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

/* What type should be used to perform memory allocation? */
type ultimate_type(type t)
{
  type nt;

  pips_debug(8, "Begins with type \"%s\"\n", type_to_string(t));

  if(type_variable_p(t)) {
    variable vt = type_variable(t);
    basic bt = variable_basic(vt);

    pips_debug(8, "and basic \"%s\"\n", basic_to_string(bt));

    if(basic_typedef_p(bt)) {
      entity e = basic_typedef(bt);
      type st = entity_type(e);

      nt = ultimate_type(st);
    }
    else
      nt = t;
  }
  else
    nt = t;

  pips_debug(8, "Ends with type \"%s\"\n", type_to_string(nt));
  ifdebug(8) {
    if(type_variable_p(nt)) {
      variable nvt = type_variable(nt);
      basic nbt = variable_basic(nvt);

      pips_debug(8, "and basic \"%s\"\n", basic_to_string(nbt));
    }
  }

  pips_assert("nt is not a typedef",
	      type_variable_p(nt)? !basic_typedef_p(variable_basic(type_variable(nt))) : TRUE);

    return nt;
}

/* The function called can have a functional type, or a typedef type or a pointer type */
type call_to_functional_type(call c)
{
  entity f = call_function(c);
  type ft = entity_type(f);
  type rt = type_undefined;

  if(type_functional_p(ft))
    rt = entity_type(f);
  else if(type_variable_p(ft)) {
    basic ftb = variable_basic(type_variable(ft));
    if(basic_pointer_p(ftb)) {
      rt = ultimate_type(basic_pointer(ftb));
    }
    else if(basic_typedef_p(ftb)) {
      entity te = basic_typedef(ftb);
      rt = ultimate_type(entity_type(te));
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
/*
 *  that is all
 */
