#include <stdio.h>
extern int fprintf();

#include "genC.h"
#include "ri.h"
#include "misc.h"

#include "ri-util.h"

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

type MakeTypeVariable(b, ld)
basic b;
cons * ld;
{
    return(make_type(is_type_variable, make_variable(b, ld)));
}

/*
 *
 */
basic MakeBasic(the_tag)
int the_tag;
{
    switch(the_tag)
    {
    case is_basic_int: 
	return(make_basic(is_basic_int, 4));
	break;
    case is_basic_float: 
	return(make_basic(is_basic_float, 4));
	break;
    case is_basic_logical: 
	return(make_basic(is_basic_logical, 4));
	break;
    case is_basic_complex: 
	return(make_basic(is_basic_complex, 8));
	break;
    case is_basic_overloaded: 
	return(make_basic(is_basic_overloaded, UU));
	break;
    case is_basic_string: 
	return(make_basic(is_basic_string, string_undefined));
	break;
    default:
	pips_error("MakeBasic", "unexpected basic tag\n");
	break;
    }
    
    return(basic_undefined);
}

/* functions on types */

type MakeTypeArray(b, ld)
basic b;
cons * ld;
{
    return(make_type(is_type_variable, make_variable(b, ld)));
}

parameter MakeOverloadedParameter()
{
    return(make_parameter((MakeTypeArray(make_basic(is_basic_overloaded, 
						    UU), NIL)),
			  make_mode(is_mode_value, UU)));
}

/* this function creates a fortran operator result, i.e. a zero
dimension variable with an overloaded basic type. */

type MakeOverloadedResult()
{
    return(MakeTypeArray(make_basic(is_basic_overloaded, UU), NIL));
}

/*
 * MakeIntegerResult
 *
 * made after MakeOverloadedResult in ri-util/type.c
 */
type MakeIntegerResult()
{
    return(MakeTypeArray(make_basic(is_basic_int, 4), NIL));
}

bool type_equal_p(t1, t2)
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
	return variable_equal_p(type_variable(t1), type_variable(t2));
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

type make_scalar_integer_type(n)
int n;
{
    type t = make_type(is_type_variable,
		       make_variable(make_basic(is_basic_int, n), NIL));
    return t;
}

bool area_equal_p(a1, a2)
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

bool variable_equal_p(v1, v2)
variable v1;
variable v2;
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
	else {
	    /* dimensions should be checked, but it's hard: the only
	       Fortran requirement is that the space allocated in
	       the callers is bigger than the space used in the
	       callee; stars represent any strictly positive integer;
	       we do not know if v1 is the caller type or the callee type;
	       I do not know what should be done; FI */
	    /* FI: I return FALSE because the exact test should never be useful
	       in the parser; 1 February 1994 */
	    return FALSE;
	    pips_error("variable_equal_p", "dimension check not implemented\n");
	}
    }
}

bool basic_equal_p(b1, b2)
basic b1;
basic b2;
{
    if(b1 == b2)
	return TRUE;
    else if (b1 == basic_undefined && b2 != basic_undefined)
	return FALSE;
    else if (b1 != basic_undefined && b2 == basic_undefined)
	return FALSE;
    else if (basic_tag(b1) != basic_tag(b2))
	return FALSE;

    /* assertion: b1 and b2 are defined and have the same tag */
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
	pips_error("basic_equal_p",
		   "string type comparison not implemented\n");
	/* could be a star or an expression; a value_equal_p() is needed! */
	return TRUE;
    default: pips_error("basic_equal_p", "unexpected tag %d\n", basic_tag(b1));
    }
    return FALSE; /* just to avoid a warning */
}

bool functional_equal_p(f1, f2)
functional f1;
functional f2;
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

bool parameter_equal_p(p1, p2)
parameter p1;
parameter p2;
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

bool mode_equal_p(m1, m2)
mode m1;
mode m2;
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

int basic_type_size(b)
basic b;
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
	pips_error("basic_type_size", "undefined for type string\n");
	break;
    default: size = basic_int(b);
	pips_error("basic_type_size", "ill. tag %d\n", basic_tag(b));
	break;
    }

    return size;
}


/*
 *
 */
basic expression_basic(expr)
expression expr;
{
    syntax the_syntax=expression_syntax(expr);

    switch(syntax_tag(the_syntax))
    {
    case is_syntax_reference:
	return(entity_basic(reference_variable(syntax_reference(the_syntax))));
	break;
    case is_syntax_range:
	/* should be int */
	return(expression_basic(range_lower(syntax_range(the_syntax))));
	break;
    case is_syntax_call:
	/*
	 * here is a little problem with pips...
	 * every intrinsics are overloaded, what is not 
	 * exactly what is desired...
	 */
    {
	return(entity_basic(call_function(syntax_call(the_syntax))));
	break;
    }
    default:
	pips_error("expression_basic", "unexpected syntax tag\n");
	break;
    }

    return(basic_undefined);
}

dimension dimension_dup(d)
dimension d;
{
    return(make_dimension(expression_dup(dimension_lower(d)),
			  expression_dup(dimension_upper(d))));
}

list ldimensions_dup(l)
list l;
{
    list 
	result = NIL ;

    MAPL(cd,
     {
	 result = CONS(DIMENSION, dimension_dup(DIMENSION(CAR(cd))), 
		       result);
     },
	 l);

    return(gen_nreverse(result));
}

dimension FindIthDimension(e, i)
entity e;
int i;
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
 * returns the string to declare a basic type.
 */
string basic_to_string(b)
basic b;
{
    switch (basic_tag(b))
    {
    case is_basic_int:
	switch(basic_int(b))
	{
	case 2: return("INTEGER*2") ;
	case 4: return("INTEGER*4") ;
	case 8: return("INTEGER*8") ;
	default: break;
	}
    case is_basic_float:
	switch(basic_float(b))
	{
	case 4: return("REAL*4") ;
	case 8: return("REAL*8") ;
	default: break;
	}
    case is_basic_logical:
	switch(basic_logical(b))
	{
	case 2: return("LOGICAL*2") ;
	case 4: return("LOGICAL*4") ;
	case 8: return("LOGICAL*8") ;
	default: break;
	}
    case is_basic_complex:
	switch(basic_complex(b))
	{
	case 8: return("COMPLEX*8") ;
	case 16: return("COMPLEX*16") ;
	default: break;
	}
    case is_basic_string:
	return("STRING");
    case is_basic_overloaded:
	return("OVERLOADED");
    default: break;
    }

    pips_error("basic_to_string", 
	       "unexpected basic: 0x%x (tag=%d)",
	       b,
	       basic_tag(b));

    return(string_undefined); /* just to avoid a gcc warning */
}

/*
 *  that is all
 */
