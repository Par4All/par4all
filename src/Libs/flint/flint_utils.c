/*
 * flint_utils.c
 * 
 * 
 * updated utils coming from size.c, eval.c, expression.c, constant.c used by
 * flint_check...
 * 
 * L. Aniort and F. Coelho 1992
 * 
 */
/*************************************************************************/

#include "local.h"

/* Print error message and exit from flint */
#define FATAL(msg,value) {fprintf(stderr,msg,value);exit(1);}

/*************************************************************************/
/* External defined functions */
extern value    EvalExpression();
extern bool     expression_integer_value();

/*************************************************************************/
/*
 * this function computes the number of elements of a variable. ld is the
 * list of dimensions of the variable
 */

bool            number_of_elements(ld, the_result)
    list            ld;
    int            *the_result;
{
    list            pc;
    int             a_temp_int;
    bool            ok = TRUE;

    (*the_result) = 1;

    for (pc = ld;
	 (pc != NULL) && (ok = size_of_dimension(DIMENSION(CAR(pc)), &a_temp_int));
	 pc = CDR(pc)) {
	(*the_result) *= a_temp_int;
    }

    return (ok);
}
/*************************************************************************/
/* this function computes the size of a dimension. */

bool            size_of_dimension(d, the_int)
    dimension       d;
    int            *the_int;
{
    int             upper_dim, lower_dim;

    if (expression_integer_value(dimension_upper(d), &upper_dim) &&
	expression_integer_value(dimension_lower(d), &lower_dim)) {
	(*the_int) = upper_dim - lower_dim + 1;
	return (TRUE);
    }
    /* else */
    return (FALSE);
}
/*************************************************************************/

/*
 * some tools to deal with basics and dimensions each function is looking for
 * a basic & a dimension if not found, it replies FALSE if found, that's
 * TRUE. read find_bd_ as "find basic and dimensions" and not find comics!
 */

bool            control_type_in_expression(a_basic, a_dim, exp)
    int             a_basic, a_dim;
    expression      exp;
{
    basic           b;
    list            d;
    int             n;
    bool            ok_dim = FALSE, ok = find_bd_expression(exp, &b, &d);

    if (ok)
	ok_dim = number_of_elements(d, &n);

    if (ok && ok_dim)
	return ((basic_tag(b) == a_basic) && (n = 1));

    /* else */

    flint_message("control type in expression",
		  "warning : cannot verify the type\n");
    return (TRUE);
}
/*******************************************/
bool            find_bd_parameter(param, base, dims)
    parameter       param;
    basic          *base;
    list           *dims;
{
    type            tp = parameter_type(param);
    return (find_bd_type_variable(tp, base, dims));
}
/*******************************************/
bool            find_bd_type_variable(tp, base, dims)
    type            tp;
    basic          *base;
    list           *dims;
{
    if (!type_variable_p(tp)) {
	flint_message("find_bd_type_var",
		 "very strange type encountered, waiting for a variable\n");
	return (FALSE);
    }
    *base = variable_basic(type_variable(tp));
    *dims = variable_dimensions(type_variable(tp));

    return (TRUE);
}
/*******************************************/
bool            find_bd_expression(exp, base, dims)
    expression      exp;
    basic          *base;
    list           *dims;
{
    syntax          s = expression_syntax(exp);
    reference       re;
    call            c;

    switch (syntax_tag(s)) {
    case is_syntax_reference:
	re = syntax_reference(s);
	return (find_bd_reference(re, base, dims));
    case is_syntax_range:
	flint_message("find_bd_expression", "no basic in this expression\n");
	return (FALSE);
    case is_syntax_call:
	c = syntax_call(s);
	return (find_bd_call(c, base, dims));
    default:
	FATAL("find_bd_expression : unexpected tag %d\n", syntax_tag(s));
    }

    return (FALSE);
}
/*******************************************/
bool            find_bd_reference(ref, base, dims)
    reference       ref;
    basic          *base;
    list           *dims;
{
    entity          var = reference_variable(ref);
    list            ind = reference_indices(ref);
    type            tp = entity_type(var);
    int             len_ind = gen_length(ind), len_dim, i;
    bool            ok;

    ok = find_bd_type_variable(tp, base, dims);
    if (!ok)
	return (FALSE);

    len_dim = gen_length((*dims));
    if (len_dim < len_ind)
	return (FALSE);

    for (i = 1; i <= len_ind; i++)
	(*dims) = CDR(*dims);
    return (TRUE);
}
/*******************************************/
bool            find_bd_call(c, base, dims)
    call            c;
    basic          *base;
    list           *dims;
{
    entity          fct = call_function(c);
    type            tp = entity_type(fct);

    if (!type_functional_p(tp)) {
	flint_message("find_bd_call",
		      "very strange function encountered\n");
	return (FALSE);
    }
    return (find_bd_type_variable(functional_result(type_functional(tp)), base, dims));
}

/*************************************************************************/
char           *flint_print_basic(b)
    basic           b;
{
    static char    *strings_of_basic[] =
    {"int", "float", "logical", "overloaded",
    "complex", "string", "unknown tag"};
    int             i = -1;

    switch (basic_tag(b)) {
    case is_basic_int:{
	    i = 0;
	    break;
	}
    case is_basic_float:{
	    i = 1;
	    break;
	}
    case is_basic_logical:{
	    i = 2;
	    break;
	}
    case is_basic_overloaded:{
	    i = 3;
	    break;
	}
    case is_basic_complex:{
	    i = 4;
	    break;
	}
    case is_basic_string:{
	    i = 5;
	    break;
	}
    default:
	pips_error("Anormal basic type of element",
		   "%d found by flinter", basic_tag(b));
    }
    return (strings_of_basic[i]);
}
/*************************************************************************/
/* End of File */
