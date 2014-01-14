/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/*
 * this function computes the number of elements of a variable. ld is the
 * list of dimensions of the variable
 */

bool            number_of_elements(ld, the_result)
    list            ld;
    intptr_t            *the_result;
{
    list            pc;
    intptr_t             a_temp_int;
    bool            ok = true;

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
    intptr_t            *the_int;
{
    intptr_t             upper_dim, lower_dim;

    if (expression_integer_value(dimension_upper(d), &upper_dim) &&
	expression_integer_value(dimension_lower(d), &lower_dim)) {
	(*the_int) = upper_dim - lower_dim + 1;
	return (true);
    }
    /* else */
    return (false);
}
/*************************************************************************/

/*
 * some tools to deal with basics and dimensions each function is looking for
 * a basic & a dimension if not found, it replies false if found, that's
 * TRUE. read find_bd_ as "find basic and dimensions" and not find comics!
 */

bool
control_type_in_expression(enum basic_utype a_basic,
			   int __attribute__ ((unused)) a_dim,
			   expression exp)
{
    basic           b;
    list            d;
    intptr_t             n;
    bool            ok_dim = false, ok = find_bd_expression(exp, &b, &d);

    if (ok)
	ok_dim = number_of_elements(d, &n);

    if (ok && ok_dim)
	return ((basic_tag(b) == a_basic) && (n = 1));

    /* else */

    flint_message("control type in expression",
		  "warning : cannot verify the type\n");
    return (true);
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
	return (false);
    }
    *base = variable_basic(type_variable(tp));
    *dims = variable_dimensions(type_variable(tp));

    return (true);
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
	return (false);
    case is_syntax_call:
	c = syntax_call(s);
	return (find_bd_call(c, base, dims));
    default:
	FATAL("find_bd_expression : unexpected tag %u\n", syntax_tag(s));
    }

    return (false);
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
	return (false);

    len_dim = gen_length((*dims));
    if (len_dim < len_ind)
	return (false);

    for (i = 1; i <= len_ind; i++)
	(*dims) = CDR(*dims);
    return (true);
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
	return (false);
    }
    return (find_bd_type_variable(functional_result(type_functional(tp)), base, dims));
}

/*************************************************************************/
/* End of File */
