/* SCCS Stuff:
 * $RCSfile: local-ri-util.c,v $ ($Date: 1994/12/06 12:07:57 $, )
 * version $Revision$
 * got on %D%, %T%
 *
 * Fabien Coelho May 1993
 */


#include <stdio.h>
#include <string.h>

extern int fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"

#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "hpfc.h"
#include "defines-local.h"

/*
 * type_variable_dup
 */
type type_variable_dup(t)
type t;
{
    if(type_variable_p(t))
    {
	variable
	    v = type_variable(t);

	return(MakeTypeVariable(variable_basic(v),
				ldimensions_dup(variable_dimensions(v))));
    }
    else
	return(t); /* !!! means sharing */
}

/*  a BASIC tag is returned for the expression
 *  this is a preliminary version. should be improved.
 */
tag suggest_basic_for_expression(e)
expression e;
{
    tag
	result = basic_tag(expression_basic(e));

    if (result==is_basic_overloaded)
    {
	syntax s = expression_syntax(e);

	/*  must be a call
	 */
	pips_assert("suggest_basic_for_expression", syntax_call_p(s));

	if (ENTITY_RELATIONAL_OPERATOR_P(call_function(syntax_call(s))))
	    result = is_basic_logical;
	else
	    /* else some clever analysis could be done
	     */
	    user_warning("suggest_basic_for_expression",
			 "an overloaded is turned into an int...\n"),
	    result = is_basic_int;
    }

    return(result);
}

/*
 * that is all
 */
