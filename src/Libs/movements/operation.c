/*
 * PACKAGE MOVEMENTS
 *
 * Corinne Ancourt  - 1995
 */


#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"

expression
make_div_expression(expression ex1,  cons * ex2)
{ 
    entity div = local_name_to_top_level_entity("idiv");

    return(make_expression(
			   make_syntax(is_syntax_call,
				       make_call(div,
						 CONS(EXPRESSION,
						      ex1,ex2))
				       ),normalized_undefined));
}

expression
make_op_expression(entity op, cons * ex2)
{
 return(make_expression(make_syntax(is_syntax_call,
				    make_call(op,ex2)), 
			normalized_undefined));
}
