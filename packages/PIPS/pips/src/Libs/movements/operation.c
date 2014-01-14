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
