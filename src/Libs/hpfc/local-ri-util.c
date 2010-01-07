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

#include "defines-local.h"

type type_variable_dup(t)
type t;
{
    if(type_variable_p(t))
    {
	variable v = type_variable(t);

	return MakeTypeVariable(variable_basic(v),
				ldimensions_dup(variable_dimensions(v)));
    }
    else
	return t; /* !!! means sharing */
}

/*  library functions...
 */
static string fortran_library[] =
{ 
  "TIME",
  (string) NULL
} ;

bool fortran_library_entity_p(e)
entity e;
{
    string *s, name=entity_local_name(e);

    if (!top_level_entity_p(e)) return(FALSE);
    for (s=fortran_library; *s!=(string) NULL; s++)
	if (same_string_p(*s, name)) return TRUE;

    return FALSE; /* else not found */
}

reference expression_to_reference(e)
expression e;
{
    syntax s = expression_syntax(e);
    message_assert("reference", syntax_reference_p(s));
    return syntax_reference(s);
}

list expression_list_to_entity_list(l)
list /* of expressions */ l;
{
    list /* of entities */ n = NIL;
    MAP(EXPRESSION, e, n = CONS(ENTITY, expression_to_entity(e), n), l);
    return gen_nreverse(n);		 
}

list entity_list_to_expression_list(list /* of entities */ l)
{
    list /* of expressions */ n = NIL;
    MAP(ENTITY, e, n = CONS(EXPRESSION, entity_to_expression(e), n), l);
    return gen_nreverse(n);
}

/* that is all
 */
