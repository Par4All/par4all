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
    string *s; const char* name=entity_local_name(e);

    if (!top_level_entity_p(e)) return(false);
    for (s=fortran_library; *s!=(string) NULL; s++)
	if (same_string_p(*s, name)) return true;

    return false; /* else not found */
}

/* that is all
 */
