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
 /* Package MOVEMENTS
 *
 * Corinne Ancourt  - juin 1990
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
   


/* This function adds a new variable to the system of constraints ps.
*/

Variable sc_add_new_variable_name(module,ps)
entity module;
Psysteme ps;
{
     static char name[ 64 ];
    string name1;
    int d;
    entity ent1;
    string full_name;
    d = ps->dimension++;
    name[0] = 'X';
    (void) sprintf(&name[1],"%d",d);
    name1 = strdup(name);

    full_name=strdup(concatenate(module_local_name(module), 
				 MODULE_SEP_STRING,
				 name1,
				 NULL));
    if ((ent1 = gen_find_tabulated(full_name,entity_domain)) == entity_undefined) {
	ent1 = make_scalar_integer_entity(name1,module_local_name(module));
	free(full_name);
    }
    ps->base = vect_add_variable (ps->base,(char *)ent1);

    return((char *)  ent1);

}
