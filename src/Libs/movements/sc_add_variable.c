 /* Package MOVEMENTS
 *
 * Corinne Ancourt  - juin 1990
 */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
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
