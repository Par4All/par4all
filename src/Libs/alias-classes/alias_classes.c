/* $Id$
 */

#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "misc.h"
#include "text.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "polyedre.h"
#include "union.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "semantics.h"
#include "transformer.h"

#include "pipsdbm.h"
#include "resources.h"

static list l_alias_lists = NIL;
static list l_alias_classes = NIL;

/*
static void
add_classes_callees(string module_name, list alias_lists)
{
}
*/

bool
alias_classes( string module_name )
{
/*
    list alias_lists;

    pips_debug(4,"begin for module %s\n",module_name);

    alias_lists = effects_to_list((effects)
				  db_get_memory_resource(DBR_ALIAS_LISTS,
							 module_name,
							 TRUE));
    MAP(LIST,alias_list,
    {
    l_alias_lists = gen_nconc(CONS(LIST,regions_dup(alias_list),NIL),l_alias_lists);
    },alias_lists);

    add_classes_callees(module_name,alias_lists);
    */

    return TRUE;
}

