/*
 * $Id$
 *
 * $Log: optimize.c,v $
 * Revision 1.1  1998/04/29 09:07:42  coelho
 * Initial revision
 *
 *
 * expression optimizations by Julien.
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "resources.h"
#include "pipsdbm.h"


#define DEBUG_NAME "TRANSFORMATION_OPTIMIZE_EXPRESSIONS_DEBUG_LEVEL"

bool optimize_expressions(string module_name)
{
    statement s;

    debug_on(DEBUG_NAME);

    /* get needed stuff.
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));

    s = get_current_module_statement();

    /* do something here.
     */
    pips_user_warning("not implemented yet!\n");

    /* return result.
     */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, s);

    reset_current_module_entity();
    reset_current_module_statement();

    debug_off();

    return TRUE;
}
