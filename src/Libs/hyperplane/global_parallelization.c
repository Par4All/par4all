/* interfaces with pipsmake for hyperplane transformation
 *
 * $Id$
 *
 * $Log: global_parallelization.c,v $
 * Revision 1.5  1998/10/09 15:50:52  irigoin
 * New function loop_hyperplane() added as a new interface with pipsmake
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "constants.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"

#include "transformations.h"

#include "hyperplane.h"

static bool 
select_loop_nest(loop l)
{
    return TRUE;
}


void global_parallelization(module_name)
string module_name;
{
    entity module = local_name_to_top_level_entity(module_name);
    statement s;

    pips_assert("global_parallelization", entity_module_p(module));
    s = (statement) db_get_memory_resource(DBR_CODE, module_name, FALSE);
	
    debug_on("HYPERPLANE_DEBUG_LEVEL");

    look_for_nested_loop_statements(s, hyperplane, select_loop_nest);

    debug_off();

    module_reorder(s);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, 
			   strdup(module_name), 
			   (char*) s);
}

bool
loop_hyperplane(string module_name)
{
    bool return_status = FALSE;

    return_status = interactive_loop_transformation(module_name, hyperplane);
    
    return return_status;
}
