 /* interface with pipsmake */

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

/* to cope with Yi-qing explicit handling of Psysteme */
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "hyperplane.h"

bool select_loop_nest(l)
loop l;
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

    look_for_nested_loop_statements(s, hyperplane,select_loop_nest);

    debug_off();

    module_reorder(s);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, 
			   strdup(module_name), 
			   (char*) s);


}
