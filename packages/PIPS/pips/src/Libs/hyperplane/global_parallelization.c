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
/* interfaces with pipsmake for hyperplane transformation
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

static bool always_select() { return true;}
void global_parallelization(module_name)
const char* module_name;
{
    entity module = local_name_to_top_level_entity(module_name);
    statement s;

    pips_assert("global_parallelization", entity_module_p(module));
    s = (statement) db_get_memory_resource(DBR_CODE, module_name, false);
	
    debug_on("HYPERPLANE_DEBUG_LEVEL");

    look_for_nested_loop_statements(s,  (statement (*)(list, bool (*)(statement)))hyperplane, always_select);

    debug_off();

    module_reorder(s);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, 
			   strdup(module_name), 
			   (char*) s);
}

bool
loop_hyperplane(const char* module_name)
{
    bool return_status = false;

    return_status = interactive_loop_transformation(module_name,  (statement (*)(list, bool (*)(statement)))hyperplane);
    
    return return_status;
}
