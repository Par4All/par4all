/*
  $Id$

  Copyright 1989-2011 MINES ParisTech
  Copyright 2011 HPC Project

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
 * This File contains the generic functions necessary for the computation of
 * live paths analyzes.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "pipsdbm.h"
#include "resources.h"

#include "pointer_values.h"
#include "effects-generic.h"



bool live_in_summary_paths_engine(const char* module_name)
{
  list l_glob = NIL, l_loc = NIL;
  statement module_stat;

  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true) );
  module_stat = get_current_module_statement();

  (*effects_computation_init_func)(module_name);

  set_live_in_paths((*db_get_live_in_paths_func)(module_name));

  l_loc = load_live_in_paths_list(module_stat);

  l_glob = (*effects_local_to_global_translation_op)(l_loc);

  (*db_put_live_in_summary_paths_func)(module_name, l_glob);

  reset_current_module_entity();
  reset_current_module_statement();
  reset_live_in_paths();

  (*effects_computation_reset_func)(module_name);

  return true;
}

bool live_out_summary_paths_engine(const char* module_name)
{
  return true;
}

bool live_paths_engine(const char *module_name)
{
  return true;
}
