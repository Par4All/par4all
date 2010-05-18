/*
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

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"

static int nb_parallel_loops = 0;

static loop outer_loop = loop_undefined;

static list privates = NIL;

static bool identify_loop (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    int threshold = get_int_property ("NESTED_PARALLELISM_THRESHOLD");
    nb_parallel_loops ++;
    if (nb_parallel_loops == threshold) {
      outer_loop = l;
      list var = loop_private_variables_as_entites (l, TRUE, TRUE);
      privates = gen_nconc (privates, var);
    } else if  (nb_parallel_loops > threshold) {
      list var = loop_private_variables_as_entites (l, TRUE, TRUE);
      privates = gen_nconc (privates, var);
    }
  }
  return TRUE;
}

static void  process_loop (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    int threshold = get_int_property ("NESTED_PARALLELISM_THRESHOLD");
    if (nb_parallel_loops == threshold) {
      gen_list_and_not (&privates, loop_locals (l));
      loop_locals (l) = gen_nconc (loop_locals (l), privates);
    } else if  (nb_parallel_loops > threshold) {
      execution_tag (loop_execution (l)) = is_execution_sequential;
    }
    nb_parallel_loops --;
  }
}

/**
**/
bool manage_nested_parallelism (const string module_name) {
  debug_on("MANAGE_PARALLEL_LOOPS_DEBUG_LEVEL");
  statement mod_stmt = statement_undefined;
  // Get the code and tell PIPS_DBM we do want to modify it
  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  // Set the current module entity and the current module statement that are
  // required to have many things working in PIPS
  set_current_module_statement(mod_stmt);
  set_current_module_entity(module_name_to_entity(module_name));

  // find the loop
  if (get_int_property ("NESTED_PARALLELISM_THRESHOLD") > 0) {
    gen_recurse(mod_stmt, loop_domain, identify_loop, process_loop);
  }

  privates = NIL;
  outer_loop = loop_undefined;
  nb_parallel_loops = 0;

  // Put the new CODE ressource into PIPS
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, mod_stmt);
  // There is no longer a current module:
  reset_current_module_statement();
  reset_current_module_entity();

  pips_debug(2, "done for %s\n", module_name);
  debug_off();

  return TRUE;
}
