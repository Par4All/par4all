/*
 Copyright 2012 MINES ParisTech
 Copyright 2012 Silkan

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

/**
 * @file taskify.c
 * Task generation
 * @author Mehdi Amini <mehdi.amini@silkan.com>
 */

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "gpu.h"
#include "accel-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "preprocessor.h"
#include "expressions.h"
#include "text-util.h"
#include "parser_private.h"

static string get_next_task_name() {
  string prefix = (string)get_string_property("TASKIFY_TASK_PREFIX");
  string task_name = build_outline_name(prefix, entity_user_name(get_current_module_entity()));
  return task_name;
}

static bool loop_found_p(loop l, bool *found) {
  *found = true;
  return false;
}

bool taskify_statement(statement s) {
  bool contains_a_loop = false;
  if(!statement_loop_p(s)) {
    gen_context_recurse(s,&contains_a_loop,loop_domain,loop_found_p,gen_false);
  }

  if( !contains_a_loop
      && !declaration_statement_p(s)
      && !return_statement_p(s)
      && !empty_statement_or_continue_p(s)
      && !statement_contains_user_call_p(s)) {
    ifdebug(2) {
      pips_debug(2,"Ouline statement into a new task : ");
      print_statement(s);
    }
    outliner(get_next_task_name(),CONS(statement,s,NIL));
    return false;
  }
  return true;
}

bool taskify(char * module_name) {

  statement module_stat = (statement)db_get_memory_resource(DBR_CODE,
                                                            module_name,
                                                            true);
  set_current_module_statement(module_stat);

  set_current_module_entity(local_name_to_top_level_entity(module_name));

  debug_on("TASKIFY_DEBUG_LEVEL");


  /* regions */
  string region = db_get_memory_resource(DBR_REGIONS, module_name, true);
  set_proper_rw_effects((statement_effects) region);
  set_cumulated_rw_effects((statement_effects) db_get_memory_resource(DBR_CUMULATED_EFFECTS,
                                                                      module_name,
                                                                      true));


  // Save the value of a property we are going to change locally:
  bool old_outline_independent_compilation_unit =
    get_bool_property("OUTLINE_INDEPENDENT_COMPILATION_UNIT");
  set_bool_property("OUTLINE_INDEPENDENT_COMPILATION_UNIT",false);

  /* Initialize set for each statement */
  gen_recurse(module_stat, statement_domain,
      taskify_statement,
      gen_identity);

  // Restore the original property value:
  set_bool_property("OUTLINE_INDEPENDENT_COMPILATION_UNIT",
        old_outline_independent_compilation_unit);


  module_reorder(get_current_module_statement());
  DB_PUT_MEMORY_RESOURCE(DBR_CODE,
      module_name,
      module_stat);

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
       compute_callees(get_current_module_statement()));


  debug_off();

  reset_proper_rw_effects();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_cumulated_rw_effects();

  return true;
}

