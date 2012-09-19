/*

 $Id $

 Copyright 1989-2012 MINES ParisTech
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
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "pipsdbm.h"
#include "text-util.h"
#include "properties.h"

static statement enclose_in_a_parallel_loop(statement module_statement) {
  if(!statement_block_p(module_statement)) {
    pips_user_error("Module statement is not a block\n");
  }
  entity loop_idx = make_new_scalar_variable(get_current_module_entity(),
                                             make_basic(is_basic_int,
                                                        (void *) 4));
  /* Loop range is created */
  range rg = make_range(int_to_expression(0),
                        int_to_expression(0),
                        int_to_expression(1));
  entity label_entity = entity_empty_label();
  instruction loop_inst = make_instruction(is_instruction_loop,
                                           make_loop(loop_idx,
                                                     rg,
                                                     module_statement,
                                                     label_entity,
                                                     make_execution_parallel(),
                                                     NIL));

  return make_statement(entity_empty_label(),
                        STATEMENT_NUMBER_UNDEFINED,
                        STATEMENT_ORDERING_UNDEFINED,
                        empty_comments,
                        loop_inst,
                        NIL,
                        NULL,
                        empty_extensions());
}

bool one_thread_parallelize(string mod_name) {
  statement module_statement = (statement) db_get_memory_resource(DBR_CODE,
                                                                  mod_name,
                                                                  true);

  set_current_module_statement(module_statement);

  /* Set the current module entity required to have many things
   working in PIPS: */
  set_current_module_entity(module_name_to_entity(mod_name));

  debug_on("ONE_THREAD_PARALLELIZE_DEBUG_LEVEL");
  pips_assert("Statement should be OK at entry...",
              statement_consistent_p(module_statement));

  module_statement = enclose_in_a_parallel_loop(module_statement);
  module_reorder(module_statement);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, (char*) module_statement);

  debug_off();

  reset_current_module_statement();
  reset_current_module_entity();

  return true;
}
