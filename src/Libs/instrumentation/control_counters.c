/*

  $Id$

  Copyright 1989-2011 MINES ParisTech

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

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "ri-util.h"
#include "top-level.h"
#include "text-util.h"
#include "properties.h"
#include "pipsdbm.h"

/* Add Control Counter recursion context
 */
typedef struct {
  entity module;
} acc_ctx;

/* generate: var = var + 1
 */
static statement make_increment_statement(entity var)
{
  // could generate var++ for C modules
  return make_assign_statement
    (entity_to_expression(var),
     MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
                    entity_to_expression(var),
                    int_to_expression(1)));
}

/* create a new integer local variable in module using name as a prefix
 */
static entity create_counter(entity module, const string name)
{
  // build name (could also insert: entity_local_name(module))
  string full = strdup(concatenate("_", name, "_", NULL));
  // create an integer counter
  entity var =
    make_new_scalar_variable_with_prefix(full, module, make_basic_int(4));
  // cleanup
  free(full), full=NULL;
  // initialized to zero
  free_value(entity_initial(var));
  entity_initial(var) = make_value_expression(int_to_expression(0));
  // add as a local variable
  AddEntityToCurrentModule(var);
  return var;
}

/* add a new counter at entry of statement s
 */
static void add_counter(acc_ctx * c, string name, statement s)
{
  entity counter = create_counter(c->module, name);
  instruction i = statement_instruction(s);
  if (instruction_sequence_p(i))
  {
    // insert counter increment ahead of the sequence
    sequence s = instruction_sequence(i);
    sequence_statements(s) =
      CONS(statement,
           make_increment_statement(counter),
           sequence_statements(s));
  }
  else
  {
    // insert a sequence in place
    statement_instruction(s) =
      make_instruction_sequence(make_sequence(
          gen_make_list(statement_domain,
                        make_increment_statement(counter),
                        instruction_to_statement(statement_instruction(s)),
                        NULL)));
  }
}

static void test_rwt(test t, acc_ctx * c) {
  add_counter(c, "if_then", test_true(t));
  add_counter(c, "if_else", test_false(t));
}

static void loop_rwt(loop l, acc_ctx * c) {
  add_counter(c, "do", loop_body(l));
}

static void whileloop_rwt(whileloop w, acc_ctx * c) {
  add_counter(c, "while", whileloop_body(w));
}

static void forloop_rwt(forloop f, acc_ctx * c) {
  add_counter(c, "for", forloop_body(f));
}

/* add control counter instrumentation
 * assumes current module entity & statement are okay.
 */
static void add_counters(entity module, statement root)
{
  acc_ctx c = { module };
  gen_context_multi_recurse
    (root, &c,
     test_domain, gen_true, test_rwt,
     loop_domain, gen_true, loop_rwt,
     whileloop_domain, gen_true, whileloop_rwt,
     forloop_domain, gen_true, forloop_rwt,
     NULL);
}

/* instrument a module with control structure counters for test & loops
 */
bool add_control_counters(string module_name)
{
  // get resources from database
  entity module = module_name_to_entity(module_name);
  statement stat =
    (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_entity(module);
  set_current_module_statement(stat);

  // do the job
  add_counters(module, stat);

  // update resource
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);

  // cleanup
  reset_current_module_entity();
  reset_current_module_statement();
  return true;
}
