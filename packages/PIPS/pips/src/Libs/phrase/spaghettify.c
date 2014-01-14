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
/**
 * The spaghettifier is used in context of PHRASE project while
 * creating "Finite State Machine"-like code portions in order to synthetise
 * them in reconfigurables units.
 *
 * This phase transforms structured code portions (eg. loops) in
 * unstructured statements.
 * 
 * To add flexibility, the behavior of \texttt{spaghettifier} is
 * controlled by the properties 
 * - DESTRUCTURE_TESTS
 * - DESTRUCTURE_LOOPS
 * - DESTRUCTURE_WHILELOOPS
 * - DESTRUCTURE_FORLOOPS
 * to allow more or less destruction power !
 *
 * spaghettify          > MODULE.code
 *       < PROGRAM.entities
 *       < MODULE.code
 *
 */

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
#include "pipsdbm.h"

#include "text-util.h"

#include "dg.h"
#include "properties.h"

#include "control.h"
#include "callgraph.h"
#include "spaghettify.h"
#include "phrase_tools.h"

/* 
 * This function is recursively called during spaghettization.
 * It takes the statement stat as parameter and return a new 
 * spaghettized statement (or the same if nothing has been done).
 * Spaguettization is done:
 * - on Tests (if property DESTRUCTURE_TESTS set to true)
 * - on Loops (if property DESTRUCTURE_LOOPS set to true)
 * - on WhileLoops (if property DESTRUCTURE_WHILELOOPS set to true)
 * - on ForLoops (if property DESTRUCTURE_FORLOOPS set to true)
 */
statement spaghettify_statement (statement stat, const char* module_name)
{
  // Defaut behaviour is to return parameter statement stat
  statement returned_statement = stat;
  instruction i = statement_instruction(stat);

  pips_debug(2,"\nSPAGHETTIFY: Module statement: =====================================\n");
  ifdebug(2) {
    print_statement(stat);
  }
  pips_debug(2,"domain number = %"PRIdPTR"\n", statement_domain_number(stat));
  pips_debug(2,"entity = UNDEFINED\n");
  pips_debug(2,"statement number = %"PRIdPTR"\n", statement_number(stat));
  pips_debug(2,"statement ordering = %"PRIdPTR"\n", statement_ordering(stat));
  if (statement_with_empty_comment_p(stat)) {
    pips_debug(2,"statement comments = EMPTY\n");
  }
  else {
    pips_debug(2,"statement comments = %s\n", statement_comments(stat));
  }
  pips_debug(2,"statement instruction = %s\n", statement_type_as_string(stat));
  switch (instruction_tag(i)) {
  case is_instruction_test: 
    {
      pips_debug(2, "TEST\n");   
    if (get_bool_property("DESTRUCTURE_TESTS")) {
      returned_statement = spaghettify_test (stat, module_name);
    }
      break;
  }
  case is_instruction_sequence: 
    {
      sequence seq = instruction_sequence(i);
      pips_debug(2, "SEQUENCE\n");   
      MAP(STATEMENT, current_stat,
      {
	statement new_stat = spaghettify_statement(current_stat, module_name);
	if (new_stat != NULL) {
	  gen_list_patch (sequence_statements(seq), current_stat, new_stat);
	}
      }, sequence_statements(seq));
      break;
    }
  case is_instruction_loop: {
    pips_debug(2, "LOOP\n");   
    if (get_bool_property("DESTRUCTURE_LOOPS")) {
      returned_statement = spaghettify_loop (stat, module_name);
    }
    break;
  }
  case is_instruction_whileloop: {
    pips_debug(2, "WHILELOOP\n");   
    if (get_bool_property("DESTRUCTURE_WHILELOOPS")) {
      returned_statement = spaghettify_whileloop (stat, module_name);
    }
    break;
  }
  case is_instruction_forloop: {
    pips_debug(2, "FORLOOP\n");   
    if (get_bool_property("DESTRUCTURE_FORLOOPS")) {
      returned_statement = spaghettify_forloop (stat, module_name);
    }
    break;
  }
  case is_instruction_call: {
    pips_debug(2, "CALL\n");   
    break;
  }
  case is_instruction_unstructured: {
    pips_debug(2, "UNSTRUCTURED\n");  
    break;
  }
  case is_instruction_goto: {
    pips_debug(2, "GOTO\n");   
    break;
  }
  default:
    pips_debug(2, "UNDEFINED\n");   
    break;
  }

  return returned_statement;
}

/*********************************************************
 * Phase main
 *********************************************************/

bool spaghettify(const char* module_name)
{
   /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      true);

  set_current_module_statement(stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  
  debug_on("SPAGUETTIFY_DEBUG_LEVEL");

  /* Now do the job */  
  stat = spaghettify_statement(stat,module_name);

  pips_assert("Statement is consistent after SPAGUETTIFY", 
	       statement_consistent_p(stat));
  
  /* Reorder the module, because new statements have been added */  
  module_reorder(stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, 
			 compute_callees(stat));
  
  /* update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
  
  debug_off();
  
  return true;
}
