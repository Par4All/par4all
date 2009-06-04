/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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


#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"

/// @brief generate pragma recursively for the given statement
/// @return void
/// @param stmt, the statement to go throught
void generate_omp_pragma (statement stmt) {
  text        t    = text_undefined;
  string      str  = string_undefined;
  statement   body = statement_undefined;
  instruction inst = statement_instruction (stmt);

  if (inst != instruction_undefined) {
    switch (instruction_tag (inst)) {
    case is_instruction_sequence:
      MAP(STATEMENT, s,
      	  {
      	    generate_omp_pragma (s);
      	  },
      	  instruction_block (inst)
      	  ); // end of MAP
      break;
    case is_instruction_loop:
      // get the pragma as text and convert to string
      t = text_omp_directive (instruction_loop (inst), 0);
      str = text_to_string (t);
      // insert the pragma as a string to the current statement
      add_pragma_to_statement (stmt, str, FALSE);
      pips_debug (5, "new pragma as an extension added: %s \n", str);
      // apply on the body
      body = loop_body (instruction_loop (inst));
      if (body != statement_undefined) generate_omp_pragma (body);
      break;
    case is_instruction_forloop:
      pips_assert ("is_instruction_forloop case need to be implemented",
		   FALSE);
      break;
    case is_instruction_call:
    case is_instruction_test:
    case is_instruction_whileloop:
    case is_instruction_goto:
    case is_instruction_expression:
      break;
    default:
      pips_assert ("not handeled for case", FALSE);
    }
  }
  return;
}

/// @brief generate a sequential code from a parrallel one Basically do only a
/// DBR_CODE(mod_name) = (DBR_CODE) DBR_PARALLELIZED_CODE(mod_name) and
/// insert some pragmas.
/// @return void
/// @param mod_name, the module to sequentialized
/// @param omp, generate omp pragma if set to TRUE
void parallel_to_sequential (char mod_name[], bool omp)
{
  statement mod_stmt;

  /* Get the parallelized code and tell PIPS_DBM we do not want to
     modify it: */
  mod_stmt = (statement) db_get_memory_resource(DBR_PARALLELIZED_CODE,
						mod_name,
						FALSE);

  // generate omp pragma if required
  if (omp == true) {
    generate_omp_pragma (mod_stmt);
  }

  /* Reorder the module, because new statements have been generated. */
  /* module_reorder(mod_stmt); */

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stmt);

  pips_debug(2,"done for %s\n", mod_name);
}


bool ompify_code (char mod_name[])
{
  debug_on("OPMIFY_CODE_DEBUG_LEVEL");

  // we want omp syntax so save and change the current PRETTYPRINT_PARALLEL
  // property
  string previous = strdup(get_string_property("PRETTYPRINT_PARALLEL"));
  set_string_property("PRETTYPRINT_PARALLEL", "omp");

  parallel_to_sequential (mod_name, TRUE);

  // Restore the previous PRETTYPRINT_PARALLEL property for the next
  set_string_property("PRETTYPRINT_PARALLEL", previous);
  free(previous);

  pips_debug(2, "done for %s\n", mod_name);
  debug_off();

  return TRUE;
}
