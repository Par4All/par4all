/*

  $Id$

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
/**
 * The spaghettifier is used in context of PHRASE project while
 * creating "Finite State Machine"-like code portions in order to synthetise
 * them in reconfigurables units.
 *
 * This file contains the code used for spaghettify whileloops.
 *
 * General syntax of whileloop in Fortran are:
 *
 * DO WHILE CONDITION
 *   STATEMENT
 * END DO
 *
 * Following code is generated:
 *
 * 10 IF (CONDITION) THEN
 *      STATEMENT
 *      GOTO 10
 * 20 CONTINUE
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

#include "text-util.h"

#include "dg.h"


#include "phrase_tools.h"
#include "spaghettify.h"


/**
 * Build and return a new control containing condition statement
 * of the unstructured whileloop
 */
static control make_condition_from_whileloop (whileloop the_whileloop,
					      statement stat)
{
  statement condition_statement;
  test condition_test
    = make_test (whileloop_condition(the_whileloop),
		 make_continue_statement(entity_empty_label()),
		 make_continue_statement(entity_empty_label()));


  condition_statement = make_statement(entity_empty_label(),
				       statement_number(stat),
				       statement_ordering(stat),
				       empty_comments,
				       make_instruction (is_instruction_test,
							 condition_test),
				       NIL,NULL,
				       statement_extensions(stat), make_synchronization_none());
  return make_control (condition_statement, NIL, NIL);
}

/**
 * Build and return a new control containing exit statement
 * of the unstructured whileloop (this is a continue statement)
 */
static control make_exit_from_whileloop ()
{
  return make_control (make_continue_statement(entity_empty_label()), NIL, NIL);
}

/**
 * Build and return a new control containing body statement
 * of the unstructured whileloop
 */
static control make_body_from_whileloop (whileloop the_whileloop,
					 const char* module_name)
{
  return make_control
    (spaghettify_statement(whileloop_body(the_whileloop),
			   module_name), NIL, NIL);
}

/**
 * Build and return a new unstructured coding the
 * "destructured" whileloop
 */
static unstructured make_unstructured_from_whileloop (whileloop the_whileloop,
						      statement stat,
						      const char* module_name)
{
  control condition = make_condition_from_whileloop (the_whileloop,stat);
  control exit = make_exit_from_whileloop();
  control body = make_body_from_whileloop(the_whileloop,module_name);

  /* The first connexion is the false one */
  //link_2_control_nodes (condition, exit); /* false condition, we exit from whileloop */
  //link_2_control_nodes (condition, body); /* true condition, we go to body */
  link_3_control_nodes (condition, body, exit);

  return make_unstructured (condition, exit);
}

/*
 * This function takes the statement stat as parameter and return a new
 * spaghettized statement, asserting stat is a WHILELOOP statement
 */
statement spaghettify_whileloop (statement stat, const char* module_name)
{
  statement returned_statement = stat;
  instruction unstructured_instruction;
  unstructured new_unstructured; 

  pips_assert("Statement is WHILELOOP in FSM_GENERATION",
	      instruction_tag(statement_instruction(stat))
	      == is_instruction_whileloop);

  pips_debug(2, "spaghettify_whileloop, module %s\n", module_name);  
  new_unstructured
    = make_unstructured_from_whileloop
    (instruction_whileloop(statement_instruction(stat)),
     stat,
     module_name);
 
  unstructured_instruction = make_instruction(is_instruction_unstructured,
					      new_unstructured);
 
  statement_instruction(returned_statement) = unstructured_instruction;
 
  return returned_statement;
}
