/**
 * The spaghettifier is used in context of PHRASE project while creating
 * "Finite State Machine"-like code portions in order to synthetise them
 * in reconfigurables units.
 *
 * This file contains the code used for spaghettify forloops.
 *
 * NOT YET IMPLEMENTED
 */

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "text-util.h"

#include "dg.h"
#include "transformations.h"
#include "properties.h"

#include "control.h"

#include "phrase_tools.h"
#include "spaghettify.h"


/* 
 * This function takes the statement stat as parameter and return a new 
 * spaghettized statement, asserting stat is a FORLOOP statement
 */
statement spaghettify_forloop (statement stat, string module_name)
{
  statement returned_statement = stat;

  pips_assert("Statement is FORLOOP in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_forloop);
  pips_debug(2, "spaghettify_forloop, module %s\n", module_name);   
  pips_user_warning("FORLOOP spaghettifier not yet implemented !!!\n");
  return returned_statement;
}
