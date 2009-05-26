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

static control make_condition_from_forloop (forloop curLoop,
					    statement stat)
{
  statement condition_statement;

  test condition_test 
    = make_test (forloop_condition(curLoop),
		 make_continue_statement(entity_empty_label()),
		 make_continue_statement(entity_empty_label()));


  condition_statement = make_statement(entity_empty_label(),
				       statement_number(stat),
				       statement_ordering(stat),
				       empty_comments,
				       make_instruction(is_instruction_test,
							condition_test),
				       NIL,NULL,
				       statement_extensions(stat));
  
  return make_control(condition_statement, NIL, NIL);
}

static control make_exit_from_forloop ()
{
  return make_control(make_continue_statement(entity_empty_label()), NIL, NIL);
}

static control make_body_from_forloop (forloop curLoop, 
				       string module_name)
{
  return make_control 
    (spaghettify_statement(forloop_body(curLoop),
			   module_name), NIL, NIL);
}

static unstructured make_unstructured_from_forloop (forloop curLoop, 
						    statement stat,
						    string module_name)
{
  control condition = make_condition_from_forloop(curLoop, stat);
  control exit = make_exit_from_forloop();
  control body = make_body_from_forloop(curLoop,module_name);

  expression loopInit = forloop_initialization(curLoop);
  expression loopCond = forloop_condition(curLoop);
  expression loopInc = forloop_increment(curLoop);

  pips_assert("syntax_call_p(expression_syntax(loopInit))",
	      syntax_call_p(expression_syntax(loopInit)));

  pips_assert("syntax_call_p(expression_syntax(loopCond))",
	      syntax_call_p(expression_syntax(loopCond)));

  pips_assert("syntax_call_p(expression_syntax(loopInc))",
	      syntax_call_p(expression_syntax(loopInc)));

  statement initStat = call_to_statement(syntax_call(expression_syntax(loopInit)));
  statement incStat = call_to_statement(syntax_call(expression_syntax(loopInc)));

  control init_control = make_control(initStat, NIL, NIL);
  control inc_control = make_control(incStat, NIL, NIL);

  link_2_control_nodes (init_control, condition);
  link_2_control_nodes (condition, exit); /* false condition, we exit from forloop */
  link_2_control_nodes (condition, body); /* true condition, we go to body */
  link_2_control_nodes(body, inc_control);
  link_2_control_nodes (inc_control, condition); /* after body, we go back to condition */

  return make_unstructured(init_control, exit);
}

/* 
 * This function takes the statement stat as parameter and return a new 
 * spaghettized statement, asserting stat is a FORLOOP statement
 */
statement spaghettify_forloop (statement stat, string module_name)
{
  statement returned_statement = stat;
  instruction unstructured_instruction;
  unstructured new_unstructured;  

  pips_assert("Statement is FORLOOP in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_forloop);

  pips_debug(2, "spaghettify_forloop, module %s\n", module_name);

  new_unstructured 
    = make_unstructured_from_forloop 
    (instruction_forloop(statement_instruction(stat)),
     stat,
     module_name);
  
  unstructured_instruction = make_instruction(is_instruction_unstructured,
					      new_unstructured);
  
  statement_instruction(returned_statement) = unstructured_instruction;

  return returned_statement;
}
