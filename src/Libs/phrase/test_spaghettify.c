/**
 * The spaghettifier is used in context of PHRASE project while creating
 * "Finite State Machine"-like code portions in order to synthetise them
 * in reconfigurables units.
 *
 * This file contains the code used for spaghettify tests.
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


/**
 * Build and return a new control containing condition statement
 * of the "destructured" test
 */
static control make_condition_from_test (test the_test,
					 statement stat)
{
  statement condition_statement;
  test condition_test 
    = make_test (test_condition(the_test),
		 make_continue_statement(entity_empty_label()),
		 make_continue_statement(entity_empty_label()));


  condition_statement = make_statement(entity_empty_label(),
				       statement_number(stat),
				       statement_ordering(stat),
				       empty_comments,
				       make_instruction (is_instruction_test,
							 condition_test),
				       NIL,NULL);  
  
  return make_control (condition_statement, NIL, NIL);
}

/**
 * Build and return a new control containing "if true" statement
 * of the "destructured" test
 */
static control make_if_true_from_test (test the_test, 
				       string module_name)
{
  return make_control (spaghettify_statement(test_true(the_test),
					     module_name),
		       NIL, NIL);
}

/**
 * Build and return a new control containing "if false" statement
 * of the "destructured" test
 */
static control make_if_false_from_test (test the_test, 
					string module_name)
{
  return make_control (spaghettify_statement(test_false(the_test),
					     module_name),
		       NIL, NIL);
}

/**
 * Build and return a new control containing exit statement
 * of the "destructured" test (this is a continue statement)
 */
static control make_exit_from_test ()
{
  return make_control (make_continue_statement(entity_empty_label()), NIL, NIL);
}
/**
 * Build and return a new unstructured coding the
 * "destructured" test
 */
static unstructured make_unstructured_from_test (test the_test, 
						 statement stat, 
						 string module_name) 
{
  control condition = make_condition_from_test (the_test,stat);
  control exit = make_exit_from_test ();
  control if_true = make_if_true_from_test (the_test,module_name);
  control if_false = make_if_false_from_test (the_test,module_name);
  
  /* The first connexion is the FALSE one */
  link_2_control_nodes (condition, if_false); 
  link_2_control_nodes (condition, if_true); 
  link_2_control_nodes (if_true, exit); 
  link_2_control_nodes (if_false, exit); 
  
  return make_unstructured (condition, exit);
}

/* 
 * This function takes the statement stat as parameter and return a new 
 * spaghettized statement, asserting stat is a TEST statement
 */
statement spaghettify_test (statement stat, string module_name)
{
  statement returned_statement = stat;
  instruction unstructured_instruction;
  unstructured new_unstructured;
  
  pips_assert("Statement is TEST in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_test);
  
  pips_debug(2, "spaghettify_test, module %s\n", module_name);
   
  new_unstructured 
    = make_unstructured_from_test 
    (instruction_test(statement_instruction(stat)),
     stat,
     module_name);
  
  unstructured_instruction = make_instruction(is_instruction_unstructured,
					      new_unstructured);
  
  statement_instruction(returned_statement) = unstructured_instruction;
  return returned_statement;
}
