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
/*
 *
 * This phase is used for PHRASE project.
 *
 * NB: The PHRASE project is an attempt to automatically (or
 * semi-automatically) transform high-level language for partial
 * evaluation in reconfigurable logic (such as FPGAs or DataPaths).
 *
 * This file provides functions used in context of FSM generation/modifications
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
#include "fsm_generation.h"

/**
 * Build and return an expression (eg. state = 23), given an entity
 * state_variable, an int value value, and an intrinsic name
 */
expression make_expression_with_state_variable(entity state_variable,
					       int value,
					       string intrinsic_name)
{
  return MakeBinaryCall (entity_intrinsic(intrinsic_name),
			 entity_to_expression (state_variable),
			 int_to_expression (value));
}

/**
 * This function creates (and add declaration) state variable.
 * The name of this variable is obtained by the concatenation of
 * string STATE_VARIABLE_NAME and name identifier.
 * If the variable doesn't exist with this name, then the variable
 * is created, added to declarations, and returned. If this variable
 * exists, then this functions search a new name by incrementing the
 * integer name_identifier
 */
entity create_state_variable (const char* module_name,
			      int name_identifier)
{
  entity module;
  entity new_state_variable;
  string state_variable_name;
  //char *buffer;

  module = module_name_to_entity(module_name);

  /* Assert that module represent a value code */
  pips_assert("it is a code", value_code_p(entity_initial(module)));

  if (name_identifier == 0) {
    state_variable_name = strdup (STATE_VARIABLE_NAME_NO_REF);
  }
  else {
    asprintf(&state_variable_name, STATE_VARIABLE_NAME, name_identifier);
  }
 
  if ((gen_find_tabulated(concatenate(module_name,
				      MODULE_SEP_STRING,
				      state_variable_name,
				      NULL),
			  entity_domain)) == entity_undefined) {
   
    /* This variable doesn't exist */
   
    new_state_variable = find_or_create_scalar_entity (state_variable_name,
						       module_name,
						       is_basic_int);
    AddEntityToDeclarations( new_state_variable,module);
    return new_state_variable;
  }
  else {
    return create_state_variable (module_name,
				  name_identifier+1);
  }
}

/**
 * This function build and return a statement representing the
 * initial assigment of the state_variable, given the UNSTRUCTURED
 * statement stat.
 */
statement make_state_variable_assignement_statement (statement stat,
						     entity state_variable,
						     int assignement_value)
{
  statement returned_statement;
  instruction new_instruction;
  call assignment_call;

  assignment_call = make_call (entity_intrinsic(ASSIGN_OPERATOR_NAME),
			       CONS(EXPRESSION,
				    entity_to_expression(state_variable),
				    CONS(EXPRESSION, int_to_expression (assignement_value), NIL)));
 
  new_instruction
    = make_instruction(is_instruction_call,
		       assignment_call);
 
  returned_statement = make_statement(entity_empty_label(),
				      /*statement_label(stat),*/
				      statement_number(stat),
				      statement_ordering(stat),
				      empty_comments,
				      new_instruction,
				      NIL,NULL,
				      statement_extensions(stat), make_synchronization_none());

  return returned_statement;
}

/**
 * Return the state variable value corresponding to the entry
 * in a unstructured statement
 */
int entry_state_variable_value_for_unstructured (statement stat)
{
  unstructured u;

  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION",
	      instruction_tag(statement_instruction(stat))
	      == is_instruction_unstructured);

  u = instruction_unstructured(statement_instruction(stat));

  return  beautify_ordering (statement_ordering(control_statement(unstructured_entry(u))));
}

/**
 * Return the state variable value corresponding to the exit
 * in a unstructured statement
 * NB: always return 0
 */
int exit_state_variable_value_for_unstructured (statement stat)
{
  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION",
	      instruction_tag(statement_instruction(stat))
	      == is_instruction_unstructured);

  return 0;
}

/**
 * This function build and return a statement representing the
 * initial assigment of the state_variable, given the UNSTRUCTURED
 * statement stat.
 */
statement make_reset_state_variable_statement (statement stat,
					       entity state_variable)
{
  return make_state_variable_assignement_statement
    (stat,
     state_variable,
     entry_state_variable_value_for_unstructured(stat));
}

/**
 * This function build a transition statement (a TEST statement)
 * corresponding to the current control current_node and the
 * root_statement root_statement. This TEST statement takes a condition on
 * the state_variable having the value matching the statement ordering
 * value, and the control statement for the test_true value. The
 * test_false value is set with a continue statement, before to be
 * eventually replaced in next control node by a new statement.
 */
statement make_transition_statement(control current_node,
				    statement root_statement,
				    entity state_variable,
				    const char* module_name)
{
  statement returned_statement = NULL;
  statement transition_statement = NULL;
  statement stat = control_statement (current_node);
  instruction test_instruction;
  instruction transition_instruction;
  sequence transition_sequence;
  test new_test;
  expression test_condition;
  int successors_nb;
  int current_transition_number;
  /*string comment;
    char buffer[50];*/

  debug_control ("TRANSITION: Module statement", current_node, 2);

  current_transition_number = beautify_ordering (statement_ordering(stat));

  test_condition
    = make_expression_with_state_variable (state_variable,
					   current_transition_number,
					   EQUAL_OPERATOR_NAME);
 
  successors_nb = gen_length(control_successors(current_node));

  if ((successors_nb == 0) || (successors_nb == 1)) {
    /* This is the exit node, or a non-test statement */
    int next_value;
    statement state_variable_assignement;
   
    if (successors_nb == 0) {
      /* This is the exit node, just generate exit code for state_variable */
      next_value = exit_state_variable_value_for_unstructured (root_statement);
    }
    else { /* successors_nb == 1 */
      /* This is a "normal" node, ie not a TEST statement, just add
	 assignement for state_variable with new value */
      next_value
	= beautify_ordering (statement_ordering
			     (control_statement
			      (CONTROL(gen_nth(0,control_successors(current_node))))));
    }
   
    state_variable_assignement
      = make_state_variable_assignement_statement
      (stat, state_variable, next_value);
   
    transition_sequence
      = make_sequence (CONS(STATEMENT,
			    fsmize_statement(stat, NULL, module_name),
			    /* NULL here because we will generate
			     * a new state variable, since the potential
			     * other FSMs are deeper */
			    CONS(STATEMENT, state_variable_assignement, NIL)));
   
    transition_instruction
      = make_instruction(is_instruction_sequence,
			 transition_sequence);
 
    transition_statement = make_statement(entity_empty_label(),
					  statement_number(stat),
					  statement_ordering(stat),
					  empty_comments,
					  transition_instruction,NIL,NULL,
					  statement_extensions(stat), make_synchronization_none());
  }
  else if (successors_nb == 2) {
    /* This is a "test" node, ie with a TEST statement, just add
       assignement for state_variable with new value after each
       statement in TEST */
    int value_if_true = beautify_ordering (statement_ordering
      (control_statement
       (CONTROL(gen_nth(0,control_successors(current_node))))));
    int value_if_false = beautify_ordering (statement_ordering
      (control_statement
       (CONTROL(gen_nth(1,control_successors(current_node))))));
    statement transition_statement_if_true;
    statement transition_statement_if_false;
    sequence transition_sequence_if_true;
    sequence transition_sequence_if_false;
    instruction transition_instruction_if_true;
    instruction transition_instruction_if_false;
    statement state_variable_assignement_if_true;
    statement state_variable_assignement_if_false;
    statement old_statement_if_true;
    statement old_statement_if_false;
    test current_test;
   
    pips_assert("Statement with 2 successors is a TEST in FSM_GENERATION",
		instruction_tag(statement_instruction(stat))
		== is_instruction_test);
   
    current_test = instruction_test (statement_instruction(stat));

    // Begin computing for the true statement

    old_statement_if_true = test_true(current_test);
   
    state_variable_assignement_if_true
      = make_state_variable_assignement_statement
      (stat, state_variable, value_if_true);
   
    transition_sequence_if_true
      = make_sequence (CONS(STATEMENT,
			    old_statement_if_true,
			    CONS(STATEMENT, state_variable_assignement_if_true, NIL)));

    transition_instruction_if_true
      = make_instruction(is_instruction_sequence,
			 transition_sequence_if_true);
 
    transition_statement_if_true = make_statement (entity_empty_label(),
						   statement_number(stat),
						   statement_ordering(stat),
						   empty_comments,
						   transition_instruction_if_true,
						   NIL,NULL,
						   statement_extensions(stat), make_synchronization_none());
   
    test_true(current_test) = transition_statement_if_true;

    // Begin computing for the false statement

    old_statement_if_false = test_false(current_test);
   
    state_variable_assignement_if_false
      = make_state_variable_assignement_statement
      (stat, state_variable, value_if_false);
   
    transition_sequence_if_false
      = make_sequence (CONS(STATEMENT,
			    old_statement_if_false,
			    CONS(STATEMENT, state_variable_assignement_if_false, NIL)));

    transition_instruction_if_false
      = make_instruction(is_instruction_sequence,
			 transition_sequence_if_false);
 
    transition_statement_if_false
      = make_statement
      (entity_empty_label(),
       statement_number(stat),
       statement_ordering(stat),
       empty_comments,
       transition_instruction_if_false,NIL,NULL,
       statement_extensions(stat), make_synchronization_none());

    test_false(current_test) = transition_statement_if_false;

    transition_statement = stat;

  }
  else {
    pips_assert("I should NOT be there :-)", 2+2 != 4); /* :-) */
  }

  new_test = make_test (test_condition, transition_statement,
			make_continue_statement(entity_undefined));
 
  test_instruction = make_instruction (is_instruction_test,new_test);

  /*sprintf (buffer,
    FSM_TRANSITION_COMMENT,
    entity_local_name(state_variable),
    current_transition_number);
    comment = strdup(buffer);*/
 
  returned_statement = make_statement (entity_empty_label(),
				       /*statement_label(root_statement),*/
				       statement_number(root_statement),
				       statement_ordering(root_statement),
				       empty_comments,
				       test_instruction,NIL,NULL,
				       statement_extensions(root_statement), make_synchronization_none());

  return returned_statement;

}

/**
 * This function build and return a statement representing the
 * transitions computation in the FSM, given the UNSTRUCTURED
 * statement stat.
 */
statement make_fsm_transitions_statement (statement stat,
					  entity state_variable,
					  const char* module_name)
{
  statement returned_statement = NULL;
  statement current_statement = NULL;
  unstructured nodes_graph;
  list blocs = NIL ;

  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION",
	      instruction_tag(statement_instruction(stat))
	      == is_instruction_unstructured);

  nodes_graph = instruction_unstructured(statement_instruction(stat));
 
  /*gen_recurse(unstructured_entry(nodes_graph), control_domain,
    transitions_filter, transitions_statements);*/
  CONTROL_MAP (current_control, {
    statement transition_statement;
    transition_statement = make_transition_statement (current_control,
						      stat,
						      state_variable,
						      module_name);
    if (returned_statement == NULL) {
      returned_statement = transition_statement;
      current_statement = returned_statement;
    }
    else {
      instruction i = statement_instruction(current_statement);
      test t;
      pips_assert("Statement is TEST in FSM_GENERATION transitions",
		  instruction_tag(i) == is_instruction_test);
      t = instruction_test(i);
      test_false (t) = transition_statement;
      current_statement = transition_statement;
    }
  }, unstructured_entry(nodes_graph), blocs);

  pips_debug(2,"blocs count = %zd\n", gen_length(blocs));
 
  return returned_statement;
}

/**
 * This function build and return a statement representing the
 * FSM code equivalent to the given unstructured statement stat.
 */
statement make_fsm_from_statement(statement stat,
				  entity state_variable,
				  const char* module_name)
{
  statement returned_statement;
  statement loop_statement;
  whileloop new_whileloop;
  expression loop_condition;
  statement loop_body;
  entity loop_entity = NULL;
  evaluation loop_evaluation = NULL;
  instruction loop_instruction;
  instruction sequence_instruction;
  sequence new_sequence;
  /*string comment;
    char buffer[256];*/
 
  /* Assert that given stat is UNSTRUCTURED */
  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION",
	      instruction_tag(statement_instruction(stat))
	      == is_instruction_unstructured);
 
  /* Create loop condition: state variable is not equal to exit value */
  loop_condition
    = make_expression_with_state_variable
    (state_variable,
     exit_state_variable_value_for_unstructured(stat),
     NON_EQUAL_OPERATOR_NAME);
 
  /* Evaluation is done BEFORE to enter the loop */
  loop_evaluation = make_evaluation_before();

  /* No label for loop */
  loop_entity = entity_empty_label();

  /* Computes the statement representing the transitions */
  loop_body = make_fsm_transitions_statement (stat,
					      state_variable,
					      module_name);

  /* Build the loop */
  new_whileloop = make_whileloop(loop_condition,
				 loop_body,
				 loop_entity,
				 loop_evaluation);

  loop_instruction = make_instruction(is_instruction_whileloop,new_whileloop);

  /*sprintf (buffer, FSM_BEGIN_COMMENT, entity_local_name(state_variable));
    comment = strdup(buffer);*/

  loop_statement = make_statement(statement_label(stat),
				  statement_number(stat),
				  statement_ordering(stat),
				  empty_comments,
				  loop_instruction,NIL,NULL,
				  statement_extensions(stat), make_synchronization_none());
 
 
  new_sequence
    = make_sequence (CONS(STATEMENT,
			  make_reset_state_variable_statement(stat,
							      state_variable),
			  CONS(STATEMENT, loop_statement, NIL)));
   
  sequence_instruction
    = make_instruction(is_instruction_sequence,
		       new_sequence);
 
  returned_statement = make_statement(entity_empty_label(),
				      statement_number(stat),
				      statement_ordering(stat),
				      empty_comments,
				      sequence_instruction,NIL,NULL,
				      statement_extensions(stat), make_synchronization_none());
  /*statement_instruction(loop_body)
    = make_instruction_block(CONS(STATEMENT,returned_statement,NIL));
  */
  return returned_statement;
}

/*
 * This function is recursively called during FSMization. It takes
 * the statement to fsmize stat as parameter, while module_name is
 * the name of the module where FSMization is applied.
 * If global variable is used for the whole module, state_variable
 * contains this element. If state_variable is null, then new
 * state_variable is created for this statement.
 */
statement fsmize_statement (statement stat,
			    entity state_variable,
			    const char* module_name)
{
  // Defaut behaviour is to return parameter statement stat
  statement returned_statement = stat;
  instruction i = statement_instruction(stat);

  pips_debug(2,"\nFSMize: Module statement: =====================================\n");
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
    // Declare the test data structure which will be used
    test current_test = instruction_test(i);
    statement true_statement, new_true_statement;
    statement false_statement, new_false_statement;

    pips_debug(2, "TEST\n");  

    // Compute new statement for true statement, and replace
    // the old one by the new one
    true_statement = test_true (current_test);
    new_true_statement = fsmize_statement(true_statement, state_variable, module_name);
    if (new_true_statement != NULL) {
      test_true (current_test) = new_true_statement;
    }

    // Do the same for the false statement
    false_statement = test_false (current_test);
    new_false_statement = fsmize_statement(false_statement, state_variable, module_name);
    if (new_false_statement != NULL) {
      test_false (current_test) = new_false_statement;
    }
   
    break;
  }
  case is_instruction_sequence:
    {
    sequence seq = instruction_sequence(i);
    pips_debug(2, "SEQUENCE\n");  
    MAP(STATEMENT, current_stat,
    {
      statement new_stat = fsmize_statement(current_stat, state_variable, module_name);
      if (new_stat != NULL) {
	gen_list_patch (sequence_statements(seq), current_stat, new_stat);
      }
    }, sequence_statements(seq));
    break;
  }
  case is_instruction_loop: {
    pips_debug(2, "LOOP\n");  
    break;
  }
  case is_instruction_whileloop: {
    pips_debug(2, "WHILELOOP\n");  
    break;
  }
  case is_instruction_forloop: {
    pips_debug(2, "FORLOOP\n");  
    break;
  }
  case is_instruction_call: {
    pips_debug(2, "CALL\n");  
    break;
  }
  case is_instruction_unstructured: {
    pips_debug(2, "UNSTRUCTURED\n"); 
    if (state_variable == NULL) {
      entity new_state_variable;
      int state_variable_identifier
	/* = beautify_ordering (statement_ordering(stat)); */
	= statement_ordering(stat);
     
      pips_debug(2, "Creating state variable with identifier %d\n",
		 state_variable_identifier);  
      new_state_variable
	= create_state_variable (module_name,
				 state_variable_identifier);
      returned_statement = make_fsm_from_statement (stat,
						    new_state_variable,
						    module_name);
    }
    else {
      returned_statement = make_fsm_from_statement (stat,
						    state_variable,
						    module_name);
    }
    pips_debug(2, "Displaying statement\n");  
    ifdebug(2) {
      print_statement (returned_statement);
    }
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
