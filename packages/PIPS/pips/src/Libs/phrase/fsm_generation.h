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
#ifndef FSM_GENERATION_DEFS
#define FSM_GENERATION_DEFS

#define STATE_VARIABLE_NAME "FSM%d_STATE"
#define STATE_VARIABLE_NAME_NO_REF "FSM_STATE"
#define FSM_BEGIN_COMMENT "! BEGIN FSM, %s\n"
#define FSM_TRANSITION_COMMENT "! Transition %s=%d\n"

/**
 * Build and return an expression (eg. state = 23), given an entity
 * state_variable, an int value value, and an intrinsic name
 */
expression make_expression_with_state_variable(entity state_variable,
					       int value,
					       string intrinsic_name);

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
			      int name_identifier);

/**
 * This function build and return a statement representing the
 * initial assigment of the state_variable, given the UNSTRUCTURED
 * statement stat.
 */
statement make_state_variable_assignement_statement (statement stat, 
						     entity state_variable, 
						     int assignement_value);
 
/**
 * Return the state variable value corresponding to the entry
 * in a unstructured statement
 */
int entry_state_variable_value_for_unstructured (statement stat);

/**
 * Return the state variable value corresponding to the exit
 * in a unstructured statement
 * NB: always return 0
 */
int exit_state_variable_value_for_unstructured (statement stat);
 
/**
 * This function build and return a statement representing the
 * initial assigment of the state_variable, given the UNSTRUCTURED
 * statement stat.
 */
statement make_reset_state_variable_statement (statement stat, 
					       entity state_variable);

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
				    const char* module_name);

/**
 * This function build and return a statement representing the
 * transitions computation in the FSM, given the UNSTRUCTURED
 * statement stat.
 */
statement make_fsm_transitions_statement (statement stat, 
					  entity state_variable,
					  const char* module_name);


/**
 * This function build and return a statement representing the
 * FSM code equivalent to the given unstructured statement stat.
 */
statement make_fsm_from_statement(statement stat, 
				  entity state_variable,
				  const char* module_name);

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
			    const char* module_name);

#endif

