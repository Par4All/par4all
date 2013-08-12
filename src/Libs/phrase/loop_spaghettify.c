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
 * This file contains the code used for spaghettify loops.
 *
 * General syntax of loop in Fortran are:
 *
 * DO INDEX=BEGIN, END, INCREMENT
 *   STATEMENT
 * END DO
 *
 * where INDEX is an entity (variable)
 * and BEGIN, END, INCREMENT are expressions.
 *
 * Be careful: Fortran interprets all the values of the loop
 * INDEX, BEGIN, END, INCREMENT before to enter the loop, and
 * those values could therefore be modified inside the loop
 * (during the execution of STATEMENT).
 * That's the reason, to generate equivalent code, why we 
 * introduce the variables: 
 * INDEX', BEGIN', END', INCREMENT'
 *
 * If INCREMENT is evaluable and POSITIVE, following code is generated:
 *
 *    BEGIN' = BEGIN
 *    END' = END
 *    INC' = INC
 *    INDEX' = BEGIN'
 * 10 IF (INDEX'.LE.END') THEN
 *      INDEX = INDEX'
 *      STATEMENT
 *      INDEX' = INDEX' + INC'
 *      GOTO 10
 *    ENDIF
 * 20 CONTINUE
 *
 *
 * If INCREMENT is evaluable and NEGATIVE, following code is generated:
 *
 *    BEGIN' = BEGIN
 *    END' = END
 *    INC' = INC
 *    INDEX' = BEGIN'
 * 10 IF (INDEX'.GE.END') THEN
 *      INDEX = INDEX'
 *      STATEMENT
 *      INDEX' = INDEX' + INC'
 *      GOTO 10
 *    ENDIF
 * 20 CONTINUE
 *
 * If INCREMENT is not evaluable, following code is generated:
 *
 *    BEGIN' = BEGIN
 *    END' = END
 *    INC' = INC
 *    INDEX' = BEGIN'
 * 10 IF (((INDEX'.LE.END').AND.(INC'.GE.0)).OR.((INDEX'.GE.END').AND.(INC'.LE.0))) THEN
 *      INDEX = INDEX'
 *      STATEMENT
 *      INDEX' = INDEX' + INC'
 *      GOTO 10
 *    ENDIF
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
 * Creates index variable for a loop the_loop of a statement stat
 */
static entity make_index_variable (loop the_loop,
				   statement stat,
				   const char* module_name) 
{
  return make_variable_from_name_and_entity (loop_index(the_loop),
					     INDEX_VARIABLE_NAME,
					     stat,
					     module_name);
}

/**
 * Creates begin variable for a loop the_loop of a statement stat
 */
static entity make_begin_variable (loop the_loop,
				   statement stat,
				   const char* module_name) 
{
  return make_variable_from_name_and_entity (loop_index(the_loop),
					     BEGIN_VARIABLE_NAME,
					     stat,
					     module_name);
}

/**
 * Creates end variable for a loop the_loop of a statement stat
 */
static entity make_end_variable (loop the_loop,
				 statement stat,
				 const char* module_name) 
{
  return make_variable_from_name_and_entity (loop_index(the_loop),
					     END_VARIABLE_NAME,
					     stat,
					     module_name);
}

/**
 * Creates increment variable for a loop the_loop of a statement stat
 */
static entity make_increment_variable (loop the_loop,
				       statement stat,
				       const char* module_name) 
{
  return make_variable_from_name_and_entity (loop_index(the_loop),
					     INCREMENT_VARIABLE_NAME,
					     stat,
					     module_name);
}

/**
 * Build and return a new control containing initialization statement
 * of the unstructured loop
 */
static control make_initialization_from_loop (loop the_loop,
					      statement stat,
					      entity index_variable,
					      entity begin_variable, 
					      entity end_variable, 
					      entity increment_variable)
{
  statement init_begin_statement;
  statement init_end_statement;
  statement init_increment_statement;
  statement init_index_statement;
  expression lower = range_lower(loop_range(the_loop));
  expression upper = range_upper(loop_range(the_loop));
  expression increment = range_increment(loop_range(the_loop));
  sequence init_sequence;

  init_begin_statement 
    = make_assignement_statement (begin_variable, lower, stat);
  init_end_statement 
    = make_assignement_statement (end_variable, upper, stat);
  init_increment_statement 
    = make_assignement_statement (increment_variable, increment, stat);
  init_index_statement 
    = make_assignement_statement (index_variable, 
				  entity_to_expression(begin_variable),
				  stat);

  init_sequence 
    = make_sequence (CONS(STATEMENT,
			  init_begin_statement,
			  CONS(STATEMENT,
			       init_end_statement,
			       CONS(STATEMENT,
				    init_increment_statement,
				    CONS(STATEMENT,
					 init_index_statement,
					 NIL)))));
				      
  return make_control (make_statement(entity_empty_label(),
				      statement_number(stat),
				      statement_ordering(stat),
				      empty_comments,
				      make_instruction(is_instruction_sequence,
						       init_sequence),
				      NIL, NULL,
				      statement_extensions(stat), make_synchronization_none()),
		       NIL, NIL);
}

/**
 * Build and return a new control containing condition statement
 * of the unstructured loop
 */
static control make_condition_from_loop (loop the_loop,
					 statement stat,
					 entity index_variable,
					 entity end_variable, 
					 entity increment_variable)
{
  statement condition_statement;
  test condition_test;
  intptr_t increment_value;
  string intrinsic_name;
  expression test_condition;

  if (expression_integer_value
      (range_increment(loop_range(the_loop)), 
       &increment_value)) {
    if (increment_value >= 0) {
      intrinsic_name = LESS_OR_EQUAL_OPERATOR_NAME;
    }
    else {
      intrinsic_name = GREATER_OR_EQUAL_OPERATOR_NAME;
    }
    test_condition 
      = MakeBinaryCall (entity_intrinsic(intrinsic_name),
			entity_to_expression (index_variable),
			entity_to_expression (end_variable));
  }
  else {
    // Generate (((INDEX'.LE.END').AND.(INC'.GE.0)).OR.((INDEX'.GE.END').AND.(INC'.LE.0)))
    test_condition 
      = MakeBinaryCall (entity_intrinsic(OR_OPERATOR_NAME),
			MakeBinaryCall (entity_intrinsic(AND_OPERATOR_NAME),
					MakeBinaryCall (entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
							entity_to_expression (index_variable),
							entity_to_expression (end_variable)),
					MakeBinaryCall (entity_intrinsic(GREATER_OR_EQUAL_OPERATOR_NAME),
							entity_to_expression (increment_variable),
							int_to_expression (0))),
			MakeBinaryCall (entity_intrinsic(AND_OPERATOR_NAME),
					MakeBinaryCall (entity_intrinsic(GREATER_OR_EQUAL_OPERATOR_NAME),
							entity_to_expression (index_variable),
							entity_to_expression (end_variable)),
					MakeBinaryCall (entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
							entity_to_expression (increment_variable),
							int_to_expression (0))));

					
  }

  condition_test 
    = make_test (test_condition,
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
 * of the unstructured loop (this is a continue statement)
 */
static control make_exit_from_loop ()
{
  return make_control (make_continue_statement(entity_empty_label()), NIL, NIL);
}

/**
 * Build and return a new control containing body statement
 * of the unstructured loop 
 */
static control make_body_from_loop (loop the_loop, 
				    const char* module_name, 
				    statement stat,
				    entity index_variable,
				    entity increment_variable)
{
  statement assignement_statement 
    = make_assignement_statement (loop_index(the_loop),
				  entity_to_expression (index_variable),
				  stat);
  statement spaghettified_body 
    = spaghettify_statement(loop_body(the_loop),
			    module_name);
  statement increment_statement
    = make_assignement_statement 
    (index_variable,
     MakeBinaryCall (entity_intrinsic(PLUS_OPERATOR_NAME),
		     entity_to_expression (index_variable),
		     entity_to_expression (increment_variable)),
     stat);

  sequence body_sequence 
    = make_sequence (CONS(STATEMENT, assignement_statement,
			  CONS(STATEMENT, spaghettified_body,
			       CONS(STATEMENT, increment_statement, NIL))));
  
  statement body_statement 
    = make_statement(entity_empty_label(),
		     statement_number(stat),
		     statement_ordering(stat),
		     empty_comments,
		     make_instruction (is_instruction_sequence,
				       body_sequence),
		     NIL,NULL,
		     statement_extensions(stat), make_synchronization_none());  
  return make_control (body_statement, NIL, NIL);
}

/**
 * Build and return a new unstructured coding the
 * "destructured" loop
 */
static unstructured make_unstructured_from_loop (loop the_loop, 
						 statement stat, 
						 const char* module_name) 
{
  entity index_variable = make_index_variable (the_loop, stat, module_name);
  entity begin_variable = make_begin_variable (the_loop, stat, module_name);
  entity end_variable = make_end_variable (the_loop, stat, module_name);
  entity increment_variable = make_increment_variable (the_loop, stat, module_name);
  control initialization 
    = make_initialization_from_loop (the_loop, 
				     stat,
				     index_variable,
				     begin_variable, 
				     end_variable, 
				     increment_variable);
  control condition = make_condition_from_loop (the_loop, 
						stat,				   
						index_variable,
						end_variable, 
						increment_variable);
  control exit = make_exit_from_loop ();
  control body = make_body_from_loop (the_loop, 
				      module_name,
				      stat,
				      index_variable,
				      increment_variable);

  link_2_control_nodes (initialization, condition); /* after init, we test */

  /* The first connexion is the false one */
  link_2_control_nodes (condition, exit); /* false condition, we exit from loop */
  link_2_control_nodes (condition, body); /* true condition, we go to body */

  link_2_control_nodes (body, condition); /* after body, we go back to condition */
  
  return make_unstructured (initialization, exit);
}

/* 
 * This function takes the statement stat as parameter and return a new 
 * spaghettized statement, asserting stat is a LOOP statement
 */
statement spaghettify_loop (statement stat, const char* module_name)
{
  statement returned_statement = stat;
  instruction unstructured_instruction;
  unstructured new_unstructured;  

  pips_assert("Statement is LOOP in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_loop);

  pips_debug(2, "spaghettify_loop, module %s\n", module_name);   
  new_unstructured = make_unstructured_from_loop (statement_loop(stat), 
						  stat,
						  module_name);
  
  unstructured_instruction = make_instruction(is_instruction_unstructured,
					      new_unstructured);
  
  statement_instruction(returned_statement) = unstructured_instruction;

  return returned_statement;
}
