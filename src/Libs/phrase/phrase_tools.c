/* 
 *
 * This phase is used for PHRASE project. 
 *
 * NB: The PHRASE project is an attempt to automatically (or
 * semi-automatically) transform high-level language for partial
 * evaluation in reconfigurable logic (such as FPGAs or DataPaths).

 * This file provides tools used in this library.
 *
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

/**
 * DEBUG FUNCTION: return a string representing the type of the
 * statement (SEQUENCE, CALL, etc...)
 */
string statement_type_as_string (statement stat)
{
  instruction i = statement_instruction(stat);
  switch (instruction_tag(i)) {
  case is_instruction_test: {
    return strdup("TEST");   
    break;
  }
  case is_instruction_sequence: {
    return strdup("SEQUENCE");   
    break;
  }
  case is_instruction_loop: {
    return strdup("LOOP");   
    break;
  }
  case is_instruction_whileloop: {
    return strdup("WHILELOOP");   
    break;
  }
  case is_instruction_forloop: {
    return strdup("FORLOOP");   
    break;
  }
  case is_instruction_call: {
    return strdup("CALL");   
    break;
  }
  case is_instruction_unstructured: {
    return strdup("UNSTRUCTURED");   
    break;
  }
  case is_instruction_goto: {
    return strdup("GOTO");   
    break;
  }
  default:
    return strdup("UNDEFINED");   
    break;
  }
}

/**
 * This function build and return an expression given
 * an entity an_entity
 */
expression make_expression_from_entity(entity an_entity)
{
  return make_entity_expression(an_entity, NIL);
  
}

/**
 * This function build and return new variable from
 * a variable a_variable, with name new_name. If an entity
 * called new_name already exists, return NULL.
 * New variable is added to declarations
 */
entity clone_variable_with_new_name(entity a_variable,
				    string new_name, 
				    string module_name)
{
  entity module;
  entity new_variable;

  module = local_name_to_top_level_entity(module_name); 
  /* Assert that module represent a value code */
  pips_assert("it is a code", value_code_p(entity_initial(module)));

  if ((gen_find_tabulated(concatenate(module_name, 
				      MODULE_SEP_STRING, 
				      new_name, 
				      NULL),
			  entity_domain)) == entity_undefined) 
    { 
      /* This entity does not exist, we can safely create it */
      new_variable = copy_entity (a_variable);
      entity_name(new_variable) 
	= strdup(concatenate(module_name, 
			     MODULE_SEP_STRING, 
			     new_name, NULL));
      add_variable_declaration_to_module(module, new_variable);
      return new_variable;
    }
  else 
    {
      /* This entity already exist, we return null */
      return NULL;
    }
}

/**
 * Build and return new entity obtained by cloning variable
 * cloned_variable, with a name obtained by the concatenation
 * of base_name and the statement ordering of statement stat.
 * If such entity already exist, increment statement ordering
 * to get first free name. We assume then that created entity's 
 * name is unique.
 */
entity make_variable_from_name_and_entity (entity cloned_variable,
					   string base_name,
					   statement stat,
					   string module_name) 
{
  string variable_name;
  entity returned_variable = NULL;
  int index = statement_ordering(stat);
  char buffer[50];
  
  while (returned_variable == NULL) {
    
    sprintf(buffer, "%d", index++);
    variable_name = strdup(concatenate(base_name,
				    strdup(buffer),
				    NULL));
    returned_variable 
      = clone_variable_with_new_name (cloned_variable,
				      variable_name,
				      module_name);
  }
  
  return returned_variable;
  
}

/**
 * Build and return new statement which is a binary call with
 * the 2 expressions expression1 and expression2, with empty
 * label, statement number and ordering of statement stat, 
 * and empty comments
 */
statement make_binary_call_statement (string operator_name,
				      expression expression1,
				      expression expression2,
				      statement stat) 
{
  call assignment_call 
    = make_call (entity_intrinsic(operator_name),
		 CONS(EXPRESSION, 
		      expression1, 
		      CONS(EXPRESSION, expression2, NIL)));
  
  return make_statement(entity_empty_label(),
			statement_number(stat),
			statement_ordering(stat),
			empty_comments,
			make_instruction (is_instruction_call,
					  assignment_call),
			NIL,NULL);  
}

/**
 * Build and return new statement which is a assignement of variable
 * a_variable with expression an_expression, with empty label, statement
 * number and ordering of statement stat, and empty comments
 */
statement make_assignement_statement (entity a_variable,
				      expression an_expression,
				      statement stat)
{
  return make_binary_call_statement (ASSIGN_OPERATOR_NAME,
				     make_expression_from_entity(a_variable),
				     an_expression,
				     stat);
}
