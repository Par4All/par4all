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
 * DEBUG FUNCTION: print debugging informations for
 * a statement stat
 */
void debug_statement (string comments, statement stat, int debug_level)
{
  pips_debug(debug_level,"%s\n",comments);
  ifdebug(debug_level) {
    print_statement(stat);
  }
  pips_debug(debug_level,"domain number         = %d\n", statement_domain_number(stat));
  pips_debug(debug_level,"entity                = UNDEFINED\n");
  pips_debug(debug_level,"statement number      = %d\n", statement_number(stat));
  pips_debug(debug_level,"statement ordering    = %d\n", statement_ordering(stat));
  if (statement_with_empty_comment_p(stat)) {
    pips_debug(debug_level,"statement comments   = EMPTY\n");
  }
  else {
    pips_debug(debug_level,"statement comments  = %s\n", statement_comments(stat));
  }
  pips_debug(debug_level,"statement instruction = %s\n", statement_type_as_string(stat));

}

/**
 * DEBUG FUNCTION: print debugging informations for
 * a control a_control
 */
void debug_control (string comments, control a_control, int debug_level) {

  debug_statement (comments, control_statement(a_control), debug_level);
  pips_debug(debug_level,"  predecessors          = %d\n", gen_length(control_predecessors(a_control)));
  pips_debug(debug_level,"  successors            = %d\n", gen_length(control_successors(a_control)));

}

/**
 * DEBUG FUNCTION: print debugging informations for
 * an unstructured an_unstructured
 */
void debug_unstructured (unstructured an_unstructured, 
			 int debug_level) 
{
  list blocs = NIL ;
  string line = "***********************************************************************\n";

  ifdebug (debug_level) {
    CONTROL_MAP (current_control, { 
      statement s = control_statement (current_control);
      string next_nodes_as_string = "";
      string previous_nodes_as_string = "";
      char title[80];
      list predecessors = control_predecessors(current_control);
      list successors = control_successors(current_control);
      char temp[50];
      int i;
      int ordering;

      for (i=0; i<gen_length(predecessors); i++) {
	ordering = statement_ordering(control_statement
				      (CONTROL(gen_nth(i,predecessors))));
	ordering = beautify_ordering (ordering);
	/*if (ordering > 65535) ordering = ordering >> 16;*/
	sprintf (temp, "[%p] ",ordering);
	previous_nodes_as_string = strdup (concatenate(previous_nodes_as_string,
						       strdup(temp),
						       NULL));
      }
      
      for (i=0; i<gen_length(successors); i++) {
	ordering = statement_ordering(control_statement
				      (CONTROL(gen_nth(i,successors))));
	ordering = beautify_ordering (ordering);
	/*if (ordering > 65535) ordering = ordering >> 16;*/
	sprintf (temp, "[%p] ",ordering);
	next_nodes_as_string = strdup (concatenate(next_nodes_as_string,
						   strdup(temp),
						   NULL));
      }
      
      ordering = beautify_ordering (ordering);
      ordering = statement_ordering(s);
      /*if (ordering > 65535) ordering = ordering >> 16;*/
      sprintf (title, "CONTROL: %p\n", ordering);
      pips_debug(debug_level, "%s\n",
		 strdup(concatenate("\n", line,
				    "* ", strdup(title),
				    line, NULL)));
      print_statement(s);
      pips_debug(debug_level, "%s\n",
		 strdup(concatenate("\n", line,
				    "NEXT: ", next_nodes_as_string, "\n",
				    "PREVIOUS: ", previous_nodes_as_string, "\n",
				    line, NULL)));
		 
    }, unstructured_entry(an_unstructured), blocs);
  }
}

/**
 * DEBUG FUNCTION: print debugging informations for
 * an unstructured an_unstructured (short version)
 */
void short_debug_unstructured (unstructured an_unstructured, 
			       int debug_level) 
{
  list blocs = NIL ;
  string entry_as_string, exit_as_string;
  char temp[50];

  ifdebug (debug_level) {
    sprintf (temp, "[%p] ",unstructured_entry(an_unstructured));
    entry_as_string = strdup(temp);
    sprintf (temp, "[%p] ",unstructured_exit(an_unstructured));
    exit_as_string = strdup(temp);
    pips_debug(debug_level, "%s\n",
	       strdup(concatenate("UNSTRUCTURED\n",
				  "ENTRY: ", entry_as_string, "\n",
				  "PREVIOUS: ", exit_as_string, "\n",
				  NULL)));

    CONTROL_MAP (current_control, { 
      string next_nodes_as_string = "";
      string previous_nodes_as_string = "";
      char title[80];

      MAP(CONTROL, c, {
	sprintf (temp, "[%p] ",c);
	previous_nodes_as_string = strdup (concatenate(previous_nodes_as_string,
						       strdup(temp),
						       NULL));
      }, control_predecessors(current_control));

      MAP(CONTROL, c, {
	sprintf (temp, "[%p] ",c);
	next_nodes_as_string = strdup (concatenate(next_nodes_as_string,
						   strdup(temp),
						   NULL));
      }, control_successors(current_control));

      sprintf (title, "CONTROL: %p\n", current_control);
      pips_debug(debug_level, "%s\n",
		 strdup(concatenate(title,
				    "NEXT: ", next_nodes_as_string, "\n",
				    "PREVIOUS: ", previous_nodes_as_string, "\n",
				    NULL)));
    }, unstructured_entry(an_unstructured), blocs);
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
      
      /* new_variable = copy_entity (a_variable);
	 entity_name(new_variable)  
	 = strdup(concatenate(module_name, 
	 MODULE_SEP_STRING, 
	 new_name, NULL)); */
      
      new_variable = make_entity (strdup(concatenate(module_name, 
						     MODULE_SEP_STRING, 
						     new_name, NULL)),
				  copy_type (entity_type(a_variable)),
				  copy_storage (entity_storage(a_variable)),
				  copy_value (entity_initial(a_variable)));
      
      /*new_variable 
	= find_or_create_scalar_entity (strdup(concatenate(module_name, 
	MODULE_SEP_STRING, 
	new_name, NULL)),module_name,
	is_basic_int);*/
      
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
    
    sprintf(buffer, base_name, index++);
    variable_name = strdup(concatenate(strdup(buffer),
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

/** 
 * Return unstructured for a statement asserting that this one
 * represent an unstructured
 */
unstructured statement_unstructured (statement stat) 
{
  pips_assert("Statement is UNSTRUCTURED", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_unstructured);
  return instruction_unstructured(statement_instruction(stat));
}

/**
 * Special function made for Ronan Keryell who likes a lot
 * when a integer number is coded on 3 bits :-)
 */
int beautify_ordering (int an_ordering) 
{
  int ordering_up = (an_ordering & 0xffff0000) >> 16;
  int ordering_down = (an_ordering & 0x0000ffff);
  return ordering_up + ((ordering_down-1) << 16);
}
