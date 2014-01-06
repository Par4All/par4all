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
#ifndef PHRASE_TOOLS_DEFS
#define PHRASE_TOOLS_DEFS

/**
 * DEBUG FUNCTION: return a string representing the type of the
 * statement (SEQUENCE, CALL, etc...)
 */
string statement_type_as_string (statement stat);

/**
 * DEBUG FUNCTION: print debugging informations for
 * a statement stat
 */
void debug_statement (const char* comments, statement stat, int debug_level);

/**
 * DEBUG FUNCTION: print debugging informations for
 * a control a_control
 */
void debug_control (const char* comments, control a_control, int debug_level);

/**
 * DEBUG FUNCTION: print debugging informations for
 * an unstructured an_unstructured
 */
void debug_unstructured (unstructured an_unstructured, 
			 int debug_level);
/**
 * DEBUG FUNCTION: print debugging informations for
 * an unstructured an_unstructured (short version)
 */
void short_debug_unstructured (unstructured an_unstructured, 
			       int debug_level);

/**
 * This function build and return an expression given
 * an entity an_entity
 */
expression make_expression_from_entity(entity an_entity);


/**
 * Build and return new statement which is a binary call with
 * the 2 expressions expression1 and expression2, with empty
 * label, statement number and ordering of statement stat, 
 * and empty comments
 */
statement make_binary_call_statement (const char* operator_name,
				      expression expression1,
				      expression expression2,
				      statement stat);
 
/**
 * This function build and return new variable from
 * a variable a_variable, with name new_name. If an entity
 * called new_name already exists, return NULL.
 * New variable is added to declarations
 */
entity clone_variable_with_new_name(entity a_variable,
				    const char* new_name, 
				    const char* module_name);

/**
 * Build and return new entity obtained by cloning variable
 * cloned_variable, with a name obtained by the concatenation
 * of base_name and the statement ordering of statement stat.
 * If such entity already exist, increment statement ordering
 * to get first free name. We assume then that created entity's 
 * name is unique.
 */
entity make_variable_from_name_and_entity (entity cloned_variable,
					   const char* base_name,
					   statement stat,
					   const char* module_name);

/**
 * Build and return new statement which is a assignement of variable
 * a_variable with expression an_expression, with empty label, statement
 * number and ordering of statement stat, and empty comments
 */
statement make_assignement_statement (entity a_variable,
				      expression an_expression,
				      statement stat);

unstructured statement_unstructured (statement stat);
				     
/**
 * Special function made for Ronan Keryell who likes a lot
 * when a integer number is coded on 3 bits :-)
 */
int beautify_ordering (int an_ordering);

void clean_statement_from_tags (const char* comment_portion,
				statement stat);

list get_statements_with_comments_containing (const char* comment_portion,
					      statement stat);

bool statement_is_contained_in_a_sequence_p (statement root_statement,
					     statement searched_stat);

statement sequence_statement_containing (statement root_statement,
					 statement searched_stat);

void replace_in_sequence_statement_with (statement old_stat, 
					 statement new_stat,
					 statement root_stat);

list references_for_regions (list l_regions);


#endif
