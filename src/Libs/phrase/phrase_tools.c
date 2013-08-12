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
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"

#include "text-util.h"

#include "dg.h"


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
void debug_statement (const char* comments, statement stat, int debug_level)
{
  ifdebug(debug_level) {
    pips_debug(debug_level,"%s\n",comments);
    print_statement(stat);
    pips_debug(debug_level,"domain number         = %"PRIdPTR"\n", statement_domain_number(stat));
    pips_debug(debug_level,"entity                = UNDEFINED\n");
    pips_debug(debug_level,"statement number      = %"PRIdPTR"\n", statement_number(stat));
    pips_debug(debug_level,"statement ordering    = %"PRIdPTR"\n", statement_ordering(stat));
    if (statement_with_empty_comment_p(stat)) {
      pips_debug(debug_level,"statement comments   = EMPTY\n");
    }
    else {
      pips_debug(debug_level,"statement comments  = %s\n", statement_comments(stat));
    }
    pips_debug(debug_level,"statement instruction = %s\n", statement_type_as_string(stat));
  }
}

/**
 * DEBUG FUNCTION: print debugging informations for
 * a control a_control
 */
void debug_control (const char* comments, control a_control, int debug_level) {

  debug_statement (comments, control_statement(a_control), debug_level);
  pips_debug(debug_level,"  predecessors          = %zd\n", gen_length(control_predecessors(a_control)));
  pips_debug(debug_level,"  successors            = %zd\n", gen_length(control_successors(a_control)));

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
      char *title;
      list predecessors = control_predecessors(current_control);
      list successors = control_successors(current_control);
      char *temp;
      int i;
      int ordering = 0;

      for (i=0; i<gen_length(predecessors); i++) {
	ordering = statement_ordering(control_statement
				      (CONTROL(gen_nth(i,predecessors))));
	ordering = beautify_ordering (ordering);
	/*if (ordering > 65535) ordering = ordering >> 16;*/
	asprintf (&temp, "[%d] ",ordering);
	previous_nodes_as_string = strdup (concatenate(previous_nodes_as_string,
						       temp,
						       NULL));
    free(temp);
      }
     
      for (i=0; i<gen_length(successors); i++) {
	ordering = statement_ordering(control_statement
				      (CONTROL(gen_nth(i,successors))));
	ordering = beautify_ordering (ordering);
	/*if (ordering > 65535) ordering = ordering >> 16;*/
	asprintf (&temp, "[%d] ",ordering);
	next_nodes_as_string = strdup (concatenate(next_nodes_as_string,
						   (temp),
						   NULL));
    free(temp);
      }
     
      ordering = beautify_ordering (ordering);
      ordering = statement_ordering(s);
      /*if (ordering > 65535) ordering = ordering >> 16;*/
      ifdebug(debug_level) {
      asprintf (&title, "CONTROL: %d\n", ordering);
	pips_debug(debug_level, "%s\n",
		   strdup(concatenate("\n", line,
				      "* ", (title),
				      line, NULL)));
	print_statement(s);
	pips_debug(debug_level, "%s\n",
		   strdup(concatenate("\n", line,
				      "NEXT: ", next_nodes_as_string, "\n",
				      "PREVIOUS: ", previous_nodes_as_string, "\n",
				      line, NULL)));
    free(title);
      }
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
  char *temp;

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
      char *title;

      MAP(CONTROL, c, {
	asprintf (&temp, "[%p] ",c);
	previous_nodes_as_string = strdup (concatenate(previous_nodes_as_string,
						       (temp),
						       NULL));
    free(temp);
      }, control_predecessors(current_control));

      MAP(CONTROL, c, {
	asprintf (&temp, "[%p] ",c);
	next_nodes_as_string = strdup (concatenate(next_nodes_as_string,
						   (temp),
						   NULL));
    free(temp);
      }, control_successors(current_control));

      asprintf (&title, "CONTROL: %p\n", current_control);
      pips_debug(debug_level, "%s\n",
		 strdup(concatenate(title,
				    "NEXT: ", next_nodes_as_string, "\n",
				    "PREVIOUS: ", previous_nodes_as_string, "\n",
				    NULL)));
      free(title);
    }, unstructured_entry(an_unstructured), blocs);
  }
}


/**
 * This function build and return new variable from
 * a variable a_variable, with name new_name. If an entity
 * called new_name already exists, return NULL.
 * New variable is added to declarations
 */
entity clone_variable_with_new_name(entity a_variable,
				    const char* new_name,
				    const char* module_name)
{
  entity module;
  entity new_variable;

  module = module_name_to_entity(module_name);
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
     
      AddEntityToDeclarations(new_variable,module);
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
					   const char* base_name,
					   statement stat,
					   const char* module_name)
{
  string variable_name;
  entity returned_variable = NULL;
  int index = statement_ordering(stat);
  //char *buffer;
 
  while (returned_variable == NULL) {
   
    asprintf(&variable_name, base_name, index++);
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
statement make_binary_call_statement (const char* operator_name,
				      expression expression1,
				      expression expression2,
				      statement stat)
{
  call assignment_call
    = make_call (entity_intrinsic(operator_name),
		 CONS(EXPRESSION,
		      expression1,
		      CONS(EXPRESSION, expression2, NIL)));

  if (stat == NULL) {
    return make_statement(entity_empty_label(),
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  empty_comments,
			  make_instruction (is_instruction_call,
					    assignment_call),
			  NIL,NULL,
			  empty_extensions (), make_synchronization_none()); 
  }
  else {
    return make_statement(entity_empty_label(),
			  statement_number(stat),
			  statement_ordering(stat),
			  empty_comments,
			  make_instruction (is_instruction_call,
					    assignment_call),
			  NIL,NULL,
			  statement_extensions(stat), make_synchronization_none());
  }
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
				     entity_to_expression(a_variable),
				     an_expression,
				     stat);
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

void clean_statement_from_tags (const char* comment_portion,
				statement stat)
{
  string comments;
  char*  next_line;
  string searched_string;

  if (!statement_with_empty_comment_p(stat)) {

    string new_comments = NULL;

    searched_string = strdup(comment_portion);
    searched_string[strcspn(comment_portion, "%s")] = '\0';
    comments = strdup(statement_comments(stat));
    next_line = strtok (comments, "\n");
    if (next_line != NULL) {
      do {
	if (strstr(next_line,searched_string) == NULL) {
	  if (new_comments != NULL) {
	    new_comments = strdup(concatenate(new_comments, next_line, "\n", NULL));
	  }
	  else {
	    new_comments = strdup(concatenate("", next_line, "\n", NULL));
	  }
	}
	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }
   
    if (new_comments != NULL) {
      statement_comments(stat) = new_comments;
    }
    else {
      statement_comments(stat) = empty_comments;
    }
  }
   
}

typedef struct {
  string searched_string;
  list list_of_statements;
} statement_checking_context;

static void check_if_statement_contains_comment(statement s, void* a_context)
{
  statement_checking_context* context = (statement_checking_context*)a_context;
  string comments;

  if (!statement_with_empty_comment_p(s)) {
   
    comments = strdup(statement_comments(s));
   
    /*pips_debug(5, "Searching comment: [%s] in [%s]\n",
      context->searched_string, comments);*/
     
    if (strstr(comments,context->searched_string) != NULL) {
      context->list_of_statements
	= CONS(STATEMENT,s,context->list_of_statements);
    }
  }
}

/**
 *
 */
list get_statements_with_comments_containing (const char* comment_portion,
					      statement stat)
{
  statement_checking_context context;

  /* First, set searched_string (we remove format information)*/
  context.searched_string = strdup(comment_portion);
  context.searched_string[strcspn(comment_portion, "%s")] = '\0';

  /* Reset list */
  context.list_of_statements = NIL;
 
  /*ifdebug(5) {
    pips_debug(5, "Searching statements with comments: %s\n",
	       context.searched_string);     
    pips_debug(5, "In statement:\n");     
    print_statement(stat);
    }*/

  gen_context_recurse(stat, &context, statement_domain, gen_true,
		      check_if_statement_contains_comment);
 
  return context.list_of_statements;
 
}

bool statement_is_contained_in_a_sequence_p (statement root_statement,
					     statement searched_stat)
{
  return (sequence_statement_containing (root_statement,
					 searched_stat) != NULL);
}

typedef struct {
  statement searched_statement;
  statement found_sequence_statement;
} sequence_searching_context;


static void search_sequence_containing (statement s,
					void* a_context)
{
  sequence_searching_context* context
    = (sequence_searching_context*)a_context;
  instruction i = statement_instruction(s);
 
  if (instruction_tag(i) == is_instruction_sequence) {
    MAP (STATEMENT, s2, {
      if (s2 == context->searched_statement) {
	context->found_sequence_statement = s;
      }
    }, sequence_statements(instruction_sequence(i)));
  }
}

statement sequence_statement_containing (statement root_statement,
					 statement searched_stat)
{
  sequence_searching_context context;

  context.searched_statement = searched_stat;
  context.found_sequence_statement = NULL;

  gen_context_recurse(root_statement, &context,
		      statement_domain, gen_true,
		      search_sequence_containing);

  return context.found_sequence_statement;
}

/**
 * Replace statement old_stat by statement new_stat, asserting that this
 * statement is contained in a sequence
 */
void replace_in_sequence_statement_with (statement old_stat,
					 statement new_stat,
					 statement root_stat)
{
  statement sequence_statement = sequence_statement_containing (root_stat,
								old_stat);
  list stats_list;
  list new_stats_list = NIL;

  pips_debug(5, "BEGIN replace_in_sequence_statement_with:\n");

  pips_assert("Statement is contained in a sequence",
	      sequence_statement != NULL);

  stats_list = sequence_statements(instruction_sequence(statement_instruction(sequence_statement)));

  MAP (STATEMENT, s, {
    ifdebug(7) {
      pips_debug(7, "Iterate on statement:\n");
      print_statement(s);
    }
    if (s == old_stat) {
      pips_debug(7, "Replace this statement:\n");
      new_stats_list = CONS(STATEMENT,new_stat,new_stats_list);
    }
    else {
      pips_debug(7, "Keep this statement:\n");
      new_stats_list = CONS(STATEMENT,s,new_stats_list);
    }
  }, stats_list);

  sequence_statements(instruction_sequence(statement_instruction(sequence_statement))) = gen_nreverse(new_stats_list);

  /*gen_insert_after (new_stat, old_stat, stats_list);
    gen_remove (&stats_list, old_stat);*/

  ifdebug(7) {
    pips_debug(7, "I've got this for the sequence\n");
    print_statement(sequence_statement);   
    pips_debug(7, "I've got this for the root statement\n");
    print_statement(root_stat);   
  }

  pips_debug(5, "END replace_in_sequence_statement_with:\n");
}

/**
 * Return a list of references corresponding to a list of regions
 */
list references_for_regions (list l_regions)
{
  list l_ref = NIL;
 
  MAP (EFFECT, reg, {
    reference ref = effect_any_reference(reg);
    l_ref = CONS (REFERENCE, ref, l_ref);
    print_reference(ref);
    pips_debug(4,"Entity: %s\n", entity_local_name(reference_variable(ref)));
  },l_regions);

  return l_ref;
}


