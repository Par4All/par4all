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
#include "safescale.h"


/**
 
 */
static void search_sequence_containing(statement s, void* a_context)
{
  sequence_searching_context* context = (sequence_searching_context*) a_context;
  instruction i = statement_instruction(s);
  
  if (instruction_tag(i) == is_instruction_sequence) 
  {
    MAP(STATEMENT, s2, {
                         if (s2 == context->searched_statement) 
                         {
	                   context->found_sequence_statement = s;
                         }
                       }, sequence_statements(instruction_sequence(i)));
  }
}


/**
 
*/
static statement sequence_statement_containing(statement root_statement, statement searched_stat)
{
  sequence_searching_context context;

  context.searched_statement = searched_stat;
  context.found_sequence_statement = NULL;

  gen_context_recurse(root_statement, &context, statement_domain, gen_true, search_sequence_containing);

  return context.found_sequence_statement;
}


/**
  
 */
static bool statement_is_contained_in_a_sequence_p(statement root_statement, statement searched_stat)
{
  return (sequence_statement_containing(root_statement, searched_stat) != NULL);
}


/**
   Add in the statement containing comments in the list of statements
 */
static void check_if_statement_contains_comment(statement s, void* a_context)
{
  statement_checking_context* context = (statement_checking_context*) a_context;
  string comments;

  if (!statement_with_empty_comment_p(s))
  {    
    comments = statement_comments(s);
    
    if (strstr(comments, context->searched_string) != NULL) 
    {
      context->list_of_statements = CONS(STATEMENT, s, context->list_of_statements);
    }
  }
}


/**
   Build a list with statements containing comments
 */
static list get_statements_with_comments_containing(string comment_portion, statement stat)
{
  string percent;
  statement_checking_context context;

  ifdebug(5)
  {
    pips_debug(5, "Comment portion: %s\n", comment_portion);
    pips_debug(5, "Statement:\n");
    print_statement(stat);
  }

  /* Set searched string */
  context.searched_string = strdup(comment_portion);
  percent = strstr(context.searched_string, "%s");

  pips_debug(5, "Percent: %s\n", percent);

  if (percent == NULL)
    pips_user_error("Malformed statement comment to search. Should be of the form 'BLAH_%%s'\n");

  *percent = '\0';

  /* Reset and get list of statements */
  context.list_of_statements = NIL;
  
  ifdebug(5) 
  {
    pips_debug(5, "Searching statements with comments: %s\n", context.searched_string);      
    pips_debug(5, "In statement:\n");      
    print_statement(stat);
  }

  gen_context_recurse(stat, &context, statement_domain, gen_true, check_if_statement_contains_comment);
  
  free(context.searched_string);

  return context.list_of_statements;  
}


/**
   Return the identified function name of the externalized portion of code by searching comment matching tag 
 */
static string get_function_name_by_searching_tag(statement stat, string tag) 
{
  string comments;
  string searched_string;
  string comment_portion = strdup(tag);
  char* function_name = NULL;
  char* next_line;
  instruction i = statement_instruction(stat);
  
  pips_debug(5, "BEGIN get_function_name_by_searching_tag [%s] on \n", tag);

  ifdebug(5) 
  {
    print_statement(stat);
  }

  if (instruction_tag(i) == is_instruction_sequence) 
  {
    stat = STATEMENT(gen_nth(0, sequence_statements(instruction_sequence(i))));
  }
  
  if (!statement_with_empty_comment_p(stat)) 
  {
    searched_string = strdup(comment_portion);
    searched_string[strcspn(comment_portion, "%s")] = '\0';
    comments = strdup(statement_comments(stat));
    next_line = strtok(comments, "\n");
  
    if (next_line != NULL) 
    {
      do 
      {
	string first_occurence = strstr(next_line, searched_string);

	if (first_occurence != NULL) 
        {
	  function_name = malloc(256);
	  sscanf(first_occurence, comment_portion, function_name);
	  pips_debug(5, "Found function: [%s]\n", function_name);
	}

	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }
  }

  free(comment_portion);
  free(comments);

  pips_debug(5, "END get_function_name_by_searching_tag [%s] on \n", tag);

  return function_name;
}


/**
   
 */
static void clean_statement_from_tags(string comment_portion, statement stat) 
{
  string comments;
  char* next_line;
  string searched_string;

  if (!statement_with_empty_comment_p(stat)) 
  {
    string new_comments = NULL;
 
    searched_string = strdup(comment_portion);
    searched_string[strcspn(comment_portion, "%s")] = '\0';
    comments = strdup(statement_comments(stat));
    next_line = strtok(comments, "\n");

    if (next_line != NULL) 
    {
      do 
      {
        if (strstr(next_line, searched_string) == NULL) 
        {
	  if (new_comments != NULL) 
          {
	    new_comments = strdup(concatenate(new_comments, next_line, "\n", NULL));
	  }
	  else 
          {
	    new_comments = strdup(concatenate("", next_line, "\n", NULL));
	  }
	}

	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }
    
    if (new_comments != NULL) 
    {
      statement_comments(stat) = new_comments;
    }
    else 
    {
      statement_comments(stat) = empty_comments;
      free(new_comments);
    }
  }    

  free(searched_string);
  free(comments);
}


/**
   Remove begin tag for a given statement and function
 */
static void remove_begin_tag(statement stat, string function_name) 
{
  char* removed_tag = malloc(256);

  sprintf(removed_tag, EXTERNALIZED_CODE_PRAGMA_BEGIN, function_name);
  pips_debug(2, "REMOVE %s from\n", removed_tag);
  print_statement(stat);
  clean_statement_from_tags(removed_tag, stat);
}


/**
   Remove end tag for given statement and function
 */
static void remove_end_tag(statement stat, string function_name) 
{
  char* removed_tag = malloc(256);

  sprintf(removed_tag, EXTERNALIZED_CODE_PRAGMA_END, function_name);
  pips_debug(2, "REMOVE %s from\n", removed_tag);
  print_statement(stat);
  clean_statement_from_tags(removed_tag, stat);
}


/**
   Transform a statement in a sequence statement with a single statement which is the embedded statement that will be distributed 
 */
static statement isolate_code_portion(statement begin_tag_statement, statement end_tag_statement, statement sequence_statement) 
{
  instruction i = statement_instruction(sequence_statement);
  list seq_stats = sequence_statements(instruction_sequence(i));
  list new_seq_stats = NIL;
  list isolated_seq_stats = NIL;
  bool statement_to_isolate = false;
  int nb_of_statements_to_isolate = 0;
  string function_name = get_function_name_by_searching_tag(begin_tag_statement, EXTERNALIZED_CODE_PRAGMA_BEGIN);

  pips_assert("sequence_statement is a sequence", instruction_tag(i) == is_instruction_sequence);
  pips_assert("function_name is not NULL", function_name != NULL);

  /* Count the number of statements to isolate in a single statement */
  MAP(STATEMENT, s, {
                      if ((statement_to_isolate && (s != end_tag_statement)) || ((!statement_to_isolate) && (s == begin_tag_statement))) 
                      {
                        nb_of_statements_to_isolate++;
                        isolated_seq_stats = CONS(STATEMENT, s, isolated_seq_stats);
                      }

                      if (statement_to_isolate && (s == end_tag_statement)) 
                      {
                        statement_to_isolate = false;
                      }

                      if ((!statement_to_isolate) && (s == begin_tag_statement)) 
                      {
                        statement_to_isolate = true;
                      }    
                    }, seq_stats);

  remove_begin_tag(begin_tag_statement, function_name);
  remove_end_tag(end_tag_statement, function_name);

  /* Insert an analyzed tag */
  {
    char *new_tag = malloc(256);

    sprintf(new_tag, strdup(concatenate("\n", EXTERNALIZED_CODE_PRAGMA_ANALYZED_PREFIX_TOP, EXTERNALIZED_CODE_PRAGMA_ANALYZED_TOP, "\n", NULL)), function_name, nb_of_statements_to_isolate);
    insert_comments_to_statement(begin_tag_statement, new_tag);
  }

  pips_debug(5, "Found %d statement to isolate\n", nb_of_statements_to_isolate);      

  if (nb_of_statements_to_isolate > 1) 
  {
    /* Build a new isolated sequence statement */    
    sequence new_sequence = make_sequence(gen_nreverse(isolated_seq_stats));
    instruction sequence_instruction = make_instruction(is_instruction_sequence, new_sequence);
    statement returned_statement = make_statement (entity_empty_label(),
						   statement_number(sequence_statement),
						   statement_ordering(sequence_statement),
						   empty_comments,
						   sequence_instruction,
						   NIL,
						   NULL,
						   statement_extensions (sequence_statement), make_synchronization_none());
    bool isolated_seq_stats_is_inserted = false;
    
    /* Build a new sequence containing isolated sequence statement */
    MAP(STATEMENT, s, {
                        if ((statement_to_isolate && (s != end_tag_statement)) || ((!statement_to_isolate) && (s == begin_tag_statement))) 
                        {
	                  if (!isolated_seq_stats_is_inserted) 
                          {
	                    new_seq_stats = CONS(STATEMENT, returned_statement, new_seq_stats);
	                    isolated_seq_stats_is_inserted = true;
	                  }
                        }
                        else 
                        {
	                  new_seq_stats = CONS(STATEMENT, s, new_seq_stats);
                        }

                        if (statement_to_isolate && (s == end_tag_statement)) 
                        {
	                  statement_to_isolate = false;
                        }

                        if ((!statement_to_isolate) && (s == begin_tag_statement)) 
                        {
	                  statement_to_isolate = true;
                        }
                      }, seq_stats); 

    /* Rebuild the sequence in the good order */
    sequence_statements(instruction_sequence(i)) = gen_nreverse(new_seq_stats);

    ifdebug(5) 
    {
      pips_debug(5,"Isolating and returning statement:\n");
      print_statement(returned_statement);
    }

    return returned_statement;
  }
  else if (nb_of_statements_to_isolate == 1) 
  {
    /* Nothing to do, the code is already isolated ! */
    ifdebug(5) 
    {
      pips_debug(5,"Isolating and returning statement:\n");
      print_statement(begin_tag_statement);
    }

    return begin_tag_statement;
  }
  else 
  {
    pips_user_warning("Malformed externalized code portion identified. No operation to do. Ignored.\n");
    return NULL;
  }
}


/**
   Return a list of statements that are marked for externalization
 */
static list identify_statements_to_distribute(statement module_stat) 
{
  list statements_containing_begin_tag;
  list statements_contained_in_a_sequence = NIL;
  list statements_to_distribute = NIL;
  
  /* Restructure code to avoid imbricated sequences if some portions are found to allow more than one INIT */
  simple_restructure_statement(module_stat);

  /* Identify statements containing a begin tag */
  statements_containing_begin_tag = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_BEGIN, module_stat); 
  
  /* Check all statements are contained in a sequence */
  MAP(STATEMENT, s, {
                      ifdebug(5) 
                      {
                        pips_debug(5, "Potential externalizable statement:\n");
                        print_statement(s);
                      }

                      if (statement_is_contained_in_a_sequence_p(module_stat,s)) 
                      {
                        statements_contained_in_a_sequence = CONS(STATEMENT, s,	statements_contained_in_a_sequence);
                      }
                      else 
                      {
                        pips_user_warning("Malformed externalized code portion identified. Ignored.\n");
                      }
                    }, statements_containing_begin_tag);
  
  /* */
  MAP(STATEMENT, s, {
                      statement sequence_statement;
                      string function_name;
                      list potential_end_statement = NIL;

                      sequence_statement = sequence_statement_containing(module_stat, s);

                      pips_debug(5, "Potential externalizable statement contained in a sequence \n");
                      print_statement(s);

                      function_name = get_function_name_by_searching_tag(s, EXTERNALIZED_CODE_PRAGMA_BEGIN);

                      if (function_name != NULL) 
                      {
                        pips_debug(5, "Name: [%s] \n", function_name);

                        potential_end_statement = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_END, sequence_statement);

                      if (gen_length(potential_end_statement) == 1) 
                      {
	                statement begin_tag_statement = s;
	                statement end_tag_statement = STATEMENT(gen_nth(0, potential_end_statement));
	                statement container_of_end_tag_statement = sequence_statement_containing(sequence_statement, end_tag_statement);

	                if (container_of_end_tag_statement == sequence_statement) 
                        {
	                  statement externalized_code = isolate_code_portion(begin_tag_statement, end_tag_statement, sequence_statement);

	                  statements_to_distribute = CONS(STATEMENT, externalized_code, statements_to_distribute);
	                }
	                else 
                        {
	                  pips_user_warning("Malformed externalized code portion identified [%s]. End tag found at a bad place!!!. Ignored.\n", function_name);
	                }
                      }
                      else 
                      {
	                pips_user_warning("Malformed externalized code portion identified [%s]. %d end tags found!!!. Ignored.\n", function_name, gen_length(potential_end_statement));
                      }
                    }
                    else 
                    {
                      pips_user_warning("Malformed externalized code portion identified [Unnamed]!!!. Ignored.\n");
                    }
                  }, statements_contained_in_a_sequence);

  return statements_to_distribute;
}
	
				      
/**
   Main phase for block code detection
*/   
bool safescale_distributor_init(const char* module_name)
{
  /* Get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  

  set_current_module_statement(stat);
  set_current_module_entity(module_name_to_entity(module_name));

  debug_on("SAFESCALE_DISTRIBUTOR_DEBUG_LEVEL");

  /* Doi the job */
  pips_debug(2, "BEGIN of SAFESCALE_DISTRIBUTOR_INIT\n");
  identify_statements_to_distribute(stat);
  pips_debug(2, "END of SAFESCALE_DISTRIBUTOR_INIT\n");

  pips_assert("Statement structure is consistent after SAFESCALE_DISTRIBUTOR_INIT", gen_consistent_p((gen_chunk*) stat));	      
  pips_assert("Statement is consistent after SAFESCALE_DISTRIBUTOR_INIT", statement_consistent_p(stat));

  /* Reorder the module because new statements have been added */  
  module_reorder(stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(stat));
  
  /* Update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
  
  debug_off();
  
  return true;
}
