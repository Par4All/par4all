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
#include "pipsdbm.h"

#include "text-util.h"
#include "properties.h"

#include "dg.h"
#include "transformer.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "semantics.h"
#include "control.h"
#include "callgraph.h"
#include "preprocessor.h"
#include "pipsmake.h"

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"

static entity create_module_with_statement (statement stat, 
					    string new_module_name,
					    list l_params,
					    list l_priv);

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tag 
 */
string get_function_name_by_searching_tag(statement stat,
					  const char* tag) 
{
  string comments;
  string searched_string;
  string comment_portion = strdup(tag);
  char* function_name = NULL;
  char*  next_line;
  instruction i = statement_instruction(stat);

  ifdebug(5) {
    pips_debug(5, "BEGIN get_function_name_by_searching_tag [%s] on \n", tag);
    print_statement(stat);
  }

  if (instruction_tag(i) == is_instruction_sequence) {
    stat = STATEMENT(gen_nth(0,sequence_statements(instruction_sequence(i))));
  }

  if (!statement_with_empty_comment_p(stat)) {
    searched_string = strdup(comment_portion);
    searched_string[strcspn(comment_portion, "%s")] = '\0';
    comments = strdup(statement_comments(stat));
    next_line = strtok (comments, "\n");
    if (next_line != NULL) {
      do {
	string first_occurence = strstr(next_line,searched_string);
	if (first_occurence != NULL) {
	  function_name = malloc(256);
	  sscanf (first_occurence, comment_portion, function_name);
	  pips_debug(5, "Found function: [%s]\n", function_name);
	}
	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }
  }

  pips_debug(5, "END get_function_name_by_searching_tag [%s] on \n", tag);
  return function_name;
}

/**
 * Remove begin tag for statement stat and function function_name
 */
static void remove_begin_tag (statement stat, string function_name)
{
  char* removed_tag ;
  asprintf (&removed_tag,EXTERNALIZED_CODE_PRAGMA_BEGIN,function_name);
  ifdebug(2) {
    pips_debug(2, "REMOVE %s from\n", removed_tag);
    print_statement (stat);
  }
  clean_statement_from_tags (removed_tag, stat);
}

/**
 * Remove end tag for statement stat and function function_name
 */
static void remove_end_tag (statement stat, string function_name)
{
  char* removed_tag ;
  asprintf (&removed_tag,EXTERNALIZED_CODE_PRAGMA_END,function_name);
  ifdebug(2) {
    pips_debug(2, "REMOVE %s from\n", removed_tag);
    print_statement (stat);
  }
  clean_statement_from_tags (removed_tag, stat);
}



/** At this point, we have a sequence statement sequence_statement which
 * contains a statement with a begin tag and a statement with a end
 * tag. The goal is to transform this statement in a sequence statement
 * with a single statement, which is the embedded statement that will be
 * distributed. 
 * This function returns this isolated statement
 */
static statement isolate_code_portion (statement begin_tag_statement, 
				       statement end_tag_statement, 
				       statement sequence_statement) 
{
  instruction i = statement_instruction(sequence_statement);
  list seq_stats = sequence_statements(instruction_sequence(i));
  list new_seq_stats = NIL;
  list isolated_seq_stats = NIL;
  bool statement_to_isolate;
  int nb_of_statements_to_isolate;
  string function_name = get_externalizable_function_name(begin_tag_statement);


  pips_assert ("sequence_statement is a sequence",
	       instruction_tag(i) == is_instruction_sequence);

  pips_assert ("function_name is not NULL",
	       function_name != NULL);

  /* First, count the number of statements to isolate in a single statement */
  statement_to_isolate = false;
  nb_of_statements_to_isolate = 0;
  MAP (STATEMENT, s, {

    if ((statement_to_isolate && (s != end_tag_statement))
	|| ((!statement_to_isolate) && (s == begin_tag_statement))) {
      nb_of_statements_to_isolate++;
      isolated_seq_stats = CONS(STATEMENT, s, isolated_seq_stats);
    }
    if (statement_to_isolate && (s == end_tag_statement)) {
      statement_to_isolate = false;
    }
    if ((!statement_to_isolate) && (s == begin_tag_statement)) {
      statement_to_isolate = true;
    }
    
  }, seq_stats);

  remove_begin_tag (begin_tag_statement, function_name);
  remove_end_tag (end_tag_statement, function_name);

  /* Insert an analyzed tag */
  {
    char* new_tag ;
    asprintf (&new_tag, 
	     (concatenate("! ",
				EXTERNALIZED_CODE_PRAGMA_ANALYZED,
				"\n",
				NULL)),
	     function_name,
	     nb_of_statements_to_isolate);
    insert_comments_to_statement(begin_tag_statement, new_tag);
  }

  pips_debug(5, "Found %d statement to isolate\n",
	     nb_of_statements_to_isolate);      

  if (nb_of_statements_to_isolate > 1) {
 
    /* Build new isolated sequence statement */
    
    sequence new_sequence = make_sequence (gen_nreverse(isolated_seq_stats));
    instruction sequence_instruction
      = make_instruction(is_instruction_sequence,
			 new_sequence);
    statement returned_statement 
      = make_statement(entity_empty_label(),
		       statement_number(sequence_statement),
		       statement_ordering(sequence_statement),
		       empty_comments,
		       sequence_instruction,NIL,NULL,
		       statement_extensions(sequence_statement), make_synchronization_none());

    bool isolated_seq_stats_is_inserted = false;
    
    /* Build new sequence containing isolated sequence statement */

    MAP (STATEMENT, s, {
      if ((statement_to_isolate && (s != end_tag_statement))
	  || ((!statement_to_isolate) && (s == begin_tag_statement))) {
	if (!isolated_seq_stats_is_inserted) {
	  new_seq_stats = CONS(STATEMENT, returned_statement, new_seq_stats);
	  isolated_seq_stats_is_inserted = true;
	}
      }
      else {
	new_seq_stats = CONS(STATEMENT, s, new_seq_stats);
      }
      if (statement_to_isolate && (s == end_tag_statement)) {
	statement_to_isolate = false;
      }
      if ((!statement_to_isolate) && (s == begin_tag_statement)) {
	statement_to_isolate = true;
      }
    }, seq_stats); 

    /* Rebuild the sequence in the GOOD order */
    sequence_statements(instruction_sequence(i)) 
      = gen_nreverse(new_seq_stats);

    ifdebug(5) {
      pips_debug(5,"Isolating and returning statement:\n");
      print_statement(returned_statement);
    }

    return returned_statement;
  }
  else if (nb_of_statements_to_isolate == 1) {
    /* Nothing to do, the code is already isolated ! */
    ifdebug(5) {
      pips_debug(5,"Isolating and returning statement:\n");
      print_statement(begin_tag_statement);
    }
    return begin_tag_statement;
  }
  else {
    pips_user_warning("Malformed externalized code portion identified. No operation to do. Ignored.\n");
    return NULL;
  }

}


/* 
 * This function return a list of statements that were previously marked
 * for externalization during phase PHRASE_DISTRIBUTOR_INIT
 */
list identify_analyzed_statements_to_distribute (statement stat) 
{
  /* We identify all the statement containing an analyzed tag */
  return get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_ANALYZED,
						 stat); 
  
}


/* 
 * This function return a list of statements that are marked for
 * externalization.  A well-formed externalizable code must be one or more
 * continuous statements defined in a sequence statement framed with
 * comments containing tags EXTERNALIZED_CODE_PRAGMA_BEGIN and
 * EXTERNALIZED_CODE_PRAGMA_END
 */
static list identify_statements_to_distribute (statement module_stat) 
{
  list statements_containing_begin_tag = NIL;
  list statements_contained_in_a_sequence = NIL;
  list statements_to_distribute = NIL;
  
  /* We identify all the statement containing a begin tag */
  statements_containing_begin_tag 
    = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_BEGIN,
					      module_stat); 
  /* We restructure the code to avoid imbricated sequences only if
   * some portions are found (to allow more than one INIT) */
  if (gen_length(statements_containing_begin_tag) > 0) {
    simple_restructure_statement(module_stat);
  }

  /* We identify again (after code restructuration) all the statement
   * containing a begin tag */
  statements_containing_begin_tag = NIL;
  statements_containing_begin_tag
    = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_BEGIN,
					      module_stat);

  /* We check that all those statements are contained in a sequence */
  MAP (STATEMENT, s, {
    ifdebug(5) {
      pips_debug(5, "Potential externalizable statement:\n");
      print_statement(s);
    }
    if (statement_is_contained_in_a_sequence_p (module_stat,s)) {
      statements_contained_in_a_sequence
	= CONS (STATEMENT,
		s,
		statements_contained_in_a_sequence);
    }
    else {
      pips_user_warning("Malformed externalized code portion identified. Ignored.\n");
    }
  }, statements_containing_begin_tag);

  /* */
  FOREACH (STATEMENT, s, statements_contained_in_a_sequence){
      statement sequence_statement;
      string function_name;
      string end_tag;
      list potential_end_statement = NIL;
      sequence_statement = sequence_statement_containing (module_stat,s);
      ifdebug(5) {
          pips_debug(5, "Potential externalizable statement contained in a sequence \n");
          print_statement(s);
      }
      function_name = get_externalizable_function_name(s);
      if (function_name != NULL) {
          pips_debug(5, "Name: [%s] \n", function_name);
          asprintf (&end_tag, EXTERNALIZED_CODE_PRAGMA_END,function_name);
          potential_end_statement
              = get_statements_with_comments_containing (end_tag,
                      sequence_statement);
          if (gen_length(potential_end_statement) == 1) {
              statement begin_tag_statement = s;
              statement end_tag_statement 
                  = STATEMENT(gen_nth(0,potential_end_statement));
              statement container_of_end_tag_statement
                  = sequence_statement_containing (sequence_statement, end_tag_statement);
              if (container_of_end_tag_statement == sequence_statement) {
                  statement externalized_code
                      = isolate_code_portion (begin_tag_statement, 
                              end_tag_statement, 
                              sequence_statement);
                  statements_to_distribute 
                      = CONS (STATEMENT,
                              externalized_code,
                              statements_to_distribute);
              }
              else {
                  pips_user_warning("Malformed externalized code portion identified [%s]. End tag found at a bad place!!!. Ignored.\n", function_name);
              }
          }
          else {
              pips_user_warning("Malformed externalized code portion identified [%s]. %d end tags found!!!. Ignored.\n", function_name, gen_length(potential_end_statement));
          }
      }
      else {
          pips_user_warning("Malformed externalized code portion identified [Unnamed]!!!. Ignored.\n");
      }
  }

  return statements_to_distribute;
}
					      
/**
 * This function is called after identification and isolation of a portion
 * of code to externalize. Externalization of specified code is done here.
 */
static void distribute_code (string function_name,
			     statement externalized_code, 
			     statement module_stat, 
			     list l_params,
			     list l_priv) 
{
  entity new_module;
  statement call_statement;
  list call_params;
  string call_comments;

  new_module 
    = create_module_with_statement (externalized_code, 
				    function_name,
				    l_params,
				    l_priv);
  
  call_params = NIL;
  MAP (REFERENCE, ref, {
    call_params = CONS(EXPRESSION, entity_to_expression(reference_variable(ref)), call_params);
  }, references_for_regions(l_params));
  
  // Insert an analyzed tag 
  {
    asprintf (&call_comments, 
	     (concatenate("! ",
				EXTERNALIZED_CODE_PRAGMA_CALL,
				"\n",
				NULL)),
	     function_name);
  }

  call_statement = make_statement(entity_empty_label(),
				  statement_number(externalized_code),
				  statement_ordering(externalized_code),
				  call_comments,
				  make_instruction(is_instruction_call,
						   make_call(new_module,call_params)),
				  NIL,NULL,
				  statement_extensions(externalized_code), make_synchronization_none());

  ifdebug(5) {
    pips_debug(5, "BEFORE REPLACING\n");
    pips_debug(5, "externalized_code=\n");
    print_statement(externalized_code);
    pips_debug(5, "call_statement=\n");
    print_statement(call_statement);
    pips_debug(5, "module_stat=\n");
    print_statement(module_stat);
  }

  replace_in_sequence_statement_with(externalized_code,
				     call_statement,
				     module_stat);

  ifdebug(5) {
    pips_debug(5, "AFTER REPLACING\n");
    pips_debug(5, "externalized_code=\n");
    print_statement(externalized_code);
    pips_debug(5, "call_statement=\n");
    print_statement(call_statement);
    pips_debug(5, "module_stat=\n");
    print_statement(module_stat);
  }

  pips_assert("Module structure is consistent after DISTRIBUTE_CODE", 
	      gen_consistent_p((gen_chunk*)new_module));
	      
  pips_assert("Statement structure is consistent after DISTRIBUTE_CODE", 
	      gen_consistent_p((gen_chunk*)externalized_code));
	      
  pips_assert("Statement is consistent after DISTRIBUTE_CODE", 
	      statement_consistent_p(externalized_code));
  
  pips_debug(5, "Code distribution for : [%s] is DONE\n", function_name);
}
/**
 * Main function for PHRASE_DISTRIBUTION: phrase distribution for main
 * module module, with root statement stat
 */
static void distribute (statement module_stat, 
			entity module) 
{
  list l_stats;
  hash_table ht_stats;
  hash_table ht_params;
  hash_table ht_private;
  hash_table ht_in_regions;
  hash_table ht_out_regions;

  l_stats = identify_analyzed_statements_to_distribute (module_stat);

  compute_distribution_context (l_stats, 
				module_stat,
				module,
				&ht_stats,
				&ht_params,
				&ht_private,
				&ht_in_regions,
				&ht_out_regions);
  
  HASH_MAP (function_name, stat, {
    distribute_code (function_name,
		     stat, 
		     module_stat, 
		     hash_get(ht_params,function_name),
		     hash_get(ht_private,function_name));
  },ht_stats);
  
  hash_table_free(ht_stats);
  hash_table_free(ht_params);
  hash_table_free(ht_private);
  hash_table_free(ht_in_regions);
  hash_table_free(ht_out_regions);
}

/**
 * Main function for PHRASE_DISTRIBUTION_INIT: phrase distribution for
 * module module_stat
 */
static void prepare_distribute (statement module_stat) 
{
  identify_statements_to_distribute (module_stat);
}

/*
 * Return EXTERNALIZED_FUNCTION_PARAM_NAME
 */
static string get_externalized_function_param_name (entity variable, int param_nb) 
{
  char *buffer;
  asprintf(&buffer,
	  EXTERNALIZED_FUNCTION_PARAM_NAME,
	  entity_local_name(variable),
	  param_nb);
  return (buffer);
}

/*
 * Return EXTERNALIZED_FUNCTION_PRIVATE_PARAM_NAME
 */
static string get_externalized_function_private_param_name (entity variable) 
{
  char *buffer;
  asprintf(&buffer,
	  EXTERNALIZED_FUNCTION_PRIVATE_PARAM_NAME,
	  entity_local_name(variable));
  return (buffer);
}

/**
 * Creates a variable declared as a parameter in specified module
 */
static entity create_parameter_variable_for_new_module (entity a_variable,
							string new_name, 
							string new_module_name,
							entity module,
							int param_nb)
{
  entity new_variable;
 
  if ((gen_find_tabulated(concatenate(new_module_name, 
				      MODULE_SEP_STRING, 
				      new_name, 
				      NULL),
			  entity_domain)) == entity_undefined) 
    { 
      /* This entity does not exist, we can safely create it */
      
      new_variable = make_entity (strdup(concatenate(new_module_name, 
						     MODULE_SEP_STRING, 
						     new_name, NULL)),
				  copy_type (entity_type(a_variable)),
				  make_storage_formal (make_formal(module, param_nb)),
				  copy_value (entity_initial(a_variable)));

      return new_variable;
    }
  else 
    {
      pips_internal_error("Entity already exist: %s", new_name);
      return NULL;
    }
}

/**
 * Creates a private variable in specified module
 */
entity create_private_variable_for_new_module (entity a_variable,
					       const char* new_name, 
					       const char* new_module_name,
					       entity module)
{
  entity new_variable;
  entity a;
  basic base;
  
  if ((gen_find_tabulated(concatenate(new_module_name, 
				      MODULE_SEP_STRING, 
				      new_name, 
				      NULL),
			  entity_domain)) == entity_undefined) 
    { 
      /* This entity does not exist, we can safely create it */
      
      new_variable = make_entity (strdup(concatenate(new_module_name, 
						     MODULE_SEP_STRING, 
						     new_name, NULL)),
				  copy_type (entity_type(a_variable)),
				  storage_undefined,
				  copy_value (entity_initial(a_variable)));
      a = FindEntity(new_module_name, DYNAMIC_AREA_LOCAL_NAME); 
      base = variable_basic(type_variable(entity_type(a_variable)));
      entity_storage(new_variable) = 
	make_storage(is_storage_ram,
		     make_ram(module, a,
			      (basic_tag(base)!=is_basic_overloaded)?
			      (add_variable_to_area(a, new_variable)):(0),
			      NIL));
      pips_debug(2, "Created new private variable: %s\n", entity_global_name(new_variable));
      return new_variable;
    }
  else 
    {
      pips_internal_error("Entity already exist: %s", new_name);
      return NULL;
    }
}


/**
 * Declare in the newly created module a new variable (which will be a
 * parameter of the module), and replace all occurences to the old
 * variable by the new created
 */
void add_parameter_variable_to_module (reference ref,
				       entity module,
				       statement stat, /* Statement of the new module */
				       string new_module_name,
				       int param_nb)
{
  parameter new_parameter;
  list module_declarations;
  list module_parameters;
  entity new_variable;
  string variable_name;

  pips_debug(2, "Registering parameter: %s\n", entity_local_name(reference_variable(ref)));

  /* Assert that entity represent a value code */
  pips_assert("It is a module", entity_module_p(module));

  /* Get the variable name */
  variable_name 
    = get_externalized_function_param_name (reference_variable(ref),param_nb); 

  new_variable = create_parameter_variable_for_new_module (reference_variable(ref),
							   variable_name,
							   new_module_name,
							   module,
							   param_nb);
  
  replace_reference (stat, ref, new_variable);
  
  module_declarations = code_declarations(value_code(entity_initial(module)));
  
  code_declarations(value_code(entity_initial(module)))
    = CONS (ENTITY, new_variable, module_declarations);
  
  new_parameter = make_parameter (entity_type(new_variable),
				  make_mode_reference(),
				  /*strdup("")*/make_dummy_identifier(new_variable));
  
  module_parameters = functional_parameters(type_functional(entity_type(module)));

  functional_parameters(type_functional(entity_type(module))) 
    = CONS(PARAMETER, new_parameter, module_parameters);

}

/**
 * Declare in the newly created module a new variable (which will be a
 * private to the module), and replace all occurences to the old
 * variable by the new created
 */
void add_private_variable_to_module (reference ref,
					    entity module,
					    statement stat, /* Statement of the new module */
					    string new_module_name)
{
  list module_declarations;
  entity new_variable;
  string variable_name;
  
  pips_debug(2, "Registering private variable: %s\n", entity_local_name(reference_variable(ref)));
  
  /* Assert that entity represent a value code */
  pips_assert("It is a module", entity_module_p(module));
  
  /* Get the variable name */
  variable_name = get_externalized_function_private_param_name (reference_variable(ref)); 

  /* Creates the variable */
  new_variable 
    = create_private_variable_for_new_module (reference_variable(ref),
					      variable_name,
					      new_module_name,
					      module);  
  
  replace_reference (stat, ref, new_variable);
    
  module_declarations = code_declarations(value_code(entity_initial(module)));
  
  code_declarations(value_code(entity_initial(module)))
    = CONS (ENTITY, new_variable, module_declarations);
  
}

/**
 * Dynamically build a new module with specified statement.
 * After creation, return it.
 */
static entity create_module_with_statement (statement stat, 
					    string new_module_name,
					    list l_params,
					    list l_priv) 
{
    entity new_module;
    int param_nb = 0;

    pips_debug(2, "Creating new module: [%s]\n", new_module_name);

    new_module = make_empty_subroutine(new_module_name,make_language_unknown());

    /* Deal with private variables */
    FOREACH (REFERENCE, ref, references_for_regions (l_priv)){
        add_private_variable_to_module (ref, 
                new_module, 
                stat, 
                new_module_name);
    }

    // Deal with parameters variables 
    param_nb = gen_length(l_params);
    FOREACH (REFERENCE, ref, references_for_regions(l_params)){
        add_parameter_variable_to_module (ref, 
                new_module, 
                stat, 
                new_module_name, 
                param_nb);
        param_nb--;
    }

    pips_debug(2, "Making new module: [%s]\n", new_module_name);
    ifdebug(5) {
        pips_debug(5, "With statement: \n");
        print_statement (stat);
    }
    text t = text_named_module(new_module, new_module, stat);
    add_new_module_from_text(new_module_name, t, fortran_module_p(get_current_module_entity()), compilation_unit_of_module(get_current_module_name()) );

    free_text(t);

    return new_module;
}

/*********************************************************
 * Phase main for PHRASE_DISTRIBUTOR_INIT
 *********************************************************/

bool phrase_distributor_init(const char* module_name)
{
  
  /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      true);
  
  
  set_current_module_statement(stat);
  set_current_module_entity(module_name_to_entity(module_name)); //FI: redundant
  
  debug_on("PHRASE_DISTRIBUTOR_DEBUG_LEVEL");

  /* Now do the job */

  pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR_INIT\n");
  prepare_distribute (stat);
  pips_debug(2, "END of PHRASE_DISTRIBUTOR_INIT\n");

  pips_assert("Statement structure is consistent after PHRASE_DISTRIBUTOR_INIT", 
	      gen_consistent_p((gen_chunk*)stat));
	      
  pips_assert("Statement is consistent after PHRASE_DISTRIBUTOR_INIT", 
	       statement_consistent_p(stat));

  
  /* Reorder the module, because new statements have been added */  
  module_reorder(stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, 
			 compute_callees(stat));
  
  /* update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
  
  debug_off();
  
  return true;
}

/*********************************************************
 * Phase main for PHRASE_DISTRIBUTOR
 *********************************************************/

static entity dynamic_area = entity_undefined;

bool phrase_distributor(const char* module_name)
{
  statement module_stat;
  entity module;
  
  /* set and get the current properties concerning regions */
  set_bool_property("MUST_REGIONS", true);
  set_bool_property("EXACT_REGIONS", true);
  get_regions_properties();
  
  /* get the resources */
  module_stat = (statement) db_get_memory_resource(DBR_CODE, 
						   module_name, 
						   true);
  
  module = module_name_to_entity(module_name);
  
  set_current_module_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name)); // FI: redundant
  
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
  module_to_value_mappings(module);
  
  /* sets dynamic_area */
  if (entity_undefined_p(dynamic_area)) {   	
    dynamic_area = FindOrCreateEntity(module_local_name(module),
				      DYNAMIC_AREA_LOCAL_NAME); 
  }

  debug_on("PHRASE_DISTRIBUTOR_DEBUG_LEVEL");

     /* Get the READ, WRITE, IN and OUT regions of the module
      */
    set_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_REGIONS, module_name, true));
    set_in_effects((statement_effects) 
	db_get_memory_resource(DBR_IN_REGIONS, module_name, true));
    set_out_effects((statement_effects) 
	db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));

  /* Now do the job */

  pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR\n");
  distribute (module_stat, module);
  pips_debug(2, "END of PHRASE_DISTRIBUTOR\n");

  //print_statement(module_stat);

  pips_assert("Statement structure is consistent after PHRASE_DISTRIBUTOR", 
	      gen_consistent_p((gen_chunk*)module_stat));
	      
  pips_assert("Statement is consistent after PHRASE_DISTRIBUTOR", 
	      statement_consistent_p(module_stat));
  
  /* Reorder the module, because new statements have been added */  
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, 
			 compute_callees(module_stat));
  
  /* update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
  dynamic_area = entity_undefined;
  reset_cumulated_rw_effects();
  reset_rw_effects();
  reset_in_effects();
  reset_out_effects();
  free_value_mappings();
  
  debug_off();
  
  return true;
}
