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

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "ricedg.h"
#include "semantics.h"
#include "control.h"

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"

static entity create_module_with_statement (statement stat, 
					    string new_module_name,
					    entity root_module,
					    list params);

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tag EXTERNALIZED_CODE_PRAGMA_BEGIN
 */
static string get_externalized_function_name(statement stat) 
{
  string comments;
  string searched_string;
  string comment_portion = strdup(EXTERNALIZED_CODE_PRAGMA_BEGIN);
  char* function_name = NULL;
  char*  next_line;
  instruction i = statement_instruction(stat);
  
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
	  /*pips_debug(5, "Found function: [%s]\n", function_name);*/
	}
	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }
  }

  return function_name;
}

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tags EXTERNALIZED_CODE_PRAGMA_ANALYZED
 * Sets the number of statements of this externalizable statement
 */
static string get_externalized_and_analyzed_function_name(statement stat,
							  int* stats_nb) 
{
  string comments;
  string searched_string;
  string searched_string2;
  string comment_portion = strdup(EXTERNALIZED_CODE_PRAGMA_ANALYZED);
  char* function_name = NULL;
  char*  next_line;
  instruction i = statement_instruction(stat);
  
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
	  pips_debug(5, "Scanning: [%s] with [%s]", first_occurence, comment_portion);
	  sscanf (first_occurence, comment_portion, function_name, stats_nb);
	  pips_debug(5, "Found function: [%s] and %d stats \n", function_name, *stats_nb);
	}
	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }
  }

  return function_name;
}

/**
 * Remove begin tag for statement stat and function function_name
 */
static void remove_begin_tag (statement stat, string function_name) 
{
  char* removed_tag = malloc(256);
  sprintf (removed_tag,EXTERNALIZED_CODE_PRAGMA_BEGIN,function_name);
  pips_debug(2, "REMOVE %s from\n", removed_tag);
  print_statement (stat);
  clean_statement_from_tags (removed_tag, stat);
}

/**
 * Remove end tag for statement stat and function function_name
 */
static void remove_end_tag (statement stat, string function_name) 
{
  char* removed_tag = malloc(256);
  sprintf (removed_tag,EXTERNALIZED_CODE_PRAGMA_END,function_name);
  pips_debug(2, "REMOVE %s from\n", removed_tag);
  print_statement (stat);
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
  statement returned_statement;
  instruction i = statement_instruction(sequence_statement);
  list seq_stats = sequence_statements(instruction_sequence(i));
  list new_seq_stats = NIL;
  list isolated_seq_stats = NIL;
  bool statement_to_isolate;
  int nb_of_statements_to_isolate;
  string function_name = get_externalized_function_name(begin_tag_statement);


  pips_assert ("sequence_statement is a sequence",
	       instruction_tag(i) == is_instruction_sequence);

  pips_assert ("function_name is not NULL",
	       function_name != NULL);

  /* First, count the number of statements to isolate in a single statement */
  statement_to_isolate = FALSE;
  nb_of_statements_to_isolate = 0;
  MAP (STATEMENT, s, {

    if ((statement_to_isolate && (s != end_tag_statement))
	|| ((!statement_to_isolate) && (s == begin_tag_statement))) {
      nb_of_statements_to_isolate++;
      isolated_seq_stats = CONS(STATEMENT, s, isolated_seq_stats);
    }
    if (statement_to_isolate && (s == end_tag_statement)) {
      statement_to_isolate = FALSE;
    }
    if ((!statement_to_isolate) && (s == begin_tag_statement)) {
      statement_to_isolate = TRUE;
    }
    
  }, seq_stats);

  remove_begin_tag (begin_tag_statement, function_name);
  remove_end_tag (end_tag_statement, function_name);

  /* Insert an analyzed tag */
  {
    char* new_tag = malloc(256);
    sprintf (new_tag, 
	     strdup(concatenate("! ",
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
		       sequence_instruction,NIL,NULL);

    bool isolated_seq_stats_is_inserted = FALSE;
    
    /* Build new sequence containing isolated sequence statement */

    MAP (STATEMENT, s, {
      if ((statement_to_isolate && (s != end_tag_statement))
	  || ((!statement_to_isolate) && (s == begin_tag_statement))) {
	if (!isolated_seq_stats_is_inserted) {
	  new_seq_stats = CONS(STATEMENT, returned_statement, new_seq_stats);
	  isolated_seq_stats_is_inserted = TRUE;
	}
      }
      else {
	new_seq_stats = CONS(STATEMENT, s, new_seq_stats);
      }
      if (statement_to_isolate && (s == end_tag_statement)) {
	statement_to_isolate = FALSE;
      }
      if ((!statement_to_isolate) && (s == begin_tag_statement)) {
	statement_to_isolate = TRUE;
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
static list identify_analyzed_statements_to_distribute (statement stat,
							string module_name) 
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
static list identify_statements_to_distribute (statement stat,
					       string module_name) 
{
  list statements_containing_begin_tag = NIL;
  list statements_contained_in_a_sequence = NIL;
  list statements_to_distribute = NIL;
  
  /* We identify all the statement containing a begin tag */
  statements_containing_begin_tag 
    = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_BEGIN,
					      stat); 
  /* We restructure the code to avoid imbricated sequences only if
   * some portions are found (to allow more than one INIT) */
  if (gen_length(statements_containing_begin_tag) > 0) {
    simple_restructure_statement(stat);
  }

  /* We identify again (after code restructuration) all the statement
   * containing a begin tag */
  statements_containing_begin_tag = NIL;
  statements_containing_begin_tag 
    = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_BEGIN,
					      stat); 
  
  /* We check that all those statements are contained in a sequence */
  MAP (STATEMENT, s, {
    ifdebug(5) {
      pips_debug(5, "Potential externalizable statement:\n");
      print_statement(s);
    }
    if (statement_is_contained_in_a_sequence_p (stat,s)) {
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
  MAP (STATEMENT, s, {
    statement sequence_statement;
    string function_name;
    string end_tag;
    list potential_end_statement = NIL;
    sequence_statement = 
      sequence_statement_containing (stat,s);
    pips_debug(5, "Potential externalizable statement contained in a sequence \n");
    print_statement(s);
    function_name = get_externalized_function_name(s);
    if (function_name != NULL) {
      pips_debug(5, "Name: [%s] \n", function_name);
      end_tag = malloc(256);
      sprintf (end_tag, EXTERNALIZED_CODE_PRAGMA_END,function_name);
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
  }, statements_contained_in_a_sequence);

  return statements_to_distribute;
}
					      
/**
 * Return list of entities which need to be passed as parameters (IN or OUT)
 */
static list compute_parameters (statement externalized_code, 
				statement module_stat, 
				entity module,
				string module_name) 
{
  list l_read, l_write, l_in, l_out;
  list l_priv = NIL;
  list l_get_refs = NIL;
  list l_set_refs = NIL;
  list l_params = NIL;

  pips_debug(3, "Compute regions\n");
  
  l_write = regions_dup
    (regions_write_regions(load_statement_local_regions(externalized_code))); 
  l_read = regions_dup
    (regions_read_regions(load_statement_local_regions(externalized_code))); 
  l_in = regions_dup(load_statement_in_regions(externalized_code));
  l_out = regions_dup(load_statement_out_regions(externalized_code));
  
  ifdebug(2)
    {
      pips_debug(3, "READ regions: \n");
      print_regions(l_read);
      pips_debug(3, "WRITE regions: \n");
      print_regions(l_write);
      pips_debug(3, "IN regions: \n");
      print_regions(l_in);
      pips_debug(3, "OUT regions: \n");
      print_regions(l_out);
      }
  
  /*pips_debug(3, "HOP1\n");*/
  l_get_refs = references_for_regions (l_read);
  /*pips_debug(3, "HOP2\n");*/
  
  l_priv = RegionsEntitiesInfDifference(l_write, l_in, w_r_combinable_p);
  l_priv = RegionsEntitiesInfDifference(l_priv, l_out, w_w_combinable_p);
  
  ifdebug(2)
    {
      pips_debug(3, "Private regions: \n");
	print_regions(l_priv);
    }
    
    /*    pips_debug(3, "HOP1\n");
	  l_priv = NIL;
	  l_priv = RegionsEntitiesInfDifference(l_write, l_in, w_r_combinable_p);
	  l_priv = RegionsEntitiesInfDifference(l_priv, l_out, w_w_combinable_p);
	  l_get_params = RegionsEntitiesInfDifference(l_read, l_priv, w_r_combinable_p);
	  pips_debug(3, "HOP2\n");
	  l_priv = NIL;
	  l_priv = RegionsEntitiesInfDifference(l_write, l_in, w_r_combinable_p);
	  l_priv = RegionsEntitiesInfDifference(l_priv, l_out, w_w_combinable_p);
	  l_set_params = RegionsEntitiesInfDifference(l_write, l_priv, w_w_combinable_p);
	  pips_debug(3, "HOP3\n");
	  l_params = RegionsEntitiesIntersection(l_get_params, l_set_params, w_r_combinable_p);
	  pips_debug(3, "HOP4\n");
	  
	  ifdebug(2)
	  {
	  pips_debug(3, "GET_PARAMS regions: \n");
	  print_regions(l_get_params);
	  pips_debug(3, "SET_PARAMS regions: \n");
	  print_regions(l_set_params);
	  pips_debug(3, "PARAMS regions: \n");
	  print_regions(l_params);
	  }
    */
  return l_get_refs;
}

/**
 * This function is called after identification and isolation of a portion
 * of code to externalize. Externalization of specified code is done here.
 * Note that specified code may be only the first statement of a sequence
 * that will automatically be externalized !!!
 */
static void distribute_code (statement analyzed_code, 
			     statement module_stat, 
			     entity module,
			     string module_name) 
{
  statement externalized_code = NULL;
  string function_name;
  entity new_module;
  statement call_statement;
  int stats_nb;
  list params;
  list call_params;

  function_name = get_externalized_and_analyzed_function_name(analyzed_code, &stats_nb);
  
  pips_debug(5, "Distribute code for : [%s] ...\n", function_name);

  if (stats_nb > 1) {
    externalized_code = sequence_statement_containing (module_stat, analyzed_code);
  }
  else if (stats_nb == 1) {
    externalized_code = analyzed_code;
  }
  else {
    pips_internal_error("Strange externalized code\n");
  }

  params = compute_parameters (externalized_code, module_stat, module, module_name);

  new_module 
    = create_module_with_statement (externalized_code, 
				    function_name,
				    module,
				    params);

  call_params = NIL;
  MAP (REFERENCE, ref, {
    call_params = CONS(EXPRESSION, make_expression_from_entity(reference_variable(ref)), call_params);
  }, params);

  call_statement = make_statement(entity_empty_label(),
				  statement_number(externalized_code),
				  statement_ordering(externalized_code),
				  empty_comments,
				  make_instruction(is_instruction_call,
						   make_call(new_module,call_params)),
				  NIL,NULL);

  replace_in_sequence_statement_with (externalized_code,
				      call_statement, 
				      module_stat);
  
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
			entity module,
			string module_name) 
{
  entity new_module1, new_module2;
  list l;

  l = identify_analyzed_statements_to_distribute (module_stat, module_name);

  MAP (STATEMENT, s, {
    distribute_code (s, module_stat, module, module_name);
  }, l);

}

/**
 * Main function for PHRASE_DISTRIBUTION_INIT: phrase distribution for
 * main module module, with root statement stat
 */
static void prepare_distribute (statement module_stat, 
				entity module,
				string module_name) 
{
  entity new_module1, new_module2;
  list l;
  
  l = identify_statements_to_distribute (module_stat, module_name);
  
}

static entity create_variable_for_new_module (entity a_variable,
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
      pips_error("Entity already exist: %s\n", new_name);
      return NULL;
    }
}

static void add_parameter_to_module (reference ref,
				     entity module,
				     statement stat, /* Statement of the new module */
				     string new_module_name,
				     entity root_module,
				     int param_nb)
{
  parameter new_parameter;
  list module_declarations;
  list module_parameters;
  entity new_variable;
  string variable_name;
  string base_name;
  char buffer[50];
  expression new_expression;

  pips_debug(2, "Registering parameter: %s\n", entity_local_name(reference_variable(ref)));

  pips_debug(2, "Indices (%d)\n", reference_indices(ref));

  /* Assert that entity represent a value code */
  pips_assert("It is a module", entity_module_p(module));

  base_name = strdup("%s_PARAM_%d");
  
  sprintf(buffer, base_name, entity_local_name(reference_variable(ref)), param_nb);
  variable_name = strdup(buffer);
  new_variable = create_variable_for_new_module(reference_variable(ref),
						/*variable_name,*/
						entity_local_name(reference_variable(ref)),
						new_module_name,
						module,
						param_nb);  
  
  StatementReplaceReference(stat, ref, make_expression_from_entity(new_variable));

  /*propagate_synonym (stat, reference_variable(ref), new_variable, FALSE);*/

  module_declarations = code_declarations(value_code(entity_initial(module)));

  code_declarations(value_code(entity_initial(module)))
    = CONS (ENTITY, new_variable, module_declarations);
  
  new_parameter = make_parameter (entity_type(new_variable),
				  make_mode(is_mode_reference, UU));
  
  module_parameters = functional_parameters(type_functional(entity_type(module)));

  functional_parameters(type_functional(entity_type(module))) 
    = CONS(PARAMETER, new_parameter, module_parameters);

}

/**
 * Dynamically build a new module with specified statement.
 * After creation, return it.
 */
static entity create_module_with_statement (statement stat, 
					    string new_module_name,
					    entity root_module,
					    list params) 
{
  entity new_module;
  string source_file;
  list params_l = NIL;
  int param_nb = 0;

  new_module = make_empty_subroutine(new_module_name);
  
  param_nb = gen_length(params);

  MAP (REFERENCE, ref, {
    add_parameter_to_module (ref, new_module, stat, new_module_name, root_module, param_nb);
    param_nb--;
  },params);
  
  /*functional_parameters(type_functional(entity_type(new_module)))
    = CONS(PARAMETER, make_parameter(MakeTypeVariable(make_basic(is_basic_int,4), NIL),
    make_mode(is_mode_reference, UU)), NIL);*/

  pips_debug(2, "Making new module: [%s]\n", new_module_name);
  ifdebug(5) {
    pips_debug(2, "With statement: \n");
    print_statement (stat);
  }

  /*module_reorder(stat);*/

  init_prettyprint(empty_text);
  make_text_resource(new_module_name,
		     DBR_SOURCE_FILE, 
		     ".f",
		     text_module(new_module,stat));
  close_prettyprint();

  source_file = db_build_file_resource_name(DBR_SOURCE_FILE, new_module_name, ".f");

  pips_debug(5, "Source file : [%s]\n", source_file);

  DB_PUT_NEW_FILE_RESOURCE (DBR_USER_FILE, new_module_name, source_file);
  DB_PUT_NEW_FILE_RESOURCE (DBR_INITIAL_FILE, new_module_name, source_file);

  return new_module;
}

/*********************************************************
 * Phase main for PHRASE_DISTRIBUTOR_INIT
 *********************************************************/

bool phrase_distributor_init(string module_name)
{
  entity module;
  
  /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      TRUE);
  
  module = local_name_to_top_level_entity(module_name);
  
  set_current_module_statement(stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  
  debug_on("PHRASE_DISTRIBUTOR_DEBUG_LEVEL");

  /* Now do the job */

  pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR_INIT\n");
  prepare_distribute (stat, module, module_name);
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
  
  return TRUE;
}

/*********************************************************
 * Phase main for PHRASE_DISTRIBUTOR
 *********************************************************/

static entity dynamic_area = entity_undefined;

bool phrase_distributor(string module_name)
{
  statement module_stat;
  entity module;
  list l_priv = NIL, l_in, l_out, l_write; 
  
  /* set and get the current properties concerning regions */
  set_bool_property("MUST_REGIONS", TRUE);
  set_bool_property("EXACT_REGIONS", TRUE);
  get_regions_properties();
  
  /* get the resources */
  module_stat = (statement) db_get_memory_resource(DBR_CODE, 
						   module_name, 
						   TRUE);
  
  module = local_name_to_top_level_entity(module_name);
  
  set_current_module_statement(module_stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
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
	db_get_memory_resource(DBR_REGIONS, module_name, TRUE));
    set_in_effects((statement_effects) 
	db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE));
    set_out_effects((statement_effects) 
	db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE));
   
    /*l_write = regions_dup
	(regions_write_regions(load_statement_local_regions(module_stat))); 
    l_in = regions_dup(load_statement_in_regions(module_stat));
    l_out = regions_dup(load_statement_out_regions(module_stat));

    ifdebug(2)
    {
	pips_debug(3, "WRITE regions: \n");
	print_regions(l_write);
	pips_debug(3, "IN regions: \n");
	print_regions(l_in);
	pips_debug(3, "OUT regions: \n");
	print_regions(l_out);
    }

    
    l_priv = RegionsEntitiesInfDifference(l_write, l_in, w_r_combinable_p);
    l_priv = RegionsEntitiesInfDifference(l_priv, l_out, w_w_combinable_p);

    ifdebug(2)
    {
	pips_debug(3, "Private regions: \n");
	print_regions(l_priv);
    }
    */

  /* Now do the job */

  pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR\n");
  distribute (module_stat, module, module_name);
  pips_debug(2, "END of PHRASE_DISTRIBUTOR\n");

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
  
  debug_off();
  
  return TRUE;
}
