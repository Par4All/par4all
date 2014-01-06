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

#ifdef lint
static char vcid[] = "$Id$";
#endif /* lint */


#include "safescale.h"
#include "preprocessor.h"


/**
   Adding in the statement containing comments in the list of statements
 */
static void check_if_statement_contains_comment(statement s, void* a_context)
{
  statement_checking_context* context = (statement_checking_context*)a_context;
  string comments;

  if (!statement_with_empty_comment_p(s)) 
  {    
    comments = strdup(statement_comments(s));
    
    if (strstr(comments, context->searched_string) != NULL) 
    {
      context->list_of_statements = CONS(STATEMENT, s, context->list_of_statements);
    }

    free(comments);
  } 
}


/**
   Building a list with statements containing comments
 */
static list get_statements_with_comments_containing(string comment_portion, statement stat)
{
  string percent;
  statement_checking_context context;

  /* Set searched string */
  context.searched_string = strdup(comment_portion);
  percent = strstr(context.searched_string, "%s");

  if (percent == NULL)
    pips_user_error("Malformed statement comment to search. Should be of the form 'BLAH_%%s'\n");

  *percent = '\0';

  /* Reset and get list of statements */
  context.list_of_statements = NIL;

  ifdebug(5) {
    pips_debug(5, "Searching statements with comments: %s\n", context.searched_string);      
    pips_debug(5, "In statement:\n");      
    print_statement(stat);
  }

  gen_context_recurse(stat, &context, statement_domain, gen_true, check_if_statement_contains_comment);
  
  free(context.searched_string);

  return context.list_of_statements;
}


/**
   Return the identified function name of the externalized portion of code by searching comment matching tags EXTERNALIZED_CODE_PRAGMA_ANALYZED_TOP
   Set the number of statements of this externalizable statement
 */
static string get_externalized_and_analyzed_function_name(statement stat, int *stats_nb) 
{
  string comments;
  string searched_string;
  string comment_portion = strdup(EXTERNALIZED_CODE_PRAGMA_ANALYZED_TOP);
  char *function_name = NULL;
  char *next_line;
  instruction i = statement_instruction(stat);

  pips_debug(7, "Statement to be processed: ");
  print_statement(stat);
  
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
	  pips_debug(5, "Scanning: [%s] with [%s]", first_occurence, comment_portion);
	  sscanf (first_occurence, comment_portion, function_name, stats_nb);
	  pips_debug(5, "Found function: [%s] and %d stats \n", function_name, *stats_nb);
	}

	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }

    free(searched_string);
    free(comments);
  }
  
  free(comment_portion);

  return function_name;
}


/**
   Get sequence containing a searched context
 */
static void search_sequence_containing(statement s, void* a_context)
{
  sequence_searching_context* context = (sequence_searching_context*)a_context;
  instruction i = statement_instruction(s);
  
  if (instruction_tag(i) == is_instruction_sequence) 
  {
    MAP (STATEMENT, s2, {
                          if (s2 == context->searched_statement) 
                          {
	                    context->found_sequence_statement = s;
                          }
    }, sequence_statements(instruction_sequence(i)));
  }
}


/**
   Get statement containing a searched statement
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
   Compute union of exact regions
 */
static list compute_regions_union(list l_in, list l_out)
{
  list l_union = gen_copy_seq(l_in);

  pips_debug(4, "BEGIN of computing regions UNION\n");

  MAP(REGION, reg, {
                     entity e = region_entity(reg);
                     bool is_already_present = false;
                     region reg_already_present = NULL;

                     MAP (REGION, union_reg, {
                                               entity e2 = region_entity(union_reg);

                                               if (same_entity_p(e, e2)) 
                                               {
            	                                 is_already_present = true;
	                                         reg_already_present = union_reg;
                                               }
                                             }, l_union);

                    if (is_already_present) 
                    {
                      if (region_scalar_p(reg)) 
                      {
	                pips_debug(6, "Found SCALAR region already present [%s]. Ignored.\n", entity_local_name(e));
                      }
                      else 
                      {
	                list new_regions;

	                pips_debug(6, "Found ARRAY region already present [%s].\n", entity_local_name(e));
            	        pips_debug(6, "Making UNION of:\n");

	                print_region(reg);
	                pips_debug(6, "and:\n");
	                print_region(reg_already_present);	
	                new_regions = region_must_union(reg,reg_already_present);

	                pips_debug(6, "Getting:\n");

	                print_regions(new_regions);

	                if (gen_length(new_regions) > 1) 
                        {
	                  pips_internal_error("Regions union must refer to only ONE region !");
	                }
	                else 
                        {
	                  gen_remove(&l_union, reg_already_present);
	                  l_union = CONS(REGION, REGION(gen_nth(0, new_regions)), l_union);
	                }
                      }
                    }
                    else 
                    {
                      pips_debug(6, "Adding region for [%s]\n", entity_local_name(e));
                      l_union = CONS(REGION, reg, l_union);
                    }
                  }, l_out);

  pips_debug(4, "END of computing regions UNION\n");

  return l_union;
}


/**
   Compute internal distribution context for statement externalized_code
 */
static bool internal_compute_distribution_context(statement externalized_code, hash_table* ht_params, hash_table* ht_private, hash_table* ht_in_regions, hash_table* ht_out_regions, void* key_value) 
{
  bool returned_value = true;
  list l_read, l_write, l_in, l_out;
  list l_params = NIL;
  list l_priv = NIL;

  pips_debug(6, "Compute regions\n");
  
  l_write = regions_dup(regions_write_regions(load_statement_local_regions(externalized_code))); 
  l_read = regions_dup(regions_read_regions(load_statement_local_regions(externalized_code))); 
  l_in = regions_dup(load_statement_in_regions(externalized_code));
  l_out = regions_dup(load_statement_out_regions(externalized_code));
  
  ifdebug(6) 
  {
    pips_debug(6, "READ regions: \n");
    print_regions(l_read);
    pips_debug(6, "WRITE regions: \n");
    print_regions(l_write);
  }
  
  l_params = compute_regions_union(l_in, l_out);    
  l_in = regions_dup(load_statement_in_regions(externalized_code));
  l_out = regions_dup(load_statement_out_regions(externalized_code));
  l_priv = RegionsEntitiesInfDifference(l_write, l_in, w_r_combinable_p);
  l_priv = RegionsEntitiesInfDifference(l_priv, l_out, w_w_combinable_p);    
  l_in = regions_dup(load_statement_in_regions(externalized_code));
  l_out = regions_dup(load_statement_out_regions(externalized_code));

  gen_sort_list(l_in, (gen_cmp_func_t)compare_effect_reference);
  gen_sort_list(l_out, (gen_cmp_func_t)compare_effect_reference);
  gen_sort_list(l_params, (gen_cmp_func_t)compare_effect_reference);
  gen_sort_list(l_priv, (gen_cmp_func_t)compare_effect_reference);

  ifdebug(2)
  {
    pips_debug(2, "IN regions: \n");
    print_regions(l_in);
    pips_debug(2, "OUT regions: \n");
    print_regions(l_out);
    pips_debug(2, "Params regions: \n");
    print_regions(l_params);
    pips_debug(2, "Private regions: \n");
    print_regions(l_priv);
  }

  /* Store results in hash_tables */
  pips_debug(2, "Storing in hash_tables with key %s: \n", (string)key_value);
  pips_debug(5, "Storing in ht_param: \n");

  if (!hash_defined_p(*ht_params, key_value)) 
  {
    hash_put(*ht_params, key_value, l_params);
  }
  else 
  {
    pips_user_warning("Multiply defined value in PARAMS hash_table!\n");    
    returned_value = false;
  }
  
  pips_debug(5, "Storing in ht_private: \n");

  if (!hash_defined_p(*ht_private, key_value)) 
  {
    hash_put(*ht_private, key_value, l_priv);
  }
  else 
  {
    pips_user_warning("Multiply defined value in PRIVATE hash_table!\n");    
    returned_value = false;
  }
  
  pips_debug(5, "Storing in ht_in_regions: \n");

  if (!hash_defined_p(*ht_in_regions, key_value)) 
  {
    hash_put(*ht_in_regions, key_value, l_in);
  }
  else 
  {
    pips_user_warning("Multiply defined value in IN_REGIONS hash_table!\n");    
    returned_value = false;
  }
  
  pips_debug(5, "Storing in ht_out_regions: \n");

  if (!hash_defined_p(*ht_out_regions, key_value)) 
  {
    hash_put(*ht_out_regions, key_value, l_out);
  }
  else 
  {
    pips_user_warning("Multiply defined value in OUT_REGIONS hash_table!\n");    
    returned_value = false;
  }

  return returned_value;
}


/**
   Compute distribution context for statement externalized_code
 */
static bool compute_distribution_context(list l_stats, statement module_stat, entity module, hash_table* ht_stats, hash_table* ht_params, hash_table* ht_private, hash_table* ht_in_regions, hash_table* ht_out_regions)
{
  bool returned_value = true;

  pips_debug(5, "[BEGIN] compute_distribution_context for %s: \n", entity_local_name(module));

  /* Init hashtables */
  *ht_stats = hash_table_make(hash_pointer, 0);
  *ht_params = hash_table_make(hash_pointer, 0);
  *ht_private = hash_table_make(hash_pointer, 0);
  *ht_in_regions = hash_table_make(hash_pointer, 0);
  *ht_out_regions = hash_table_make(hash_pointer, 0);

  MAP(STATEMENT, s, {
                      statement externalized_code = NULL;
                      int stats_nb;
                      string function_name = get_externalized_and_analyzed_function_name(s, &stats_nb);

                      pips_debug(5, "Funtion name: %s\n", function_name);
                      pips_debug(5, "Number of statements: %d\n", stats_nb);

                      if (stats_nb > 1)
                      {
                        externalized_code = sequence_statement_containing(module_stat, s);
                      }
                      else if (stats_nb == 1) 
                      {
                        externalized_code = s;
                      }
                      else 
                      {
                        pips_internal_error("Strange externalized code!");
                      }

                      /* Register new externalized function */
                      pips_debug(5, "Register externalized function %s: \n", function_name);

                      if (!hash_defined_p(*ht_stats, function_name)) 
                      {
                        hash_put(*ht_stats, function_name, externalized_code);
                      }
                      else 
                      {
                        pips_user_warning("Multiply defined value in STATS hash_table!\n");    
                        returned_value = false;
                      }

                      pips_debug(3, "ANALYSING function named [%s]..................\n", function_name);

                      if (!internal_compute_distribution_context(externalized_code, ht_params, ht_private, ht_in_regions, ht_out_regions, function_name)) 
                      {
                        returned_value = false;
                      }
                    }, l_stats);

  pips_debug(5, "[END] compute_distribution_context for %s: \n", entity_local_name(module));

  return returned_value;
}


/**
   Create a variable declared as a parameter in specified module
 */
static entity create_parameter_variable_for_new_module(entity a_variable, string new_name, string new_module_name, entity module, int param_nb)
{
  entity new_variable;

  /* Test if entity exists and create it if not */
  if ((gen_find_tabulated(concatenate(new_module_name, MODULE_SEP_STRING, new_name, NULL), entity_domain)) == entity_undefined) 
  { 
    new_variable = make_entity(strdup(concatenate(new_module_name, MODULE_SEP_STRING, new_name, NULL)), copy_type(entity_type(a_variable)), make_storage_formal(make_formal(module, param_nb)), copy_value(entity_initial(a_variable)));

    return new_variable;
  }
  else 
  {
    pips_internal_error("Entity already exist: %s", new_name);
  
    return NULL;
  }
}


/**
   Create a private variable in specified module
 */
static entity create_private_variable_for_new_module(entity a_variable, string new_name, string new_module_name, entity module)
{
  entity new_variable;
  entity a;
  basic base;
  
  /* Test if entity exists and create it if not */
  if ((gen_find_tabulated(concatenate(new_module_name, MODULE_SEP_STRING, new_name, NULL), entity_domain)) == entity_undefined) 
  { 
    new_variable = make_entity(strdup(concatenate(new_module_name, MODULE_SEP_STRING, new_name, NULL)), copy_type(entity_type(a_variable)), storage_undefined, copy_value(entity_initial(a_variable)));

    a = FindEntity(new_module_name, DYNAMIC_AREA_LOCAL_NAME); 
    base = variable_basic(type_variable(entity_type(a_variable)));
    entity_storage(new_variable) = make_storage(is_storage_ram, make_ram(module, a, (basic_tag(base) != is_basic_overloaded) ? (add_variable_to_area(a, new_variable)):(0), NIL));

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
   Return EXTERNALIZED_FUNCTION_PARAM_NAME
 */
static string get_externalized_function_param_name(entity variable, int param_nb) 
{
  char buffer[256];

  sprintf(buffer, EXTERNALIZED_FUNCTION_PARAM_NAME, entity_user_name(variable), param_nb);

  return strdup(buffer);
}


/**
   Return EXTERNALIZED_FUNCTION_PRIVATE_PARAM_NAME
 */
static string get_externalized_function_private_param_name(entity variable) 
{
  char buffer[256];

  sprintf(buffer, EXTERNALIZED_FUNCTION_PRIVATE_PARAM_NAME, entity_user_name(variable));

  return strdup(buffer);
}



/**
   Declare in the newly created module a new variable and replace all occurences to the old variable by the new created
 */
static void add_parameter_variable_to_module(reference ref, entity module, statement stat, string new_module_name, int param_nb)
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
  variable_name = get_externalized_function_param_name(reference_variable(ref), param_nb); 

  new_variable = create_parameter_variable_for_new_module(reference_variable(ref), variable_name, new_module_name, module, param_nb);  
  replace_reference(stat, ref, new_variable);  
  module_declarations = code_declarations(value_code(entity_initial(module)));  
  code_declarations(value_code(entity_initial(module))) = CONS(ENTITY, new_variable, module_declarations);
  
  new_parameter = make_parameter(entity_type(new_variable), make_mode(is_mode_reference, UU), make_dummy_unknown()); //FI: could be make_dummy_identifier(new_variable)
  module_parameters = module_functional_parameters(module);
  module_functional_parameters(module) = CONS(PARAMETER, new_parameter, module_parameters);
}


/**
   Declare in the newly created module a new variable and replace all occurences to the old variable by the new created
 */
static void add_private_variable_to_module(reference ref, entity module, statement stat, string new_module_name)
{
  list module_declarations;
  entity new_variable;
  string variable_name;
  
  pips_debug(2, "Registering private variable: %s\n", entity_local_name(reference_variable(ref)));
  
  /* Assert that entity represent a value code */
  pips_assert("It is a module", entity_module_p(module));
  
  /* Get the variable name */
  variable_name = get_externalized_function_private_param_name(reference_variable(ref)); 

  /* Creates the variable */
  new_variable = create_private_variable_for_new_module(reference_variable(ref), variable_name, new_module_name, module);  
  
  replace_reference(stat, ref, new_variable);    
  module_declarations = code_declarations(value_code(entity_initial(module)));
  code_declarations(value_code(entity_initial(module))) = CONS(ENTITY, new_variable, module_declarations);
  
}


/**
   Return a list of references corresponding to a list of regions
 */
static list references_for_regions(list l_regions)
{
  list l_ref = NIL;
  
  MAP(EFFECT, reg, {
                     reference ref = effect_any_reference(reg);

                     l_ref = CONS(REFERENCE, ref, l_ref);
                     print_reference(ref);
                     pips_debug(4,"Entity: %s\n", entity_local_name(reference_variable(ref)));
                   }, l_regions);

  return l_ref;
}


/**
   Dynamically build a new module with specified statement
 */
static entity create_module_with_statement(statement stat, string new_module_name, list l_params, list l_priv) 
{
  entity new_module;
  //string source_file;
  //text text_code;
  int param_nb = 0;

  pips_debug(5, "[BEGIN] create_module_with_statement\n");
  pips_debug(2, "Creating new module: [%s]\n", new_module_name);

  new_module = make_empty_subroutine(new_module_name,copy_language(module_language(get_current_module_entity())));
  
  /* Deal with private variables */
  MAP(REFERENCE, ref, {
                        add_private_variable_to_module(ref, new_module, stat, new_module_name);
                      }, references_for_regions(l_priv));
  
  /* Deal with parameter variables */ 
  param_nb = gen_length(l_params);

  MAP(REFERENCE, ref, {
                        add_parameter_variable_to_module(ref, new_module, stat, new_module_name, param_nb);
                        param_nb--;
                      }, references_for_regions(l_params));
  
  pips_debug(2, "Making new module: [%s]\n", new_module_name);

  ifdebug(5)
  {
    pips_debug(5, "With statement: \n");
    print_statement(stat);
  }
  
  add_new_module(new_module_name, new_module, stat,
		 prettyprint_language_is_fortran_p ());
  
  pips_debug(5, "[END] create_module_with_statement\n");

  return new_module;
}


/**
   Replace statement old_stat by statement new_stat and assert that this statement is contained in a sequence
 */
static void replace_in_sequence_statement_with(statement old_stat, statement new_stat, statement root_stat) 
{
  statement sequence_statement = sequence_statement_containing(root_stat, old_stat);
  list stats_list = sequence_statements(instruction_sequence(statement_instruction(sequence_statement))); 
  list new_stats_list = NIL;

  pips_debug(5, "BEGIN replace_in_sequence_statement_with:\n");
  pips_assert("Statement is contained in a sequence", sequence_statement != NULL);

  MAP(STATEMENT, s, {
                      pips_debug(7, "Iterate on statement:\n");

                      print_statement(s);    
                     
                      if (s == old_stat) 
                      {
                        pips_debug(7, "Replace this statement:\n");

                        new_stats_list = CONS(STATEMENT, new_stat, new_stats_list);
                      }
                    else 
                    {
                      pips_debug(7, "Keep this statement:\n");

                      new_stats_list = CONS(STATEMENT, s, new_stats_list);
                    }
                  }, stats_list);

  sequence_statements(instruction_sequence(statement_instruction(sequence_statement))) = gen_nreverse(new_stats_list);

  ifdebug(7) 
  {
    pips_debug(7, "I've got this for the sequence\n");
    print_statement(sequence_statement);    
    pips_debug(7, "I've got this for the root statement\n");
    print_statement(root_stat);    
  }

  pips_debug(5, "END replace_in_sequence_statement_with:\n");
}


/**
   Externalize a specified code portion
 */
static void distribute_code(string function_name, statement externalized_code, statement module_stat, list l_params, list l_priv)  
{
  entity new_module = create_module_with_statement(externalized_code, function_name, l_params, l_priv);
  statement call_statement;
  list call_params = NIL;
  string call_comments;

  pips_debug(5, "[BEGIN] distribute_code\n");

  MAP(REFERENCE, ref, {
                        call_params = CONS(EXPRESSION, make_entity_expression(reference_variable(ref), NIL), call_params);
                      }, references_for_regions(l_params));
  
  /* Insert an analyzed tag */ 
  {
    char* new_tag = malloc(256);

    sprintf(new_tag, strdup(concatenate("\n! ", EXTERNALIZED_CODE_PRAGMA_CALL, "\n", NULL)), function_name);
    call_comments = strdup(new_tag);
  }

  call_statement = make_statement(entity_empty_label(),
				  statement_number(externalized_code),
				  statement_ordering(externalized_code),
				  call_comments,
				  make_instruction(is_instruction_call,
						   make_call(new_module,
							     call_params)),
				  NIL,
				  NULL,
				  statement_extensions (externalized_code), make_synchronization_none());
  
  pips_debug(7, "BEFORE REPLACING\n");
  pips_debug(7, "externalized_code=\n");
  print_statement(externalized_code);
  pips_debug(7, "call_statement=\n");
  print_statement(call_statement);
  pips_debug(7, "module_stat=\n");
  print_statement(module_stat);

  replace_in_sequence_statement_with(externalized_code, call_statement, module_stat);

  pips_debug(7, "AFTER REPLACING\n");
  pips_debug(7, "externalized_code=\n");
  print_statement(externalized_code);
  pips_debug(7, "call_statement=\n");
  print_statement(call_statement);
  pips_debug(7, "module_stat=\n");
  print_statement(module_stat);

  pips_assert("Module structure is consistent after DISTRIBUTE_CODE", gen_consistent_p((gen_chunk*) new_module));	      
  pips_assert("Statement structure is consistent after DISTRIBUTE_CODE", gen_consistent_p((gen_chunk*) externalized_code));	      
  pips_assert("Statement is consistent after DISTRIBUTE_CODE", statement_consistent_p(externalized_code));
  
  pips_debug(7, "Code distribution for : [%s] is DONE\n", function_name);
  pips_debug(5, "[END] distribute_code\n");

  free(call_comments);
}


/**
   Distribute for main module module with root statement stat
 */
static void distribute(statement module_stat, entity module) 
{
  list l_stats = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_ANALYZED_TOP, module_stat);
  hash_table ht_stats;
  hash_table ht_params;
  hash_table ht_private;
  hash_table ht_in_regions;
  hash_table ht_out_regions;

  pips_debug(5, "[BEGIN] distribute\n");
  pips_debug(5, "Number of analyzed statements to distribute: %td\n", gen_length(l_stats));

  compute_distribution_context(l_stats,	module_stat, module, &ht_stats, &ht_params, &ht_private, &ht_in_regions, &ht_out_regions);

  HASH_MAP(function_name, stat, {
                                  distribute_code(function_name, stat, module_stat, hash_get(ht_params, function_name), hash_get(ht_private, function_name));
                                }, ht_stats);

  hash_table_free(ht_stats);
  hash_table_free(ht_params);
  hash_table_free(ht_private);
  hash_table_free(ht_in_regions);
  hash_table_free(ht_out_regions);

  pips_debug(5, "[END] distribute\n");
}


/**
   Main phase for block code externalization
 */
static entity dynamic_area = entity_undefined;

bool safescale_distributor(const char* module_name)
{
  statement module_stat;
  entity module;

  /* Set and get the current properties concerning regions */
  set_bool_property("MUST_REGIONS", true);
  set_bool_property("EXACT_REGIONS", true);
  get_regions_properties();
  
  /* Get the resources */
  module_stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  module = module_name_to_entity(module_name);

  set_current_module_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));

  set_cumulated_rw_effects((statement_effects) db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
  module_to_value_mappings(module);
  
  /* Set dynamic_area */
  if (entity_undefined_p(dynamic_area)) 
  {   	
    dynamic_area = FindOrCreateEntity(module_local_name(module), DYNAMIC_AREA_LOCAL_NAME); 
  }

  debug_on("SAFESCALE_DISTRIBUTOR_DEBUG_LEVEL");

  /* Get the READ, WRITE, IN and OUT regions of the module */
  set_rw_effects((statement_effects) db_get_memory_resource(DBR_REGIONS, module_name, true));
  set_in_effects((statement_effects) db_get_memory_resource(DBR_IN_REGIONS, module_name, true));
  set_out_effects((statement_effects) db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));

  /* Do the job */
  pips_debug(2, "BEGIN of SAFESCALE_DISTRIBUTOR\n");
  /* Restructuring code to avoid imbricated sequences if some portions are found to allow more than one INIT */
  simple_restructure_statement(module_stat);
  distribute(module_stat, module);
  pips_debug(2, "END of SAFESCALE_DISTRIBUTOR\n");

  print_statement(module_stat);

  pips_assert("Statement structure is consistent after SAFESCALE_DISTRIBUTOR", gen_consistent_p((gen_chunk*) module_stat));
  pips_assert("Statement is consistent after SAFESCALE_DISTRIBUTOR", statement_consistent_p(module_stat));
  
  /* Reorder the module because new statements have been added */  
  module_reorder(module_stat);  

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(module_stat));

  /* Update/release resources */
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
