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
#include "prettyprint.h"

#include "dg.h"

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

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "transformer.h"

#include "phrase_distribution.h"

/**
 * Create new variable parameter for a newly created module
 */
entity create_parameter_for_new_module (variable var,
					const char* parameter_name,
					const char* module_name,
					entity module,
					int param_nb)
{
  entity new_variable;
  parameter new_parameter;
  list module_declarations;
  list module_parameters;
 
  if ((gen_find_tabulated(concatenate(module_name,
				      MODULE_SEP_STRING,
				      parameter_name,
				      NULL),
			  entity_domain)) == entity_undefined)
    {
      /* This entity does not exist, we can safely create it */
     
      new_variable = make_entity (strdup(concatenate(module_name,
						     MODULE_SEP_STRING,
						     parameter_name, NULL)),
				  make_type_variable(copy_variable(var)),
				  make_storage_formal (make_formal(module, param_nb)),
				  make_value_unknown());
     
  module_declarations = code_declarations(value_code(entity_initial(module)));
 
  code_declarations(value_code(entity_initial(module)))
    = CONS (ENTITY, new_variable, module_declarations);
 
  new_parameter = make_parameter (entity_type(new_variable),
				  make_mode_reference(),
				  make_dummy_identifier(new_variable)/*SG used to be strdup("")*/);
 
  module_parameters = functional_parameters(type_functional(entity_type(module)));
 
  functional_parameters(type_functional(entity_type(module)))
    = CONS(PARAMETER, new_parameter, module_parameters);
 
  return new_variable;
    }
  else
    {
      pips_internal_error("Entity already exist: %s", parameter_name);
      return NULL;
    }
}

/**
 * Create new integer variable parameter for a newly created module
 */
entity create_integer_parameter_for_new_module (const char* parameter_name,
						const char* module_name,
						entity module,
						int param_nb)
{
  return create_parameter_for_new_module
    (make_variable(MakeBasic(is_basic_int), NIL, NIL),
     parameter_name,
     module_name,
     module,
     param_nb);
}

/**
 * Store (PIPDBM) newly created module module with module_statement
 * as USER_FILE by saving pretty printing
 */
void store_new_module (const char* module_name,
		       entity module,
		       statement module_statement)
{
  string source_file;
  text code;

  pips_debug(2, "[BEGIN] store_new_module [%s]\n", module_name);
  ifdebug(2) {
    entity set_entity = get_current_module_entity();
    reset_current_module_entity();
    set_current_module_entity(module);
    pips_debug(2, "Statement for module: ");
    print_statement(module_statement);
    pips_debug(7, "Declarations for module: \n");
    MAP (ENTITY, e, {
      pips_debug(2, "Declared entity %s\n", entity_global_name(e));
    },code_declarations(value_code(entity_initial(module))));
    fprint_environment(stderr, module);
    reset_current_module_entity();
    set_current_module_entity(set_entity);
  }

  init_prettyprint(empty_text);
  code = text_module(module, module_statement);
  make_text_resource(module_name,
		     DBR_SOURCE_FILE,
		     ".f",
		     code);
  close_prettyprint();
 
  source_file = db_build_file_resource_name(DBR_SOURCE_FILE, module_name, ".f");
 
  pips_debug(5, "Source file : [%s]\n", source_file);
 
  DB_PUT_NEW_FILE_RESOURCE (DBR_USER_FILE, module_name, source_file);
  DB_PUT_NEW_FILE_RESOURCE (DBR_INITIAL_FILE, module_name, source_file);

  init_prettyprint(empty_text);
  make_text_resource(module_name,
		     DBR_INITIAL_FILE,
		     ".f_initial",
		     code);
  close_prettyprint();

  pips_debug(2, "[END] store_new_module [%s]\n", module_name);
}
	
/**
 * Creates and declares a new variable for a newly created common
 */
entity create_new_common_variable(string name, entity module, entity common, variable var)
{
  int old_size, variable_size;

  string var_global_name = strdup(concatenate(module_local_name(module),MODULE_SEP_STRING,
					      name,NULL));
  type var_type = make_type(is_type_variable, var);
  storage var_storage = storage_undefined;
  value var_value = make_value_unknown();
  entity e = make_entity(var_global_name,var_type,var_storage,var_value);
  list old_layout = area_layout(type_area(entity_type(common)));
  old_size = area_size(type_area(entity_type(common)));
  variable_size = storage_space_of_variable(e);
  pips_debug(2, "New variable %s created with size %d (old_size was:%d)\n", name, variable_size, old_size);
  entity_storage (e) = make_storage(is_storage_ram,
				    (make_ram(module,common,old_size,NIL)));
  area_layout(type_area(entity_type(common))) = gen_nconc(old_layout,CONS(ENTITY,e,NIL));
  /* gen_nreverse(CONS(ENTITY,e,old_layout)); */
  area_size(type_area(entity_type(common))) = old_size+variable_size;
  AddEntityToDeclarations( e,module);
  return e;
}

/**
 * Creates and declares a new scalar variable for a newly created common
 */
entity create_new_scalar_common_variable(string name, entity module, entity common, basic b)
{
  return create_new_common_variable( name,  module,  common, make_variable(b, NIL,NIL));
}

/**
 * Creates and declares a new integer scalar variable for a newly created common
 */
entity create_new_integer_scalar_common_variable(string name, entity module, entity common)
{
  return create_new_scalar_common_variable (name, module, common, MakeBasic(is_basic_int));
}

/**
 * Creates all the things that need to be created in order to declare common
 * in module (all the variable are created)
 */
void declare_common_variables_in_module (entity common, entity module)
{
  list new_variables = NIL;
  list primary_variables = NIL;
  bool is_primary_area = true;
  int totalized_offset = -1;

  /* Compute the primary variables */
  MAP (ENTITY, v, {
    if (is_primary_area) {
      int offset = ram_offset(storage_ram(entity_storage(v)));
      if (offset > totalized_offset) {
	totalized_offset = offset;
	primary_variables = CONS (ENTITY,v,primary_variables);
      }
      else {
	is_primary_area = false;
      }
    }
  }, area_layout(type_area(entity_type(common))));
 
  primary_variables = gen_nreverse(primary_variables);

  ifdebug(4) {
    pips_debug(4, "Current layout for %s\n", entity_global_name(common));    
    MAP(ENTITY, v, {
      pips_debug(4, "[%s] offset %"PRIdPTR"\n", entity_global_name(v),ram_offset(storage_ram(entity_storage(v))));    
   }, area_layout(type_area(entity_type(common))));
    pips_debug(4, "Primary variables for %s\n", entity_global_name(common));    
    MAP(ENTITY, v, {
      pips_debug(4, "[%s] offset %"PRIdPTR" PRIMARY\n", entity_global_name(v),ram_offset(storage_ram(entity_storage(v))));    
   }, primary_variables);
  }
 
  MAP (ENTITY, v, {

    /* We iterate on the primary variables declared in the common and
       create a new variable mapping the one declared in common */

    entity new_variable; 
    const char* name = entity_local_name(v);
    int v_offset = ram_offset(storage_ram(entity_storage(v)));
   
    /* Creates the name for the new variable */
    string var_global_name = strdup(concatenate(module_local_name(module),
						MODULE_SEP_STRING,
						name,NULL));

    /* Copy type of variable */
    type var_type = copy_type(entity_type(v));

    /* Create storage for new variable */
    storage var_storage = make_storage(is_storage_ram,
				       (make_ram(module,
						 common,
						 v_offset,
						 NIL)));
    /* Copy initial value of variable */
    value var_value = copy_value(entity_initial(v));

    pips_debug(7, "Build variable %s\n", var_global_name);

    /* Build the new variable */
    new_variable = make_entity(var_global_name,var_type,var_storage,var_value);

    /* Mark for addition */
    new_variables = gen_nconc(new_variables,CONS(ENTITY,new_variable,NIL));

    pips_debug(7, "Add to declarations %s\n", var_global_name);

    /* Add to declarations.... */
    AddEntityToDeclarations( new_variable,module);
   
    pips_debug(7, "New common variable %s declared\n", var_global_name);
   
  }, primary_variables);

  /* Add those new variable to common layout */
  {
    list old_layout = area_layout(type_area(entity_type(common)));
    area_layout(type_area(entity_type(common))) = gen_nconc(old_layout,new_variables);
  }

  AddEntityToDeclarations( common,module);
  pips_debug(3, "Common %s declared in module %s\n",
	     entity_local_name(common),
	     entity_local_name(module));
}

/*
 * Return CONTROLIZED_STATEMENT_COMMENT
 */
string get_controlized_statement_comment (entity function)
{
  char *buffer;
  asprintf(&buffer,
	  CONTROLIZED_STATEMENT_COMMENT,
	  entity_local_name(function));
  return (buffer);
}

/*
 * Return IN_PARAM_ID_NAME
 */
string get_in_param_id_name (entity variable, entity function)
{
  char *buffer;
  asprintf(&buffer,
	  IN_PARAM_ID_NAME,
	  entity_local_name(variable),
	  entity_local_name(function));
  return (buffer);
}

/*
 * Return OUT_PARAM_ID_NAME
 */
string get_out_param_id_name (entity variable, entity function)
{
  char *buffer;
  asprintf(&buffer,
	  OUT_PARAM_ID_NAME,
	  entity_local_name(variable),
	  entity_local_name(function));
  return (buffer);
}

/*
 * Return FUNCTION_ID_NAME
 */
string get_function_id_name (entity function)
{
  char *buffer;
  asprintf(&buffer,
	  FUNCTION_ID_NAME,
	  entity_local_name(function));
  return strdup(buffer);
}

/*
 * Return FUNCTION_COMMON_NAME
 */
static string get_function_common_name (entity function)
{
  char *buffer;
  asprintf(&buffer,
	  FUNCTION_COMMON_NAME,
	  entity_local_name(function));
  return strdup(buffer);
}

/*
 * Return COMMON_PARAM_NAME
 */
string get_common_param_name (entity variable, entity function)
{
  char *buffer;
  asprintf(&buffer,
	  COMMON_PARAM_NAME,
	  entity_local_name(variable),
	  entity_local_name(function));
  return strdup(buffer);
}

/*
 * Return UNIT_ID_NAME
 */
static string get_unit_id_name (int unit)
{
  char *buffer;
  asprintf(&buffer,UNIT_ID_NAME,unit);
  return strdup(buffer);
}

/*
 * Return SEND_PARAMETER_MODULE_NAME
 */
string get_send_parameter_module_name (variable var)
{
  char *buffer;
  asprintf(&buffer,SEND_PARAMETER_MODULE_NAME,variable_to_string(var));
  return strdup(buffer);
}

/*
 * Return RECEIVE_PARAMETER_MODULE_NAME
 */
string get_receive_parameter_module_name (variable var)
{
  char *buffer;
  asprintf(&buffer,RECEIVE_PARAMETER_MODULE_NAME,variable_to_string(var));
  return strdup(buffer);
}

/**
 * Return entity named name in specified module
 */
entity entity_in_module (const char* name, entity module)
{
  /* Is it the main module ? */
  if (strchr(entity_local_name(module),'%') != NULL) {
    return FindEntity(entity_local_name(module)+1,name);
  }
  else {
    return FindEntity(entity_local_name(module),name);
  }

}

/**
 * Build and return CONTROL_DATA global common used to store global
 * information on phrase distribution controlization and initialization of
 * values contained in CONTROL_DATA common
 */
static statement
make_global_common_and_initialize (entity main_module,
				   statement module_stat,
				   entity *global_common,
				   int number_of_deployment_units,
				   hash_table ht_calls,
				   hash_table ht_in_regions,
				   hash_table ht_out_regions)
{
  list stats_list = CONS(STATEMENT,module_stat,NIL);
  statement new_stat;
  sequence new_sequence;
  int i;
  list l_in, l_out;
  const char* function_name;
  int id_function, id_param;
  entity units_nb_variable;
  entity functions_nb_variable;
 
  *global_common = make_new_common(CONTROL_DATA_COMMON_NAME,
				     main_module);
  units_nb_variable = create_new_integer_scalar_common_variable(UNITS_NB_NAME,
								main_module,
								*global_common);
  new_stat = make_assignement_statement
    (units_nb_variable,
     int_to_expression (number_of_deployment_units), NULL);
  stats_list = CONS (STATEMENT, new_stat,stats_list);

  if (number_of_deployment_units > 1) {
    for (i=1; i<= number_of_deployment_units; i++) {
      entity unit_id_variable =
	create_new_integer_scalar_common_variable(get_unit_id_name(i),
						  main_module,
						  *global_common);
      new_stat = make_assignement_statement
	(unit_id_variable,
	 int_to_expression (i), NULL);
      stats_list = CONS (STATEMENT, new_stat,stats_list);
    }
  }
 
 
  functions_nb_variable =
    create_new_integer_scalar_common_variable(FUNCTIONS_NB_NAME,
					      main_module,
					      *global_common);

  new_stat = make_assignement_statement
    (functions_nb_variable,
     int_to_expression (hash_table_entry_count(ht_calls)), NULL);
  stats_list = CONS (STATEMENT, new_stat,stats_list);
 
  id_function = 1;

  HASH_MAP (externalized_function, l_calls_for_function, {
   
    entity function = (entity)externalized_function;
    entity function_id_variable;
    function_name = entity_local_name(function);
    pips_debug(2, "Function [%s]\n",function_name);
     
    function_id_variable =
    create_new_integer_scalar_common_variable
    (get_function_id_name(function),
     main_module,
     *global_common);
    new_stat = make_assignement_statement
    (function_id_variable,
     int_to_expression (id_function++), NULL);
    stats_list = CONS (STATEMENT, new_stat,stats_list);
     
    id_param = 1;
   
    l_in = (list)hash_get(ht_in_regions,function);
    l_out = (list)hash_get(ht_out_regions,function);

    /*l_in = regions_dup(load_statement_in_regions(s));
      l_out = regions_dup(load_statement_out_regions(s));*/
     
    MAP (REGION, reg, {
      reference ref = effect_any_reference(reg);
      entity variable = reference_variable(ref);
      entity param_id_variable =
	create_new_integer_scalar_common_variable
	(get_in_param_id_name(variable,function),
	 main_module,
	 *global_common);
      new_stat = make_assignement_statement
	(param_id_variable,
	 int_to_expression (id_param++), NULL);
      stats_list = CONS (STATEMENT, new_stat,stats_list);
    }, l_in);
   
    MAP (REGION, reg, {
      reference ref = effect_any_reference(reg);
      entity variable = reference_variable(ref);
      entity param_id_variable =
	create_new_integer_scalar_common_variable
	(get_out_param_id_name(variable,function),
	 main_module,
	 *global_common);
      new_stat = make_assignement_statement
	(param_id_variable,
	 int_to_expression (id_param++), NULL);
      stats_list = CONS (STATEMENT, new_stat,stats_list);
    }, l_out);
   
  }, ht_calls);
 
  new_sequence = make_sequence (stats_list);
 
  return make_statement (entity_empty_label(),
			 STATEMENT_NUMBER_UNDEFINED,
			 STATEMENT_ORDERING_UNDEFINED,
			 empty_comments,
			 make_instruction (is_instruction_sequence,
					   new_sequence),
			 NIL,NULL,
			 empty_extensions (), make_synchronization_none());
 
}
	
/**
 * Creates and returns a common used to store variable for communications
 * between control code and externalized code
 */
static entity create_externalized_function_common (entity main_module,
						   entity externalized_function,
						   list params_regions,
						   int number_of_deployment_units)
{
  entity returned_common;
 
  returned_common
    = make_new_common(get_function_common_name (externalized_function),
			main_module);
 
  /* Creates params variables */
  MAP (REGION, reg, {
    entity in_variable = region_entity(reg);
    variable var = copy_variable(type_variable(entity_type(in_variable)));
    /* If many deployment units, add dimension to handle those different units */
    if (number_of_deployment_units > 1) {
      list l_dimensions = variable_dimensions(var);
      variable_dimensions(var)
	= gen_nconc(l_dimensions,
		    CONS(DIMENSION,
			 make_dimension (int_to_expression(1),int_to_expression (number_of_deployment_units)),
			 NIL));
    }
    create_new_common_variable(get_common_param_name(in_variable,externalized_function),
			       main_module,
			       returned_common,
			       var);
  }, params_regions);
 
  return returned_common;
}

			   
/**
 * Main function for PHRASE_DISTRIBUTION_CONTROL_CODE
 */
static statement controlize_distribution (statement module_stat,
					  entity module)
{
  statement returned_statement = module_stat;
  list l_calls;
  const char* function_name;

  entity start_ru_module;
  entity wait_ru_module;
  statement start_ru_module_statement;
  statement wait_ru_module_statement;

  entity global_common;
  entity externalized_fonction_common;
  list l_commons = NIL;

  hash_table ht_calls;
  hash_table ht_params;
  hash_table ht_private;
  hash_table ht_in_regions;
  hash_table ht_out_regions;
   
  hash_table ht_in_communications = hash_table_make(hash_pointer, 0);
  hash_table ht_out_communications = hash_table_make(hash_pointer, 0);

  int number_of_deployment_units;
  int unit_id;

  /* We identify all the statements containing an externalized-call tag */
  l_calls = get_statements_with_comments_containing(EXTERNALIZED_CODE_PRAGMA_CALL,
						    module_stat);
 
  /* Compute the context of distribution, before to controlize */
  compute_distribution_controlization_context (l_calls,
					       module_stat,
					       module,
					       &ht_calls,
					       &ht_params,
					       &ht_private,
					       &ht_in_regions,
					       &ht_out_regions);
 

  if (hash_table_entry_count (ht_calls)) {
     
    /* OK, be begin deployement... */
   
    string resp;
   
    /* Get the loop label form the user */
    resp = user_request("Deployment on how many externalized units ?\n"
			"(give its number > 0): ");   

    if (sscanf(resp,"%d",&number_of_deployment_units) > 0) {
     
      pips_debug(2, "Deployment on %d units\n", number_of_deployment_units);
     
      /* Declare global common for controlization and make initializations */
      returned_statement
	= make_global_common_and_initialize (module,
					     module_stat,
					     &global_common,
					     number_of_deployment_units,
					     ht_calls,
					     ht_in_regions,
					     ht_out_regions);
     
      /* Let's begin to iterate on all externalized functions,
       * in order to build commons and controlization modules */
     
      pips_debug(2, "Found %d externalized functions\n", hash_table_entry_count(ht_calls));

      HASH_MAP (externalized_function, calls_for_f, {
	
	entity f = (entity)externalized_function;
	
	/* Get the function name */
	function_name = entity_local_name(f);

	pips_debug(2, "Found %zd calls for externalized function %s\n",
		   gen_length((list)calls_for_f),
		   function_name);

	/* Creates a common used to store variable for communications
	 * between control code and externalized code */
	externalized_fonction_common =
	create_externalized_function_common (module,
					     f,
					     (list)hash_get(ht_params,f),
					     number_of_deployment_units);
	
	/* And register it to the list of commons */
	l_commons = CONS(ENTITY, externalized_fonction_common, l_commons);
	
	pips_debug(2, "Register input communications\n");
	register_scalar_communications (&ht_in_communications,
					f,
					(list)hash_get(ht_in_regions,f));
	
	pips_debug(2, "Register output communications\n");
	register_scalar_communications (&ht_out_communications,
					f,
					(list)hash_get(ht_out_regions,f));
	
	pips_debug(2, "Register and create ARRAY input communications\n");
	make_send_array_params_modules (f,
					(list)hash_get(ht_in_regions,f),
					global_common,
					externalized_fonction_common,
					number_of_deployment_units);

	pips_debug(2, "Register and create ARRAY output communications\n");
	make_receive_array_params_modules (f,
					   (list)hash_get(ht_out_regions,f),
					   global_common,
					   externalized_fonction_common,
					   number_of_deployment_units);
	
      }, ht_calls);
     
      /* Build START_RU module */
      start_ru_module = make_start_ru_module (ht_params,
					      &start_ru_module_statement,
					      number_of_deployment_units,
					      global_common,
					      l_commons);
     
      /* Build WAIT_RU module */
      wait_ru_module = make_wait_ru_module (&wait_ru_module_statement,
					    number_of_deployment_units,
					    global_common,
					    l_commons);
     
#if 0
      /* Build SEND_PARAMS modules */
      list send_params_modules = make_send_scalar_params_modules (ht_in_communications,
							     number_of_deployment_units,
							     global_common,
							     l_commons);
     
      /* Build RECEIVE_PARAMS modules */
      list receive_params_modules = make_receive_scalar_params_modules (ht_out_communications,
								   number_of_deployment_units,
								   global_common,
								   l_commons);
#endif

      /* Let's begin to iterate on all externalized functions and statements,
       * in order to add controlization code  */
     
      HASH_MAP (externalized_function, calls_for_f, {
	
	entity f = (entity)externalized_function;
	list l_stats = NIL;
	entity func_id_variable;
	entity unit_id_variable = NULL;
	
	/* Get the function name */
	function_name = entity_local_name(f);
	
	/* Retrieve variables func_id_variable and unit_id_variable */
	func_id_variable = entity_in_module(get_function_id_name(f), module);

	/* Iterate on all the statements where externalized function is called */
	
	MAP (STATEMENT, s, {
	 
	  l_stats = NIL;
	  unit_id = 0;
	 
	  if (number_of_deployment_units > 1) {
	    char *question;
	    unit_id = 0;
	    do {
	      asprintf (&question, "Deployment of function %s on which units ?\n(give its number 1-%d):", function_name, number_of_deployment_units);
	      resp = user_request(question); 
	      sscanf(resp,"%d",&unit_id);	   
          free(question);
	    }
	    while ((unit_id <1) || (unit_id > number_of_deployment_units));
	   
	    /* Some debug */
	    pips_debug(2, "Externalized function [%s] being executed on unit %d:\n", function_name, unit_id);
	    ifdebug(2) {
	      pips_debug(2, "Current statement is:\n");
	      print_statement(s);
	    } 

	    unit_id_variable = entity_in_module (get_unit_id_name(unit_id), module);

	  }
	 
#if 0
	  /* Get the called module */
	  statement called_module_stat = (statement) db_get_memory_resource(DBR_CODE,
								  function_name,
								  true);
	  entity called_module = module_name_to_entity(function_name);
#endif
	 
	 
	  /* SEND PARAMS calls */
	  MAP (REGION, reg, {

	    entity param = entity_in_module(entity_local_name(region_entity(reg)),module);
	    entity send_param_module;
	    entity param_id_variable;
	    statement new_stat;

	    list call_params = NIL;

	    send_param_module  = module_name_to_entity(get_send_param_module_name(f, reg));
	    pips_debug(7, "Call to [%s]\n", entity_local_name(send_param_module));
	    pips_debug(7, "Concerned entity is [%s]\n", entity_local_name(param));

	    call_params = CONS (EXPRESSION,
				entity_to_expression(func_id_variable),
				call_params);
	    if (number_of_deployment_units > 1) {
	      call_params = CONS (EXPRESSION,
				  entity_to_expression(unit_id_variable),
				  call_params);
	    }
	    param_id_variable = entity_in_module (get_in_param_id_name(param,f),
						  module);
	    call_params = CONS (EXPRESSION,
				entity_to_expression(param_id_variable),
				call_params);
	    call_params = CONS (EXPRESSION,
				entity_to_expression(param),
				call_params);

	   
	    if (!region_scalar_p(reg))
	      {
		/* Add dynamic variables */
		
		list l_reg_params; /* list of entities: phi1, phi2,... */
		list l_reg_variables; /* list of dynamic variables....*/
		compute_region_variables(reg,&l_reg_params,&l_reg_variables);
		
		MAP (ENTITY, dyn_var, {
		  call_params = CONS (EXPRESSION,
				      entity_to_expression(dyn_var),
				      call_params);
		}, l_reg_variables);
	      }


	    new_stat = make_statement(entity_empty_label(),
				      STATEMENT_NUMBER_UNDEFINED,
				      STATEMENT_ORDERING_UNDEFINED,
				      empty_comments,
				      make_instruction(is_instruction_call,
						       make_call(send_param_module,
								 gen_nreverse(call_params))),
				      NIL,NULL,
				      empty_extensions (), make_synchronization_none());

	    l_stats = CONS (STATEMENT,
			    new_stat,
			    l_stats);

	  }, (list)hash_get(ht_in_regions,f));
	 
	  /* START_RU_CALL */
	  {
	    list start_ru_call_params = CONS(EXPRESSION,
				    entity_to_expression(func_id_variable),
				    (number_of_deployment_units>1)?CONS(EXPRESSION,
									entity_to_expression(unit_id_variable),
									NIL):NIL);
	    l_stats = CONS (STATEMENT,
			    make_statement(entity_empty_label(),
					   STATEMENT_NUMBER_UNDEFINED,
					   STATEMENT_ORDERING_UNDEFINED,
					   empty_comments,
					   make_instruction(is_instruction_call,
							    make_call(start_ru_module,
								      start_ru_call_params)),
					   NIL,NULL,
					   empty_extensions (), make_synchronization_none()),
			    l_stats);
	  }

	  /* WAIT_RU_CALL */
	  {
	    list wait_ru_call_params = CONS(EXPRESSION,
				       entity_to_expression(func_id_variable),
				       (number_of_deployment_units>1)?CONS(EXPRESSION,
									   entity_to_expression(unit_id_variable),
									   NIL):NIL);
	    l_stats = CONS (STATEMENT,
			    make_statement(entity_empty_label(),
					   STATEMENT_NUMBER_UNDEFINED,
					   STATEMENT_ORDERING_UNDEFINED,
					   empty_comments,
					   make_instruction(is_instruction_call,
							    make_call(wait_ru_module,
								      wait_ru_call_params)),
					   NIL,NULL,
					   empty_extensions (), make_synchronization_none()),
			    l_stats);
	  }
	 
	  /* RECEIVE PARAMS calls */

	  MAP (REGION, reg, {

	    entity param = entity_in_module(entity_local_name(region_entity(reg)),module);
	    entity receive_param_module;
	    entity param_id_variable;
	    statement new_stat;

	    list call_params = NIL;

	    receive_param_module  = module_name_to_entity(get_receive_param_module_name(f, reg));
	    pips_debug(7, "Call to [%s]\n", entity_local_name(receive_param_module));
	    pips_debug(7, "Concerned entity is [%s]\n", entity_local_name(param));

	    call_params = CONS (EXPRESSION,
				entity_to_expression(func_id_variable),
				call_params);
	    if (number_of_deployment_units > 1) {
	      call_params = CONS (EXPRESSION,
				  entity_to_expression(unit_id_variable),
				  call_params);
	    }
	    param_id_variable = entity_in_module (get_out_param_id_name(param,f),
						  module);
	    call_params = CONS (EXPRESSION,
				entity_to_expression(param_id_variable),
				call_params);
	    call_params = CONS (EXPRESSION,
				entity_to_expression(param),
				call_params);


	    if (!region_scalar_p(reg))
	      {
		/* Add dynamic variables */
		
		list l_reg_params; /* list of entities: phi1, phi2,... */
		list l_reg_variables; /* list of dynamic variables....*/
		compute_region_variables(reg,&l_reg_params,&l_reg_variables);
		
		MAP (ENTITY, dyn_var, {
		  call_params = CONS (EXPRESSION,
				      entity_to_expression(dyn_var),
				      call_params);
		}, l_reg_variables);
	      }

	    new_stat = make_statement(entity_empty_label(),
				      STATEMENT_NUMBER_UNDEFINED,
				      STATEMENT_ORDERING_UNDEFINED,
				      empty_comments,
				      make_instruction(is_instruction_call,
						       make_call(receive_param_module,
								 gen_nreverse(call_params))),
				      NIL,NULL,
				      empty_extensions (), make_synchronization_none());

	    l_stats = CONS (STATEMENT,
			    new_stat,
			    l_stats);

	  }, (list)hash_get(ht_out_regions,f));
	 
	  /* Now, just inverse list of statements */

	  l_stats = gen_nreverse(l_stats);
	 
	  /* And replace CALL instruction by SEQUENCE instruction */

	  statement_comments(s) = get_controlized_statement_comment(f);
	  statement_instruction(s) = make_instruction(is_instruction_sequence,
						      make_sequence(l_stats));

	  ifdebug(7) {
	    pips_debug(7, "After controlization, statement is\n");
	    print_statement(s);
	  }

	}, (list)calls_for_f);
	
	pips_debug(2, "Controlization is done on [%s]\n", entity_local_name(f));
	
      }, ht_calls);
     
     
      hash_table_free(ht_calls);
      hash_table_free(ht_params);
      hash_table_free(ht_private);
      hash_table_free(ht_in_regions);
      hash_table_free(ht_out_regions);

      HASH_MAP(k,v,{
	hash_table_free(v);	
      },ht_in_communications);
      hash_table_free(ht_in_communications);

      HASH_MAP(k,v,{
	hash_table_free(v);	
      },ht_out_communications);
      hash_table_free(ht_out_communications);

    }
   
    else {
      pips_debug(2, "Invalid number. Operation aborted\n");
    }
   
  }
  return returned_statement;
}
 

/*********************************************************
 * Phase main for PHRASE_DISTRIBUTOR_CONTROL_CODE
 *********************************************************/

static entity dynamic_area = entity_undefined;

bool phrase_distributor_control_code(const char* module_name)
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

  pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR_CONTROL_CODE\n");
  module_stat = controlize_distribution (module_stat, module);
  pips_debug(2, "END of PHRASE_DISTRIBUTOR_CONTROL_CODE\n");

  /* Display the statement before to check consistency */
  ifdebug(4) {
    print_statement(module_stat);
  }

  /* Check the coherency of data */

  pips_assert("Statement structure is consistent after PHRASE_DISTRIBUTOR_CONTROL_CODE",
	      gen_consistent_p((gen_chunk*)module_stat));
	     
  pips_assert("Statement is consistent after PHRASE_DISTRIBUTOR_CONTROL_CODE",
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

