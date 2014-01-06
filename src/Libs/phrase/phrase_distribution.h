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
#ifndef PHRASE_DISTRIBUTOR_DEFS
#define PHRASE_DISTRIBUTOR_DEFS

#define EXTERNALIZED_CODE_PRAGMA_BEGIN "BEGIN_FPGA_%s"
#define EXTERNALIZED_CODE_PRAGMA_END "END_FPGA_%s"
#define EXTERNALIZED_CODE_PRAGMA_ANALYZED "ANALYZED_FPGA_%s (%d statements)"
#define EXTERNALIZED_CODE_PRAGMA_CALL "CALL_FPGA_%s"

/* Stuff for distribution controlization */
#define EXTERNALIZED_FUNCTION_PARAM_NAME "%s_PARAM_%d"
#define EXTERNALIZED_FUNCTION_PRIVATE_PARAM_NAME "%s_PRIV"
#define CONTROL_DATA_COMMON_NAME "CONTROL_DATA"
#define FUNCTION_COMMON_NAME "%s_COMMON"
#define COMMON_PARAM_NAME "%s_%s"
#define DYN_VAR_PARAM_NAME "%s_DV_PARAM"
#define REF_VAR_PARAM_NAME "%s_REF_PARAM"
#define UNITS_NB_NAME "UNITS_NB"
#define UNIT_ID_NAME "UNIT%d"
#define FUNCTIONS_NB_NAME "FUNCTIONS_NB"
#define FUNCTION_ID_NAME "%s_FUNCTION"
#define IN_PARAM_ID_NAME "%s_%s_IN_PARAM"
#define OUT_PARAM_ID_NAME "%s_%s_OUT_PARAM"
#define CONTROLIZED_STATEMENT_COMMENT "! CONTROLIZED CALL TO %s\n"

/* Stuff for START_RU(...) subroutine generation */
#define START_RU_MODULE_NAME "START_RU"
#define START_RU_PARAM1_NAME "FUNC_ID"
#define START_RU_PARAM2_NAME "UNIT_ID"

/* Stuff for WAIT_RU(...) subroutine generation */
#define WAIT_RU_MODULE_NAME "WAIT_RU"
#define WAIT_RU_PARAM1_NAME "FUNC_ID"
#define WAIT_RU_PARAM2_NAME "UNIT_ID"

/* Stuff for SEND_PARAM....(...) and RECEIVE_PARAM....(...)
 * subroutines generation */
#define VARIABLE_NAME_FORMAT "_%"PRIdPTR"_%"PRIdPTR
#define SEND_PARAMETER_MODULE_NAME "SEND_%s_PARAMETER"
#define RECEIVE_PARAMETER_MODULE_NAME "RECEIVE_%s_PARAMETER"
#define SEND_ARRAY_PARAM_MODULE_NAME "SEND_%s_%s_PARAMETER"
#define RECEIVE_ARRAY_PARAM_MODULE_NAME "RECEIVE_%s_%s_PARAMETER"
#define COM_MODULE_PARAM1_NAME "FUNC_ID"
#define COM_MODULE_PARAM2_NAME "UNIT_ID"
#define COM_MODULE_PARAM3_NAME "PARAM_ID"
#define COM_MODULE_PARAM4_NAME "PARAM"

#define RU_SEND_FLOAT_PARAM_MODULE_NAME "RU_SEND_FLOAT_PARAM"
#define RU_RECEIVE_FLOAT_PARAM_MODULE_NAME "RU_RECEIVE_FLOAT_PARAM"

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tag 
 */
string get_function_name_by_searching_tag (statement stat,const char* tag);

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tag EXTERNALIZED_CODE_PRAGMA_BEGIN
 */
string get_externalizable_function_name(statement stat); 

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tag EXTERNALIZED_CODE_PRAGMA_CALL
 */
string get_externalized_function_name(statement stat); 

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tags EXTERNALIZED_CODE_PRAGMA_ANALYZED
 * Sets the number of statements of this externalizable statement
 */
string get_externalized_and_analyzed_function_name(statement stat,
						   int* stats_nb);
 
bool compute_distribution_context (list l_stats, 
				      statement module_stat,
				      entity module,
				      hash_table* ht_stats,
				      hash_table* ht_params,
				      hash_table* ht_private,
				      hash_table* ht_out_regions,
				      hash_table* ht_in_regions);

bool compute_distribution_controlization_context (list l_calls, 
						     statement module_stat,
						     entity module,
						     hash_table* ht_calls,
						     hash_table* ht_params,
						     hash_table* ht_private,
						     hash_table* ht_out_regions,
						     hash_table* ht_in_regions);
void register_scalar_communications (hash_table* ht_communications,
				     entity function,
				     list l_regions);

string variable_to_string (variable var);

/**
 * Build and store new module START_RU.
 * Create statement module_statement
 */
entity make_start_ru_module (hash_table ht_params, 
			     statement* module_statement, 
			     int number_of_deployment_units,
			     entity global_common,
			     list l_commons);
 
/**
 * Build and store new module WAIT_RU.
 * Create statement module_statement
 */
entity make_wait_ru_module (statement* module_statement, 
			    int number_of_deployment_units,
			    entity global_common,
			    list l_commons);

 

/**
 * Return SEND_PARAM module name for function and region
 */
string get_send_param_module_name(entity function, region reg);


/**
 * Return RECEIVE_PARAM module name for function and region
 */
string get_receive_param_module_name(entity function, region reg);

/**
 * Build and return list of modules used for INPUT communications
 * (SEND_PARAMETERS...)
 */
list make_send_scalar_params_modules (hash_table ht_in_communications,
				      int number_of_deployment_units,
				      entity global_common,
				      list l_commons); 

/**
 * Build and return list of modules used for OUTPUT communications
 * (RECEIVE_PARAMETERS...)
 */
list make_receive_scalar_params_modules (hash_table ht_out_communications,
					 int number_of_deployment_units,
					 entity global_common,
					 list l_commons);


/**
 * Make all SEND_PARAM communication modules for non-scalar regions for a
 * given function
 */
list make_send_array_params_modules (entity function,
				     list l_regions,
				     entity global_common,
				     entity externalized_fonction_common,
				     int number_of_deployment_units);

/**
 * Make all RECEIVE_PARAM communication modules for non-scalar regions for a
 * given function
 */
list make_receive_array_params_modules (entity function,
					list l_regions,
					entity global_common,
					entity externalized_fonction_common,
					int number_of_deployment_units);

/**
 * Creates a private variable in specified module
 */
entity create_private_variable_for_new_module (entity a_variable,
					       const char* new_name, 
					       const char* new_module_name,
					       entity module);

/**
 * Create new variable parameter for a newly created module
 */
entity create_parameter_for_new_module (variable var,
					const char* parameter_name, 
					const char* module_name,
					entity module,
					int param_nb);

/**
 * Create new integer variable parameter for a newly created module
 */
entity create_integer_parameter_for_new_module (const char* parameter_name, 
						const char* module_name,
						entity module,
						int param_nb);

/**
 * Store (PIPDBM) newly created module module with module_statement 
 * as USER_FILE by saving pretty printing
 */
void store_new_module (const char* module_name,
		       entity module,
		       statement module_statement); 

/*
 * Return COMMON_PARAM_NAME
 */
string get_common_param_name (entity variable, entity function);
 
/*
 * Return FUNCTION_ID_NAME
 */
string get_function_id_name (entity function);
 
/*
 * Return SEND_PARAMETER_MODULE_NAME
 */
string get_send_parameter_module_name (variable var);
 
/*
 * Return RECEIVE_PARAMETER_MODULE_NAME
 */
string get_receive_parameter_module_name (variable var);
 
/*
 * Return IN_PARAM_ID_NAME
 */
string get_in_param_id_name (entity variable, entity function); 

/*
 * Return OUT_PARAM_ID_NAME
 */
string get_out_param_id_name (entity variable, entity function);

/**
 * Build and return parameters (PHI1,PHI2) and dynamic variables for
 * region reg.  
 * NOT IMPLEMENTED: suppress unused dynamic variables !!!!
 */
void compute_region_variables (region reg,
			       list* l_reg_params,
			       list* l_reg_variables);

/**
 * Creates all the things that need to be created in order to declare common
 * in module (all the variable are created)
 */
void declare_common_variables_in_module (entity common, entity module);

#endif
