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
/**
 * General computation for PHRASE distribution
 * DISTRIBUTION CONTEXT
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

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"

bool internal_compute_distribution_context (statement externalized_code,
						      hash_table* ht_params,
						      hash_table* ht_private,
						      hash_table* ht_in_regions,
						      hash_table* ht_out_regions,
						      void* key_value); 
/**
 * This function is called during PHRASE distribution
 *
 * The goal here is to compute the context of the distribution from a
 * given list of statements (after distribution initialization). Those
 * results are stored in following hashtable:
 *
 * HT_STATS: This hashtable stores the statement (values)
 * associated to names (keys) of externalized function. 
 *
 * HT_PARAMS: This hashtable stores the list of regions (values)
 * associated to names (keys) of externalized function. Those regions are
 * computed as union of IN and OUT regions of specified statement. If two
 * or more regions refer to the same variable (entity), those regions are
 * transformed into ONE region (which could be converted to a MAY region).
 * This hashtable is used to compute the parameters of specified
 * externalized function.
 *
 * HT_PRIVATE: This hashtable stores the list of privates regions (values)
 * associated to names (keys) of externalized function. This hashtable is
 * used to compute the private working area in specified externalized
 * function.
 *
 * HT_IN_REGIONS: This hashtable stores the list of IN regions (values)
 * associated to names (keys) of externalized function. This hashtable is
 * used to compute OUTPUT communication after externalized function
 * execution.
 *
 * HT_OUT_REGIONS: This hashtable stores the list of OUT regions (values)
 * associated to names (keys) of externalized function. This hashtable is
 * used to compute OUTPUT communication after externalized function
 * execution.
 *
 * Return true if everything is OK
 *
 * NB: All lists are sorted using externalized fonction name
 */
bool compute_distribution_context (list l_stats, 
				      statement module_stat,
				      entity module,
				      hash_table* ht_stats,
				      hash_table* ht_params,
				      hash_table* ht_private,
				      hash_table* ht_in_regions,
				      hash_table* ht_out_regions)
{
  bool returned_value = true;

  pips_debug(5, "[BEGIN] compute_distribution_context for %s: \n",
	     entity_local_name(module));

  *ht_stats = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_params = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_private = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_in_regions = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_out_regions = hash_table_make (hash_pointer, 0); /* lazy init */

  MAP (STATEMENT, s, {
    statement externalized_code = NULL;
    int stats_nb;
    string function_name = get_externalized_and_analyzed_function_name(s, &stats_nb);
    if (stats_nb > 1) {
      externalized_code = sequence_statement_containing (module_stat, s);
    }
    else if (stats_nb == 1) {
      externalized_code = s;
    }
    else {
      pips_internal_error("Strange externalized code");
    }

    /* Register new externalized function */
    pips_debug(5, "Register externalized function %s: \n",
	       function_name);
    if (!hash_defined_p(*ht_stats,function_name)) {
      hash_put(*ht_stats,function_name,externalized_code);
    }
    else {
      pips_user_warning("Multiply defined value in STATS hash_table!\n");    
      returned_value = false;
    }

    pips_debug(3, "ANALYSING function named [%s]..................\n",
		   function_name);

    if (!internal_compute_distribution_context (externalized_code,
						ht_params,
						ht_private,
						ht_in_regions,
						ht_out_regions,
						function_name)) {
      returned_value = false;
    }
  }, l_stats);

  pips_debug(5, "[END] compute_distribution_context for %s: \n",
	     entity_local_name(module));

  return returned_value;
}

/**
 * This function is called during PHRASE distribution controlization
 *
 * The goal here is to compute the context of the distribution from a
 * given list of call statements (after distribution). Those results are
 * stored in following hashtable:
 *
 * HT_CALLS: This hashtable stores the list of call statements (values)
 * associated to entity representing the external function (keys). This is
 * a list because a same function can be called from multiple points of
 * original program
 *
 * HT_PARAMS: This hashtable stores the list of regions (values)
 * associated to entity representing the external function (keys). Those
 * regions are computed as union of IN and OUT regions of specified
 * statement. If two or more regions refer to the same variable (entity),
 * those regions are transformed into ONE region (which could be converted
 * to a MAY region). This hashtable is used to compute the parameters of
 * specified externalized function.
 *
 * HT_PRIVATE: This hashtable stores the list of privates regions (values)
 * associated to entity representing the external function (keys). This
 * hashtable is used to compute the private working area in specified
 * externalized function.
 *
 * HT_IN_REGIONS: This hashtable stores the list of IN regions (values)
 * associated to entity representing the external function (keys). This
 * hashtable is used to compute OUTPUT communication after externalized
 * function execution.
 *
 * HT_OUT_REGIONS: This hashtable stores the list of OUT regions (values)
 * associated to entity representing the external function (keys). This
 * hashtable is used to compute OUTPUT communication after externalized
 * function execution.
 *
 * Return true if everything is OK
 *
 * NB: All lists are sorted using externalized fonction name
 */
bool compute_distribution_controlization_context (list l_calls, 
						     statement module_stat,
						     entity module,
						     hash_table* ht_calls,
						     hash_table* ht_params,
						     hash_table* ht_private,
						     hash_table* ht_in_regions,
						     hash_table* ht_out_regions)
{
  bool returned_value = true;
  string function_name;
  entity externalized_function;

  pips_debug(5, "[BEGIN] compute_distribution_controlization_context for %s: \n",
	     entity_local_name(module));
  ifdebug(9) {
    print_statement(module_stat);
  }

  *ht_calls = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_params = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_private = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_in_regions = hash_table_make (hash_pointer, 0); /* lazy init */
  *ht_out_regions = hash_table_make (hash_pointer, 0); /* lazy init */


  MAP (STATEMENT, s, {

    function_name = get_externalized_function_name(s);

    pips_debug(5, "Register statement calling externalized function %s: \n",
	       function_name);

    externalized_function = module_name_to_entity(function_name);

    /* Register new externalized function */

    if (!hash_defined_p(*ht_calls,externalized_function)) {
      /* This function has NOT been already defined, 
	 Register it */
      pips_debug(2, "Register statement for NEW function %s\n", function_name);
      hash_put(*ht_calls,externalized_function,CONS(STATEMENT, s, NIL));
      if (!internal_compute_distribution_context (s,
						  ht_params,
						  ht_private,
						  ht_in_regions,
						  ht_out_regions,
						  externalized_function)) {
	returned_value = false;
      }
    }
    else {
      /* This function has already been defined, 
	 add this statement to the list */
      list l_stats = (list)hash_get(*ht_calls,externalized_function);
      hash_put(*ht_calls,externalized_function,CONS(STATEMENT, s, l_stats));
      /* Check that IN and OUT regions match ! */
      /* NOT IMPLEMENTED Yet ! */
      pips_debug(2, "Adding statement to function %s\n", function_name);
    }

    pips_debug(3, "ANALYSING function named [%s]..................\n",
		   function_name);

  }, l_calls);

  pips_debug(5, "[END] compute_distribution_controlization_context for %s: \n",
	     entity_local_name(module));

  return returned_value;
}

/**
 * Compute union of exact regions.
 */
list compute_regions_union (list l_in, list l_out)
{
  list l_union = gen_copy_seq (l_in);

  pips_debug(4, "BEGIN of computing regions UNION\n");
  MAP (REGION, reg, {
    entity e = region_entity (reg);
    bool is_already_present = false;
    region reg_already_present = NULL;
    MAP (REGION, union_reg, {
      entity e2 = region_entity (union_reg);
      if (same_entity_p(e, e2)) {
	is_already_present = true;
	reg_already_present = union_reg;
      }
    }, l_union);
    if (is_already_present) {
      if (region_scalar_p(reg)) {
	pips_debug(6, "Found SCALAR region already present [%s]. Ignored.\n",
		   entity_local_name(e));
      }
      else {
	list new_regions;
	pips_debug(6, "Found ARRAY region already present [%s].\n",
		   entity_local_name(e));
	pips_debug(6, "Making UNION of:\n");
	print_region(reg);
	pips_debug(6, "and:\n");
	print_region(reg_already_present);	
	new_regions = region_must_union(reg,reg_already_present);
	pips_debug(6, "Getting:\n");
	print_regions(new_regions);
	if (gen_length(new_regions) > 1) {
	  pips_internal_error("Regions union must refer to only ONE region !");
	}
	else {
	  gen_remove (&l_union, reg_already_present);
	  l_union = CONS (REGION, REGION(gen_nth(0,new_regions)), l_union);
	}
      }
    }
    else {
      pips_debug(6, "Adding region for [%s]\n", entity_local_name(e));
      l_union = CONS(REGION, reg, l_union);
    }
  }, l_out);

  pips_debug(4, "END of computing regions UNION\n");
  return l_union;
}

/**
 * Internally used to compute distribution context for statement
 * externalized_code
 */
bool internal_compute_distribution_context (statement externalized_code,
						      hash_table* ht_params,
						      hash_table* ht_private,
						      hash_table* ht_in_regions,
						      hash_table* ht_out_regions,
						      void* key_value) 
{
  bool returned_value = true;

  list l_read, l_write, l_in, l_out;
  list l_params = NIL;
  list l_priv = NIL;

  pips_debug(6, "Compute regions\n");
  
  l_write = regions_dup
    (regions_write_regions(load_statement_local_regions(externalized_code))); 
  l_read = regions_dup
    (regions_read_regions(load_statement_local_regions(externalized_code))); 
  l_in = regions_dup(load_statement_in_regions(externalized_code));
  l_out = regions_dup(load_statement_out_regions(externalized_code));
  
  ifdebug(6) {
    pips_debug(6, "READ regions: \n");
    print_regions(l_read);
    pips_debug(6, "WRITE regions: \n");
    print_regions(l_write);
  }
  
  l_params = compute_regions_union (l_in, l_out);
    
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

  /* Storing results in hash_tables */

  pips_debug(2, "Storing in hash_tables with key %s: \n", (string)key_value);

  pips_debug(5, "Storing in ht_param: \n");
  if (!hash_defined_p(*ht_params,key_value)) {
    hash_put(*ht_params,key_value,l_params);
  }
  else {
    pips_user_warning("Multiply defined value in PARAMS hash_table!\n");    
    returned_value = false;
  }
  
  pips_debug(5, "Storing in ht_private: \n");
  if (!hash_defined_p(*ht_private,key_value)) {
    hash_put(*ht_private,key_value,l_priv);
  }
  else {
    pips_user_warning("Multiply defined value in PRIVATE hash_table!\n");    
    returned_value = false;
  }
  
  pips_debug(5, "Storing in ht_in_regions: \n");
  if (!hash_defined_p(*ht_in_regions,key_value)) {
    hash_put(*ht_in_regions,key_value,l_in);
  }
  else {
    pips_user_warning("Multiply defined value in IN_REGIONS hash_table!\n");    
    returned_value = false;
  }
  
  pips_debug(5, "Storing in ht_out_regions: \n");
  if (!hash_defined_p(*ht_out_regions,key_value)) {
    hash_put(*ht_out_regions,key_value,l_out);
  }
  else {
    pips_user_warning("Multiply defined value in OUT_REGIONS hash_table!\n");    
    returned_value = false;
  }

  return returned_value;
}

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tag EXTERNALIZED_CODE_PRAGMA_BEGIN
 */
string get_externalizable_function_name(statement stat) 
{
  return get_function_name_by_searching_tag
    (stat,EXTERNALIZED_CODE_PRAGMA_BEGIN);
}

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tag EXTERNALIZED_CODE_PRAGMA_CALL
 */
string get_externalized_function_name(statement stat) 
{
  return get_function_name_by_searching_tag
    (stat,EXTERNALIZED_CODE_PRAGMA_CALL);
}

/**
 * Return the identified function name of the externalized portion of code
 * by searching comment matching tags EXTERNALIZED_CODE_PRAGMA_ANALYZED
 * Sets the number of statements of this externalizable statement
 */
string get_externalized_and_analyzed_function_name(statement stat,
						   int* stats_nb) 
{
  string comments;
  string searched_string;
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
 * Return a unique (regarding variable_equal_p(var1,var2)) string
 * representation of a variable var
 */
string variable_to_string (variable var)
{
  string returned_value;
  intptr_t low, up;

  returned_value = basic_to_string(variable_basic(var));
  (strchr(returned_value,'*'))[0] = '\0';
  FOREACH (DIMENSION, dim, variable_dimensions(var)){
    if ((expression_integer_value(dimension_lower(dim), &low))
	&& (expression_integer_value(dimension_upper(dim), &up))) {
      char buffer[256];
      sprintf (buffer, VARIABLE_NAME_FORMAT, low, up);
      returned_value = strdup(concatenate(returned_value,strdup(buffer),NULL));
    }
    else {
      returned_value = strdup(concatenate(returned_value,":UNDEFINED",NULL));
    }
  }

  return returned_value;
}

/**
 * Build an HASHTABLE where keys are VARIABLE and values are HASHTABLE
 * where keys are modules or externalized function (ENTITY) and values are
 * list of regions
 */
void register_scalar_communications (hash_table* ht_communications,
				     entity function,
				     list l_regions)
{

  /* Iterate on all regions of the given list */
  MAP (REGION, reg, {

    if (region_scalar_p(reg)) {
      
      bool already_present = false;
      variable already_registered_variable = NULL;
      
      /* Get the variable type */
      variable var = type_variable(entity_type(region_entity(reg)));
      
      pips_debug(2, "Variable %s %zd \n", basic_to_string(variable_basic(var)), gen_length(variable_dimensions(var)));
      print_region(reg);
      
      /* Look if this variable is already registered */
      HASH_MAP (var2, l2, {
	if (variable_equal_p(var,var2)) {
	  already_present = true;
	  already_registered_variable = var2;
	}
      },*ht_communications);
      
      if (already_present) {
	hash_table ht_for_variable 
	  = (hash_table)hash_get(*ht_communications,
				 already_registered_variable);
	if (hash_defined_p(ht_for_variable,function)) {
	  list old_regs = hash_get(ht_for_variable,
				   function);
	  hash_update(ht_for_variable, function,
		      CONS(REGION,reg,old_regs));
	  pips_debug(2, "Add region for existing function\n");
	}
	else {
	  hash_put(ht_for_variable, function,
		   CONS(REGION,reg,NIL));
	  pips_debug(2, "New region for existing function\n");
	}
      }
      else {
	hash_table new_ht_for_variable
	  = hash_table_make(hash_pointer,0);
	hash_put(new_ht_for_variable, function,
		 CONS(REGION,reg,NIL));
	hash_put(*ht_communications,var,new_ht_for_variable);
	pips_debug(2, "New function\n");
      } 
    }
  }, l_regions);
}
