/* regions package :  Be'atrice Creusillet, october 1995
 *
 * array_privatization.c
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the functions computing the private regions.
 *
 */

#include <stdio.h>
#include <string.h>
#include "genC.h"
#include "ri.h"
#include "database.h"
#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "transformer.h"
#include "semantics.h"
#include "effects.h"
#include "regions.h"
#include "pipsdbm.h"
#include "resources.h"

/* =============================================================================== 
 *
 * INTRAPROCEDURAL PRIVATE REGIONS ANALYSIS
 *
 * =============================================================================== */

/* void array_privatizer(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
void array_privatizer(module_name)
char *module_name;
{
    entity module;
    statement module_stat;
    /* set and get the current properties concerning regions */
    (void) set_bool_property("MUST_REGIONS", TRUE);
    (void) set_bool_property("EXACT_REGIONS", TRUE);
    (void) get_regions_properties();

    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();
    
    /* Get the transformers and preconditions of the module. (Necessary ?) */
    set_transformer_map( (statement_mapping) 
	db_get_memory_resource(DBR_TRANSFORMERS, module_name, TRUE) );
    set_precondition_map( (statement_mapping) 
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE) );

    /* Get the READ, WRITE, IN and OUT regions of the module */
    set_local_regions_map( effectsmap_to_listmap( (statement_mapping) 
	db_get_memory_resource(DBR_REGIONS, module_name, TRUE) ) );
    set_in_regions_map( effectsmap_to_listmap( (statement_mapping) 
	db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE) ) );
    set_out_regions_map( effectsmap_to_listmap( (statement_mapping) 
	db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE) ) );

    /* predicates defining summary regions from callees have to be 
       translated into variables local to module */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();

    set_cumulated_effects_map( effectsmap_to_listmap((statement_mapping)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE)) );
    module_to_value_mappings(module);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_precondition_map();
    reset_cumulated_effects_map();
    reset_local_regions_map();
    reset_in_regions_map();
    reset_out_regions_map();
 
}
