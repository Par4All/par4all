/* transformationss package 
 *
 * coarse_grain_parallelization.c :  Beatrice Creusillet, october 1996
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the functions parallelizing loops from the 
 * regions of their bodies.
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
#include "parser_private.h"
#include "syntax.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "transformer.h"
#include "semantics.h"
#include "effects.h"
#include "regions.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"

static void
coarse_grain_loop_parallelization(statement module_parallelized_stat)
{
    gen_multi_recurse(module_parallelized_stat,
		      loop_domain, gen_null, gen_null, 
		      NULL);    
}

bool coarse_grain_parallelization(string module_name)
{
    statement module_stat, module_parallelized_stat;
    entity module;

    /* Get the code of the module. */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();
    set_cumulated_effects_map( effectsmap_to_listmap((statement_mapping)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE)) );
    module_to_value_mappings(module);
 
    /* Local regions */
    set_local_regions_map( effectsmap_to_listmap( (statement_mapping) 
	db_get_memory_resource(DBR_REGIONS, module_name, TRUE) ) );

    debug_on("COARSE_GRAIN_PARALLELIZATION_DEBUG_LEVEL");

    module_parallelized_stat = copy_statement(module_stat);
    coarse_grain_loop_parallelization(module_parallelized_stat);
 
    debug_off();    

    DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE,
			   strdup(module_name),
			   (char*) module_parallelized_stat);
	
    
    free_local_regions_map();
    return(TRUE);
}


