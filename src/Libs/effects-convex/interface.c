/* package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: interface.c
 * ~~~~~~~~~~~~~~~~~
 *
 * This File contains the interfaces with pipsmake which compute the various
 * types of convex regions by using the generic functions.
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

#include "effects-generic.h"
#include "effects-convex.h"

#include "pipsdbm.h"
#include "resources.h"

/*********************************************************************************/
/* CONVEX R/W REGIONS                                                            */
/*********************************************************************************/

/* bool summary_regions(char *module_name): computes the global
 * regions of a module : global regions only use formal or common variables.
 */
bool summary_regions(char *module_name)
{

    list sum_regions;

    /* Get the regions of the module */
    set_rw_effects( (statement_effects) 
	db_get_memory_resource(DBR_REGIONS, module_name, TRUE));

    sum_regions = NIL;

    DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_REGIONS, 
			   strdup(module_name),
			   (char*) list_to_effects(sum_regions));

    reset_rw_effects();
    return(TRUE);
}



/* bool may_regions(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
bool may_regions(char *module_name)
{
    set_bool_property("MUST_REGIONS", FALSE);
    regions(module_name);

    return(TRUE);
}


/* bool must_regions(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
bool must_regions(char *module_name)
{
    set_bool_property("MUST_REGIONS", TRUE);
    regions(module_name);    

    return(TRUE);
}



/* bool regions(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
bool regions(char *module_name)
{

    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    
    /* Get the transformers and preconditions of the module. */
    set_transformer_map( (statement_mapping)
	db_get_memory_resource(DBR_TRANSFORMERS, module_name, TRUE) );	
    set_precondition_map( (statement_mapping) 
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE) );

    /* predicates defining summary regions from callees have to be 
       translated into variables local to module */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );


    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(get_current_module_entity());

    /* Compute the regions of the module. */
    init_rw_effects();
    init_invariant_rw_effects();
    init_proper_rw_effects();
  
    debug_on("REGIONS_DEBUG_LEVEL");
    debug(1, "regions", "begin\n");

    debug(1, "regions", "end\n");
    debug_off();



    DB_PUT_MEMORY_RESOURCE(DBR_REGIONS, 
			   strdup(module_name),
			   (char*) get_rw_effects());

    DB_PUT_MEMORY_RESOURCE(DBR_INV_REGIONS, 
			   strdup(module_name),
			   (char*) get_invariant_rw_effects());

    DB_PUT_MEMORY_RESOURCE(DBR_PROPER_REGIONS, 
			   strdup(module_name),
			   (char*) get_proper_rw_effects());


    reset_current_module_entity();
    reset_current_module_statement();
    reset_transformer_map();
    reset_precondition_map();
    reset_cumulated_rw_effects();

    reset_rw_effects();
    reset_invariant_rw_effects();
    reset_proper_rw_effects();


    return(TRUE);
}



/*********************************************************************************/
/* CONVEX IN REGIONS                                                             */
/*********************************************************************************/


/* bool in_summary_regions(char *module_name): 
 * input    : the name of the current module.
 * output   : nothing !
 * modifies : the database.
 * comment  : computes the summary in regions of the current module, using the
 *            regions of its embedding statement.	
 */
bool in_summary_regions(module_name)
char *module_name;
{

    list in_sum_regions = NIL; 


      /* Get the in_regions of the module: they are computed by the function
       * in_regions() (see below).
       */
    set_rw_effects( (statement_effects) 
	db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE) );
    
    DB_PUT_MEMORY_RESOURCE(DBR_IN_SUMMARY_REGIONS, 
			   strdup(module_name),
			   (char*) list_to_effects(in_sum_regions));

    reset_rw_effects();

    return(TRUE);
}

/* bool in_regions(char *module_name): 
 * input    : the name of the current module.
 * output   : nothing !
 * modifies : the database.
 * comment  : computes the in regions of the current module.	
 */
bool in_regions(module_name)
char *module_name;
{
    entity module;
    statement module_stat;
    list l_in;

    debug_on("REGIONS_DEBUG_LEVEL");
    debug(1, "in_regions", "begin\n");

    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();
    
    /* Get the transformers, preconditions and regions of the module. */
    set_transformer_map( (statement_mapping)
	db_get_memory_resource(DBR_TRANSFORMERS, module_name, TRUE) );	
    set_precondition_map( (statement_mapping) 
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE) );

    set_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_REGIONS, module_name, TRUE) );
    set_invariant_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_INV_REGIONS, module_name, TRUE));


    /* initialise the maps for in regions */
    init_in_effects();
    init_invariant_in_effects();
    init_cumulated_in_effects();
  
    /* predicates defining summary regions from callees have to be 
       translated into variables local to module */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();

    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(module);

    /* Compute the regions of the module. */
    l_in = NIL;      


    debug(1, "in_regions", "end\n");

    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_IN_REGIONS, 
			   strdup(module_name),
			   (char*) get_in_effects());

    DB_PUT_MEMORY_RESOURCE(DBR_INV_IN_REGIONS, 
			   strdup(module_name),
			   (char*) get_invariant_in_effects());

    DB_PUT_MEMORY_RESOURCE(DBR_CUMULATED_IN_REGIONS, 
			   strdup(module_name),
			   (char*) get_cumulated_in_effects()); 

    reset_current_module_entity();
    reset_current_module_statement();
    reset_transformer_map();
    reset_precondition_map();

    reset_cumulated_rw_effects();

    reset_rw_effects();
    reset_invariant_rw_effects();

    reset_in_effects();
    reset_invariant_in_effects();
    reset_cumulated_in_effects();


    return(TRUE);
}


/*********************************************************************************/
/* CONVEX OUT REGIONS                                                            */
/*********************************************************************************/

/* bool out_summary_regions(module_name)
 * input    : the name of the current module.
 * output   : TRUE.
 * modifies : stores the out_summary_regions of the current module
 * comment  : new correct version: computes the out_summary_regions from
 *            the out_regions at the different call sites.
 */
bool
out_summary_regions(char * module_name)
{
    list l_reg = NIL;
    /* Look for all call sites in the callers */
    callees callers = (callees) db_get_memory_resource(DBR_CALLERS,
						       module_name,
						       TRUE);
    entity callee = local_name_to_top_level_entity(module_name);

    debug_on("SUMMARY_REGIONS_DEBUG_LEVEL");

    set_current_module_entity(callee);

    l_reg = NIL;
	
    DB_PUT_MEMORY_RESOURCE(DBR_OUT_SUMMARY_REGIONS, 
			   strdup(module_name),
			   (char*) list_to_effects(l_reg));

    reset_current_module_entity();
    debug_off();
    return(TRUE);
}


/* bool out_regions(char *module_name): 
 * input    : the name of the current module.
 * output   : nothing !
 * modifies : the database.
 * comment  : computes the out regions of the current module.	
 */
bool out_regions(module_name)
char *module_name;
{
    entity module;
    statement module_stat;
    list out_sum_regions = NIL;

    debug_on("REGIONS_DEBUG_LEVEL");
    debug(1, "out_regions", "begin\n");


    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();
    
    /* Get the transformers, preconditions, regions and cumulated in_regions 
       of the module. */
    set_transformer_map( (statement_mapping)
	db_get_memory_resource(DBR_TRANSFORMERS, module_name, TRUE) );	
    set_precondition_map( (statement_mapping) 
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE) );

    set_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_REGIONS, module_name, TRUE));
    set_invariant_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_INV_REGIONS, module_name, TRUE));

    set_cumulated_in_effects( (statement_effects) 
	db_get_memory_resource(DBR_CUMULATED_IN_REGIONS, module_name, TRUE) );
    set_in_effects( (statement_effects) 
	db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE) );
    set_invariant_in_effects( (statement_effects) 
	db_get_memory_resource(DBR_INV_IN_REGIONS, module_name, TRUE) );

    /* Get the out_summary_regions of the current module */
    out_sum_regions = effects_to_list((effects)
	    db_get_memory_resource(DBR_OUT_SUMMARY_REGIONS, module_name, TRUE));

    /* initialise the map for out regions */
    init_out_effects();
  
    /* predicates defining summary regions from callees have to be 
       translated into variables local to module */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();

    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(module);

    /* Compute the out_regions of the module. */

    debug(1, "out_regions", "end\n");

    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_OUT_REGIONS, 
			   strdup(module_name),
			   (char*) get_out_effects() );

    reset_current_module_entity();
    reset_current_module_statement();
    reset_transformer_map();
    reset_precondition_map();

    reset_cumulated_rw_effects();

    reset_rw_effects();
    reset_invariant_rw_effects();

    reset_in_effects();
    reset_cumulated_in_effects();
    reset_invariant_in_effects();
    reset_out_effects();

    return(TRUE);
}


/*********************************************************************************/
/* END                                                                           */
/*********************************************************************************/


