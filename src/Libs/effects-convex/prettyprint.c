/* package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: prettyprint.c
 * ~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the prettyprinting functions.
 *
 */

#include <stdio.h>
#include <string.h>
#include <values.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "top-level.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-convex.h"

#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"

#define REGION_BUFFER_SIZE 2048

#define REGION_FORESYS_PREFIX "C$REG"
#define PIPS_NORMAL_PREFIX "C"

static boolean in_out_regions_p = FALSE;
static boolean is_user_view_p = FALSE;
static hash_table nts = hash_table_undefined;

static bool print_code_with_regions(string, string, string, string);
static text get_any_regions_text(string, string, string, bool);

text
get_text_regions(string module_name)
{
    is_user_view_p = FALSE;
    in_out_regions_p = FALSE;

    return get_any_regions_text(module_name,
				DBR_REGIONS,
				DBR_SUMMARY_REGIONS,
				FALSE);
}

text
get_text_in_regions(string module_name)
{
    is_user_view_p = FALSE;
    in_out_regions_p = TRUE;

    return get_any_regions_text(module_name,
				DBR_IN_REGIONS,
				DBR_IN_SUMMARY_REGIONS,
				FALSE);
}

text
get_text_out_regions(string module_name)
{
    is_user_view_p = FALSE;
    in_out_regions_p = TRUE;

    return get_any_regions_text(module_name,
				DBR_OUT_REGIONS,
				DBR_OUT_SUMMARY_REGIONS,
				FALSE);
}

/* bool print_source_regions(string module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the original source code with the corresponding regions.	
 */
bool
print_source_regions(module_name)
string module_name;
{
    is_user_view_p = TRUE;
    in_out_regions_p = FALSE;

    return print_code_with_regions(module_name,
				   DBR_REGIONS,
				   DBR_SUMMARY_REGIONS,
				   USER_REGION_SUFFIX);
}

/* bool print_source_in_regions(string module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the original source code with the corresponding in regions.	
 */
bool
print_source_in_regions(module_name)
string module_name;
{
    is_user_view_p = TRUE;
    in_out_regions_p = TRUE;

    return print_code_with_regions(module_name,
				   DBR_IN_REGIONS,
				   DBR_IN_SUMMARY_REGIONS,
				   USER_IN_REGION_SUFFIX);
}

/* bool print_source_out_regions(string module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the original source code with the corresponding out regions.	
 */
bool
print_source_out_regions(module_name)
string module_name;
{
    is_user_view_p = TRUE;
    in_out_regions_p = TRUE;

    return print_code_with_regions(module_name,
				   DBR_OUT_REGIONS,
				   DBR_OUT_SUMMARY_REGIONS,
				   USER_OUT_REGION_SUFFIX);
}



/* bool print_code_regions(string module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the source code with the corresponding regions.	
 */
bool
print_code_regions(module_name)
string module_name;
{
    is_user_view_p = FALSE;
    in_out_regions_p = FALSE;

    return print_code_with_regions(module_name,
				   DBR_REGIONS,
				   DBR_SUMMARY_REGIONS,
				   SEQUENTIAL_REGION_SUFFIX);
}


/* bool  print_code_in_regions(string module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the source code with the corresponding in regions.	
 */
bool
print_code_in_regions(module_name)
string module_name;
{
    is_user_view_p = FALSE;
    in_out_regions_p = TRUE;

    return print_code_with_regions(module_name,
				   DBR_IN_REGIONS,
				   DBR_IN_SUMMARY_REGIONS,
				   SEQUENTIAL_IN_REGION_SUFFIX);
}

/* bool print_code_out_regions(string module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the source code with the corresponding out regions.	
 */
bool
print_code_out_regions(module_name)
string module_name;
{
    is_user_view_p = FALSE;
    in_out_regions_p = TRUE;

    return print_code_with_regions(module_name,
				   DBR_OUT_REGIONS,
				   DBR_OUT_SUMMARY_REGIONS,
				   SEQUENTIAL_OUT_REGION_SUFFIX);
}

/* bool print_code_proper_regions(string module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the source code with the corresponding proper regions
 *            (and summary regions for compatibility purposes).	
 */
bool
print_code_proper_regions(module_name)
string module_name;
{
    is_user_view_p = FALSE;
    in_out_regions_p = FALSE;

    return print_code_with_regions(module_name,
				   DBR_PROPER_REGIONS,
				   DBR_SUMMARY_REGIONS,
				   SEQUENTIAL_PROPER_REGION_SUFFIX);
}




/* bool print_code_with_regions(string module_name, list summary_regions)
 * input    : the name of the current module, the name of the region and
 *            summary region resources and the file suffix
 *            the regions are in the global variable local_regions_map.
 * modifies : nothing.
 * comment  : prints the source code with the corresponding regions.	
 */
static bool
print_code_with_regions(string module_name,
			string resource_name,
			string summary_resource_name,
			string file_suffix)
{
    char *file_name, *file_resource_name;
    bool success = TRUE;

    file_name = strdup(concatenate(file_suffix,
                                  get_bool_property
				  ("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? 
				  GRAPH_FILE_EXT : "",
                                  NULL));
    file_resource_name = get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	DBR_GRAPH_PRINTED_FILE : 
	    (is_user_view_p ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE);

    /*
    begin_attachment_prettyprint();
    */  
    success = make_text_resource(module_name, file_resource_name,
				 file_name,
				 get_any_regions_text(module_name,
				    resource_name,
				    summary_resource_name,
				    TRUE));
    /*
    end_attachment_prettyprint();
    */
    free(file_name);
    return(TRUE);
}

static text 
get_any_regions_text(string module_name,
		     string resource_name,
		     string summary_resource_name,
		     bool give_code_p)
{
    list summary_regions;
    entity module;
    statement module_stat, user_stat = statement_undefined;
    text txt = make_text(NIL);

    debug_on("REGIONS_DEBUG_LEVEL");

    /* load regions corresponding to the current module */
    set_rw_effects((statement_effects) 
			   db_get_memory_resource
			   (resource_name, module_name, TRUE) );

    summary_regions = 
	effects_to_list((effects) db_get_memory_resource
			(summary_resource_name,
			 module_name, TRUE));

    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();

    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, TRUE));

    module_stat = get_current_module_statement();

    /* To set up the hash table to translate value into value names */       
    set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(module);

    debug_off();

    reset_current_module_entity();
    reset_current_module_statement();
    reset_rw_effects(); 

    reset_cumulated_rw_effects();

    return txt;
}

