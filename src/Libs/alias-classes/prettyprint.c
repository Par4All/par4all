
#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"

#include "effects.h"
#include "regions.h"
#include "semantics.h"

#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"

#define BACKWARD TRUE
#define FORWARD FALSE


static text alias_pairs_text(string module_name,string resource_name)
{
    list l_pairs;
    entity module;
    text txt = make_text(NIL);

/*    debug_on("REGIONS_DEBUG_LEVEL"); */

    l_pairs = (list) db_get_memory_resource(resource_name, module_name, TRUE);    

    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();

    /* To set up the hash table to translate value into value names */       
    set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(module);

    MAP(LIST,pair,
	{
	    /* MERGE_TEXTS(txt, text_array_regions(pair)); */
    ADD_SENTENCE_TO_TEXT(txt,make_sentence(is_sentence_formatted,strdup("\n")));
	},
    l_pairs);

/*    debug_off(); */

    reset_current_module_entity();
    reset_cumulated_rw_effects();

    return txt;
}


static bool
print_alias_pairs( string module_name, string resource_name, string file_extn )
{
    char *file_resource_name;
    bool success = TRUE;

    file_resource_name = DBR_ALIAS_FILE;

    /*
    begin_attachment_prettyprint();
    */  
    success = make_text_resource(module_name, file_resource_name,
				 file_extn,
				 alias_pairs_text(module_name,
				    resource_name));
    /*
    end_attachment_prettyprint();
    */
    return(TRUE);
}


bool
print_in_alias_pairs( string module_name )
{
return print_alias_pairs(module_name,DBR_IN_ALIAS_PAIRS,".in_alias");
}


bool
print_out_alias_pairs( string module_name )
{
return(TRUE);
}
