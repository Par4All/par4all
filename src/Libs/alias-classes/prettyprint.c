/* $Id$
 */

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

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

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
    text txt_reg = make_text(NIL);

    pips_debug(8,"module %s resource %s\n",module_name,resource_name);

/*    l_pairs = (list) db_get_memory_resource(resource_name, module_name, TRUE); */

    l_pairs = effects_to_list((effects)
			      db_get_memory_resource(resource_name, module_name, TRUE));

    pips_debug(8,"got pairs\n");

    /* To set up the hash table to translate value into value names */       

    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();
    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, TRUE));
    set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(module);

    pips_debug(8,"hash table set up\n");

    set_action_interpretation(ACTION_IN,ACTION_OUT);

    MAP(LIST,pair,
	{
	    pips_debug(9,"start 1st map\n");

	    if (pair != (list) HASH_UNDEFINED_VALUE && pair != list_undefined) 
	    {
/* was
		MAP(EFFECT, reg,
		    {
			txt_reg = text_region(reg);
			MERGE_TEXTS(txt, text_region(reg));
		    },
			pair);
			*/

		txt_reg = text_region(EFFECT(CAR(pair)));

		pips_debug(9,"done text_region\n");

		MERGE_TEXTS(txt,txt_reg);

		pips_debug(9,"done MERGE_TEXTS\n");

		ADD_SENTENCE_TO_TEXT(
		    txt,
		    make_sentence(
			is_sentence_formatted,
			strdup("can't print translation?\n"))
		    );

		pips_debug(9,"done ADD_SENTENCE_TO_TEXT\n");

		ADD_SENTENCE_TO_TEXT(
		    txt,
		    make_sentence(is_sentence_formatted,strdup("\n"))
		    );

		pips_debug(9,"done ADD_SENTENCE_TO_TEXT\n");
	    }
	},
	    l_pairs);

    pips_debug(8,"made text\n");

    reset_action_interpretation();

    reset_current_module_entity();
    reset_cumulated_rw_effects();

    return txt;
}


static bool
print_alias_pairs( string module_name, string resource_name, string file_extn )
{
    char *file_resource_name;
    bool success = TRUE;

    pips_debug(8,"module %s resource %s file extn %s\n",
	       module_name,resource_name,file_extn);

    file_resource_name = DBR_ALIAS_FILE;

    success = make_text_resource(module_name,
				 file_resource_name,
				 file_extn,
				 alias_pairs_text(module_name,resource_name));
    return(success);
}


bool
print_in_alias_pairs( string module_name )
{
    bool success = TRUE;

    debug_on("ALIAS_DEBUG_LEVEL");
    pips_debug(8,"module %s\n",module_name);

    success = print_alias_pairs(module_name,DBR_IN_ALIAS_PAIRS,".in_alias");

    debug_off();

    return(TRUE);
}


bool
print_out_alias_pairs( string module_name )
{
    bool success = TRUE;

    debug_on("ALIAS_DEBUG_LEVEL");
    pips_debug(8,"module %s\n",module_name);

    success = print_alias_pairs(module_name,DBR_OUT_ALIAS_PAIRS,".out_alias");

    debug_off();

    return(TRUE);
}
