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

#include "transformer.h"

#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"

#include "properties.h"
#define REGION_BUFFER_SIZE 2048
#define REGION_FORESYS_PREFIX "C$REG"
#define PIPS_NORMAL_PREFIX "C"

/*
#define BACKWARD TRUE
#define FORWARD FALSE
*/


/* list words_region_no_action(effect reg)
 * input    : a region.
 * output   : a list of strings representing the region.
 * modifies : nothing.
 * comment  :	because of 'buffer', this function cannot be called twice
 * before
 * its output is processed. Also, overflows in relation_to_string() 
 * cannot be prevented. They are checked on return.
 * COPIED FROM THE FUNCTION words_region IN FILE effects-convex/prettyprint.c
 * AND MODIFIED TO NOT PRINT ACTION (IN/OUT)
 */
static list
words_region_no_action(region reg)
{
    static char buffer[REGION_BUFFER_SIZE];
    
    list pc = NIL;
    reference r = effect_reference(reg);
/*    action ac = effect_action(reg); */
    approximation ap = effect_approximation(reg);
    boolean foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    Psysteme sc = region_system(reg);

    buffer[0] = '\0';

    if(!region_empty_p(reg) && !region_rn_p(reg))
    {
	Pbase sorted_base = region_sorted_base_dup(reg);
	Psysteme sc = sc_dup(region_system(reg));
	
      /* sorts in such a way that constraints with phi variables come first */
	region_sc_sort(sc, sorted_base);

	strcat(buffer, "-");	
	region_sc_to_string(buffer, sc);
	sc_rm(sc);
	base_rm(sorted_base);

    }
    else
    {
	strcat(buffer, "-");	
	region_sc_to_string(buffer, sc);
    }
    pips_assert("words_region", strlen(buffer) < REGION_BUFFER_SIZE );

    if (foresys)
    {
      pc = gen_nconc(pc, words_reference(r));
      pc = CHAIN_SWORD(pc, ", RGSTAT(");
/*      pc = CHAIN_SWORD(pc, action_read_p(ac) ? "R," : "W,"); */
      pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "MAY), " : "EXACT), ");
      pc = CHAIN_SWORD(pc, buffer);
    }
    else /* PIPS prettyprint */
    {
	pc = CHAIN_SWORD(pc, "<");
	pc = gen_nconc(pc, effect_words_reference(r));
	pc = CHAIN_SWORD(pc, "-");
/*	pc = CHAIN_SWORD(pc, action_interpretation(action_tag(ac)));
 *      pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "-MAY" : "-EXACT");
*/
        pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "MAY" : "EXACT");
	pc = CHAIN_SWORD(pc, buffer);
	pc = CHAIN_SWORD(pc, ">");
    }

    return pc;
}


/* text text_region_no_action(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries, 
 *            representing the region
 * modifies : nothing
 * COPIED FROM THE FUNCTION tex_region IN FILE effects-convex/prettyprint.c
 * AND MODIFIED TO NOT PRINT ACTION (IN/OUT)
 */
static text 
text_region_no_action(effect reg)
{
    text t_reg = make_text(NIL);
    boolean foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    string str_prefix;

    if (foresys)
	str_prefix = REGION_FORESYS_PREFIX;
    else
	str_prefix = PIPS_NORMAL_PREFIX;
    
    if(reg == effect_undefined)
    {
	ADD_SENTENCE_TO_TEXT(t_reg, 
			     make_pred_commentary_sentence
			     (strdup("<REGION_UNDEFINED>"),
			      str_prefix));
	user_log("[region_to_string] unexpected effect undefined\n");
    }
    else
    {
	gen_free(t_reg);
	t_reg =
	    words_predicate_to_commentary(words_region_no_action(reg),
					  str_prefix);
    }

    return(t_reg);   
}


static text
aliases_text(string module_name, string resource_name)
{
    list alias_lists;
    list al = NIL;
    entity module;
    text txt = make_text(NIL);

    pips_debug(4,"module %s resource %s\n",module_name,resource_name);

/*alias_lists = (list) db_get_memory_resource(resource_name,module_name,TRUE);
*/

/*    alias_lists = effects_to_list(
	(effects)
	db_get_memory_resource(resource_name, module_name, TRUE));
*/

    alias_lists = effects_classes_classes(
	(effects_classes)
	db_get_memory_resource(resource_name, module_name, TRUE));

    pips_debug(9,"got aliases\n");

    /* ATTENTION: all this is necessary to call module_to_value_mappings
     * to set up the hash table to translate value into value names
     * before the call to text_region below
     */       
    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();
    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, TRUE));
    set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    /* that's it, but we musn't forget to rest everything after the call
     */

    module_to_value_mappings(module);

    pips_debug(9,"hash table set up\n");

/*    set_action_interpretation(ACTION_IN,ACTION_OUT); */

    MAP(LIST,alias_list,
	{
	    pips_debug(9,"make text for alias list\n");

	    if (alias_list != (list) HASH_UNDEFINED_VALUE
		&& alias_list != list_undefined) 
	    {
		/* have to copy alias_list here */
		al = alias_list;
		MAP(EFFECT,alias,
		    {
			pips_debug(9,"make text for alias\n");

			MERGE_TEXTS(txt,text_region_no_action(alias));
		    },
			al);

		ADD_SENTENCE_TO_TEXT(
		    txt,
		    make_sentence(is_sentence_formatted,strdup("\n"))
		    );

		pips_debug(9,"made text for alias list\n");
	    }
	},
	    alias_lists);

    pips_debug(4,"end\n");

/*    reset_action_interpretation(); */
    free_value_mappings();
    reset_cumulated_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();

    return txt;
}


static bool
print_aliases( string module_name, string resource_name, string file_extn)
{
    char *file_resource_name;
    bool success = TRUE;

    pips_debug(4,"module %s resource %s file extn %s\n",
	       module_name,resource_name,file_extn);

    file_resource_name = DBR_ALIAS_FILE;

    success = 
	make_text_resource(module_name,
			   file_resource_name,
			   file_extn,
			   aliases_text(module_name,resource_name));

    pips_debug(4,"end\n");

    return(success);
}


bool
print_in_alias_pairs( string module_name )
{
    bool success = TRUE;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);

    success = print_aliases(module_name,DBR_IN_ALIAS_PAIRS,".in_alias");

    pips_debug(4,"end\n");
    debug_off();

    return(TRUE);
}


bool
print_out_alias_pairs( string module_name )
{
    bool success = TRUE;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);

    success = print_aliases(module_name,DBR_OUT_ALIAS_PAIRS,".out_alias");

    pips_debug(4,"end\n");
    debug_off();

    return(TRUE);
}

bool
print_alias_lists( string module_name )
{
    bool success = TRUE;

    debug_on("ALIAS_LISTS_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);

    success = print_aliases(module_name,DBR_ALIAS_LISTS,".alias_lists");

    pips_debug(4,"end\n");
    debug_off();

    return(TRUE);
}

bool
print_alias_classes( string module_name )
{
    bool success = TRUE;

    debug_on("ALIAS_CLASSES_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);


    success = print_aliases(module_name,DBR_ALIAS_CLASSES,".alias_classes");

    pips_debug(4,"end\n");
    debug_off();

    return(TRUE);
}
