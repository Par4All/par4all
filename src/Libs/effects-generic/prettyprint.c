/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: prettyprint.c
 * ~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the prettyprint of 
 * all types of effects.
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "top-level.h"
#include "text.h"

#include "database.h"
#include "resources.h"
#include "pipsdbm.h"
#include "properties.h"

#include "text-util.h"
#include "prettyprint.h"

#include "effects-generic.h"

/***************************************** READ/WRITE ACTION INTERPRETATION */

static string read_action_interpretation = string_undefined;
static string write_action_interpretation = string_undefined;

void 
set_action_interpretation(string r, string w)
{
    read_action_interpretation = r;
    write_action_interpretation = w;
}

void
reset_action_interpretation()
{
    read_action_interpretation = string_undefined;
    write_action_interpretation = string_undefined;
}


string 
action_interpretation(action a)
{
    return action_read_p(a) ? 
	read_action_interpretation : write_action_interpretation;
}


/****************************************************************************/

static bool is_user_view_p = FALSE;
static hash_table nts = hash_table_undefined;

void
set_is_user_view_p(bool user_view_p)
{
    is_user_view_p = user_view_p;
}

static bool prettyprint_with_attachments_p = FALSE;

void
set_prettyprint_with_attachments(bool attachments_p)
{
    prettyprint_with_attachments_p = attachments_p;
}


static text
text_statement_any_effect_type(entity module, int margin, statement stat)
{
    list l_eff = NIL;
    text l_eff_text;

    if (is_user_view_p)
    {
	statement i;

	if (!statement_undefined_p
	    (i = apply_number_to_statement(nts, statement_number(stat))))
	{
	    l_eff = load_rw_effects_list(i);
	}
	else
	    l_eff = (list) HASH_UNDEFINED_VALUE;
    }
    else
    {
	l_eff = load_rw_effects_list(stat);
	ifdebug(1)
	    {
		if (l_eff != (list) HASH_UNDEFINED_VALUE &&
		    l_eff != list_undefined) 
		{
		    pips_debug(1, "current effects:\n");
		    (*effects_prettyprint_func)(l_eff);
		}
	    }
    }

    l_eff_text = (*effects_to_text_func)(l_eff);

    /* (*attach_effects_decoration_to_text_func)(the_effect_text); */

    return l_eff_text;
}


text
get_any_effect_type_text(
    string module_name,
    string resource_name,
    string summary_resource_name,
    bool give_code_p)
{
    list l_summary_eff = list_undefined;
    entity module;
    statement
	module_stat,
	user_stat = statement_undefined;
    text txt = make_text(NIL);


    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();

    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, TRUE));
    module_stat = get_current_module_statement();

    (*effects_computation_init_func)(module_name);

    /* load regions corresponding to the current module */
    set_rw_effects((statement_effects) 
		   db_get_memory_resource(resource_name, module_name, TRUE));

    debug_on("EFFECTS_DEBUG_LEVEL");

    if (!string_undefined_p(summary_resource_name))
    {
	l_summary_eff = 
	    effects_to_list((effects) db_get_memory_resource
			    (summary_resource_name, module_name, TRUE));
	ifdebug(1)
	{
	    pips_debug(1, "summary effects:\n");
	    (*effects_prettyprint_func)(l_summary_eff);
	}
    }


    if(is_user_view_p) 
    {
	user_stat =  (statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, TRUE);

	nts = allocate_number_to_statement();
	nts = build_number_to_statement(nts, module_stat);

	ifdebug(1)
	    print_number_to_statement(nts);
    }

    /* prepare the prettyprinting */
    init_prettyprint(text_statement_any_effect_type);

    /* summary regions first */
    MERGE_TEXTS(txt, (*effects_to_text_func)(l_summary_eff));

    if (give_code_p)
	/* then code with effects, using text_statement_any_effect_type */
	MERGE_TEXTS(txt, text_module(module,
				     is_user_view_p? user_stat : module_stat));

    if(is_user_view_p)
    {
	hash_table_free(nts);
	nts = hash_table_undefined;
    }

    close_prettyprint();

    debug_off();

    reset_current_module_entity();
    reset_current_module_statement();
    reset_rw_effects(); 
    (*effects_computation_reset_func)(module_name);

    return txt;
}

bool
print_source_or_code_with_any_effects_engine(
    string module_name,
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
    file_resource_name = 
	get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	DBR_GRAPH_PRINTED_FILE : 
	    (is_user_view_p ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE);

    if (prettyprint_with_attachments_p)
	begin_attachment_prettyprint();
    
    success = make_text_resource
	(module_name,
	 file_resource_name,
	 file_name,
	 get_any_effect_type_text(module_name,
				  resource_name,
				  summary_resource_name,
				  TRUE));

    if (prettyprint_with_attachments_p)
	end_attachment_prettyprint();

    return(TRUE);
}


/* 
 * made from words_reference
 * this function can print entity_name instead of entity_local_name,
 * when the entity is not called in the current program.
 */
list effect_words_reference(reference obj)
{
    cons *pc = NIL;
    string begin_attachment;
    entity e = reference_variable(obj);

    pc = CHAIN_SWORD(pc, entity_minimal_name(e));
    begin_attachment = STRING(CAR(pc));

    if (reference_indices(obj) != NIL) {
	pc = CHAIN_SWORD(pc,"(");
	MAPL(pi, {
	    pc = gen_nconc(pc, words_expression(EXPRESSION(CAR(pi))));
	    if (CDR(pi) != NIL)
		pc = CHAIN_SWORD(pc,",");
	}, reference_indices(obj));
	pc = CHAIN_SWORD(pc,")");
    }
    else {
	int d;
	if( (d=variable_entity_dimension(reference_variable(obj))) != 0) {
	    int i;
	    pc = CHAIN_SWORD(pc,"(*");
	    for(i = 1; i < d; i++)
		pc = CHAIN_SWORD(pc,",*");
	    pc = CHAIN_SWORD(pc,")");
	}
    }
    attach_reference_to_word_list(begin_attachment, STRING(CAR(gen_last(pc))),
				  obj);

    return(pc);
}
