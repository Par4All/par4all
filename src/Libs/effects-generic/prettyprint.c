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

#include "linear.h"

#include "genC.h"

#include "text.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "top-level.h"

#include "database.h"
#include "resources.h"
#include "pipsdbm.h"
#include "properties.h"

#include "text-util.h"
#include "prettyprint.h"

#include "effects-generic.h"

/*****************Written by phamdat**********************/
/*text my_text_statement_any_effect_type(entity module, int margin, statement stat)
{
  text result = make_text(NIL);
  list l;
  for (l=lp; l; POP(l)) {
    p_prettyprint_stuff pps = (p_prettyprint_stuff) STRING(CAR(l));
    MERGE_TEXTS(result, resource_text(module, margin, stat, pps));
  }
  return result;
}*/

text my_get_any_effects_text(string module_name)
{
  entity module;
  statement module_stat;
  text txt = make_text(NIL);

  /* current entity
   */
  set_current_module_entity( local_name_to_top_level_entity(module_name));
  module = get_current_module_entity();

  /* current statement
   */
  set_current_module_statement((statement) db_get_memory_resource
			       (DBR_CODE, module_name, TRUE));
  module_stat = get_current_module_statement();

  /* resources to be prettyprinted...
   */
  load_resources(module_name);
  
  debug_on("EFFECTS_DEBUG_LEVEL");

  /* prepare the prettyprinting */
  init_prettyprint(my_text_statement_any_effect_type);
  
  /* summary regions first */
  MERGE_TEXTS(txt, my_text_summary_any_effect_type(module));
    
  /* then code with effects, using text_statement_any_effect_type */
  MERGE_TEXTS(txt, text_module(module,  module_stat));

  close_prettyprint();
  
  debug_off();
  
  reset_current_module_entity();
  reset_current_module_statement();
  
  return txt;
}

text my_get_text_proper_effects(string module_name)
{
  text t;

  set_is_user_view_p(FALSE);
  set_methods_for_rw_effects_prettyprint(module_name);
  push_prettyprints(resource_name, summary_resource_name);
  /*add_a_generic_prettyprint(DBR_PROPER_EFFECTS, FALSE, effects_to_text_func, effects_prettyprint_func, attach_effects_decoration_to_text_func);*/
  t = my_get_any_effects_text(module_name);
  reset_generic_prettyprints();
  reset_methods_for_effects_prettyprint(module_name);
  return t;
}
/*********************************************************/

/***************************************************** ACTION INTERPRETATION */

static string read_action_interpretation = "read";
static string write_action_interpretation = "write";
void set_action_interpretation(string r, string w)
{
    read_action_interpretation = r;
    write_action_interpretation = w;
}
void reset_action_interpretation(void)
{
    read_action_interpretation = "read";
    write_action_interpretation = "write";
}
string action_interpretation(int tag)
{
    return tag==is_action_read ? 
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


/****************************************************** PRETTYPRINT STUFFS */

typedef struct 
{
    string name;
    bool is_a_summary;
    gen_chunk * resource;
    generic_text_function get_text;
    generic_prettyprint_function prettyprint;
    generic_attachment_function attach;
}
    prettyprint_stuff, *p_prettyprint_stuff;

/* the required prettyprints are stored in the specified list items.
 * there can be any number of them. They are stored. 
 */
static list /* of p_prettyprint_stuff */ lp = NIL;

void 
reset_generic_prettyprints(void)
{
    gen_map(free, lp);
    gen_free_list(lp);
    lp = NIL;
}

void 
set_a_generic_prettyprint(
    string resource_name,
    bool is_a_summary,
    gen_chunk* res,
    generic_text_function tf,
    generic_prettyprint_function tp,
    generic_attachment_function ta)
{
    p_prettyprint_stuff pps = 
	(p_prettyprint_stuff) malloc(sizeof(prettyprint_stuff));

    pps->name         = resource_name;
    pps->is_a_summary = is_a_summary;
    pps->resource     = res;
    pps->get_text     = tf;
    pps->prettyprint  = tp;
    pps->attach       = ta;

    lp = CONS(STRING, (char*) pps, lp); /* hum... */
}

void 
add_a_generic_prettyprint(
    string resource_name,
    bool is_a_summary,
    generic_text_function tf,
    generic_prettyprint_function tp,
    generic_attachment_function ta)
{
    set_a_generic_prettyprint
	(resource_name, is_a_summary, gen_chunk_undefined, tf, tp, ta);
}

static void 
load_resources(string module_name)
{
    list l;
    for (l=lp; l; POP(l))
    {
	p_prettyprint_stuff pps = (p_prettyprint_stuff) STRING(CAR(l));
	pps->resource = 
	    (gen_chunk*) db_get_memory_resource(pps->name, module_name, TRUE);
    }
}

static list 
load_list(statement_effects m, statement s)
{
    return effects_effects(apply_statement_effects(m, s));
}

/********************************************************************* TEXT */

/* returns the text associated to a specified prettyprint and statement 
 */
static text
resource_text(
    entity module, 
    int margin, 
    statement stat,
    p_prettyprint_stuff pps)
{
    list l_eff = NIL;
    text l_eff_text;

    pips_assert("must not be a summary", !pps->is_a_summary);

    if (is_user_view_p)
    {
	statement i;

	if (!statement_undefined_p
	    (i = apply_number_to_statement(nts, statement_number(stat))))
	{
	    l_eff = load_list(pps->resource, i);
	}
	else
	    l_eff = (list) HASH_UNDEFINED_VALUE;
    }
    else
    {
	l_eff = load_list(pps->resource, stat);
	ifdebug(1)
	 {
	     if (l_eff != (list) HASH_UNDEFINED_VALUE &&
		 l_eff != list_undefined) 
	     {
		 pips_debug(1, "current effects:\n");
		 (*(pps->prettyprint))(l_eff);
	     }
	 }
    }

    l_eff_text = (*(pps->get_text))(l_eff);

    /* (*attach_effects_decoration_to_text_func)(the_effect_text); */

    return l_eff_text;
}

/* returns the text of all required summaries
 */
static text
text_summary_any_effect_type(
    entity module)
{
    text result = make_text(NIL);
    list l;
    for (l=lp; l; POP(l))
    {
	p_prettyprint_stuff pps = (p_prettyprint_stuff) STRING(CAR(l));
	if (pps->is_a_summary) {
	    pips_debug(5, "considering resource %s\n", pps->name);
	    MERGE_TEXTS(result, (*(pps->get_text))
			(effects_effects( (effects) pps->resource)));
	}
    }

    return result;
}

/* returns the text of all required effects associated to statement stat
 */
static text
text_statement_any_effect_type(
    entity module,
    int margin,
    statement stat)
{
    text result = make_text(NIL);
    list l;
    for (l=lp; l; POP(l))
    {
	p_prettyprint_stuff pps = (p_prettyprint_stuff) STRING(CAR(l));
	if (!pps->is_a_summary) {
	    pips_debug(5, "considering resource %s\n", pps->name);
	    MERGE_TEXTS(result, resource_text(module, margin, stat, pps));
	    /*{
	      string filename = "/users/tmp/phamdat/textout";
	      FILE * my_file = safe_fopen(filename, "w");
	      print_text(my_file, result);
	      safe_fclose(my_file, filename);
	      free(filename);
	      }*/
	}
    }

    return result;
}

static text
get_any_effects_text(
    string module_name,
    bool give_code_p)
{
    entity module;
    statement module_stat, user_stat = statement_undefined;
    text txt = make_text(NIL);

    /* current entity
     */
    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();

    /* current statement
     */
    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, TRUE));
    module_stat = get_current_module_statement();

    /* resources to be prettyprinted...
     */
    load_resources(module_name);

    debug_on("EFFECTS_DEBUG_LEVEL");

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
    MERGE_TEXTS(txt, text_summary_any_effect_type(module));

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

    return txt;
}

/* generic engine for prettyprinting effects.
 */
bool
print_source_or_code_effects_engine(
    string module_name,
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
    
    success = make_text_resource_and_free
	(module_name,
	 file_resource_name,
	 file_name,
	 get_any_effects_text(module_name, TRUE));

    if (prettyprint_with_attachments_p)
	end_attachment_prettyprint();

    return TRUE;
}




/********************************************************************* MISC */

/* made from words_reference
 * this function can print entity_name instead of entity_local_name,
 * when the entity is not called in the current program.
 */
list /* of string */ effect_words_reference(reference obj)
{
    list pc = NIL;
    string begin_attachment;
    entity e = reference_variable(obj);
   
    if (get_bool_property("PRETTYPRINT_WITH_COMMON_NAMES")  
	&& entity_in_common_p(e)) {
	pc = CHAIN_SWORD(pc, (string) entity_and_common_name(e));
    } else 
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


/************************************************************ OLD INTERFACES */

static void 
push_prettyprints(
    string resource_name,
    string summary_resource_name)
{

    if (!string_undefined_p(resource_name))
	add_a_generic_prettyprint(resource_name, 
				  FALSE, 
				  effects_to_text_func,
				  effects_prettyprint_func, 
				  attach_effects_decoration_to_text_func);

    if (!string_undefined_p(summary_resource_name))
	add_a_generic_prettyprint(summary_resource_name, 
				  TRUE,
				  effects_to_text_func,
				  effects_prettyprint_func, 
				  attach_effects_decoration_to_text_func);
}

/* get the text
 */
text
get_any_effect_type_text(
    string module_name,
    string resource_name,
    string summary_resource_name,
    bool give_code_p)
{
    text txt;
    push_prettyprints(resource_name, summary_resource_name);
    txt = get_any_effects_text(module_name, give_code_p);
    reset_generic_prettyprints();
    return txt;
}

/* initial engine
 */
bool
print_source_or_code_with_any_effects_engine(
    string module_name,
    string resource_name,
    string summary_resource_name,
    string file_suffix)
{
    bool ok;
    push_prettyprints(resource_name, summary_resource_name);
    ok = print_source_or_code_effects_engine(module_name, file_suffix);
    reset_generic_prettyprints();
    return ok;
}
