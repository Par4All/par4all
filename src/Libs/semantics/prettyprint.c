 /* package semantics - prettyprint interface */

#include <stdio.h>
extern int fprintf();
#include <string.h>
/* #include <stdlib.h> */

#include "types.h"

#include "genC.h"

#include "text.h"
#include "text-util.h"

#include "constants.h"

#include "ri.h"
#include "ri-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"

#include "prettyprint.h"

#include "transformer.h"
#include "effects.h"

#include "semantics.h"

DEFINE_CURRENT_MAPPING(semantic, transformer)

static bool is_transformer;
static bool is_user_view;
static hash_table nts = hash_table_undefined;

void print_code_transformers(module_name)
char *module_name;
{
    is_user_view = FALSE;
    is_transformer = TRUE;
    print_code_semantics(module_name);
}

void print_code_preconditions(module_name)
char *module_name;
{
    is_user_view = FALSE;
    is_transformer = FALSE;
    print_code_semantics(module_name);
}

void print_source_transformers(module_name)
char *module_name;
{
    is_user_view = TRUE;
    is_transformer = TRUE;
    print_code_semantics(module_name);
}

void print_source_preconditions(module_name)
char *module_name;
{
    is_user_view = TRUE;
    is_transformer = FALSE;
    print_code_semantics(module_name);
}

void print_code_semantics(module_name)
char *module_name;
{
    text r = make_text(NIL);
    char *filename;
    FILE *fd;
    entity mod;
    statement mod_stat;
    transformer summary = transformer_undefined;
    statement user_stat;

    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    mod = get_current_module_entity();

    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    mod_stat = get_current_module_statement();

    if(is_user_view) {
	user_stat =  (statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, TRUE);

	debug_on("SEMANTICS_DEBUG_LEVEL");

	nts = allocate_number_to_statement();
	nts = build_number_to_statement(nts, mod_stat);

	ifdebug(1) {
	    print_number_to_statement(nts);
	}
	debug_off();
    }

    set_semantic_map( (statement_mapping)
	db_get_memory_resource(
			       is_transformer? DBR_TRANSFORMERS 
			       : DBR_PRECONDITIONS,
			       module_name,
			       TRUE) );

    summary = (transformer)
	db_get_memory_resource(
			       is_transformer? DBR_SUMMARY_TRANSFORMER
			       : DBR_SUMMARY_PRECONDITION,
			       module_name,
			       TRUE);

    if(string_undefined_p((string) summary)) {
	pips_error("print_code_semantics",
		   "Summary information %s unavailable\n",
		   is_transformer? "DBR_SUMMARY_TRANSFORMER"
		   : "DBR_SUMMARY_PRECONDITION");
    }

    /* still necessary ? BA, September 1993 */
    set_cumulated_effects_map( effectsmap_to_listmap((statement_mapping) 
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE)));
    
    filename = strdup(concatenate(db_get_current_program_directory(), 
				  "/",
				  module_name,
				  is_transformer?
				  (is_user_view? USER_TRANSFORMER_SUFFIX : SEQUENTIAL_TRANSFORMER_SUFFIX ) :
				  (is_user_view? USER_PRECONDITION_SUFFIX : SEQUENTIAL_PRECONDITION_SUFFIX),
				  NULL));
    fd = safe_fopen(filename, "w");

    init_prettyprint(semantic_to_text);

    debug_on("SEMANTICS_DEBUG_LEVEL");

    module_to_value_mappings(mod);

    /* initial version; to be used again when prettyprint really prettyprints*/
    /* print_text(fd, text_statement(mod, 0, mod_stat)); */

    /* new version */
    ADD_SENTENCE_TO_TEXT(r, 
			 make_sentence(is_sentence_formatted,
				       is_transformer? 
				       transformer_to_string(summary)
				       : precondition_to_string(summary)));

    debug_off();

    ADD_SENTENCE_TO_TEXT(r, 
			 make_sentence(is_sentence_formatted, 
				       code_decls_text(entity_code(mod))));
    MERGE_TEXTS(r, text_statement(mod, 0, is_user_view? user_stat:mod_stat));
    ADD_SENTENCE_TO_TEXT(r, sentence_tail());
    print_text(fd, r);
    /* end of new version */

    safe_fclose(fd, filename);

    /* Let's assume that value_mappings will be somehow sometime freed... */

    DB_PUT_FILE_RESOURCE(is_user_view? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE,
			 strdup(module_name),
 			 filename);

    if(is_user_view) {
	hash_table_free(nts);
	nts = hash_table_undefined;
    }
    reset_cumulated_effects_map();
    reset_semantic_map();
    reset_current_module_entity();
    reset_current_module_statement();
}

/* this function name is VERY misleading - it should be changed, sometime FI */
text semantic_to_text(module, margin, stmt)
entity module;
int margin;
statement stmt;
{
    transformer t;
    text txt = make_text(NIL);

    if(is_user_view) {
	statement i = (statement) hash_get(nts, (char *) statement_number(stmt));

	if(i!=(statement) HASH_UNDEFINED_VALUE) {
	    t = load_statement_semantic(i);
	}
	else
	    t = (transformer) HASH_UNDEFINED_VALUE;
    }
    else
	t = load_statement_semantic(stmt);

    if(t != (transformer) HASH_UNDEFINED_VALUE) {
	ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
						is_transformer? 
						transformer_to_string(t)
						: precondition_to_string(t)));
    }
    return txt; 
}



/* ---------------------------------------------------------------- */
/* to convert strings containing predicates to text of commentaries */
/* BA, april 1994                                                   */
/* ---------------------------------------------------------------- */

#define MAX_PRED_COMMENTARY_STRLEN 70
#define MAX_PRED_STRLEN (MAX_PRED_COMMENTARY_STRLEN - 3)


/* text words_predicate_to_commentary(list w_pred)
 * input    : a list of strings, one of them representing a predicate.
 * output   : a text of several lines of commentaries containing 
 *            this list of strings.
 * modifies : nothing.
 */
text words_predicate_to_commentary(w_pred)
list w_pred ;
{
    text t_pred = make_text(NIL);
    string str_pred, str_suiv;
    char str_tmp[MAX_PRED_STRLEN];
    int len, new_str_pred_len, longueur_max;
    sentence sent_pred;
    boolean premiere_ligne = TRUE;
    
    longueur_max = MAX_PRED_STRLEN;

    /* str_pred is the string corresponding to the concatenation
     * of the strings in w_pred */
    str_pred = words_to_string(w_pred);
    
    /* if str_pred is too long, it must be splitted in several lines; 
     * the hyphenation must be done only between the constraints of the
     * predicate, when there is a ",". A space is added at the beginning
     * of extra lines, for indentation. */
    while((len = strlen(str_pred)) > 0) {
	if (len > longueur_max) {

	    /* search the maximal substring which length 
	     * is less than longueur_max */
	    str_tmp[0] = '\0';
	    (void) strncat(str_tmp, str_pred, longueur_max);

	    str_suiv = strrchr(str_tmp, ',');
	    new_str_pred_len = (strlen(str_tmp) - strlen(str_suiv)) + 1;
	    str_suiv = strdup(&(str_pred[new_str_pred_len]));

	    str_tmp[0] = '\0';
	    if (!premiere_ligne)
		(void) strcat(str_tmp, " "); 
	    (void) strncat(str_tmp, str_pred, new_str_pred_len);

	    /* add it to the text */
	    ADD_SENTENCE_TO_TEXT(t_pred, sent_pred = make_pred_commentary_sentence(strdup(str_tmp)));
	    str_pred =  str_suiv;
	}
	else {
	    /* if the remaining string fits in one line */
	    str_tmp[0] = '\0';
	    if (!premiere_ligne)
		(void) strcat(str_tmp, " "); 
	    (void) strcat(str_tmp, str_pred);

	    ADD_SENTENCE_TO_TEXT(t_pred, make_pred_commentary_sentence(str_tmp));
	    str_pred[0] = '\0';
	}
	
	if (premiere_ligne) {
	    premiere_ligne = FALSE;
	    longueur_max = longueur_max - 1;
	}
    }
    
    return(t_pred);
}


/* sentence make_pred_commentary_sentence(string str_pred) 
 * input    : a substring formatted to be a commentary
 * output   : a sentence, containing the commentary form of this string.
 * modifies : nothing
 */
sentence make_pred_commentary_sentence(str_pred)
string str_pred;
{
    char str_tmp[MAX_PRED_COMMENTARY_STRLEN + 1];
    sentence sent_pred;

    str_tmp[0] = '\0';
    (void) strcat(str_tmp, "C  "); 
    (void) strcat(str_tmp, str_pred);
    (void) strcat(str_tmp, "\n"); 

    sent_pred = make_sentence(is_sentence_formatted, strdup(str_tmp));
    free(str_tmp);
    return(sent_pred);
}

