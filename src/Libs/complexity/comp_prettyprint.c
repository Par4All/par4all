/* comp_prettyprint.c */

/* Modif:
  -- entity_local_name is replaced by module_local_name. LZ 230993
*/

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "complexity_ri.h"
#include "database.h"     /* useful */
#include "pipsdbm.h"      /* useful */
#include "resources.h"    /* useful  */
#include "ri-util.h"
#include "text.h"
#include "matrice.h"      /* useful */
#include "complexity.h"
#include "properties.h"
#include "misc.h"
#include "prettyprint.h"
#include "text-util.h"
#include "top-level.h"

static bool is_user_view;	/* print_code_complexities or print_source_complexities */
static hash_table nts = hash_table_undefined;

text text_complexity(module, margin, stat)
entity module;
int margin;
statement stat;
{
    complexity stat_comp = load_statement_complexity(stat);
    int print_stats_level = get_int_property("COMPLEXITY_PRINT_STATISTICS");
    bool print_stats_p = ((print_stats_level == 2) ||
			  ((print_stats_level == 1) &&
			   !statement_call_p(stat)));
    cons *pc ;
    char *r ;
    int nblanks ;
    instruction ins = statement_instruction(stat);
#define TEXT_COMPLEXITY_BUFFER_SIZE 1024
    static char s[TEXT_COMPLEXITY_BUFFER_SIZE];
    text t = make_text(NIL);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "statement %s, ordering (%d %d)\n", 
		module_local_name(statement_label(stat)),
		ORDERING_NUMBER(statement_ordering(stat)), 
		ORDERING_STATEMENT(statement_ordering(stat)));
    }

    complexity_check_and_warn("text_complexity", stat_comp);
    pc = CHAIN_SWORD(NIL, complexity_sprint(stat_comp, print_stats_p,
					    PRINT_LOCAL_NAMES));
    r = words_to_string(pc);
    nblanks = 65-strlen(r);

    if (nblanks<1) 
	nblanks = 1;
    if ( instruction_block_p(ins) )
	sprintf(s, "C    %*s%s (BLOCK)\n", nblanks, "", r);
    else if ( instruction_test_p(ins) )
	sprintf(s, "C    %*s%s (TEST) \n", nblanks, "", r);
    else if ( instruction_loop_p(ins) )
	sprintf(s, "C    %*s%s (DO)   \n", nblanks, "", r);
    else if ( instruction_call_p(ins) )
	sprintf(s, "C    %*s%s (STMT) \n", nblanks, "", r);
    else if ( instruction_unstructured_p(ins) )
	sprintf(s, "C    %*s%s (UNSTR)\n", nblanks, "", r);
    else
	pips_error("text_complexity", "Never occur!");

    pips_assert("text_complexity", strlen(s) < TEXT_COMPLEXITY_BUFFER_SIZE);

    ADD_SENTENCE_TO_TEXT(t, make_sentence(is_sentence_formatted,
					  strdup(s)));

    return (t);
}

void print_code_complexities(module_name)
char *module_name;
{
  is_user_view = FALSE;
  print_code_or_source_comp(module_name);
}

void print_source_complexities(module_name)
char *module_name;
{
  is_user_view = TRUE;
  print_code_or_source_comp(module_name);
}

void print_code_or_source_comp(module_name)
char *module_name;
{
    FILE *fd;
    entity mod;
    statement mod_stat,user_stat;
    char *fn = strdup(concatenate(db_get_current_program_directory(), 
				  "/",
				  module_name,
				  is_user_view? ".ucomp" : ".comp",
                                  get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? GRAPH_FILE_EXT : "",
				  NULL));
    text txt = make_text(NIL);

    
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    mod_stat = get_current_module_statement();
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    mod = get_current_module_entity();
  
    if(is_user_view) {
	user_stat =  (statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, TRUE);

	nts = allocate_number_to_statement();
	nts = build_number_to_statement(nts, mod_stat);
    }

  
    set_complexity_map( (statement_mapping)
	db_get_memory_resource(DBR_COMPLEXITIES, module_name, TRUE));

    init_prettyprint(text_complexity);

    fd = (FILE *)safe_fopen(fn, "w");
    MERGE_TEXTS(txt, text_summary_complexity( get_current_module_entity() ));
    MERGE_TEXTS(txt, text_module(mod,get_current_module_statement()));

    print_text(fd, txt);
    safe_fclose(fd, fn);

    DB_PUT_FILE_RESOURCE(get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? DBR_GRAPH_PRINTED_FILE
                         : is_user_view ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE,
			 strdup(module_name),
			 fn);

    if(is_user_view) {
	hash_table_free(nts);
	nts = hash_table_undefined;
    }
    reset_complexity_map();
    reset_current_module_entity();
    reset_current_module_statement();
}

text text_summary_complexity(module)
entity module;
{
    string module_name = module_local_name(module);
    complexity stat_comp = (complexity)
	db_get_memory_resource(DBR_SUMMARY_COMPLEXITY, module_name, TRUE);
    cons *pc = CHAIN_SWORD(NIL, complexity_sprint(stat_comp, FALSE,
						  PRINT_LOCAL_NAMES));
    char *r = words_to_string(pc);
    int nblanks = 65-strlen(r);
#define TEXT_SUMMARY_COMPLEXITY 1024
    static char s[TEXT_SUMMARY_COMPLEXITY];
    text t = make_text(NIL);

    if (nblanks < 1) 
	nblanks = 1;
    sprintf(s, "C    %*s%s (SUMMARY)\n", nblanks, "", r);
    pips_assert("text_summary_complexity", strlen(s) < TEXT_SUMMARY_COMPLEXITY);
    ADD_SENTENCE_TO_TEXT(t, make_sentence(is_sentence_formatted,
					  strdup(s)));

    return (t);
}












