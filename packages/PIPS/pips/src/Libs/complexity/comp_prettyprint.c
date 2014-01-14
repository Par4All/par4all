/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "database.h"     /* useful */
#include "resources.h"    /* useful  */

#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"      /* useful */

#include "matrice.h"      /* useful */
#include "properties.h"
#include "misc.h"
#include "prettyprint.h"
#include "text-util.h"
#include "top-level.h"
#include "complexity.h"

static bool is_user_view;	/* print_code_complexities or print_source_complexities */
static hash_table nts = hash_table_undefined;

text text_complexity(entity module __attribute__ ((__unused__)),
		     int margin __attribute__ ((__unused__)),
		     statement stat)
{
    complexity stat_comp = complexity_undefined;
    int print_stats_level = get_int_property("COMPLEXITY_PRINT_STATISTICS");
    bool print_stats_p = ((print_stats_level == 2) ||
			  ((print_stats_level == 1) &&
			   !statement_call_p(stat)));
    cons *pc ;
    char *r ;
    int nblanks ;
    instruction ins = statement_instruction(stat);
    char *s;
    text t = make_text(NIL);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "statement %s, ordering (%td %td)\n", 
		module_local_name(statement_label(stat)),
		ORDERING_NUMBER(statement_ordering(stat)), 
		ORDERING_STATEMENT(statement_ordering(stat)));
    }

    if(is_user_view) {
	statement i = apply_number_to_statement(nts, statement_number(stat));

	if(!statement_undefined_p(i)) {
	    stat_comp = load_statement_complexity(i);
	}
	else
	    stat_comp = (complexity) HASH_UNDEFINED_VALUE;
    }
    else
	stat_comp = load_statement_complexity(stat);

    if(stat_comp != (complexity) HASH_UNDEFINED_VALUE) {
      string it = string_undefined;

	complexity_check_and_warn("text_complexity", stat_comp);
	pc = CHAIN_SWORD(NIL, complexity_sprint(stat_comp, print_stats_p,
						PRINT_LOCAL_NAMES));
	r = words_to_string(pc);
	nblanks = 65-strlen(r);

	if (nblanks<1)
	    nblanks = 1;

	//Becher Molka : Replacing the control structure 'if' by selective structure 'switch case' +  Updating of instruction's tags.
    switch (instruction_tag(ins))
    {
        case is_instruction_sequence:
            {
	        it = "(SEQ)";
                break;
            }
        case is_instruction_test:
            {
	        it = "(TEST)";
                break;
            }
        case is_instruction_loop:
            {
                it = "(DO)";
                break;
            }
        case is_instruction_whileloop:
            {
                it = "(WHILE)";
                break;
            }
        case is_instruction_goto:
            {
                it = "(GOTO)";
                break;           
	    }
        case is_instruction_call:
            {
                it = "(STMT)";
                break;
            }
        case is_instruction_unstructured:
            {
                it = "(UNSTR)";
                break;
            }
        case is_instruction_forloop:
            {
                it = "(FOR)";
                break;
            }
        case is_instruction_expression:
	    {
	        it = "(EXPR)";
		break;
	    }
        case is_instruction_multitest:
	    {
	        it = "(MTEST)";
		break;
	    }
        default:
            pips_internal_error("Never occur!");
            break;
    }


	asprintf(&s, "%s    %*s%s %s\n", get_comment_sentinel(), nblanks, "", r, it);

	ADD_SENTENCE_TO_TEXT(t, make_sentence(is_sentence_formatted,
					      s));
    }

    return (t);
}

bool print_code_complexities(module_name)
char *module_name;
{
  is_user_view = false;
  return print_code_or_source_comp(module_name);
}

bool print_source_complexities(module_name)
char *module_name;
{
  is_user_view = true;
  return print_code_or_source_comp(module_name);
}

bool print_code_or_source_comp(module_name)
char *module_name;
{
    bool success = true;
    entity mod;
    statement mod_stat,user_stat;
    char *file_ext = strdup
	(concatenate(
	  is_user_view? ".ucomp" : ".comp",
	  get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? 
	  GRAPH_FILE_EXT : "",
	  NULL));
    char * resource_name =
	get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	    DBR_GRAPH_PRINTED_FILE
		: is_user_view ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE;
    text txt = make_text(NIL);

    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, true) );
    mod_stat = get_current_module_statement();
    set_current_module_entity(module_name_to_entity(module_name) );
    mod = get_current_module_entity();

    if(is_user_view) {
	user_stat =  (statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, true);

	nts = allocate_number_to_statement();
	nts = build_number_to_statement(nts, mod_stat);
    }


    set_complexity_map( (statement_mapping)
	db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));

    init_prettyprint(text_complexity);

    MERGE_TEXTS(txt, text_summary_complexity( get_current_module_entity() ));
    MERGE_TEXTS(txt, text_module(mod, is_user_view ? user_stat : mod_stat));

    close_prettyprint();
    success = make_text_resource_and_free
	(module_name, resource_name, file_ext, txt);
    free(file_ext);

    if(is_user_view) {
	hash_table_free(nts);
	nts = hash_table_undefined;
    }
    reset_complexity_map();
    reset_current_module_entity();
    reset_current_module_statement();

    return success;
}

text text_summary_complexity(module)
entity module;
{
    const char* module_name = module_local_name(module);
    complexity stat_comp = (complexity)
	db_get_memory_resource(DBR_SUMMARY_COMPLEXITY, module_name, true);
    cons *pc = CHAIN_SWORD(NIL, complexity_sprint(stat_comp, false,
						  PRINT_LOCAL_NAMES));
    char *r = words_to_string(pc);
    int nblanks = 65-strlen(r);
    char *s;
    text t = make_text(NIL);

    if (nblanks < 1)
	nblanks = 1;
    asprintf(&s, "%s    %*s%s (SUMMARY)\n",
	     fortran_module_p(module)? "C" : "//",
	     nblanks, "", r);
    ADD_SENTENCE_TO_TEXT(t, make_sentence(is_sentence_formatted,
					  s));

    return (t);
}

text get_text_complexities(module_name)
const char *module_name;
{
    /* FI: different kind of complexities should later be made
       available.  Instead of the module intrinsic complexity, it would
       be interesting to have its contextual complexity. The same is
       true for the icfg
       */
    entity module = module_name_to_entity(module_name);

    return text_summary_complexity(module);
}
