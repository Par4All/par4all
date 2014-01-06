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
/* package continuation :  Be'atrice Creusillet, 1996
 *
 * This File contains the functions to prettyprint continuation conditions 
 * of a module (over- and under-approximations.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"

#include "text.h"
#include "text-util.h"

#include "top-level.h"

#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"
#include "properties.h"

#include "prettyprint.h"

#include "transformer.h"
#include "effects-generic.h"

#include "semantics.h"
#include "continuation.h"

#define PREC_FORESYS_PREFIX "C$PREC"
#define TRAN_FORESYS_PREFIX "C$TRAN"

static bool is_user_view_p;
static hash_table nts = hash_table_undefined;

/* The strange argument type is required by qsort(), deep down in the calls */
static int 
is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
    /* The constant term is given the highest weight to push constant
       terms at the end of the constraints and to make those easy
       to compare. If not, constant 0 will be handled differently from
       other constants. However, it would be nice to give constant terms
       the lowest weight to print simple constraints first...

       Either I define two comparison functions, or I cheat somewhere else.
       Let's cheat? */
    int is_equal = 0;
    
    if (term_cst(*pvarval1) && !term_cst(*pvarval2))
	is_equal = 1;
    else if (term_cst(*pvarval1) && term_cst(*pvarval2))
	is_equal = 0;
    else if(term_cst(*pvarval2))
	is_equal = -1;
    else
	is_equal = 
	    strcmp(pips_user_value_name((entity) vecteur_var(*pvarval1)),
		   pips_user_value_name((entity) vecteur_var(*pvarval2)));


    return is_equal; 
}

#define continuation get_comment_continuation()
#define append(s) add_to_current_line(crt_line, s, continuation, txt)

/* text text_continuation(transformer tran) 
 * input    : a transformer representing a transformer or a precondition 
 * output   : a text containing commentaries representing the transformer
 * modifies : nothing.
 *
 * Modification: AP, Nov 10th, 1995. Instead of building a (very long)
 * string, I directly use the transformer to build the prettyprint in text
 * format. This is to avoid the problem occuring when the buffer used in
 * transformer[precondition]_to_string() is too small. I also use a static
 * buffer to build each constraint; we are restricted to constraints of
 * lengths smaller than the line length.
 */
static text 
text_continuation(transformer cont, bool is_must)
{
    text txt = make_text(NIL);
    char crt_line[MAX_LINE_LENGTH];

    crt_line[0] = '\0'; 
    append(get_comment_sentinel());
    append(" ");

    if(cont != (transformer) HASH_UNDEFINED_VALUE ) 
    {
	if(cont==transformer_undefined)
	{
	    append(" CONTINUATION: TRANSFORMER_UNDEFINED");
	}
	else
	{  
	    Psysteme ps = predicate_system(transformer_relation(cont));
	    sc_lexicographic_sort(ps, is_inferior_pvarval); 

	    append(is_must? "C-MUST-":"C-MAY-");
	    system_text_format(crt_line, continuation, txt, ps, 
			       (get_variable_name_t) pips_user_value_name,
			       false);
	}
      
	close_current_line(crt_line, txt,continuation);
    }
    /* else an empty text is returned. 
     */

    return txt; 
}

static text 
text_continuation_conditions(
    transformer must_cont_t,
    transformer may_cont_t) 
{
    
    text cont_text = make_text(NIL);
    bool loose_p = get_bool_property("PRETTYPRINT_LOOSE");
    
    if ((must_cont_t ==(transformer) HASH_UNDEFINED_VALUE) &&
	(may_cont_t ==(transformer) HASH_UNDEFINED_VALUE) )
	return(cont_text);

    if (loose_p)
    {
	ADD_SENTENCE_TO_TEXT(cont_text, 
			     make_sentence(is_sentence_formatted, 
					   strdup("\n")));
    }
    
    /* First: must continuation conditions */
    if (must_cont_t !=(transformer) HASH_UNDEFINED_VALUE)
    {
	MERGE_TEXTS(cont_text, text_continuation(must_cont_t,true));
    }
    
    /* Then: may continuation conditions */
    if (may_cont_t !=(transformer) HASH_UNDEFINED_VALUE)
    {
	MERGE_TEXTS(cont_text, text_continuation(may_cont_t,false));
    }
    if (loose_p)
	ADD_SENTENCE_TO_TEXT(cont_text, 
			     make_sentence(is_sentence_formatted, 
					   strdup("\n")));

    return(cont_text);
}

static text 
text_statement_continuation_conditions(
    entity module, 
    int margin,
    statement stat)
{
    transformer must_cont_t, may_cont_t;
    statement s;

    s = is_user_view_p? 
	(statement) hash_get(nts, (char *) statement_number(stat)) :
	stat;

    if (is_user_view_p)
    {
	s = (statement) hash_get(nts, (char *) statement_number(stat));
    }

    
    if (s != (statement) HASH_UNDEFINED_VALUE)
    {
	must_cont_t = load_statement_must_continuation(s);
	may_cont_t = load_statement_may_continuation(s);
    }
    else
    {
	must_cont_t = (transformer) HASH_UNDEFINED_VALUE;
	may_cont_t = (transformer) HASH_UNDEFINED_VALUE;
    }
    
    return text_continuation_conditions(must_cont_t, may_cont_t);
}


static text 
get_continuation_condition_text(const char* module_name, bool give_code_p)
{
    entity module;
    statement module_stat, user_stat = statement_undefined;
    text txt = make_text(NIL);
    transformer must_sum_cont_t, may_sum_cont_t;   

    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();
    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, true));
    module_stat = get_current_module_statement();

    /* To set up the hash table to translate value into value names */       
    set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    module_to_value_mappings(module);


    if(is_user_view_p) 
    {
	user_stat =  (statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, true);

	nts = allocate_number_to_statement();
	nts = build_number_to_statement(nts, module_stat);

	ifdebug(1)
	{
	    print_number_to_statement(nts);
	}
    }

    debug_on("CONTINUATION_DEBUG_LEVEL");

    set_must_continuation_map( (statement_mapping) 
	db_get_memory_resource(DBR_MUST_CONTINUATION, module_name, true) );
    set_may_continuation_map( (statement_mapping) 
	db_get_memory_resource(DBR_MAY_CONTINUATION, module_name, true) );
    must_sum_cont_t = (transformer) 
     db_get_memory_resource(DBR_MUST_SUMMARY_CONTINUATION, module_name, true);
    may_sum_cont_t = (transformer) 
     db_get_memory_resource(DBR_MAY_SUMMARY_CONTINUATION, module_name, true);

    /* prepare the prettyprinting */
    init_prettyprint(text_statement_continuation_conditions);
    /* summary information first */
    MERGE_TEXTS(txt,text_continuation(must_sum_cont_t, true)); 
    MERGE_TEXTS(txt,text_continuation(may_sum_cont_t, false)); 

    if (give_code_p)
	/* then code with regions, 
	 * using text_statement_continuation_conditions */
	MERGE_TEXTS(txt, text_module(module,
				     is_user_view_p? user_stat : module_stat));

    debug_off();

    if(is_user_view_p)
    {
	hash_table_free(nts);
	nts = hash_table_undefined;
    }

    close_prettyprint();

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_must_continuation_map();
    reset_may_continuation_map();
    free_value_mappings();

    return txt;
}

static bool 
print_continuation_conditions(const char* module_name)
{
    char *file_name, *file_resource_name;
    bool success = true;

    file_name = strdup(concatenate(".cont",
				   get_bool_property
				  ("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? 
				  GRAPH_FILE_EXT : "",
                                  NULL));
    file_resource_name = 
	get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	DBR_GRAPH_PRINTED_FILE : 
	(is_user_view_p ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE);

    success = 
	make_text_resource_and_free(
	    module_name,
	    file_resource_name,
	    file_name,
	    get_continuation_condition_text(module_name,true));

    free(file_name);
    return(success);
}

bool print_code_continuation_conditions(const char* module_name)
{
    is_user_view_p = false;
    return print_continuation_conditions(module_name);
}


bool print_source_continuation_conditions(const char* module_name)
{
    is_user_view_p = true;
    return print_continuation_conditions(module_name);
}

