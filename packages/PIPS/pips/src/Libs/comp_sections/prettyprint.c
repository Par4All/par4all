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
/* print complementary sections */
#include <stdlib.h>
#include <stdlib.h>

#include "all.h"
#include "text-util.h"
#include "text.h"
#include "prettyprint.h"
#include "top-level.h"

/* {{{ banner */

/* 
 * package comp_regions : Alexis Platonoff, 5 September 1990,  
 *                   Beatrice Creusillet, April 1994
 *
 * -------------- 
 * prettyprint.c
 * --------------
 *
 * This file contains the functions specific to the prettyprinting of comp_regions.
 *
 */

/* }}} */


#define REGION_BUFFER_SIZE 2048

#define REGION_FORESYS_PREFIX "C$REG"
#define PIPS_NORMAL_PREFIX "C"

/* {{{ function prototype */

static bool in_out_comp_regions_p = false;
static bool is_user_view_p = false;
static hash_table nts = hash_table_undefined;

static text text_statement_array_comp_regions(entity, int, statement);
static text text_array_comp_regions(list);
static bool print_code_with_comp_regions(const char*, string, string, string);
static text get_any_comp_regions_text(const char*, string, string, bool);

/* }}} */

text get_text_comp_regions(const char* module_name)
{
    is_user_view_p = false;
    in_out_comp_regions_p = false;

    return get_any_comp_regions_text(module_name,
				DBR_REGIONS,
				DBR_SUMMARY_REGIONS,
				false);
}


/* bool print_source_regions(const char* module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the original source code with the corresponding regions.	
 */
bool print_source_comp_regions(module_name)
const char* module_name;
{
    is_user_view_p = true;
    in_out_comp_regions_p = false;

    return print_code_with_comp_regions(module_name,
				   DBR_REGIONS,
				   DBR_SUMMARY_REGIONS,
				   USER_REGION_SUFFIX);
}


/* bool print_code_comp_regions(const char* module_name)
 * input    : the name of the current module
 * modifies : nothing.
 * comment  : prints the source code with the corresponding regions.	
 */
bool print_code_comp_regions(module_name)
const char* module_name;
{
    is_user_view_p = false;
    in_out_comp_regions_p = false;

    return print_code_with_comp_regions(module_name,
				   DBR_COMPSEC,
				   DBR_SUMMARY_COMPSEC,
				   SEQUENTIAL_COMPSEC_SUFFIX);
}

/* bool print_code_with_comp_regions(const char* module_name, list summary_comp_regions)
 * input    : the name of the current module, the name of the region and
 *            summary region resources and the file suffix
 *            the comp_regions are in the global variable local_regions_map.
 * modifies : nothing.
 * comment  : prints the source code with the corresponding comp_regions.	
 */
static bool print_code_with_comp_regions(const char* module_name,
				    string resource_name,
				    string summary_resource_name,
				    string file_suffix)
{
    char *file_name, *file_resource_name;

    file_name = strdup(concatenate(file_suffix,
                                  get_bool_property
				  ("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? 
				  GRAPH_FILE_EXT : "",
                                  NULL));
    file_resource_name = get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	DBR_GRAPH_PRINTED_FILE : 
	    (is_user_view_p ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE);

    bool success = make_text_resource(module_name, file_resource_name,
				 file_name,
				 get_any_comp_regions_text(module_name,
				    resource_name,
				    summary_resource_name,
				    true));

    free(file_name);
    return success ;
}

/*{{{  get any comp_regions text*/
static text get_any_comp_regions_text(const char* module_name,
			     string resource_name,
			     string summary_resource_name,
			     bool give_code_p)
{
    list summary_comp_regions;
    entity module;
    statement module_stat, user_stat = statement_undefined;
    text txt = make_text(NIL);

    debug_on("COMP_REGIONS_DEBUG_LEVEL");

    /* load comp_regions corresponding to the current module */
    /* change later */

    set_local_comp_regions_map(comp_secs_map_to_listmap
			  ((statement_mapping) 
			   db_get_memory_resource
			   (resource_name, module_name, true)) );

    summary_comp_regions = 
	     comp_desc_set_to_list((comp_desc_set) db_get_memory_resource
			(summary_resource_name,
			 module_name, true));
    
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

	ifdebug(5)
	{
	    print_number_to_statement(nts);
	}
    }

    /* prepare the prettyprinting */
    /* summary comp_regions first */
    init_prettyprint(text_statement_array_comp_regions);
    MERGE_TEXTS(txt, text_array_comp_regions(summary_comp_regions));

    if (give_code_p)
	/* then code with comp_regions, using text_array_comp_regions */
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
    free_local_comp_regions_map(); 
    reset_cumulated_rw_effects();

    return txt;
}
/*}}}*/
/*{{{  text statement array comp_regions*/
/* static text text_statement_array_comp_regions(entity module, int margin, statement stat)
 * output   : a text representing the list of array comp_regions associated with the
 *            statement stat.
 * comment  : if the number of array comp_regions is not nul, then empty lines are
 *            added before and after the text of the list of comp_regions.
 */
static text
text_statement_array_comp_regions(entity __attribute__ ((unused)) module,
				  int __attribute__ ((unused)) margin,
				  statement stat)
{
    list l_reg = NIL;

    if (is_user_view_p) {
	statement i;
 
	i = (statement) hash_get(nts, (char *) statement_number(stat));

	if (i != (statement) HASH_UNDEFINED_VALUE) {
	    l_reg = load_statement_local_comp_regions(i);
	}
	else
	    l_reg = (list) HASH_UNDEFINED_VALUE;
    }
    else
	l_reg = load_statement_local_comp_regions(stat);

    
    /* Necessary because of unreachable statements - In this case, no comp_regions
     * are stored in the statement_mapping, and their values are thus 
     * HASH_UNDEFINED_VALUE or list_undefined. BC. 25/07/95. */
    /* GO 31/7/95: I replace it by a different test in text_array_comp_regions */

    return text_array_comp_regions(l_reg);
}

/*}}}*/
/*{{{  text array comp_regions*/

/* static text text_array_comp_regions(list l_reg)
 * input    : a list of comp_regions
 * output   : a text representing this list of comp_regions.
 * comment  : if the number of array comp_regions is not nul, and if 
 *            PRETTYPRINT_LOOSE is true, then empty lines are
 *            added before and after the text of the list of comp_regions.
 */
static text text_array_comp_regions(l_reg)
list l_reg;
{
    text reg_text = make_text(NIL);
    /* in case of loose_prettyprint, at least one region to print? */
    bool loose_p = get_bool_property("PRETTYPRINT_LOOSE");
    bool one_p = false;  

    /* GO: No redundant test anymore, see  text_statement_array_comp_regions */
    if (l_reg != (list) HASH_UNDEFINED_VALUE && l_reg != list_undefined) 
    {
/*
	MAP(COMP_DESC, reg,
	{
	    entity ent = effect_entity(reg);
	    if ( get_bool_property("PRETTYPRINT_SCALAR_comp_regions") || 
		! entity_scalar_p(ent)) 
	    {
		if (loose_p && !one_p )
		{
		    ADD_SENTENCE_TO_TEXT(reg_text, 
					 make_sentence(is_sentence_formatted, 
						       strdup("\n")));
		    one_p = true;
		}
		MERGE_TEXTS(reg_text, text_comp_region(reg));
	    }	
	},
	    l_reg);
	    */

	if (loose_p && one_p)
	    ADD_SENTENCE_TO_TEXT(reg_text, 
				 make_sentence(is_sentence_formatted, 
					       strdup("\n")));
    }
    return(reg_text);
}
/*}}}*/
/*{{{  text all comp_regions*/

/* text text_all_comp_regions(list l_reg)
 * input    : a list of comp_regions
 * output   : a text representing this list (with non-array comp_regions)
 */
text text_all_comp_regions(l_reg)
list l_reg;
{
    text reg_text = make_text(NIL);

    MAP(EFFECT, reg,
    {
	MERGE_TEXTS(reg_text, text_region(reg));
    },
	l_reg);
    return(reg_text);
}
/*}}}*/
/*{{{  text comp_regions*/
/* text text_comp_regions(list l_reg)
 * input    : a list of comp_regions
 * output   : a text representing this list (with non-array comp_regions)
 */
text text_comp_regions(l_reg)
list l_reg;
{

    text reg_text = make_text(NIL);
  /* change later */
    return(reg_text);

    MAP(EFFECT, reg,
    {
	entity ent = effect_entity(reg);
	if (! entity_scalar_p(ent)) 
	{	    
	    MERGE_TEXTS(reg_text, text_comp_region(reg));
	}	
    },
	l_reg);       

   return(reg_text);
}
/*}}}*/
/*{{{  text region*/
/* text text_region(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries, 
 *            representing the region
 * modifies : nothing
 */
text text_comp_region(reg)
effect reg;
{
    text t_reg = make_text(NIL);
    bool foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    string str_prefix;

    if (foresys)
	str_prefix = REGION_FORESYS_PREFIX;
    else
	str_prefix = PIPS_NORMAL_PREFIX;
    
    if(reg == effect_undefined)
    {
	ADD_SENTENCE_TO_TEXT(t_reg, 
			     make_pred_commentary_sentence(strdup("<REGION_UNDEFINED>"),
							   str_prefix));
	user_log("[region_to_string] unexpected effect undefined\n");
    }
    else
    {
	free_text(t_reg);
	t_reg = words_predicate_to_commentary(words_effect(reg), str_prefix);
    }

    return(t_reg);   
}
#if 0

/* list words_comp_region(comp_desc Dad)
 * input    : a region.
 * output   : a list of strings representing the region.
 * modifies : nothing.
 * comment  :	because of 'buffer', this function cannot be called twice before
 * its output is processed. Also, overflows in relation_to_string() 
 * cannot be prevented. They are checked on return.
 */
list words_comp_region(comp_desc Dad)
{
  char buffer[REGION_BUFFER_SIZE];

  int Rank;
  int i;
  tag RefType;
  list pc = NIL;

  Rank = context_info_rank(simple_section_context
			   (comp_sec_hull(comp_desc_section(Dad))));

  buffer[0] = '\0';


  sprintf(buffer, "\nReference Template :: \n[ ");
  for (i=0; i < Rank; i++) {
    RefType = GetRefTemp(comp_desc_section(Dad),i);
    switch(RefType) {
      case is_rtype_lininvariant:
        		sprintf(buffer, "INV");
						break;
      case is_rtype_linvariant:
        		sprintf(buffer, "VAR");
						break;
      case is_rtype_nonlinear:
        		sprintf(buffer, "NON");
						break;
      default:
        		sprintf(buffer, "NUL");
						break;
    }

		if( i < (Rank-1) ) 
			sprintf(buffer, ", ");
		else
	  	sprintf(buffer, " ]\n");
  }
  sprintf(buffer,"\n");
   
  pc = CHAIN_SWORD(pc, buffer);

  return(pc);
}
#endif

/*}}}*/









