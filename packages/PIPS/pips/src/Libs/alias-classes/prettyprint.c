/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
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


/* text text_region_no_action(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries, 
 *            representing the region BUT WITHOUT THE ACTION TAG (IN/OUT)
 * modifies : nothing
 * COPIED FROM THE FUNCTION text_region IN FILE effects-convex/prettyprint.c
 * AND MODIFIED TO NOT PRINT ACTION (IN/OUT)
 */
#define append(s) add_to_current_line(line_buffer, s, str_prefix, t_reg)

static text 
text_region_no_action(effect reg)
{
    text t_reg;
    bool foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    string str_prefix = foresys? 
	FORESYS_CONTINUATION_PREFIX: get_comment_continuation();
    char line_buffer[MAX_LINE_LENGTH];
    reference r;
/*    action ac; */
    approximation ap;
    Psysteme sc;
    Pbase sorted_base;
    list /* of string */ ls;

    if(effect_undefined_p(reg))
    {
	user_log("[text_region] unexpected effect undefined\n");
	return
	  make_text(CONS(SENTENCE, make_sentence(is_sentence_formatted,
	    strdup(concatenate(str_prefix, "<REGION_UNDEFINED>\n", NULL))),
			 NIL));
    }
    /* else the effect is defined...
     */

    /* PREFIX
     */
    t_reg = make_text(NIL);
    strcpy(line_buffer, foresys? REGION_FORESYS_PREFIX: get_comment_sentinel());
    if (!foresys) append("  <");

    /* REFERENCE
     */
    r = effect_any_reference(reg);
    ls = foresys? words_reference(r, NIL): effect_words_reference(r);

    MAP(STRING, s, append(s), ls);
    gen_map(free, ls); gen_free_list(ls); ls = NIL;

    /* ACTION and APPROXIMATION
     */
/*    ac = effect_action(reg); */
    ap = effect_approximation(reg);

    if (foresys)
    {
	append(", RGSTAT(");
/*	append(action_read_p(ac) ? "R," : "W,"); */
	append(approximation_may_p(ap) ? "MAY), " : "EXACT), ");
    }
    else /* PIPS prettyprint */
    {
/*	append("-");
	append(action_interpretation(action_tag(ac))); */
	append(approximation_may_p(ap) ? "-MAY" : "-EXACT");
	append("-");
    }

    /* SYSTEM
     * sorts in such a way that constraints with phi variables come first.
     */
    sorted_base = region_sorted_base_dup(reg);
    sc = sc_dup(region_system(reg));
    region_sc_sort(sc, sorted_base);

    system_sorted_text_format(line_buffer, str_prefix, t_reg, sc,
	       (get_variable_name_t) pips_region_user_name,
 vect_contains_phi_p, foresys);

    sc_rm(sc);
    base_rm(sorted_base);

    /* CLOSE */
    if (!foresys) append(">");
    close_current_line(line_buffer, t_reg,str_prefix);

    return t_reg;
}


static text
aliases_text(const char* module_name, string resource_name)
{
    list alias_lists;
    list al = NIL;
    entity module;
    text txt = make_text(NIL);

    pips_debug(4,"module %s resource %s\n",module_name,resource_name);

    alias_lists = effects_classes_classes(
	(effects_classes)
	db_get_memory_resource(resource_name, module_name, true));

    pips_debug(9,"got aliases\n");

    /* ATTENTION: all this is necessary to call module_to_value_mappings
     * to set up the hash table to translate value into value names
     * before the call to text_region below
     */       
    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();
    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, true));
    set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    /* that's it, but we musn't forget to rest everything after the call
     */

    module_to_value_mappings(module);

    pips_debug(9,"hash table set up\n");

/*    set_action_interpretation(ACTION_IN,ACTION_OUT); */

    MAP(EFFECTS,alias_list_effects,
	{
	    list alias_list = effects_effects(alias_list_effects);

	    pips_debug(9,"make text for alias list\n");

	    if (alias_list != (list) HASH_UNDEFINED_VALUE
		&& alias_list != list_undefined) 
	    {
		/* have to copy alias_list here */
		al = alias_list;
		MAP(EFFECT,alias,
		    {
			pips_debug(9,"make text for alias:\n");

			ifdebug(9)
			    {
				set_action_interpretation(ACTION_IN,ACTION_OUT);
				print_region(alias);
				reset_action_interpretation();
			    }

/*		    set_action_interpretation(ACTION_IN,ACTION_OUT);
			MERGE_TEXTS(txt,text_region(alias));
		    reset_action_interpretation();
		    */

			MERGE_TEXTS(txt,text_region_no_action(alias));
		    },
			al);

		ADD_SENTENCE_TO_TEXT(
		    txt,
		    make_sentence(is_sentence_formatted,strdup("\n"))
		    );

		pips_debug(9,"made text for alias list\n");
	    }
	},alias_lists);

    pips_debug(4,"end\n");

/*    reset_action_interpretation(); */
    free_value_mappings();
    reset_cumulated_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();

    return txt;
}


static bool
print_aliases( const char* module_name, string resource_name, string file_extn)
{
    char *file_resource_name;
    bool success = true;

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
print_in_alias_pairs( const char* module_name )
{
    bool success = true;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);

    success = print_aliases(module_name,DBR_IN_ALIAS_PAIRS,".in_alias");

    pips_debug(4,"end\n");
    debug_off();

    return success;
}


bool
print_out_alias_pairs( const char* module_name )
{
    bool success = true;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);

    success = print_aliases(module_name,DBR_OUT_ALIAS_PAIRS,".out_alias");

    pips_debug(4,"end\n");
    debug_off();

    return success;
}

bool
print_alias_lists( const char* module_name )
{
    bool success = true;

    debug_on("ALIAS_LISTS_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);

    success = print_aliases(module_name,DBR_ALIAS_LISTS,".alias_lists");

    pips_debug(4,"end\n");
    debug_off();

    return success;
}

bool
print_alias_classes( const char* module_name )
{
    bool success = true;

    debug_on("ALIAS_CLASSES_DEBUG_LEVEL");
    pips_debug(4,"module %s\n",module_name);


    success = print_aliases(module_name,DBR_ALIAS_CLASSES,".alias_classes");

    pips_debug(4,"end\n");
    debug_off();

    return success;
}

#include "alias-classes.h"

/* For debugging: prettyprint many different kinds of newgen objects */
void dprint(expression x)
{
  if(expression_undefined_p(x))
    (void) fprintf(stderr, "UNDEFINED NEWGEN OBJECT\n");
  else if(x==0)
    (void) fprintf(stderr, "EMPTY LIST\n");
  else {
    int ot = expression_domain_number(x);
    if(ot==0)
      (void) fprintf(stderr,"PROBABLY AN EMPTY LIST\n");
    else if(expression_undefined_p(x))
      (void) fprintf(stderr,"UNDEFINED NEWGEN OBJECT\n");
    else if(ot==expression_domain)
      print_expression( x);
    else if(ot==reference_domain)
      print_reference((reference) x);
    else if(ot==points_to_domain)
      print_points_to((points_to) x); // See also print_points_to_relation
    else if(ot==cell_domain) {
      print_points_to_cell((cell) x);
      fprintf(stderr, "\n");
    }
    else if(ot==type_domain)
      print_type((type) x);
    else if(ot==statement_domain)
      print_statement((statement) x);
    else if(ot==effect_domain)
      print_effect((effect) x);
    else if(ot==points_to_list_domain)
      print_points_to_list((points_to_list) x);
    else if(ot==points_to_graph_domain)
      print_points_to_graph((points_to_graph) x);
    else if(ot==text_domain)
      print_text(stderr, (text) x);
    else if(ot==entity_domain) {
      entity m = get_current_module_entity();
      entity mx = module_name_to_entity(entity_module_name((entity)x));
      if(m!=mx)
	fprintf(stderr,"%s" MODULE_SEP_STRING, entity_local_name(mx));
      fprintf(stderr, "%s\n", entity_local_name((entity) x));
    }
    else if(ot==basic_domain) {
      string s = basic_to_string((basic) x);
      fprintf(stderr, "%s\n", s);
      free(s);
    }
    else if(0<=ot && ot<1000)
      (void) fprintf(stderr, "Unprocessed Newgen Object with tag %d\n", ot);
    else if(ot>1000 || ot<=0) {
      // FI: I do not know how to get the largest Newgen type
      // We could assume that the object is a list and look for the type
      // of the first object...
      (void) fprintf(stderr,"NOT A NEWGEN OBJECT. MAYBE A LIST\n");
      expression cx = EXPRESSION(CAR((list) x));
      int cot = expression_domain_number(cx);
      if(cot==expression_domain)
	print_expressions((list) x);
      else if(cot==reference_domain)
	print_references((list) x);
      else if(cot==cell_domain)
	print_points_to_cells((list) x);
      else if(cot==type_domain)
	print_types((list) x);
      else if(cot==statement_domain)
	print_statements((list) x);
      else if(cot==effect_domain)
	print_effects((list) x);
      else if(cot==points_to_domain)
	print_points_to_relations((list) x);
      else if(cot==entity_domain) {
	// print_entities((list) x);
	list el = (list) x;
      entity m = get_current_module_entity();
	FOREACH(ENTITY, e, el) {
	  entity me = module_name_to_entity(entity_module_name(e));
	  if(m!=me)
	    fprintf(stderr,"%s" MODULE_SEP_STRING, entity_local_name(me));
	  fprintf(stderr, "%s\n", entity_local_name(e));
	}
      }
      else
	(void) fprintf(stderr, "If a list, a list of unknown objects: tag=%d\n", (int) cot);
    }
  }
}
