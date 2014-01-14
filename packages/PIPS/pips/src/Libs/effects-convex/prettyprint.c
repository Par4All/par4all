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
/* package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: prettyprint.c
 * ~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the prettyprinting functions.
 *
 */

#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "top-level.h"
#include "properties.h"

#include "effects-generic.h"
#include "effects-convex.h"

#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"

#define REGION_FORESYS_PREFIX "C$REG"


string
region_sc_to_string(string __attribute__ ((unused)) s,
		    Psysteme __attribute__ ((unused)) ps)
{
    pips_internal_error("implementation dropped");
    return string_undefined;
}



#define append(s) add_to_current_line(line_buffer, s, str_prefix, t_reg)

/* text text_region(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries,
 *            representing the region
 * modifies : nothing
 */
text text_region(effect reg)
{
    text t_reg = make_text(NIL);

    if(store_effect_p(reg)
       || !get_bool_property("PRETTYPRINT_MEMORY_EFFECTS_ONLY")) {
      bool foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
      string str_prefix =
	foresys ? FORESYS_CONTINUATION_PREFIX
	: get_comment_continuation();
      char line_buffer[MAX_LINE_LENGTH];
      reference r;
      action ac;
      action_kind ak;
      approximation ap;
      descriptor ed = effect_descriptor(reg);
      Psysteme sc;
      list /* of string */ ls;

      if(effect_undefined_p(reg))
	{
	  free_text(t_reg); // t_reg should be used instead of make_text()
	  user_log("[text_region] unexpected effect undefined\n");
	  return make_text(CONS(SENTENCE,
				make_sentence(is_sentence_formatted,
					      strdup(concatenate(str_prefix, "<REGION_UNDEFINED>\n", NULL))),
				NIL));
	}
      /* else the effect is defined...
       */

      /* PREFIX
       */
      strcpy(line_buffer, foresys? REGION_FORESYS_PREFIX: get_comment_sentinel());
      if (!foresys) append("  <");

      /* REFERENCE
       */
      r = effect_any_reference(reg);
      ls = foresys? words_reference(r, NIL): effect_words_reference(r);

      MAP(STRING, s, append(s), ls);
      gen_free_string_list(ls); ls = NIL;

      /* ACTION and APPROXIMATION
       */
      ac = effect_action(reg);
      ak = action_to_action_kind(ac);
      ap = effect_approximation(reg);

      if (foresys)
	{
	  append(", RGSTAT(");
	  append(action_read_p(ac) ? "R," : "W,");
	  append(approximation_may_p(ap) ? "MAY), " : "EXACT), ");
	}
      else /* PIPS prettyprint */
	{
	  append("-");
	  append(action_interpretation(action_tag(ac)));
	  /* To preserve the current output, actions on store are
	     implicit, actions on environment and type declaration are
	     specified */
	  if(!action_kind_store_p(ak))
	    append(action_kind_to_string(ak));
	  append(approximation_may_p(ap) ? "-MAY" : "-EXACT");
	  append("-");
	}

      /* SYSTEM
       * sorts in such a way that constraints with phi variables come first.
       */
      if(descriptor_none_p(ed)) {
	/* FI: there is no system; it's equivalent to an empty one... */
	  append("{}");
      } else {
	sc = sc_copy(region_system(reg));
	sc_lexicographic_sort(sc, is_inferior_cell_descriptor_pvarval);
	system_sorted_text_format(line_buffer, str_prefix, t_reg, sc,
				  (get_variable_name_t) pips_region_user_name,
				  vect_contains_phi_p, foresys);

	sc_rm(sc);
      }
      /* CLOSE
       */
      if (!foresys) append(">");
      close_current_line(line_buffer, t_reg,str_prefix);
    }

    return t_reg;
}

/* print the constraint system of a region
 *
 * Used for debugging, e.g. called from gdb
 */
void print_region_sc(effect r)
{
  descriptor d = effect_descriptor(r);
  if(descriptor_none_p(d)) {
    fprintf(stderr, "No descriptor\n");
  }
  else {
    Psysteme sc = region_system(r);
    sc_print(sc, (get_variable_name_t) pips_region_user_name);
  }
}


/* text text_array_regions(list l_reg, string ifread, string ifwrite)
 * input    : a list of regions
 * output   : a text representing this list of regions.
 * comment  : if the number of array regions is not nul, and if
 *            PRETTYPRINT_LOOSE is true, then empty lines are
 *            added before and after the text of the list of regions.
 */
static text
text_array_regions(list l_reg, string ifread, string ifwrite)
{
    text reg_text = make_text(NIL);
    /* in case of loose_prettyprint, at least one region to print? */
    bool loose_p = get_bool_property("PRETTYPRINT_LOOSE");
    bool one_p = false;

    set_action_interpretation(ifread, ifwrite);

    /* GO: No redundant test anymore, see  text_statement_array_regions */
    if (l_reg != (list) HASH_UNDEFINED_VALUE && l_reg != list_undefined)
    {
      gen_sort_list(l_reg, (int (*)(const void *,const void *)) effect_compare);
	FOREACH(EFFECT, reg, l_reg)
	{
	  if(store_effect_p(reg)
	     || !get_bool_property("PRETTYPRINT_MEMORY_EFFECTS_ONLY")) {
	    entity ent = effect_entity(reg);
	    if (  anywhere_effect_p(reg)
		  || malloc_effect_p(reg)
		  || get_bool_property("PRETTYPRINT_SCALAR_REGIONS")
		  || ! entity_non_pointer_scalar_p(ent))
	    {
		if (loose_p && !one_p )
		{
		    ADD_SENTENCE_TO_TEXT(reg_text,
					 make_sentence(is_sentence_formatted,
						       strdup("\n")));
		    one_p = true;
		}
		MERGE_TEXTS(reg_text, text_region(reg));
	    }
	  }
	}

	if (loose_p && one_p)
	    ADD_SENTENCE_TO_TEXT(reg_text,
				 make_sentence(is_sentence_formatted,
					       strdup("\n")));
    }

    reset_action_interpretation();
    return reg_text;
}

/* practical interfaces
 */
text text_inout_array_regions(list l)
{ return text_array_regions(l, ACTION_IN, ACTION_OUT);}

text text_rw_array_regions(list l)
{ return text_array_regions(l, ACTION_READ, ACTION_WRITE);}

text text_copyinout_array_regions(list l)
{ return text_array_regions(l, ACTION_COPYIN, ACTION_COPYOUT);}

text text_private_array_regions(list l)
{ return text_array_regions(l, ACTION_PRIVATE, ACTION_PRIVATE);}

/*********************************************************** ABSOLETE MAYBE? */

/* CALLGRAPH/ICFG stuff (should be OBSOLETE?)
 */
static text 
get_text_regions_for_module(
    const char* module_name, 
    string resource_name,
    string ifread,
    string ifwrite)
{
    text t;
    entity mod;
    list /* of effect */ le = effects_effects((effects) 
	db_get_memory_resource(resource_name, module_name, true));

    /* the current entity may be used for minimal names... */
    mod = module_name_to_entity(module_name);
    set_current_module_entity(mod);
    t = text_array_regions(le, ifread, ifwrite);
    reset_current_module_entity();
    return t;
}

text 
get_text_regions(const char* module_name)
{
    return get_text_regions_for_module
	(module_name, DBR_SUMMARY_REGIONS, ACTION_READ, ACTION_WRITE);
}

text 
get_text_in_regions(const char* module_name)
{
    return get_text_regions_for_module
	(module_name, DBR_IN_SUMMARY_REGIONS, ACTION_IN, ACTION_OUT);
}

text 
get_text_out_regions(const char* module_name)
{
    return get_text_regions_for_module
	(module_name, DBR_OUT_SUMMARY_REGIONS, ACTION_IN, ACTION_OUT);
}

/*********************************************************** DEBUG FUNCTIONS */

/* void print_regions(list pc)
 * input    : a list of regions.
 * modifies : nothing.
 * comment  : prints the list of regions on stderr .
 */
static void print_regions_with_action(list pc, string ifread, string ifwrite)
{
  set_action_interpretation(ifread, ifwrite);

  if (pc == NIL) {
    fprintf(stderr,"\t<NONE>\n");
  }
  else {
    FOREACH(EFFECT, ef, pc) {
      print_region(ef);
      fprintf(stderr,"\n");
    }
  }

  reset_action_interpretation();
}

/* external interfaces
 *
 * NW:
 * before calling
 * "print_inout_regions"
 * or "print_rw_regions"
 * or "print_copyinout_regions"
 * or "print_private_regions"
 *
 * "module_to_value_mappings" must be called to set up the
 * hash table to translate value into value names
 */
void print_rw_regions(list l)
{ print_regions_with_action(l, ACTION_READ, ACTION_WRITE);}

void print_inout_regions(list l)
{ print_regions_with_action(l, ACTION_IN, ACTION_OUT);}

void print_copyinout_regions(list l)
{ print_regions_with_action(l, ACTION_COPYIN, ACTION_COPYOUT);}

void print_private_regions(list l)
{ print_regions_with_action(l, ACTION_PRIVATE, ACTION_PRIVATE);}

void print_regions(list l) { print_rw_regions(l);}

/* void print_regions(effect r)
 * input    : a region.
 * modifies : nothing.
 * comment  : prints the region on stderr using words_region.
 *
 * NW:
 * before calling "print_region" or "text_region"
 *
 * "module_to_value_mappings" must be called to set up the
 * hash table to translate value into value names
 * (see comment for "module_to_value_mappings" for what must be done
 * before that is called)
 *
 * and also "set_action_interpretation" with arguments:
 * ACTION_READ, ACTION_WRITE to label regions as R/W
 * ACTION_IN, ACTION_OUT to label regions as IN/OUT
 * ACTION_COPYIN, ACTION_COPYOUT to label regions as COPYIN/COPYOUT
 * ACTION_PRIVATE, ACTION_PRIVATE to label regions as PRIVATE
 *
 * like this:
 *
 * const char* module_name;
 * entity module;
 * ...
 * (set up call to module_to_value_mappings as indicated in its comments)
 * ...
 * module_to_value_mappings(module);
 * set_action_interpretation(ACTION_IN, ACTION_OUT);
 *
 * (that's it, but after the call to "print_region" or "text_region",
 * don't forget to do:)
 *
 * reset_action_interpretation();
 * (resets after call to module_to_value_mappings as indicated in its comments)
 *
 * FI: Regions/Effects related to store and environment mutations are not
 * displayed because they have no descriptors, but by two LFs.
 */
void print_region(effect r)
{
    fprintf(stderr,"\n");
    if(effect_region_p(r)) {
	text t = text_region(r);
	print_text(stderr, t);
	free_text(t);
    }
    /* FI: uncommented to introduce environment and type declaration
       regions/effects */
    /* else print_words(stderr, words_effect(r)); */
    else {
      // FI: this is not homogeneous with the regions
      //print_words(stderr, words_effect(r));
      // This is even worse
      /*
      list el = CONS(EFFECT, r, NIL);
      text t = simple_rw_effects_to_text(el);
      print_text(stderr, t);
      free_text(t);
      gen_free_list(el);
      */
      // FI: let's homogeneize the outputs...
	text t = text_region(r);
	print_text(stderr, t);
	free_text(t);
    }
    fprintf(stderr,"\n");
}



/************************************************* STATISTICS FOR OPERATORS */


void print_regions_op_statistics(char __attribute__ ((unused)) *mod_name,
				 int regions_type)
{
    /*
    string prefix = string_undefined;

    switch (regions_type) {
    case R_RW :
	prefix = "rrw-";
	break;
    case R_IN :
	prefix = "rin-";
	break;
    case R_OUT :
	prefix = "rout-";
	break;
    }

    print_proj_op_statistics(mod_name, prefix);
    print_umust_statistics(mod_name, prefix);
    print_umay_statistics(mod_name, prefix);
    print_dsup_statistics(mod_name, prefix); */
    /* print_dinf_statistics(mod_name, prefix); */

}


/***************************************************************** SORTING */
