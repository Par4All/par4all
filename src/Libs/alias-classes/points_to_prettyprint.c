/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/*
 * (prettyg)print of POINTS TO.
 *
 * AM, August 2009.
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "text.h"
#include "text-util.h"

#include "top-level.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"


#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "preprocessor.h"
#include "misc.h"

#include "prettyprint.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"


#include "alias-classes.h"




#define PT_TO_SUFFIX ".points_to"
#define PT_TO_DECO "points to = "
#define SUMMARY_PT_TO_SUFFIX ".summary_points_to"
#define SUMMARY_PT_TO_DECO "summary points to = "

/****************************************************** STATIC INFORMATION */
GENERIC_GLOBAL_FUNCTION(printed_points_to_list, statement_points_to)

/************************************************************* BASIC WORDS */

text text_points_to(entity module __attribute__ ((__unused__)),int margin __attribute__ ((__unused__)), statement s)
{

  text t;
  t = bound_printed_points_to_list_p(s)?
    words_predicate_to_commentary
    (words_points_to_list(PT_TO_DECO,
			  load_printed_points_to_list(s)),
			      get_comment_sentinel())
    :words_predicate_to_commentary
    (CONS(STRING,PT_TO_DECO, CONS(STRING, strdup("{}"), NIL)),
        get_comment_sentinel());


  return t;
}


 text text_code_points_to(statement s)
{
  text t;
  debug_on("PRETTYPRINT_DEBUG_LEVEL");
  init_prettyprint(text_points_to);
  t = text_module(get_current_module_entity(), s);
  close_prettyprint();
  debug_off();
  return t;
}

/* bool print_code_points_to(const char* module_name, */
/* 			  string resource_name __attribute__ ((__unused__)), */
/* 			  string file_suffix) */
/* { */
/*   list wl = list_undefined; */
/*   text t, st = text_undefined; */
/*   bool res; */
/*   debug_on("POINTS_TO_DEBUG_LEVEL"); */
/*   set_current_module_entity(local_name_to_top_level_entity(module_name)); */
/*   points_to_list summary_pts_to = (points_to_list) */
/*     db_get_memory_resource(DBR_SUMMARY_POINTS_TO_LIST, module_name, true); */
/*   wl = words_points_to_list(PT_TO_DECO, summary_pts_to); */
/*   pips_debug(1, "considering module %s \n", */
/* 	     module_name); */

/*   /\*  FI: just for debugging *\/ */
/*   // check_abstract_locations(); */
/*   set_printed_points_to_list((statement_points_to) */
/* 			     db_get_memory_resource(DBR_POINTS_TO_LIST, module_name, true)); */
/*   statement_points_to_consistent_p(get_printed_points_to_list()); */
/*   set_current_module_statement((statement) */
/* 			       db_get_memory_resource(DBR_CODE, */
/* 						      module_name, */
/* 						      true)); */
/*   // FI: should be language neutral... */

/*   st = words_predicate_to_commentary(wl, get_comment_sentinel()); */
/*   t = text_code_points_to(get_current_module_statement()); */
/*   MERGE_TEXTS(st, t); */
/*   res= make_text_resource_and_free(module_name,DBR_PRINTED_FILE,file_suffix, st); */
/*   reset_current_module_entity(); */
/*   reset_current_module_statement(); */
/*   reset_printed_points_to_list(); */
/*   debug_off(); */
/*   return true; */
/* } */
bool print_code_points_to(const char* module_name,
			  string resource_name __attribute__ ((__unused__)),
			  string file_suffix)
{
  //list wl = list_undefined;
  bool res;
  debug_on("POINTS_TO_DEBUG_LEVEL");
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  
  
  /* Load IN  pts-to */
  points_to_list pts_to_in = (points_to_list)
    db_get_memory_resource(DBR_POINTS_TO_IN, module_name, true);
  //list l_pt_to_in = points_to_list_list(pts_to_in);

 /* Load OUT  pts-to */
  points_to_list pts_to_out = (points_to_list)
    db_get_memory_resource(DBR_POINTS_TO_OUT, module_name, true);
  //list l_pt_to_out = points_to_list_list(pts_to_out);

  pips_debug(1, "considering module %s \n",
	     module_name);

  /*  FI: just for debugging */
  // check_abstract_locations();
  set_printed_points_to_list((statement_points_to)
			     db_get_memory_resource(DBR_POINTS_TO, module_name, true));
  /* statement_points_to_consistent_p(get_printed_points_to_list()); */
  set_current_module_statement((statement)
			       db_get_memory_resource(DBR_CODE,
						      module_name,
						      true));

  init_prettyprint(text_pt_to);
  /* text sum_tex = text_points_to_relations(l_sum_pt_to, "Points To:"); */
  text in_tex = text_points_to_relations(pts_to_in, "Points To IN:");
  text out_tex = text_points_to_relations(pts_to_out, "Points To OUT:");
  text t = make_text(NIL);
  /* MERGE_TEXTS( t, sum_tex ); */
  MERGE_TEXTS( t, in_tex );
  MERGE_TEXTS( t, out_tex );
  MERGE_TEXTS(t, text_module(get_current_module_entity(),
			     get_current_module_statement()));
  res = make_text_resource_and_free(module_name,DBR_PRINTED_FILE,file_suffix, t);
  close_prettyprint();
  reset_current_module_entity();
  reset_current_module_statement();
  reset_printed_points_to_list();
  debug_off();
  return res;
}

void print_points_to_list(points_to_list ptl)
{
  text t = text_points_to_relations(ptl, "");
  print_text(stderr, t);
  free_text(t);
}

void print_points_to_graph(points_to_graph ptg)
{
  bool ok = !points_to_graph_bottom(ptg);
  if(!ok)
    fprintf(stderr, "Points-to graph does not exist, dead code\n");
  else {
    set s = points_to_graph_set(ptg);
    print_points_to_set("", s);
  }
}

//Handlers for PIPSMAKE
bool print_code_points_to_list(const char* module_name)
{
	return print_code_points_to(module_name,
				    DBR_POINTS_TO,
				    PT_TO_SUFFIX);
}


list words_points_to(points_to pt)
{
  cell source = points_to_source(pt);
  cell sink = points_to_sink(pt);

  pips_assert("there should not be preference cells in points to (source) \n",
	      !cell_preference_p(source));
  pips_assert("there should not be preference cells in points to (sink) \n",
	      !cell_preference_p(sink));

  pips_assert("gaps not handled yet (source)", !cell_gap_p(source));
  pips_assert("gaps not handled yet (sink)", !cell_gap_p(sink));


  list w = NIL;

  reference source_ref = cell_reference(source);
  reference sink_ref = cell_reference(sink);
  approximation ap = points_to_approximation(pt);
  pips_assert("approximation is not must\n", !approximation_exact_p(ap));

  w = gen_nconc(w, effect_words_reference(source_ref));
  w = CHAIN_SWORD(w," -> ");
  if(!nowhere_cell_p) {
    w = gen_nconc(w, effect_words_reference(sink_ref));
  }
  else {
    string undef = "undefined" ;
    w = gen_nconc(w, CONS(STRING,undef,NIL));
  }
    
  w = CHAIN_SWORD(w, approximation_may_p(ap) ? " (may)" : " (exact)" );
  return (w);
}

#define append(s) add_to_current_line(line_buffer, s, str_prefix, tpt_to)

/* text text_region(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries,
 *            representing the region
 * modifies : nothing
 */
text text_points_to_relation(points_to pt_to)
{
  text tpt_to = text_undefined;

  bool foresys = false;
  string str_prefix = get_comment_continuation();
  char line_buffer[MAX_LINE_LENGTH];
  Psysteme sc;
  list /* of string */ ls;

  if (points_to_undefined_p(pt_to))
    {
      ifdebug(1)
	{
	  return make_text(CONS(SENTENCE,
				make_sentence(is_sentence_formatted,
					      strdup(concatenate(str_prefix,
								 "undefined points to relation\n",
								 NULL))),
				NIL));
	}
      else
	pips_user_warning("unexpected points-to relation undefined\n");
    }
  else
    tpt_to = make_text(NIL);

  cell source = points_to_source(pt_to);
  cell sink = points_to_sink(pt_to);

  pips_assert("there should not be preference cells in points to relation (source) \n",
	      !cell_preference_p(source));
  pips_assert("there should not be preference cells in points to relation (sink) \n",
	      !cell_preference_p(sink));

  pips_assert("gaps not handled yet (source)", !cell_gap_p(source));
  pips_assert("gaps not handled yet (sink)", !cell_gap_p(sink));


  reference source_r = cell_reference(source);
  reference sink_r = cell_reference(sink);
  approximation ap = points_to_approximation(pt_to);
  descriptor d = points_to_descriptor(pt_to);

  /* PREFIX
   */
  strcpy(line_buffer, get_comment_sentinel());
  append(" ");

  /* REFERENCES */
  ls = effect_words_reference(source_r);

  FOREACH(STRING, s, ls) {append(s);}
  gen_free_string_list(ls); ls = NIL;

  append(" -> ");

  /* Change nowhere cells into undefined to comply with the C standard */
  entity e = reference_variable(sink_r);
  if(! entity_typed_nowhere_locations_p(e)) {
    ls = effect_words_reference(sink_r);
  }
  else {
    string undef = "undefined" ;
    ls =  CONS(STRING,strdup(undef),NIL);
  }
  //  ls = effect_words_reference(sink_r);
  /* if (points_to_second_address_of_p(pt_to)) */
  /*   append("&"); */

  FOREACH(STRING, s, ls) {append(s);}
  gen_free_string_list(ls); ls = NIL;

  /* DESCRIPTOR */
  /* sorts in such a way that constraints with phi variables come first.
   */
  if(!descriptor_none_p(d))
    {
      sc = sc_copy(descriptor_convex(d));
      sc_lexicographic_sort(sc, is_inferior_cell_descriptor_pvarval);
      system_sorted_text_format(line_buffer, str_prefix, tpt_to, sc,
				(get_variable_name_t) pips_region_user_name,
				vect_contains_phi_p, foresys);
      sc_rm(sc);
    }

  /* APPROXIMATION */
  append(approximation_may_p(ap) ? " , MAY" : " , EXACT");

  /* CLOSE */
  close_current_line(line_buffer, tpt_to, str_prefix);

  return tpt_to;
}

text
text_points_to_relations(points_to_list ptl, string header)
{
  list l_pt_to = points_to_list_list(ptl);
  bool bottom_p = points_to_list_bottom(ptl);

    text tpt_to = make_text(NIL);
    /* in case of loose_prettyprint, at least one region to print? */
    bool loose_p = get_bool_property("PRETTYPRINT_LOOSE");

    /* GO: No redundant test anymore, see  text_statement_array_regions */
    if (l_pt_to != (list) HASH_UNDEFINED_VALUE && l_pt_to != list_undefined)
    {
      /* header first */
      char line_buffer[MAX_LINE_LENGTH];
      string str_prefix = get_comment_continuation();
      if (loose_p)
	{
	  strcpy(line_buffer,"\n");
	  append(get_comment_sentinel());
	}
      else
	{
	  strcpy(line_buffer,get_comment_sentinel());
	}
      append(" ");
      append(header);
      if(bottom_p)
	append(" unreachable\n");
      else if(ENDP(l_pt_to))
	append(" none\n");
      else
	append("\n");
      ADD_SENTENCE_TO_TEXT(tpt_to,
			   make_sentence(is_sentence_formatted,
					 strdup(line_buffer)));
      l_pt_to = points_to_list_sort(l_pt_to);
      FOREACH(points_to, pt_to, l_pt_to)
	{
	  MERGE_TEXTS(tpt_to, text_points_to_relation(pt_to));
	}

      if (loose_p)
	ADD_SENTENCE_TO_TEXT(tpt_to,
			     make_sentence(is_sentence_formatted,
					   strdup("\n")));
    }
    return tpt_to;
}

/* print a points-to arc, print_points_to() or print_points_to_arc() */
void print_points_to_relation(points_to pt_to)
{
  text t = text_points_to_relation(pt_to);
  print_text(stderr, t);
  free_text(t);
}

/* print a list of points-to arcs */
void print_points_to_relations(list l_pt_to)
{
  fprintf(stderr,"\n");
  if (ENDP(l_pt_to))
    fprintf(stderr,"<none>");
  else
    {
      FOREACH(POINTS_TO, pt, l_pt_to)
	{
	  print_points_to_relation(pt);
	}
    }
  fprintf(stderr,"\n");
}

text text_pt_to(entity __attribute__ ((unused)) module_name, int __attribute__ ((unused)) margin, statement s)
{
  points_to_list ptl = load_printed_points_to_list(s);
  text t = text_points_to_relations(ptl, "Points To:");
  return t;
}

