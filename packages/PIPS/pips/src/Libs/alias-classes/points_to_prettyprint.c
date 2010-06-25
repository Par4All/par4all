/*

  $Id: prettyprint.c 14802 2009-08-12 12:27:53Z mensi $

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

text text_points_to(entity module,int margin, statement s)
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

bool print_code_points_to(string module_name,
		      string resource_name,
		      string file_suffix)
{
  
  list wl = list_undefined;
  text t, st;
  bool res;
  debug_on("POINTS_TO_DEBUG_LEVEL");
  //init_printed_points_to_list();
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  points_to_list summary_pts_to = (points_to_list)
    db_get_memory_resource(DBR_SUMMARY_POINTS_TO_LIST, module_name, TRUE);
  wl = words_points_to_list(SUMMARY_PT_TO_SUFFIX, summary_pts_to);
  pips_debug(1, "considering module %s \n",
	     module_name);

  /*  FI: just for debugging */
  // check_abstract_locations();

  //init_printed_points_to_list();
  set_printed_points_to_list((statement_points_to)
			     db_get_memory_resource(DBR_POINTS_TO_LIST, module_name, TRUE));
  // statement_points_to_consistent_p(get_printed_points_to_list());
  statement_points_to_consistent_p(get_printed_points_to_list());
  set_current_module_statement((statement)
			       db_get_memory_resource(DBR_CODE,
						      module_name,
						      TRUE));
  // FI: should be language neutral...
  st = words_predicate_to_commentary(wl, get_comment_sentinel());
  t = text_code_points_to(get_current_module_statement());
  //print_text(stderr,t);
  //st = text_code_summary_points_to(get_current_module_statement());
  MERGE_TEXTS(st, t);
  //print_text(stderr,t);
  res= make_text_resource_and_free(module_name,DBR_PRINTED_FILE,file_suffix, st);
  reset_current_module_entity();
  reset_current_module_statement();
  reset_printed_points_to_list();
  
 
  

  //reset_printed_points_to_list();
  
 
  //free(t);
  debug_off();
  return TRUE;
}

//Handlers for PIPSMAKE
bool print_code_points_to_list(string module_name)
{
	return print_code_points_to(module_name,
				    DBR_POINTS_TO_LIST,
				    PT_TO_SUFFIX);
}
