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
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"

#include "text.h"
#include "text-util.h"

#include "top-level.h"

#include "linear.h"
#include "ri.h"
#include "ri-util.h"


#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"
#include "properties.h"

#include "prettyprint.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"


#include "alias-classes.h"




#define PT_TO_SUFFIX ".points_to"
#define PT_TO_DECO "points to = "


/****************************************************** STATIC INFORMATION */
GENERIC_GLOBAL_FUNCTION(printed_points_to_list, statement_points_to)

/************************************************************* BASIC WORDS */
reference access_reference1(access acc)
{
  points_to_path p = points_to_path_undefined;
  reference r = reference_undefined;
  if(access_referencing_p(acc)){
    p = access_referencing(acc);
    r = points_to_path_reference(p);
  }
  if(access_dereferencing_p(acc)){
    p = access_dereferencing(acc);
    r = points_to_path_reference(p);
  }
  if(access_addressing_p(acc)){
    p = access_addressing(acc);
    r = points_to_path_reference(p);
  }

  return copy_reference(r);

}

static bool variable_p(entity e)
{
	return type_undefined_p(entity_type(e));
}


list
words_reference_1(reference obj)
{
  list pc = NIL;
  string begin_attachment;

  entity e = reference_variable(obj);

  pc = CHAIN_SWORD(pc, entity_name(e));
  begin_attachment = STRING(CAR(pc));

  if (reference_indices(obj) != NIL) {
    if (prettyprint_is_fortran)
      {
	pc = CHAIN_SWORD(pc,"(");
	MAPL(pi, {
	  expression subscript = EXPRESSION(CAR(pi));
	  syntax ssubscript = expression_syntax(subscript);

	  if(syntax_range_p(ssubscript)) {
			pc = gen_nconc(pc, words_subscript_range(syntax_range(ssubscript),NIL));
	  }
	  else {
			pc = gen_nconc(pc, words_subexpression(subscript, 0, TRUE,NIL));
	  }

	  if (CDR(pi) != NIL)
	    pc = CHAIN_SWORD(pc,",");
	}, reference_indices(obj));
	pc = CHAIN_SWORD(pc,")");
      }
    else
      {
	MAPL(pi, {
	  expression subscript = EXPRESSION(CAR(pi));
	  syntax ssubscript = expression_syntax(subscript);
	  pc = CHAIN_SWORD(pc, "[");
	  if(syntax_range_p(ssubscript)) {
			pc = gen_nconc(pc, words_subscript_range(syntax_range(ssubscript),NIL));
	  }
	  else {
			pc = gen_nconc(pc, words_subexpression(subscript, 0, TRUE, NIL));
	  }
	  pc = CHAIN_SWORD(pc, "]");
	}, reference_indices(obj));
      }
  }
  /* attach_reference_to_word_list(begin_attachment," ", */
/*   			obj); */

  return(pc);
}


list word_points_to(points_to pt)
{
  list l2 = NIL;
  if( points_to_undefined_p(pt))
    {
      
    }
 access a1 = points_to_source(pt);
 access a2 = points_to_sink(pt);
 reference r1 = access_reference1(copy_access(a1));
 reference r2 = access_reference1(copy_access(a2));
 approximation rel = points_to_relation(pt);
 string l3 = "-MAY-";;
 if (approximation_exact_p(rel))
   l3 = "-Exact-";
 if(!variable_p(reference_variable(r2)))
	 l2 = words_reference(r2,NIL);
 else
   l2 = words_reference_1(r2);
 
 list l1 = words_reference(r1, NIL);
 list rlt1 = gen_nconc((CONS(STRING,strdup("("), NIL)),l1);
 rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(","), NIL)));
 
 rlt1 = gen_nconc(rlt1, l2);
 rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(","), NIL)));
 rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(l3), NIL)));
 rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(")"), NIL)));
 rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(";"), NIL)));
 return rlt1;
}

//extern void print_points_to(
list words_points_to_list( string note, points_to_list s)
{
	list l = NIL;

	FOREACH (POINTS_TO, j,points_to_list_list(s))
	{
		// points_to_consistent_p(pt);
		l = gen_nconc(l,word_points_to(j));
	}
	l = gen_nconc((CONS(STRING,strdup("{"), NIL)),l);
	l = gen_list_head(&l,gen_length(l)-1);
	l = gen_nconc(l,(CONS(STRING,strdup("}"), NIL)));
	
	return l? CONS(STRING, strdup(note), l): NIL;
}


text text_points_to(entity module,int margin, statement s)
{

  text t;
  t = bound_printed_points_to_list_p(s)?
	  words_predicate_to_commentary
	  (words_points_to_list(PT_TO_DECO,
				load_printed_points_to_list(s)),
	   PIPS_COMMENT_SENTINEL)
	  :make_text(NIL);

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


 bool
 print_code_points_to(string module_name,
											string resource_name,
											string file_suffix)
{
 text t;

 debug_on("POINTS_TO_DEBUG_LEVEL");
 pips_debug(1, "considering module %s \n",
			module_name);
 set_current_module_entity(local_name_to_top_level_entity(module_name));
 set_proper_rw_effects((statement_effects)
		       db_get_memory_resource(DBR_PROPER_EFFECTS,
					      module_name, TRUE));
 set_printed_points_to_list((statement_points_to)
 db_get_memory_resource(DBR_POINTS_TO_LIST, module_name, TRUE));
 // statement_points_to_consistent_p(get_printed_points_to_list());
 //statement_points_to_consistent_p(get_printed_points_to_list());
 set_current_module_statement((statement)
			      db_get_memory_resource(DBR_CODE,
						     module_name,
						     TRUE));
 t = text_code_points_to(get_current_module_statement());
// print_text(stderr,t);
 bool res= make_text_resource_and_free(module_name, DBR_PRINTED_FILE,
				file_suffix, t);

 reset_current_module_entity();
 reset_current_module_statement();
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

