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
#include "points_to_private.h"

#include "ri-util.h"
#include "effects-util.h"

#include "database.h"
#include "resources.h"
#include "properties.h"
#include "preprocessor.h"
#include "misc.h"



/************* EFFECT REFERENCES */

/* made from words_reference
 * this function can print entity_name instead of entity_local_name,
 * when the entity is not called in the current program.
 */
list /* of string */ effect_words_reference(reference obj)
{
  list pc = NIL;
  string begin_attachment;
  entity e = reference_variable(obj);

  if (get_bool_property("PRETTYPRINT_WITH_COMMON_NAMES")
      && entity_in_common_p(e)) {
    pc = CHAIN_SWORD(pc, entity_and_common_name(e));
  } else if (get_bool_property("PRETTYPRINT_EFFECT_WITH_FULL_ENTITY_NAME")) {
    pc = CHAIN_SWORD(pc, entity_name(e));
  } else {
    pc = CHAIN_SWORD(pc, entity_minimal_name(e));
  }

  begin_attachment = STRING(CAR(pc));

  if (reference_indices(obj) != NIL)
    {
      string before_first_index = string_undefined;
      string before_index = string_undefined;
      string after_index = string_undefined;
      string after_last_index = string_undefined;
      string before_field = string_undefined;
      string after_field = string_undefined;

      switch(get_prettyprint_language_tag())
	{
	case is_language_fortran95:
	case is_language_fortran:
	  before_first_index = "(";
	  before_index = "";
	  after_index = ",";
	  after_last_index = ")";
	  before_field = "";
	  after_field = "";
	  break;
	case is_language_c:
	  before_first_index = "[";
	  before_index = "[";
	  after_index = "]";
	  after_last_index = "]";
	  before_field = ".";
	  after_field = "";
	  break;
	default:
	  pips_internal_error("Language unknown !");
	  break;
	}

      bool first = true;
      for(list pi = reference_indices(obj); !ENDP(pi); POP(pi))
	{
	  expression ind_exp = EXPRESSION(CAR(pi));
	  syntax s = expression_syntax(ind_exp);
	  if (syntax_reference_p(s) &&
	      entity_field_p(reference_variable(syntax_reference(s))))
	    {
	      // add a '.' to disambiguate field names from variable names
	      pc = CHAIN_SWORD(pc, before_field);
	      pc = gen_nconc(pc, words_expression(ind_exp,NIL));
	      pc = CHAIN_SWORD(pc, after_field);
	    }
	  else
	    {
	      if (first)
		{
		  pc = CHAIN_SWORD(pc,before_first_index);
		  first = false;
		}
	      else
		pc = CHAIN_SWORD(pc,before_index);

	      pc = gen_nconc(pc, words_expression(ind_exp,NIL));
	      if (CDR(pi) != NIL)
		pc = CHAIN_SWORD(pc,after_index);
	      else
		pc = CHAIN_SWORD(pc,after_last_index);
	    }
	}
    }

  attach_reference_to_word_list(begin_attachment, STRING(CAR(gen_last(pc))),
				obj);
  return(pc);
}

string effect_reference_to_string(reference ref)
{
  return words_to_string(effect_words_reference(ref));
}

/************* CELL DESCRIPTORS */

/* char * pips_region_user_name(entity ent)
 * output   : the name of entity.
 * modifies : nothing.
 * comment  : allows to "catch" the PHIs entities, else, it works like
 *            pips_user_value_name() (see semantics.c).
 */
const char *
pips_region_user_name(entity ent)
{
    /* external_value_name cannot be used because there is no need for
       the #new suffix, but the #old one is necessary */
    const char* name;
    if(ent == NULL)
	/* take care of the constant term TCST */
	name = "";
    else {
	char *ent_name = entity_name(ent);

	if (strncmp(ent_name, REGIONS_MODULE_NAME, 7) == 0)
	    /* ent is a PHI entity from the regions module */
	    name = entity_local_name(ent);
	else
	  {
	    /* if (!hash_entity_to_values_undefined_p() && !entity_has_values_p(ent)) */
/* 	      name = external_value_name(ent); */
/* 	    else */
	      name = entity_minimal_name(ent);
	  }
    }

    return name;
}



/********** POINTS_TO *************/



#define PT_TO_SUFFIX ".points_to"
#define PT_TO_DECO "points to = "
#define SUMMARY_PT_TO_SUFFIX ".summary_points_to"
#define SUMMARY_PT_TO_DECO "summary points to = "

/*Already exist in cprettyprint but in mode static. To be removed later.*/
static bool variable_p(entity e)
{
	return type_undefined_p(entity_type(e));
}

/* To modelize the heap locations we manufacture fictious reference,
 * that triggered a bug when it appears as an argument of entity_user_name(). */
list
words_fictious_reference(reference obj)
{
  list pc = NIL;
  entity e = reference_variable(obj);
  pc = CHAIN_SWORD(pc, entity_name(e));
  return(pc);
}

#if 0
/* For debugging points-to: use this function in points_to_words_reference() */
static string entity_full_name(entity e)
{
  return entity_name(e);
}
#endif

/* Specific handling of references appearing in points_to */
list points_to_words_reference(reference r)
{
  extern const char* entity_minimal_user_name(entity);

  // Normal implementation, used for validation:
  return words_any_reference(r,NIL, entity_minimal_user_name);
  // To ease debugging, use:
  //return words_any_reference(r,NIL, entity_full_name);
}

list word_points_to(points_to pt)
{
  list l2 = NIL, l1 = NIL, rlt1 = NIL;

  pips_assert("pt is defined", !points_to_undefined_p(pt));
  points_to_consistent_p(pt);
  cell c1 = points_to_source(pt);
  cell c2 = points_to_sink(pt);
  // FI->AM: check all your copy_xxxx(); you often copy a large object
  // to obtain a small part of it
  reference r1 = cell_to_reference(c1);
  reference r2 = cell_to_reference(c2);
  approximation rel = points_to_approximation(pt);
  string l3 = "-MAY-";

  if (approximation_exact_p(rel))
    l3 = "-Exact-";
  if(variable_p(reference_variable(r2)))
    l2 = words_fictious_reference(r2);
  else
    l2 = points_to_words_reference(r2);

  l1 = points_to_words_reference(r1);

  rlt1 = gen_nconc((CONS(STRING,strdup("("), NIL)),l1);
  rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(","), NIL)));

  rlt1 = gen_nconc(rlt1, l2);
  rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(","), NIL)));
  rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(l3), NIL)));
  rlt1 = gen_nconc(rlt1,(CONS(STRING,strdup(")"), NIL)));
  return rlt1;
}

/* Comparison of two points-to arcs based on their source and sink nodes.
 *
 * This comparison function is used to sort a list of points-to before
 * storage and print-out.
 *
 * It must return -1, 0 or 1 like strcmp(). It should avoid 0 because
 * we want a total order to avoid validation problems. Hence the
 * exploitation of the references, number of indices, subscript
 * expressions, etc. if the entity names are not sufficient to
 * disambiguate the references.
 *
 * When subscript expressions are used, fields are replaced by the
 * corresponding field number. So the sort is based on the field ranks
 * in the data structure and not on the the field names.
 *
 * For abstract locations, the local name is used for the sort and the
 * global names is sometimes used in the prettyprint. Hence, the
 * alphabetical order is not obvious in the print-out.
 */
int points_to_compare_cells(const void * vpt1, const void * vpt2)
{
  int i = 0;

  points_to pt1 = *((points_to *) vpt1);
  points_to pt2 = *((points_to *) vpt2);

  cell c1so = points_to_source(pt1);
  cell c2so = points_to_source(pt2);
  cell c1si = points_to_sink(pt1);
  cell c2si = points_to_sink(pt2);

  //cell c1 = CELL(CAR(vc1));
  //cell c2 = CELL(CAR(vc2));
  // FI: bypass of GAP case
  reference r1so = cell_to_reference(c1so);
  reference r2so = cell_to_reference(c2so);
  reference r1si = cell_to_reference(c1si);
  reference r2si = cell_to_reference(c2si);

  entity v1so = reference_variable(r1so);
  entity v2so = reference_variable(r2so);
  entity v1si = reference_variable(r1si);
  entity v2si = reference_variable(r2si);
  list sl1 = NIL, sl2 = NIL, sli1 = NIL, sli2 = NIL ;
  // FI: memory leak? generation of a new string?
  extern const char* entity_minimal_user_name(entity);
  string n1so =   entity_abstract_location_p(v1so)?
    entity_local_name(v1so) : entity_minimal_user_name(v1so);
  string n2so =   entity_abstract_location_p(v2so)?
    entity_local_name(v2so) : entity_minimal_user_name(v2so);
  string n1si =   entity_abstract_location_p(v1si)?
    entity_local_name(v1si) : entity_minimal_user_name(v1si);
  string n2si =   entity_abstract_location_p(v2si)?
    entity_local_name(v2si) : entity_minimal_user_name(v2si);

  i = strcmp(n1so, n2so);
  if(i==0) {
    i = strcmp(n1si, n2si);
    if(i==0) {
      sl1 = reference_indices(r1so);
      sl2 = reference_indices(r2so);
      int i1 = gen_length(sl1);
      int i2 = gen_length(sl2);

      i = i2>i1? 1 : (i2<i1? -1 : 0);

      if(i==0) {
	sli1 = reference_indices(r1si);
	sli2 = reference_indices(r2si);
	int i1 = gen_length(sli1);
	int i2 = gen_length(sli2);

	i = i2>i1? 1 : (i2<i1? -1 : 0);
	//	if(i==0) {
	for(;i==0 && !ENDP(sl1); POP(sl1), POP(sl2)){
	  expression se1 = EXPRESSION(CAR(sl1));
	  expression se2 = EXPRESSION(CAR(sl2));
	  if(expression_constant_p(se1) && expression_constant_p(se2)){
	    int i1 = expression_to_int(se1);
	    int i2 = expression_to_int(se2);
	    i = i2>i1? 1 : (i2<i1? -1 : 0);
	    if(i==0){
	      i = strcmp(entity_minimal_user_name(v1si), entity_minimal_user_name(v2si));
	      for(;i==0 && !ENDP(sli1); POP(sli1), POP(sli2)){
		expression sei1 = EXPRESSION(CAR(sli1));
		expression sei2 = EXPRESSION(CAR(sli2));
		if(expression_constant_p(sei1) && expression_constant_p(sei2)){
		  int i1 = expression_to_int(sei1);
		  int i2 = expression_to_int(sei2);
		  i = i2>i1? 1 : (i2<i1? -1 : 0);
		}else{
		  string s1 = words_to_string(words_expression(se1, NIL));
		  string s2 = words_to_string(words_expression(se2, NIL));
		  i = strcmp(s1, s2);
		}
	      }
	    }
	  } else {
	    string s1 = words_to_string(words_expression(se1, NIL));
	    string s2 = words_to_string(words_expression(se2, NIL));
	    i = strcmp(s1, s2);
	  }
	}
      }
    }
  }
  // }
  return i;
}

/* Allocate a copy of ptl and sort it. It might be better to admit a
   side effect on ptl and to let the caller copy the liste before
   sorting. */
list points_to_list_sort(list ptl)
{
  list sptl = gen_full_copy_list(ptl);

  gen_sort_list(sptl, /* (gen_cmp_func_t) */ points_to_compare_cells);

  return sptl;
}

//extern void print_points_to(
/* Make sure that points-to are fully ordered before prettyprinting
   them or validation will be in trouble sooner or later. The sort
   could occur before storing the points-to information into the hash
   table or just before prettypriting it. */
list words_points_to_list(__attribute__((unused))string note, points_to_list s)
{
  list l = NIL;
  int i = 0;
  list ptl = points_to_list_list(s);
  list sptl = points_to_list_sort(ptl);

  FOREACH(POINTS_TO, j, sptl) {
    if(i>0)
      l = gen_nconc(l, (CONS(STRING,strdup(";"), NIL)));
    else
      i++;
    l = gen_nconc(l,word_points_to(j));
  }
  l = CONS(STRING,strdup("{"), l);
  l = gen_nconc(l,(CONS(STRING,strdup("}"), NIL)));

  gen_full_free_list(sptl);

  return l;
}


/************* POINTER VALUES */

list words_pointer_value(cell_relation pv)
{
  cell first_c = cell_relation_first_cell(pv);
  cell second_c = cell_relation_second_cell(pv);

  pips_assert("there should not be preference cells in pointer values (first) \n",
	      !cell_preference_p(first_c));
  pips_assert("there should not be preference cells in pointer values (second) \n",
	      !cell_preference_p(second_c));

  pips_assert("gaps not handled yet (first)", !cell_gap_p(first_c));
  pips_assert("gaps not handled yet (second)", !cell_gap_p(second_c));

  pips_assert("the first cell must have value_of interpretation\n",
	      cell_relation_first_value_of_p(pv));

  list w = NIL;

  reference first_r = cell_reference(first_c);
  reference second_r = cell_reference(second_c);
  approximation ap = cell_relation_approximation(pv);
  pips_assert("approximation is not must\n", !approximation_exact_p(ap));

  w= gen_nconc(w, effect_words_reference(first_r));
  w = CHAIN_SWORD(w," == ");
  w= gen_nconc(w, effect_words_reference(second_r));
  w = CHAIN_SWORD(w, approximation_may_p(ap) ? " (may)" : " (exact)" );
  return (w);
}

string approximation_to_string(approximation a)
{
  string as = string_undefined;
  if(approximation_may_p(a))
    as = "may";
  else if(approximation_must_p(a))
    as = "must"; // could be "exact"
  else if(approximation_exact_p(a))
    as = "exact";
  else
    pips_internal_error("Unknown approximation tag.\n");
  return as;
}

#define append(s) add_to_current_line(line_buffer, s, str_prefix, tpv)

/* text text_region(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries,
 *            representing the region
 * modifies : nothing
 */
text text_pointer_value(cell_relation pv)
{
  text tpv = text_undefined;

  bool foresys = false;
  string str_prefix = get_comment_continuation();
  char line_buffer[MAX_LINE_LENGTH];
  Psysteme sc;
  list /* of string */ ls;

  if (cell_relation_undefined_p(pv))
    {
      ifdebug(1)
	{
	  return make_text(CONS(SENTENCE,
				make_sentence(is_sentence_formatted,
					      strdup(concatenate(str_prefix,
								 "undefined pointer value\n",
								 NULL))),
				NIL));
	}
      else
	pips_user_warning("unexpected pointer value undefined\n");
    }
  else
    tpv = make_text(NIL);

  cell first_c = cell_relation_first_cell(pv);
  cell second_c = cell_relation_second_cell(pv);

  pips_assert("there should not be preference cells in pointer values (first) \n",
	      !cell_preference_p(first_c));
  pips_assert("there should not be preference cells in pointer values (second) \n",
	      !cell_preference_p(second_c));

  pips_assert("gaps not handled yet (first)", !cell_gap_p(first_c));
  pips_assert("gaps not handled yet (second)", !cell_gap_p(second_c));

  pips_assert("the first cell must have value_of interpretation\n",
	      cell_relation_first_value_of_p(pv));


  reference first_r = cell_reference(first_c);
  reference second_r = cell_reference(second_c);
  approximation ap = cell_relation_approximation(pv);
  descriptor d = cell_relation_descriptor(pv);

  /* PREFIX
   */
  strcpy(line_buffer, get_comment_sentinel());
  append(" ");

  /* REFERENCES */
  ls = effect_words_reference(first_r);

  FOREACH(STRING, s, ls) {append(s);}
  gen_free_string_list(ls); ls = NIL;

  append(" == ");

  ls = effect_words_reference(second_r);
  if (cell_relation_second_address_of_p(pv))
    append("&");

  FOREACH(STRING, s, ls) {append(s);}
  gen_free_string_list(ls); ls = NIL;

  /* DESCRIPTOR */
  /* sorts in such a way that constraints with phi variables come first.
   */
  if(!descriptor_none_p(d))
    {
      sc = sc_copy(descriptor_convex(d));
      sc_lexicographic_sort(sc, is_inferior_cell_descriptor_pvarval);
      system_sorted_text_format(line_buffer, str_prefix, tpv, sc,
				(get_variable_name_t) pips_region_user_name,
				vect_contains_phi_p, foresys);
      sc_rm(sc);
    }

  /* APPROXIMATION */
  append(approximation_may_p(ap) ? " (may);" : " (exact);");

  /* CLOSE */
  close_current_line(line_buffer, tpv,str_prefix);

  return tpv;
}


text
text_pointer_values(list lpv, string header)
{
    text tpv = make_text(NIL);
    /* in case of loose_prettyprint, at least one region to print? */
    bool loose_p = get_bool_property("PRETTYPRINT_LOOSE");

    /* GO: No redundant test anymore, see  text_statement_array_regions */
    if (lpv != (list) HASH_UNDEFINED_VALUE && lpv != list_undefined)
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
      if(ENDP(lpv))
	append(" none\n");
      else
	append("\n");
      ADD_SENTENCE_TO_TEXT(tpv,
			   make_sentence(is_sentence_formatted,
					 strdup(line_buffer)));
      gen_sort_list(lpv, (int (*)(const void *,const void *)) pointer_value_compare);
      FOREACH(CELL_RELATION, pv, lpv)
	{
	  MERGE_TEXTS(tpv, text_pointer_value(pv));
	}

      if (loose_p)
	ADD_SENTENCE_TO_TEXT(tpv,
			     make_sentence(is_sentence_formatted,
					   strdup("\n")));
    }
    return tpv;
}

void print_pointer_value(cell_relation pv)
{
  text t = text_pointer_value(pv);
  print_text(stderr, t);
  free_text(t);
}

void print_pointer_values(list lpv)
{
  fprintf(stderr,"\n");
  if (ENDP(lpv))
    fprintf(stderr,"<none>");
  else
    {
      FOREACH(CELL_RELATION, pv, lpv)
	{
	  print_pointer_value(pv);
	}
    }
  fprintf(stderr,"\n");
}
