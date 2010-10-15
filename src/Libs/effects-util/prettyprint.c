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
  } else
    pc = CHAIN_SWORD(pc, entity_minimal_name(e));

  begin_attachment = STRING(CAR(pc));

  if (reference_indices(obj) != NIL) {
    string beg = string_undefined;
    string mid = string_undefined;
    string end = string_undefined;

    switch(get_prettyprint_language_tag()) {
      case is_language_fortran95:
      case is_language_fortran:
        beg = "(";
        mid = ",";
        end = ")";
        break;
      case is_language_c:
        beg = "[";
        mid = "][";
        end = "]";
        break;
      default:
        pips_internal_error("Language unknown !");
        break;
    }

    pc = CHAIN_SWORD(pc,beg);
    for(list pi = reference_indices(obj); !ENDP(pi); POP(pi))
      {
	expression ind_exp = EXPRESSION(CAR(pi));
	syntax s = expression_syntax(ind_exp);
	if (syntax_reference_p(s) &&
	    entity_field_p(reference_variable(syntax_reference(s))))
	  {
	    // add a '.' to disambiguate field names from variable names
	    pc = CHAIN_SWORD(pc, ".");
	  }
	pc = gen_nconc(pc, words_expression(ind_exp,NIL));
	if (CDR(pi) != NIL)
	  pc = CHAIN_SWORD(pc,mid);
      }
    pc = CHAIN_SWORD(pc,end);
  }

  attach_reference_to_word_list(begin_attachment, STRING(CAR(gen_last(pc))),
				obj);
  return(pc);
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
	    name = entity_minimal_name(ent);
    }

    return name;
}

/** @brief weight function for Pvecteur passed as argument to
 *         sc_lexicographic_sort in prettyprint functions involving cell descriptors.
 *
 * The strange argument type is required by qsort(), deep down in the calls.
 * This function is an adaptation of is_inferior_pvarval in semantics
 */
int
is_inferior_cell_descriptor_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
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
    else if(variable_phi_p((entity) vecteur_var(*pvarval1))
	    && !variable_phi_p((entity) vecteur_var(*pvarval2)))
      is_equal = -1;
    else  if(variable_phi_p((entity) vecteur_var(*pvarval2))
	    && !variable_phi_p((entity) vecteur_var(*pvarval1)))
      is_equal = 1;
    else
	is_equal =
	    strcmp(pips_region_user_name((entity) vecteur_var(*pvarval1)),
		   pips_region_user_name((entity) vecteur_var(*pvarval2)));

    return is_equal;
}


/********** POINT_TO *************/



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

/* must return -1, 0 or 1. Should avoid 0 because we want a total
   order to avoid validation problems. */
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

  // FI: memory leak? generation of a new string?
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1so), entity_minimal_user_name(v2so));
  if(i==0) {
    i = strcmp(entity_minimal_user_name(v1si), entity_minimal_user_name(v2si));
    if(i==0) {
      list sl1 = reference_indices(r1so);
      list sl2 = reference_indices(r2so);
      int i1 = gen_length(sl1);
      int i2 = gen_length(sl2);

      i = i2>i1? 1 : (i2<i1? -1 : 0);
      if(i==0) {
	list sl1 = reference_indices(r1si);
	list sl2 = reference_indices(r2si);
	int i1 = gen_length(sl1);
	int i2 = gen_length(sl2);

	i = i2>i1? 1 : (i2<i1? -1 : 0);
	if(i==0) {
	  pips_internal_error("Further reference comparison not implemented...");
	}
      }
    }
  }

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

int cell_compare(cell *c1, cell *c2)
{
  int c1_pos = 0; /* result */
  pips_assert("gaps not handled yet (ppv1 first)", !cell_gap_p(*c1));
  pips_assert("gaps not handled yet (ppv2 first)", !cell_gap_p(*c2));

  reference r1 = cell_reference(*c1);
  reference r2 = cell_reference(*c2);
  entity e1 = reference_variable(r1);
  entity e2 = reference_variable(r2);

  if(same_entity_p(e1, e2))
    {
      /* same entity, sort on indices values */
      list dims1 = reference_indices(r1);
      list dims2 = reference_indices(r2);

      size_t nb_dims1 = gen_length(dims1);
      size_t nb_dims2 = gen_length(dims2);

      for(;!ENDP(dims1) && !ENDP(dims2) && c1_pos == 0; POP(dims1), POP(dims2))
	{
	  expression e1 = EXPRESSION(CAR(dims1));
	  expression e2 = EXPRESSION(CAR(dims2));

	  if(unbounded_expression_p(e1))
	    if(unbounded_expression_p(e2))
	      c1_pos = 0;
	    else
	      c1_pos = 1;
	  else
	    if(unbounded_expression_p(e2))
	      c1_pos = -1;
	    else
	      {
		syntax s1 = expression_syntax(e1);
		syntax s2 = expression_syntax(e2);
		if (syntax_reference_p(s1)
		    && entity_field_p(reference_variable(syntax_reference(s1)))
		    && syntax_reference_p(s2)
		    && entity_field_p(reference_variable(syntax_reference(s2))))
		  {
		    entity fe1 = reference_variable(syntax_reference(s1));
		    entity fe2 = reference_variable(syntax_reference(s2));
		    if (!same_entity_p(fe1, fe2))
		      c1_pos = strcmp(entity_name(fe1),entity_name(fe2));
		  }
		else
		  {
		    intptr_t i1 = 0;
		    intptr_t i2 = 0;
		    intptr_t diff = 0;

		    int r1 = expression_integer_value(e1, &i1);
		    int r2 = expression_integer_value(e2, &i2);

		    if (r1 && r2)
		      {
			diff = i1 - i2;
			c1_pos = diff==0? 0 : (diff>0?1:-1);
		      }
		  }
	      }
	}

      if (c1_pos == 0)
	c1_pos = (nb_dims1 < nb_dims2) ? -1 : ( (nb_dims1 > nb_dims2) ? 1 : 0);
    }
  else
    {
      /* not same entity, sort on entity name */
      /* sort on module name */
      c1_pos = strcmp(entity_module_name(e1), entity_module_name(e2));

      /* if same module name: sort on entity local name */
      if (c1_pos == 0)
	  c1_pos = strcmp(entity_user_name(e1), entity_user_name(e2));
      /* else: current module comes first, others in lexicographic order */
      else
	{
	  entity module = get_current_module_entity();

	  if (strcmp(module_local_name(module), entity_module_name(e1)) == 0)
	    c1_pos = -1;
	  if (strcmp(module_local_name(module), entity_module_name(e2)) == 0)
	    c1_pos = 1;
	}
    }

  return c1_pos;
}

/* Compares two pointer values for sorting. The first criterion is based on names.
 * Local entities come first; then they are sorted according to the
 * lexicographic order of the module name, and inside each module name class,
 * according to the local name lexicographic order. Then for a given
 * entity name, a read effect comes before a write effect. It is assumed
 * that there is only one effect of each type per entity. bc.
 */
int
pointer_value_compare(cell_relation *ppv1, cell_relation *ppv2)
{
  int ppv1_pos = 0; /* result */
  /* compare first references of *ppv1 and *ppv2 */

  cell ppv1_first_c = cell_relation_first_cell(*ppv1);
  cell ppv2_first_c = cell_relation_first_cell(*ppv2);

  pips_assert("there should not be preference cells in pointer values (ppv1 first) \n",
	      !cell_preference_p(ppv1_first_c));
  pips_assert("there should not be preference cells in pointer values (ppv2 first) \n",
	      !cell_preference_p(ppv2_first_c));

  pips_assert("the first cell must have value_of interpretation (ppv1)\n",
	      cell_relation_first_value_of_p(*ppv1));
  pips_assert("the first cell must have value_of interpretation (ppv2)\n",
	      cell_relation_first_value_of_p(*ppv2));

  ppv1_pos = cell_compare(&ppv1_first_c, &ppv2_first_c);

  if (ppv1_pos == 0)       /* same first cells */
    {
      /* put second cells value_of before address_of */
      bool ppv1_second_value_of_p = cell_relation_second_value_of_p(*ppv1);
      bool ppv2_second_value_of_p = cell_relation_second_value_of_p(*ppv2);

      ppv1_pos = (ppv1_second_value_of_p ==  ppv2_second_value_of_p) ? 0 :
	(ppv1_second_value_of_p ? -1 : 1);

      if (ppv1_pos == 0) /* both are value_of or address_of*/
	{
	  /* compare second cells */
	  cell ppv1_second_c = cell_relation_second_cell(*ppv1);
	  cell ppv2_second_c = cell_relation_second_cell(*ppv2);
	  ppv1_pos = cell_compare(&ppv1_second_c, &ppv2_second_c);

	}
    }
  return(ppv1_pos);
}


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
  pips_assert("approximation is not must\n", !approximation_must_p(ap));

  w= gen_nconc(w, effect_words_reference(first_r));
  w = CHAIN_SWORD(w," == ");
  w= gen_nconc(w, effect_words_reference(second_r));
  w = CHAIN_SWORD(w, approximation_may_p(ap) ? " (may)" : " (exact)" );
  return (w);
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

  boolean foresys = FALSE;
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
    boolean loose_p = get_bool_property("PRETTYPRINT_LOOSE");

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
