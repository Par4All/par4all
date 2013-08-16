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
/*
 * package semantics - prettyprint interface
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
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
#include "properties.h"

#include "prettyprint.h"

#include "transformer.h"
#include "effects-generic.h"

#include "semantics.h"
#include "preprocessor.h"

#define PREC_FORESYS_PREFIX "C$PREC"
#define TRAN_FORESYS_PREFIX "C$TRAN"

DEFINE_CURRENT_MAPPING(semantic, transformer)

static bool is_transformer;
static bool is_total_precondition;
static bool is_user_view;
static bool is_transformer_filtered;

void set_prettyprint_transformer(void)
{
  is_transformer = true;
  is_total_precondition = false;
  is_transformer_filtered = false;
}

static hash_table nts = hash_table_undefined;
//static string non_feasible_system;

static bool print_code_semantics();
static text get_semantic_text(const char*,bool);

bool print_code_transformers(const char* module_name)
{
  is_user_view = false;
  is_transformer = true;
  is_total_precondition = false;
  is_transformer_filtered = false;
  return print_code_semantics(module_name);
}

bool print_code_preconditions(const char* module_name)
{
  is_user_view = false;
  is_transformer = false;
  is_total_precondition = false;
  is_transformer_filtered =
    get_bool_property("SEMANTICS_FILTERED_PRECONDITIONS");
  return print_code_semantics(module_name);
}

bool print_code_total_preconditions(const char* module_name)
{
  bool success;

  is_user_view = false;
  is_transformer = false;
  is_total_precondition = true;
  is_transformer_filtered =
    get_bool_property("SEMANTICS_FILTERED_PRECONDITIONS");
  success = print_code_semantics(module_name);
  return success;
}

bool print_source_transformers(const char* module_name)
{
  is_user_view = true;
  is_transformer = true;
  is_total_precondition = false;
  is_transformer_filtered = false;
  return print_code_semantics(module_name);
}

bool print_source_preconditions(const char* module_name)
{
  is_user_view = true;
  is_transformer = false;
  is_total_precondition = false;
  is_transformer_filtered =
    get_bool_property("SEMANTICS_FILTERED_PRECONDITIONS");
  return print_code_semantics(module_name);
}

bool print_source_total_preconditions(const char* module_name)
{
  is_user_view = true;
  is_transformer = false;
  is_total_precondition = true;
  is_transformer_filtered =
    get_bool_property("SEMANTICS_FILTERED_PRECONDITIONS");
  return print_code_semantics(module_name);
}

text get_text_transformers(const char* module_name)
{
  is_user_view = false;
  is_transformer = true;
  is_total_precondition = false;
  is_transformer_filtered = false;
  return get_semantic_text(module_name,false);
}

text get_text_preconditions(const char* module_name)
{
  is_user_view = false;
  is_transformer = false;
  is_total_precondition = false;
  is_transformer_filtered =
    get_bool_property("SEMANTICS_FILTERED_PRECONDITIONS");
  return get_semantic_text(module_name,false);
}

text get_text_total_preconditions(const char* module_name)
{
  is_user_view = false;
  is_transformer = false;
  is_total_precondition = true;
  is_transformer_filtered =
    get_bool_property("SEMANTICS_FILTERED_PRECONDITIONS");
  return get_semantic_text(module_name,false);
}

static bool print_code_semantics(const char* module_name)
{
    bool success = true;
    text t = text_undefined;

    char * file_ext =
      strdup(concatenate(is_transformer?
			 (is_user_view? USER_TRANSFORMER_SUFFIX :
			  SEQUENTIAL_TRANSFORMER_SUFFIX )
		     :
			 (is_total_precondition?
			  (is_user_view? USER_TOTAL_PRECONDITION_SUFFIX :
			   SEQUENTIAL_TOTAL_PRECONDITION_SUFFIX) :
			  (is_user_view? USER_PRECONDITION_SUFFIX :
			   SEQUENTIAL_PRECONDITION_SUFFIX)),

			 get_bool_property
			 ("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
			 GRAPH_FILE_EXT : "",

			 /* To exploit the language sensitive prettyprint ability of the display */
			 c_module_p(module_name_to_entity(module_name))? ".c" : ".f",

			 NULL));

    char * resource_name =
	get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	    DBR_GRAPH_PRINTED_FILE :
		(is_user_view? DBR_PARSED_PRINTED_FILE :
		 DBR_PRINTED_FILE);

    begin_attachment_prettyprint();
    t = get_semantic_text(module_name,true);
    success = make_text_resource(module_name, resource_name, file_ext, t);
    end_attachment_prettyprint();

    free_text(t);
    free(file_ext);
    return success;
}

static text get_semantic_text(const char* module_name, bool give_code_p)
{
  text r = make_text(NIL), txt_summary;
  entity mod;
  statement mod_stat;
  transformer summary = transformer_undefined;
  statement user_stat = statement_undefined;

  set_current_module_entity(module_name_to_entity(module_name));
  mod = get_current_module_entity();

  set_current_module_statement
    ((statement)db_get_memory_resource(DBR_CODE, module_name, true) );
  mod_stat = get_current_module_statement();

  /* To set up the hash table to translate value into value names */
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource
			   (DBR_CUMULATED_EFFECTS, module_name, true));

  debug_on("SEMANTICS_PRINT_DEBUG_LEVEL");

  module_to_value_mappings(mod);

  if(is_user_view) {
    user_stat =  (statement)
      db_get_memory_resource(DBR_PARSED_CODE, module_name, true);

    nts = allocate_number_to_statement();
    nts = build_number_to_statement(nts, mod_stat);

    ifdebug(1) print_number_to_statement(nts);
  }

  set_semantic_map((statement_mapping)
      db_get_memory_resource(is_transformer? DBR_TRANSFORMERS:
			     (is_total_precondition?
			      DBR_TOTAL_PRECONDITIONS : DBR_PRECONDITIONS),
			     module_name, true));

  summary = (transformer)
      db_get_memory_resource(is_transformer? DBR_SUMMARY_TRANSFORMER:
			     (is_total_precondition?
		    DBR_SUMMARY_TOTAL_PRECONDITION: DBR_SUMMARY_PRECONDITION),
			     module_name, true);
  /* The summary precondition may be in another module's frame */
  translate_global_values(mod, summary);

  init_prettyprint(semantic_to_text);

  /* initial version; to be used again when prettyprint really prettyprints*/
  /* print_text(fd, text_statement(mod, 0, mod_stat)); */

  /* summary information first */
  txt_summary = text_transformer(summary);
  ifdebug(7){
    dump_text(txt_summary);
    pips_debug(7, "summary text consistent? %s\n",
	       text_consistent_p(txt_summary)? "YES":"NO");
  }
  MERGE_TEXTS(r,txt_summary );
  attach_decoration_to_text(r);
  if (is_transformer) {
    attach_transformers_decoration_to_text(r);
  }
  else if(is_total_precondition) {
    attach_total_preconditions_decoration_to_text(r);
  }
  else {
    attach_preconditions_decoration_to_text(r);
  }

  if (give_code_p == true) {
    MERGE_TEXTS(r, text_module(mod, is_user_view? user_stat:mod_stat));
  }

  debug_off();

  if(is_user_view) {
    hash_table_free(nts);
    nts = hash_table_undefined;
  }

  close_prettyprint();

  reset_semantic_map();
  reset_current_module_entity();
  reset_current_module_statement();
  reset_cumulated_rw_effects();

  free_value_mappings();

  return r;
}

/* this function name is VERY misleading - it should be changed, sometime FI */
text semantic_to_text(
  entity module,
  int margin,
  statement stmt)
{
    transformer t;
    text txt;

    pips_assert("To please the compiler", module==module && margin==margin);

    if(is_user_view) {
	statement i = apply_number_to_statement(nts, statement_number(stmt));

	if(!statement_undefined_p(i)) {
	    t = load_statement_semantic(i);
	}
	else
	    t = (transformer) HASH_UNDEFINED_VALUE;
    }
    else
	t = load_statement_semantic(stmt);

    if(is_transformer_filtered && !transformer_undefined_p(t)
	&& t != (transformer) HASH_UNDEFINED_VALUE) {
	list ef = load_cumulated_rw_effects_list(stmt);
	t = filter_transformer(t, ef);
    }

    txt = text_transformer(t);

    if (is_transformer)
	attach_transformers_decoration_to_text(txt);
    else if (is_total_precondition)
	attach_total_preconditions_decoration_to_text(txt);
    else
	attach_preconditions_decoration_to_text(txt);

    return txt;
}

/* The strange argument type is required by qsort(), deep down in the
 * calls.
 *
 * FI: what is it supposed to do? Is it allowed to return with
 * equal==0, which leads to an assert failure later above
 * vect_lexicographic_unsafe_compare_generic ().
 */
static int
semantics_is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
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
    else {
	is_equal =
	    strcmp(pips_user_value_name((entity) vecteur_var(*pvarval1)),
		   pips_user_value_name((entity) vecteur_var(*pvarval2)));
	if(is_equal==0 && !vect_equal(*pvarval1, *pvarval2))
	  is_equal =
	    strcmp(entity_name((entity) vecteur_var(*pvarval1)),
		   entity_name((entity) vecteur_var(*pvarval2)));
    }

    return is_equal;
}

#define append(s) add_to_current_line(crt_line, s, str_prefix, txt)

static bool __attribute__ ((unused))
value_is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
  bool is_inferior = true;

  if (term_cst(*pvarval1))
    is_inferior = false;
  else if(term_cst(*pvarval2))
    is_inferior = true;
  else
    is_inferior = (strcmp(external_value_name((entity) vecteur_var(*pvarval1)),
			  external_value_name((entity) vecteur_var(*pvarval2)))
		   > 0 );

    return is_inferior;
}

/* text text_transformer(transformer tran)
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
text text_transformer(transformer tran)
{
  text txt = make_text(NIL);
  bool foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
  string str_prefix;
  char crt_line[MAX_LINE_LENGTH];

  /* If in EMACS mode, does not add any separator line: */
  if (get_bool_property("PRETTYPRINT_ANALYSES_WITH_LF")
      && !get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES"))
    ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
					    strdup("\n")));

  str_prefix = foresys? FORESYS_CONTINUATION_PREFIX: get_comment_continuation();
  crt_line[0] = '\0';

  if (foresys)
    append(is_transformer? TRAN_FORESYS_PREFIX: PREC_FORESYS_PREFIX);
  else
    append(get_comment_sentinel());

  if(tran != (transformer) HASH_UNDEFINED_VALUE &&
     tran != (transformer) list_undefined) {
    if(tran==transformer_undefined) {
      if (is_transformer)
	append(" TRANSFORMER: TRANSFORMER_UNDEFINED");
      else if(is_total_precondition)
	append(" TOTAL PRECONDITION: TRANSFORMER_UNDEFINED");
      else
	append(" PRECONDITION: TRANSFORMER_UNDEFINED");
    }
    else {
      Psysteme ps;
      list args = transformer_arguments(tran);

      append(is_transformer? "  T(": (is_total_precondition? " TP(" : "  P("));

      entity_list_text_format(crt_line, str_prefix, txt,
			      args, entity_minimal_name);

      append(")");
      if (foresys) append(",");
      append(" ");

      ps = sc_copy(predicate_system(transformer_relation(tran)));
      sc_lexicographic_sort(ps, semantics_is_inferior_pvarval);

      ifdebug(7) {
	pips_debug(7, "sys %p\n", ps);
	sc_syst_debug(ps);
      }

      system_text_format(crt_line, str_prefix, txt, ps,
			 (char * (*)(Variable)) pips_user_value_name, foresys);

      sc_rm(ps);
    }

    close_current_line(crt_line, txt, str_prefix);
  }

  if (get_bool_property("PRETTYPRINT_ANALYSES_WITH_LF")
      && !get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES"))
    ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
					    strdup("\n")));

  ifdebug(7){
    pips_debug(7, "final txt: \n");
    dump_text(txt);
  }

  return txt;
}

/* call this one from outside.
 */
text text_for_a_transformer(transformer tran, bool is_a_transformer)
{
  bool save_is_transformer = is_transformer;
  bool save_is_total_precondition = is_total_precondition;
  text t = text_undefined;

  is_transformer = is_a_transformer;
  is_total_precondition = false;
  t = text_transformer(tran);
  is_transformer = save_is_transformer;
  is_total_precondition = save_is_total_precondition;
  return t;
}


/* ---------------------------------------------------------------- */
/* to convert strings containing predicates to text of commentaries */
/* BA, april 1994                                                   */
/* ---------------------------------------------------------------- */

#define MAX_PRED_COMMENTARY_STRLEN 70


/* text string_predicate_to_commentary(string str_pred, string comment_prefix)
 * input    : a string, part of which represents a predicate.
 * output   : a text consisting of several lines of commentaries,
 *            containing the string str_pred, and beginning with
 *            comment_prefix.
 * modifies : str_pred;
 */
text string_predicate_to_commentary(str_pred, comment_prefix)
string str_pred;
string comment_prefix;
{
  text t_pred = make_text(NIL);
  string str_suiv = NULL;
  string str_prefix = comment_prefix;
  char str_tmp[MAX_PRED_COMMENTARY_STRLEN];
  int len, new_str_pred_len, longueur_max;
  bool premiere_ligne = true;
  bool foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
  longueur_max = MAX_PRED_COMMENTARY_STRLEN - strlen(str_prefix) - 2;

  /* if str_pred is too long, it must be splitted in several lines;
   * the hyphenation must be done only between the constraints of the
   * predicate, when there is a "," or a ")". A space is added at the beginning
   * of extra lines, for indentation. */
  while((len = strlen(str_pred)) > 0) {
    if (len > longueur_max) {

      /* search the maximal substring which length
       * is less than longueur_max */
      str_tmp[0] = '\0';
      (void) strncat(str_tmp, str_pred, longueur_max);

      if (foresys)
	str_suiv = strrchr(str_tmp, ')');
      else
	str_suiv = strrchr(str_tmp, ',');

      new_str_pred_len = (strlen(str_tmp) - strlen(str_suiv)) + 1;
      str_suiv = strdup(&(str_pred[new_str_pred_len]));

      str_tmp[0] = '\0';
      if (!premiere_ligne)
	(void) strcat(str_tmp, " ");
      (void) strncat(str_tmp, str_pred, new_str_pred_len);

      /* add it to the text */
      ADD_SENTENCE_TO_TEXT(t_pred,
			   make_pred_commentary_sentence(strdup(str_tmp),
							 str_prefix));
      str_pred =  str_suiv;
    }
    else {
      /* if the remaining string fits in one line */
      str_tmp[0] = '\0';
      if (!premiere_ligne)
	(void) strcat(str_tmp, " ");
      (void) strcat(str_tmp, str_pred);

      ADD_SENTENCE_TO_TEXT(t_pred,
			   make_pred_commentary_sentence(str_tmp,
							 str_prefix));
      str_pred[0] = '\0';
    }

    if (premiere_ligne) {
      premiere_ligne = false;
      longueur_max = longueur_max - 1;
      if (foresys){
	int i;
	int nb_espaces = strlen(str_prefix) -
	  strlen(FORESYS_CONTINUATION_PREFIX);

	str_prefix = strdup(str_prefix);
	str_prefix[0] = '\0';
	(void) strcat(str_prefix, FORESYS_CONTINUATION_PREFIX);
	for (i=1; i <= nb_espaces; i++)
	  (void) strcat(str_prefix, " ");
      }
    }
  }

  return(t_pred);
}

/* text words_predicate_to_commentary(list w_pred, string comment_prefix)
 * input    : a list of strings, one of them representing a predicate.
 * output   : a text of several lines of commentaries containing
 *            this list of strings, and beginning with comment_prefix.
 * modifies : nothing.
 */
text words_predicate_to_commentary(w_pred, comment_prefix)
list w_pred;
string comment_prefix;
{
  string str_pred;
  text t_pred;

  /* str_pred is the string corresponding to the concatenation
   * of the strings in w_pred */
  str_pred = words_to_string(w_pred);

  t_pred = string_predicate_to_commentary(str_pred, comment_prefix);

  return(t_pred);
}


/* sentence make_pred_commentary_sentence(string str_pred,
 *                                        string comment_prefix)
 * input    : a substring formatted to be a commentary
 * output   : a sentence, containing the commentary form of this string,
 *            beginning with the comment_prefix.
 * modifies : nothing
 */
sentence
make_pred_commentary_sentence(str_pred, comment_prefix)
string str_pred;
string comment_prefix;
{
  /*
  char str_tmp[MAX_PRED_COMMENTARY_STRLEN + 1];

  str_tmp[0] = '\0';
  (void) strcat(str_tmp, comment_prefix);
  (void) strcat(str_tmp, "  ");
  (void) strcat(str_tmp, str_pred);
  (void) strcat(str_tmp, "\n");
  */
  char * str_tmp = NULL;
  sentence sent_pred = sentence_undefined;

  str_tmp = strdup(concatenate(comment_prefix, "  ", str_pred, "\n", NULL));
  sent_pred = make_sentence_formatted(str_tmp);
  return(sent_pred);
}
