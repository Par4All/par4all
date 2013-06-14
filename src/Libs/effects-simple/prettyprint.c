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
/* package simple effects :  Be'atrice Creusillet 5/97
 *
 * File: prettyprint.c
 * ~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the intanciation of the generic functions
 * necessary
 * for the computation of all types of simple effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "properties.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"

#include "text-util.h"
#include "database.h"
#include "resources.h"

#include "effects-generic.h"
#include "effects-simple.h"

static string continuation = string_undefined;
#define CONTINUATION (string_undefined_p(continuation)? \
 strdup(concatenate(get_comment_continuation(), "                           ", NULL)) \
 : continuation)

/* new definitions for action interpretation
 */
#define ACTION_read		"read   "
#define ACTION_write		"written"
#define ACTION_in		"imported"
#define ACTION_out		"exported"
#define ACTION_live_in		"alive (in)"
#define ACTION_live_out		"alive (out)"
/* Can be used both for environment and type declaration */
#define ACTION_declared		"declared"
#define ACTION_referenced	"referenced"



/* Try to factorize code from Alexis Platonoff :

   Append an_effect_string to an_effect_text and position the
   exist_flag at TRUE.

   The procedure is intended to be called for the 4 different types of
   effects. RK

   Now, for 12 types of effects, but I do not see the exist_flag
   positionning (FI)
*/
static void
update_an_effect_type(
    text an_effect_text,
    string current_line_in_construction,
    string an_effect_string)
{
    add_to_current_line(current_line_in_construction, " ",
			CONTINUATION, an_effect_text);
    add_to_current_line(current_line_in_construction, an_effect_string,
			CONTINUATION, an_effect_text);
}


/* text simple_effects_to_text(list sefs_list)
 *
 * Updated version of store_text_line for the generic implementation of
 * effects computation. BC, June 17th, 1997.
 *
 * New version of store_text_line() by AP, Nov 4th, 1995.
 *
 * This function builds the prettyprint in text format of the list of
 * effects given in arguments.
 *
 * These effects are split into four categories : R-MAY, W-MAY, R-EXACT and
 * W-EXACT. Then, we first build four texts, each containing the
 * prettyprint of one kind. Finally, we merge all the texts into a single
 * one.
 *
 * For a given kind of effect, we use a static buffer (of size the length
 * of a line of prettyprint) and a boolean, and we build a text in which
 * each sentence corresponds to a line of prettyprint. If the
 * concatenation of a new effect in the buffer is impossible (overfull),
 * we save the buffer as a new sentence in the text and begin a new line.
 * Initially, the bool is set to false, and is turned to true as soon
 * as we found one effect of this kind in the list. Finally, we merge the
 * text into the global one only if the bool is TRUE.
 *
 * Moreover, we sort the effects list in lexicographic order on the
 * references. We use gen_sort_list().
 *
 * Modification: AP, Nov 10th, 1995. I have replaced the call to
 * effect_to_string() by adirect transformation of an effect to its
 * prettyprint format. This is to avoid the problem occuring when the
 * buffer used in effect_to_string() is too small.
 */

#define is_may		"             <may be "
#define is_exact		"             <    is "
#define exact_end	">:"
#define may_end		exact_end

static text
simple_effects_to_text(
    list /* of effect */ sefs_list,
    string ifread,
    string ifwrite,
    string ifdeclared,
    string ifreferenced)
{
  /* FI: althoug the internal representation does distinguish between
     variable declarations and type declarations, this print-out
     ignores the difference. */
  text sefs_text = make_text(NIL), rt, wt, Rt, Wt, dt, Dt, ut, Ut;
  char r[MAX_LINE_LENGTH], w[MAX_LINE_LENGTH],
    R[MAX_LINE_LENGTH],  W[MAX_LINE_LENGTH],
    d[MAX_LINE_LENGTH], D[MAX_LINE_LENGTH], u[MAX_LINE_LENGTH], U[MAX_LINE_LENGTH];
  bool rb = false, Rb = false,
    wb = false, Wb = false,
    db = false, Db = false,
    ub = false, Ub = false;
  list ce = list_undefined;

  if (sefs_list == (list) HASH_UNDEFINED_VALUE ||
      sefs_list == list_undefined)
    {
      pips_debug(9, "Effects list empty\n");
      return sefs_text;
    }

  /* These eight buffers are used to build the current line of prettyprint
     for a given type of effect. */

  r[0] = '\0'; strcat(r, concatenate(get_comment_sentinel(), is_may, ifread, may_end, NULL));
  R[0] = '\0'; strcat(R, concatenate(get_comment_sentinel(), is_exact, ifread, exact_end, NULL));
  w[0] = '\0'; strcat(w, concatenate(get_comment_sentinel(), is_may, ifwrite, may_end, NULL));
  W[0] = '\0'; strcat(W, concatenate(get_comment_sentinel(), is_exact, ifwrite, exact_end, NULL));
  d[0] = '\0'; strcat(d, concatenate(get_comment_sentinel(), is_may, ifdeclared, may_end, NULL));
  D[0] = '\0'; strcat(D, concatenate(get_comment_sentinel(), is_exact, ifdeclared, exact_end, NULL));
  u[0] = '\0'; strcat(u, concatenate(get_comment_sentinel(), is_may, ifreferenced, may_end, NULL));
  U[0] = '\0'; strcat(U, concatenate(get_comment_sentinel(), is_exact, ifreferenced, exact_end, NULL));

  /* These eight "texts" are used to build all the text of prettyprint
     for a given type of effect. Each sentence contains one line. */
  rt = make_text(NIL);
  Rt = make_text(NIL);
  wt = make_text(NIL);
  Wt = make_text(NIL);
  dt = make_text(NIL);
  Dt = make_text(NIL);
  ut = make_text(NIL);
  Ut = make_text(NIL);

  /* We sort the list of effects in lexicographic order */
  if (get_bool_property("PRETTYPRINT_WITH_COMMON_NAMES"))
    gen_sort_list(sefs_list, (gen_cmp_func_t)compare_effect_reference_in_common);
  else
    gen_sort_list(sefs_list, (gen_cmp_func_t)compare_effect_reference);

  /* Walk through all the effects */
  for(ce = sefs_list; !ENDP(ce); POP(ce))
    {
      effect eff = EFFECT(CAR(ce));
      if(store_effect_p(eff)
	 || !get_bool_property("PRETTYPRINT_MEMORY_EFFECTS_ONLY")) {
	reference ref = effect_any_reference(eff);
	action ac = effect_action(eff);
	approximation ap = effect_approximation(eff);
	list /* of string */ ls = effect_words_reference(ref);
	string t;

	/* We build the string containing the effect's reference */
	/* Be careful about attachments since the words references are
	 * heavily moved around in the following. words_to_string is now
	 * attachment safe. RK
	 */
	t = words_to_string(ls);
	gen_free_string_list(ls);

	/* We now proceed to the addition of this effect to the current line
	   of prettyprint. First, we select the type of effect : R-MAY, W-MAY,
	   R-EXACT, W-EXACT. Then, if this addition results in a line too long,
           we save the current line, and begin a new one. */
	if (action_read_p(ac) && approximation_may_p(ap))
	  if(store_effect_p(eff))
	    update_an_effect_type(rt, r, t), rb = true;
	  else
	    update_an_effect_type(ut, u, t), ub = true;
	else if (!action_read_p(ac) && approximation_may_p(ap))
	  if(store_effect_p(eff))
	    update_an_effect_type(wt, w, t), wb = true;
	  else
	    update_an_effect_type(dt, d, t), db = true;
	else if (action_read_p(ac) && !approximation_may_p(ap))
	  if(store_effect_p(eff))
	    update_an_effect_type(Rt, R, t), Rb = true;
	  else
	    update_an_effect_type(Ut, U, t), Ub = true;
	else if (!action_read_p(ac) && !approximation_may_p(ap))
	  if(store_effect_p(eff))
	    update_an_effect_type(Wt, W, t), Wb = true;
	  else
	    update_an_effect_type(Dt, D, t), Db = true;
	else
	  pips_internal_error("unrecognized effect");

	free(t);
      }
    }

  close_current_line(r, rt, CONTINUATION);
  close_current_line(R, Rt, CONTINUATION);
  close_current_line(w, wt, CONTINUATION);
  close_current_line(W, Wt, CONTINUATION);
  close_current_line(d, dt, CONTINUATION);
  close_current_line(D, Dt, CONTINUATION);
  close_current_line(u, ut, CONTINUATION);
  close_current_line(U, Ut, CONTINUATION);

  if (rb) { MERGE_TEXTS(sefs_text, rt); } else free_text(rt);
  if (wb) { MERGE_TEXTS(sefs_text, wt); } else free_text(wt);
  if (ub) { MERGE_TEXTS(sefs_text, ut); } else free_text(ut);
  if (db) { MERGE_TEXTS(sefs_text, dt); } else free_text(dt);
  if (Rb) { MERGE_TEXTS(sefs_text, Rt); } else free_text(Rt);
  if (Wb) { MERGE_TEXTS(sefs_text, Wt); } else free_text(Wt);
  if (Ub) { MERGE_TEXTS(sefs_text, Ut); } else free_text(Ut);
  if (Db) { MERGE_TEXTS(sefs_text, Dt); } else free_text(Dt);

  return sefs_text;
}

/* external interfaces
 */
text simple_rw_effects_to_text(list l)
{ return simple_effects_to_text(l, ACTION_read, ACTION_write, ACTION_declared, ACTION_referenced);}

text simple_inout_effects_to_text(list l)
{ return simple_effects_to_text(l, ACTION_in, ACTION_out, ACTION_declared, ACTION_referenced);}

text simple_live_in_paths_to_text(list l)
{ return simple_effects_to_text(l, ACTION_live_in, ACTION_write, ACTION_referenced, ACTION_referenced);}

text simple_live_out_paths_to_text(list l)
{ return simple_effects_to_text(l, ACTION_live_out, ACTION_write, ACTION_referenced, ACTION_referenced);}

string
effect_to_string(effect eff)
{
    list /* of string */ ls = effect_words_reference(effect_any_reference(eff));
    string result = words_to_string(ls);
    gen_free_string_list(ls);
    return result;
}

/* Assumed that the cell is a preference and not a reference. */
list words_effect_generic(effect obj, bool approximation_p)
{
    list pc = NIL;
    reference r = effect_any_reference(obj);
    action ac = effect_action(obj);
    approximation ap = effect_approximation(obj);

    pc = CHAIN_SWORD(pc,"<");
    pc = gen_nconc(pc, effect_words_reference(r));
    pc = CHAIN_SWORD(pc,"-");
    pc = CHAIN_SWORD(pc, full_action_to_short_string(ac));
    if(approximation_p)
      pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "-MAY>" : "-EXACT>" );
    return (pc);
}

list words_effect(effect obj)
{
  return words_effect_generic(obj, true);
}

list words_effect_without_approximation(effect obj)
{
  return words_effect_generic(obj, false);
}

void print_effect(effect e)
{
  if(!effect_undefined_p(e))
    {
      effect_consistent_p(e);
      fprintf(stderr, "\t");
      print_words(stderr, words_effect(e));
      fprintf(stderr, "\n");
    }
  else
    fprintf(stderr,"\t effect undefined\n");
}

/* FI: could be placed in effect-generic */
void print_effects( list pc)
{
  if (pc != NIL) {
    FOREACH(EFFECT, e, pc) {
      descriptor d = effect_descriptor(e);
      if(descriptor_none_p(d))
	print_effect(e);
      else {
	void print_region(effect);
	print_region(e);
      }
    }
  }
  else
    fprintf(stderr, "\t<NONE>\n");
}

void print_memory_effects( list pc)
{
  if (pc != NIL) {
    FOREACH(EFFECT, e, pc)
      {
	action a = effect_action(e);
	action_kind ak = action_read_p(a)? action_read(a) : action_write(a);
	if(action_kind_store(ak)
	   || !get_bool_property("PRETTYPRINT_MEMORY_EFFECTS_ONLY"))
	  print_effect(e);
      }
  }
  else
    fprintf(stderr, "\t<NONE>\n");
}

