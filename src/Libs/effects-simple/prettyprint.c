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
/* package simple effects :  Be'atrice Creusillet 5/97
 *
 * File: prettyprint.c
 * ~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the intanciation of the generic functions necessary 
 * for the computation of all types of simple effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "properties.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "text.h"

#include "text-util.h"
#include "prettyprint.h"
#include "database.h"
#include "resources.h"
#include "pipsdbm.h"

#include "effects-generic.h"
#include "effects-simple.h"

static string continuation = string_undefined;
#define CONTINUATION (string_undefined_p(continuation)? \
 strdup(concatenate(PIPS_COMMENT_CONTINUATION, "                              ", NULL)) \
 : continuation)

/* new definitions for action interpretation
 */
#define ACTION_read 	"read   "
#define ACTION_write	"written"
#define ACTION_in   	"imported"
#define ACTION_out	"exported"

/* int compare_effect_reference(e1, e2):
 *
 * returns -1 if "e1" is before "e2" in the alphabetic order, else
 * +1. "e1" and "e2" are pointers to effect, we compare the names of their
 * reference's entity. */
int
compare_effect_reference(effect * e1, effect * e2)
{
  reference r1 = effect_any_reference(*e1);
  reference r2 = effect_any_reference(*e2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  int n1, n2;
  /* FI: might not be best... entity_unambiguous_user_name()? */
  string s1 = entity_name_without_scope(v1);
  string s2 = entity_name_without_scope(v2);
  int result;

  n1 = (v1==(entity)NULL),
  n2 = (v2==(entity)NULL);
  result = (n1 || n2)?  (n2-n1): strcmp(s1,s2);
  free(s1);
  free(s2);
  if(result==0) {
    list ind1 = reference_indices(r1);
    list ind2 = reference_indices(r2);
    list cind1 = list_undefined;
    list cind2 = list_undefined;
    int diff = 0;

    if(ENDP(ind1))
      if(ENDP(ind2))
	diff = 0;
      else
	diff = -1;
    else
      if(ENDP(ind2))
	diff = 1;
      else
	diff = 0;

    for(cind1 = ind1, cind2 = ind2; !ENDP(cind1) && !ENDP(cind2) && diff ==0; POP(cind1), POP(cind2)) {
      expression e1 = EXPRESSION(CAR(cind1));
      expression e2 = EXPRESSION(CAR(cind2));

      if(unbounded_expression_p(e1))
	if(unbounded_expression_p(e2))
	  diff = 0;
	else
	  diff = 1;
      else
	if(unbounded_expression_p(e2))
	  diff = -1;
	else {
	  int i1 = 0;
	  int i2 = 0;

	  /* FI: This is not enough as effects are not summarized
	     right away. It may be impossible to find an integer value
	     for e1 and/or e2. The output is till not deterministic. */
	  expression_integer_value(e1, &i1);
	  expression_integer_value(e2, &i2);
	  diff = i1 - i2;
	}
    }
    result = diff==0? 0 : (diff>0?1:-1);
  }
  return result;
}

/* int compare_effect_reference_in_common(e1, e2):
 *
 * returns -1 if "e1" is before "e2" in the alphabetic order, else
 * +1. "e1" and "e2" are pointers to effect, we compare the names of their
 * reference's entity with the common name in first if the entity belongs  
 * to a common */
int
compare_effect_reference_in_common(effect * e1, effect * e2)
{
  entity v1, v2;
  int n1, n2 ,result;
  string name1, name2;
  v1 = reference_variable(effect_any_reference(*e1));
  v2 = reference_variable(effect_any_reference(*e2));
  n1 = (v1==(entity)NULL),
  n2 = (v2==(entity)NULL);
  name1= strdup((entity_in_common_p(v1)) ? 
      (string) entity_and_common_name(v1):
      entity_name(v1));
  name2=  strdup((entity_in_common_p(v2)) ? 
      (string) entity_and_common_name(v2):
      entity_name(v2));
  
  result =  (n1 || n2)?  (n2-n1): strcmp(name1,name2);
  free(name1);free(name2);  
  return result;
}


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
 * These effects are split into four categories : R-MAY, W-MAY, R-MUST and
 * W-MUST. Then, we first build four texts, each containing the
 * prettyprint of one kind. Finally, we merge all the texts into a single
 * one.
 *
 * For a given kind of effect, we use a static buffer (of size the length
 * of a line of prettyprint) and a boolean, and we build a text in which
 * each sentence corresponds to a line of prettyprint. If the
 * concatenation of a new effect in the buffer is impossible (overfull),
 * we save the buffer as a new sentence in the text and begin a new line.
 * Initially, the boolean is set to FALSE, and is turned to TRUE as soon
 * as we found one effect of this kind in the list. Finally, we merge the
 * text into the global one only if the boolean is TRUE.
 *
 * Moreover, we sort the effects list in lexicographic order on the
 * references. We use gen_sort_list().
 *
 * Modification: AP, Nov 10th, 1995. I have replaced the call to
 * effect_to_string() by adirect transformation of an effect to its
 * prettyprint format. This is to avoid the problem occuring when the
 * buffer used in effect_to_string() is too small.
 */

#define may_be 		"               <may be "
#define must_be 	"               <must be "
#define must_end	">:"
#define may_end  	" " must_end

static text
simple_effects_to_text(
    list /* of effect */ sefs_list, 
    string ifread, 
    string ifwrite)
{
    text sefs_text = make_text(NIL), rt, wt, Rt, Wt;
    char r[MAX_LINE_LENGTH], w[MAX_LINE_LENGTH], 
	R[MAX_LINE_LENGTH],  W[MAX_LINE_LENGTH];
    bool rb = FALSE, wb = FALSE, Rb = FALSE, Wb = FALSE;
    list ce;
    
    if (sefs_list == (list) HASH_UNDEFINED_VALUE ||
	sefs_list == list_undefined) 
    {
	pips_debug(9, "Effects list empty\n");
	return sefs_text;
    }

    /* These four buffers are used to build the current line of prettyprint
     for a given type of effect. */
  
    r[0] = '\0'; strcat(r, concatenate(PIPS_COMMENT_SENTINEL, may_be, ifread, may_end, NULL));
    R[0] = '\0'; strcat(R, concatenate(PIPS_COMMENT_SENTINEL, must_be, ifread, must_end, NULL));
    w[0] = '\0'; strcat(w, concatenate(PIPS_COMMENT_SENTINEL, may_be, ifwrite, may_end, NULL));
    W[0] = '\0'; strcat(W, concatenate(PIPS_COMMENT_SENTINEL, must_be, ifwrite, must_end, NULL));

    /* These four "texts" are used to build all the text of prettyprint
       for a given type of effect. Each sentence contains one line. */
    rt = make_text(NIL);
    wt = make_text(NIL);
    Rt = make_text(NIL);
    Wt = make_text(NIL);

    /* We sort the list of effects in lexicographic order */
     if (get_bool_property("PRETTYPRINT_WITH_COMMON_NAMES")) 
	 gen_sort_list(sefs_list, compare_effect_reference_in_common);
     else 
	 gen_sort_list(sefs_list, compare_effect_reference);
  
    /* Walk through all the effects */
    for(ce = sefs_list; !ENDP(ce); POP(ce)) 
    {
	effect eff = EFFECT(CAR(ce));
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
	   R-MUST, W-MUST. Then, if this addition results in a line too long,
           we save the current line, and begin a new one. */
	if (action_read_p(ac) && approximation_may_p(ap))
	    update_an_effect_type(rt, r, t), rb = TRUE;
	else if (!action_read_p(ac) && approximation_may_p(ap))
	    update_an_effect_type(wt, w, t), wb = TRUE;
	else if (action_read_p(ac) && !approximation_may_p(ap))
	    update_an_effect_type(Rt, R, t), Rb = TRUE;
	else if (!action_read_p(ac) && !approximation_may_p(ap))
	    update_an_effect_type(Wt, W, t), Wb = TRUE;
	else
	    pips_internal_error("unrecognized effect");

	free(t);
    }
    
    close_current_line(r, rt, CONTINUATION);
    close_current_line(w, wt, CONTINUATION);
    close_current_line(R, Rt, CONTINUATION);
    close_current_line(W, Wt, CONTINUATION);

    if (rb) { MERGE_TEXTS(sefs_text, rt); } else free_text(rt);
    if (wb) { MERGE_TEXTS(sefs_text, wt); } else free_text(wt);
    if (Rb) { MERGE_TEXTS(sefs_text, Rt); } else free_text(Rt);
    if (Wb) { MERGE_TEXTS(sefs_text, Wt); } else free_text(Wt);

    return sefs_text;
}

/* external interfaces
 */
text simple_rw_effects_to_text(list l) 
{ return simple_effects_to_text(l, ACTION_read, ACTION_write);}

text simple_inout_effects_to_text(list l)
{ return simple_effects_to_text(l, ACTION_in, ACTION_out);}

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
    pc = CHAIN_SWORD(pc, action_read_p(ac) ? "-R" : "-W" );  
    if(approximation_p)
      pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "-MAY>" : "-MUST>" );
    return (pc);
}

list words_effect(effect obj)
{
  return words_effect_generic(obj, TRUE);
}

list words_effect_without_approximation(effect obj)
{
  return words_effect_generic(obj, FALSE);
}

void 
print_effects( list pc)
{
    if (pc != NIL) {
        while (pc != NIL) {
            effect e = EFFECT(CAR(pc));
            fprintf(stderr, "\t");
            print_words(stderr, words_effect(e));
            fprintf(stderr, "\n");
            pc = CDR(pc);
        }
    }
    else 
	fprintf(stderr, "\t<NONE>\n");
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
