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

#define MAX_LINE_LENGTH 60
#define LINE_PREFIX "C\t\t                  "
#define LINE_SUFFIX "\n"

/* new definitions for action interpretation
 */
#define ACTION_read 	"read   "
#define ACTION_write	"written"
#define ACTION_in   	"imported"
#define ACTION_out	"exported"

/* ======================================================================== */
/* int compare_effect_reference(e1, e2):
 *
 * returns -1 if "e1" is before "e2" in the alphabetic order, else
 * +1. "e1" and "e2" are pointers to effect, we compare the names of their
 * reference's entity. */
int
compare_effect_reference(effect * e1,
			 effect * e2)
{
  entity v1, v2;
  int n1, n2;

  v1 = reference_variable(effect_reference(*e1));
  v2 = reference_variable(effect_reference(*e2));
  n1 = (v1==(entity)NULL),
  n2 = (v2==(entity)NULL);
  if (n1 || n2) 
    return(n2-n1);
  else
    return(strcmp(entity_name(v1), entity_name(v2)));
}

/* ======================================================================== */
/* Try to factorize code from Alexis Platonoff :

   Append an_effect_string to an_effect_text an position the
   exist_flag at TRUE. The procedure is intended to be called for the
   4 different types of effects. RK
   */
void
update_an_effect_type(bool * exist_flag,
		      text an_effect_text,
		      string current_line_in_construction,
		      string an_effect_string)
{
    /* There is at least one effect of this type that will need to be
       displayed: */
    *exist_flag = TRUE;
    if (strlen(current_line_in_construction) + strlen(an_effect_string)
	> MAX_LINE_LENGTH - 3) {
	(void) strcat(current_line_in_construction, LINE_SUFFIX);
	ADD_SENTENCE_TO_TEXT(an_effect_text,
			     make_sentence(is_sentence_formatted,
					   strdup_and_migrate_attachments(current_line_in_construction)));
	/* Reset the line under construction: */
	current_line_in_construction[0] = '\0';
	(void) strcat(current_line_in_construction, LINE_PREFIX); 
	if (strlen(current_line_in_construction) + strlen(an_effect_string)
	    > MAX_LINE_LENGTH - 3) {
	    user_warning("store_text_line",
			 "read-must line buffer too small..."
			 "effect discarded!");
	    return;
	}
    }
    (void) strcat(current_line_in_construction, " ");
    (void) strcat_word_and_migrate_attachments(current_line_in_construction,
					       an_effect_string);
}


/* ======================================================================== */
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

#define may_be 		"C\t\t<may be "
#define must_be 	"C\t\t<must be "
#define must_end	">:"
#define may_end  	" " must_end

static text
simple_effects_to_text(list sefs_list, string ifread, string ifwrite)
{
  text sefs_text = make_text(NIL);
  text rt, wt, Rt, Wt;
  char *t = NULL;
  char r[MAX_LINE_LENGTH]; /* !!! HUM... */
  char w[MAX_LINE_LENGTH];
  char R[MAX_LINE_LENGTH];
  char W[MAX_LINE_LENGTH];
  list ce;

  /* These booleans are used to say if a given type of effect is not
     empty, i.e. there is some information to prettyprint. */
  boolean rb = FALSE, wb = FALSE, Rb = FALSE, Wb = FALSE;

  if (sefs_list == (list) HASH_UNDEFINED_VALUE ||
      sefs_list == list_undefined) {

    debug (9,"store_text_line", "Effects list empty\n");
    return sefs_text;
  }

  /* These four buffers are used to build the current line of prettyprint
     for a given type of effect. */
  
  r[0] = '\0'; (void) strcat(r, concatenate(may_be, ifread, may_end, NULL));
  R[0] = '\0'; (void) strcat(R, concatenate(must_be, ifread, must_end, NULL));
  w[0] = '\0'; (void) strcat(w, concatenate(may_be, ifwrite, may_end, NULL));
  W[0] = '\0'; (void) strcat(W, concatenate(must_be, ifwrite, must_end, NULL));


  /* These four "texts" are used to build all the text of prettyprint
     for a given type of effect. Each sentence contains one line. */
  rt = make_text(NIL);
  wt = make_text(NIL);
  Rt = make_text(NIL);
  Wt = make_text(NIL);

  /* We sort the list of effects in lexicographic order */
  gen_sort_list(sefs_list, compare_effect_reference);
  
  /* Walk through all the effects */
  for(ce = sefs_list; !ENDP(ce); POP(ce)) {
    effect eff = EFFECT(CAR(ce));
    reference ref = effect_reference(eff);
    action ac = effect_action(eff);
    approximation ap = effect_approximation(eff);

    /* We build the string containing the effect's reference */
    /* Be careful about attachments since the words references are
       heavily moved around in the following. words_to_string is now
       attachment safe. RK */
    t = words_to_string(effect_words_reference(ref));

    /* We now proceed to the addition of this effect to the current line
       of prettyprint. First, we select the type of effect : R-MAY, W-MAY,
       R-MUST, W-MUST. Then, if this addition results in a line too long,
       we save the current line, and begin a new one. */
    if (action_read_p(ac) && approximation_may_p(ap))
	update_an_effect_type(&rb, rt, r, t);
    else if (!action_read_p(ac) && approximation_may_p(ap))
	update_an_effect_type(&wb, wt, w, t);
    else if (action_read_p(ac) && !approximation_may_p(ap))
	update_an_effect_type(&Rb, Rt, R, t);
    else if (!action_read_p(ac) && !approximation_may_p(ap))
	update_an_effect_type(&Wb, Wt, W, t);
    else
	pips_assert("unrecognized effect", 0);
  }

  /* For each kind of effect, save the last line and then the text. Only
     done if it is not empty. */
  if(rb) {
    (void) strcat(r, LINE_SUFFIX);
    ADD_SENTENCE_TO_TEXT(rt, make_sentence(is_sentence_formatted,
					   strdup_and_migrate_attachments(r)));
    MERGE_TEXTS(sefs_text, rt);
  }
  if(wb) {
    (void) strcat(w, LINE_SUFFIX);
    ADD_SENTENCE_TO_TEXT(wt, make_sentence(is_sentence_formatted,
					   strdup_and_migrate_attachments(w)));
    MERGE_TEXTS(sefs_text, wt);
  }
  if(Rb) {
    (void) strcat(R, LINE_SUFFIX);
    ADD_SENTENCE_TO_TEXT(Rt, make_sentence(is_sentence_formatted,
					   strdup_and_migrate_attachments(R)));
    MERGE_TEXTS(sefs_text, Rt);
  }
  if(Wb) {
    (void) strcat(W, LINE_SUFFIX);
    ADD_SENTENCE_TO_TEXT(Wt, make_sentence(is_sentence_formatted,
					   strdup_and_migrate_attachments(W)));
    MERGE_TEXTS(sefs_text, Wt);
  }

  return (sefs_text);
}

/* external interfaces
 */
text simple_rw_effects_to_text(list l) 
{ return simple_effects_to_text(l, ACTION_read, ACTION_write);}

text simple_inout_effects_to_text(list l)
{ return simple_effects_to_text(l, ACTION_in, ACTION_out);}

string
effect_to_string(eff)
effect eff;
{
    static char buffer[1024];
    buffer[0] = '\0';

    if(eff == effect_undefined)
	pips_error("effect_to_string", "unexpected effect undefined");
    else {
	(void) strcat(buffer, words_to_string(words_effect(eff)));
    }
    pips_assert("effect_to_string", strlen(buffer)<1023);

    return(strdup(buffer));
}

list
words_effect(obj)
effect obj;
{
    list pc = NIL;
    reference r = effect_reference(obj);
    action ac = effect_action(obj);
    approximation ap = effect_approximation(obj);

    pc = CHAIN_SWORD(pc,"<");
    pc = gen_nconc(pc, effect_words_reference(r));
    pc = CHAIN_SWORD(pc, action_read_p(ac) ? "-R" : "-W" );  
    pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "-MAY>" : "-MUST>" );
    return (pc);
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

 
