/* 	@(#) prettyprint.c 1.48@(#) (97/01/24, 09:12:03) version 1.48, got on 97/01/24, 09:12:08 [/a/chailly/export/users/export/users/pips/Pips/Development/Libs/ri-util/SCCS/s.prettyprint.c].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char lib_ri_util_prettyprint_c_vcid[] = "@(#) prettyprint.c 1.48@(#) (97/01/24, 09:12:03) version 1.48, got on 97/01/24, 09:12:08 [/a/chailly/export/users/export/users/pips/Pips/Development/Libs/ri-util/SCCS/s.prettyprint.c].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */

 /*
  * Prettyprint unstructured
  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"



text
text_unstructured(entity module,
                  string label,
                  int margin,
                  unstructured u, int num)
{
   text r = make_text(NIL);
   hash_table labels = hash_table_make(hash_pointer, 0) ;
   list trail = NIL;
   control cexit = unstructured_exit(u) ;
   control centry = unstructured_control(u) ;

   debug(2, "text_unstructured", "Begin for unstructured %x\n",
	 (unsigned int) u);

   if (get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH"))
   {
      output_a_graph_view_of_the_unstructured
	  (r, module, label, margin, u, num);
   }
   else {
       list pbeg, pend;

       if(get_bool_property("PRETTYPRINT_UNSTRUCTURED")) {
	   pbeg = CHAIN_SWORD(NIL, "BEGIN UNSTRUCTURED");
	    
	   u = make_unformatted(strdup("C"), num, margin, pbeg);
	   ADD_SENTENCE_TO_TEXT(r, 
				make_sentence(is_sentence_unformatted, u));
       }

       /* build an arbitrary trail of control nodes */
       trail = CONS(CONTROL, centry, NIL);
       build_trail(trail);
       trail = gen_nreverse(trail);

       /* decorate control nodes with labels when necessary */
       decorate_trail(module, trail, labels);

       /* generate text with labels and goto's */

       MERGE_TEXTS(r, text_trail(margin, trail, labels));

       if(get_bool_property("PRETTYPRINT_UNSTRUCTURED")) {
	   pend = CHAIN_SWORD(NIL, "END UNSTRUCTURED");
	    
	   u = make_unformatted(strdup("C"), num, margin, pend);
	   ADD_SENTENCE_TO_TEXT(r, 
				make_sentence(is_sentence_unformatted, u));
       }
   }
   
   ifdebug(9) {
      fprintf(stderr,"Unstructured %x (%x, %x)\n", 
              (unsigned int) u, (unsigned int) ct, (unsigned int) cexit ) ;
      CONTROL_MAP( n, {
         statement st = control_statement(n) ;

         fprintf(stderr, "\n%*sNode %x (%s)\n--\n", margin, "", 
                 (unsigned int) n, control_slabel(module, n, labels)) ;
         print_text(stderr, text_statement(module,margin,st));
         fprintf(stderr, "--\n%*sPreds:", margin, "");
         MAPL(ps,{fprintf(stderr,"%x ", (unsigned int) CONTROL(CAR(ps)));},
         control_predecessors(n));
         fprintf(stderr, "\n%*sSuccs:", margin, "") ;
         MAPL(ss,{fprintf(stderr,"%x ", (unsigned int) CONTROL(CAR(ss)));},
         control_successors(n));
         fprintf(stderr, "\n\n") ;
      }, ct , blocs) ;
      gen_free_list(blocs);
   }
   hash_table_free(labels) ;
   set_free(trail) ;

   debug(2, "text_unstructured", "End for unstructured %x\n",
	 (unsigned int) u);

   return(r) ;
}

/* Any heuristics can be used to build the trail, depth or width first,
 * true or false branch first, lower statement numbering first or not,
 * sorted by statement numbering (but some statements have no number...).
 * No simple heuristics seems bullet proof.
 *
 * For CONS convenience, the list is built in reverse order (and reversed
 * by the caller).
 */
static list
build_trail(list l)
{
  control c = control_undefined;
  control succ = control_undefined;

  pips_assert("build trail", !ENDP(l));

  c = CONTROL(CAR(l));
  nsucc = gen_length(control_successors(c));
  switch(nsucc) {
  case 0:
    break;
  case 1:
    succ = CONTROL(CAR(control_successors(c)));
    if(!control_in_trail_p(l,c))
      l = CONS(CONTROL,succ,l);
      l = build_trail(l);
    }
    break;
  case 2:
    /* Follow the false branch in depth first, assuming that IF GOTO's 
     * mainly handle exceptions */
    succ = CONTROL(CAR(CDR(control_successors(c))));
    if(!control_in_trail_p(l,c)) {
      l = CONS(CONTROL,succ,l);
      l = build_trail(l);
    }
    succ = CONTROL(CAR(control_successors(c)));
    if(!control_in_trail_p(l,c)) {
      l = CONS(CONTROL,succ,l);
      l = build_trail(l);
    }
    break;
  default:
    pips_error("build_trail", "Too many successors (%d) for a control node\n",
	       nsucc);
  }
  return l;
}

/* OK, a hash table could be used, as Pierre used too... but the order
 * is lost. You need both order and direct access. Easy to add later if too
 * sloow
 */
static bool
control_in_trail_p(list l, control c)
{
  found = gen_find_eq(l, c) != gen_chunk_undefined;
}

static void
decorate_trail(entity module, list trail, hash_table labels)
{
  list cc = NIL;

  pips_assert("trail cannot be empty", !ENDP(trail));

  for(cc=trail; !ENDP(cc); POP(cc)) {
    control c = CONTROL(CAR(cc));
    int nsucc = gen_length(control_successors(c));

    switch(nsucc) {
    case 0:
      /* No need for a label, it must be the exit node */
      break;
    case 1: {
      control succ = CONTROL(CAR(control_successors(c)));
      /* If the statement "really" has a continuation (e.g. not a STOP
       * or a RETURN
       */
      if(statement_does_return(control_statement(c))) {
	if(!ENDP(cc)) {
	  control tsucc = CONTROL(CAR(CDR(cc)));
	  if(tsucc==succ) {
	    /* the control successor is the textual successor */
	    break;
	  }
	}
	/* A label must be associated with the control successor */
	control_slabel(module, succ, labels);
      }
      break;
    }
    case 2: {
      control succ1 = CONTROL(CAR(control_successors(c)));
      control succ2 = CONTROL(CAR(CDR(control_successors(c))));

      /* Is there a textual successor? */
      if(!ENDP(cc)) {
	control tsucc = CONTROL(CAR(CDR(cc)));
	if(tsucc==succ1) {
	  if(tsucc==succ2) {
	    /* This may happen after restructuring */
	    ;
	  }
	  else {
	    /* succ2 must be labelled */
	    control_slabel(module, succ2, labels);
	  }
	  if(tsucc==succ2) {
	    /* succ1 must be labelled */
	    control_slabel(module, succ1, labels);
	  }
	  else {
	    /* Both successors must be labelled */
	    control_slabel(module, succ1, labels);
	    control_slabel(module, succ2, labels);
	  }
	}
      }
      else {
	/* Both successors must be textual predecessors */
	pips_assert("succ1 before c", appears_first_in_trail(trail, succ1, c));
	pips_assert("succ1 before c", appears_first_in_trail(trail, succ2, c));
	control_slabel(module, succ1, labels);
	control_slabel(module, succ2, labels);
      }
      break;
    }
    default:
      pips_internal_error("Too many successors for a control node\n");
    }
  }
}

/* returns TRUE if c1 is encountered before c2 in trail, or if c1 == c2
 * is encountered in trail, FALSE is c2 is encountered first and c2 != c1,
 * core dumps otherwise, if neither c1 nor c2 is in trail
 */
static bool
appears_first_in_trail(list trail, control c1, control c2)
{
  bool first = FALSE;

  MAPL(cc, {
    control c = CONTROL(CAR(cc));

    if(c==c1) {
      first = TRUE;
      break;
    }
    else if(c==c2) {
      first = FALSE;
      break;
    }
  }, trail);
  pips_assert("At least one control should appear in trail", c==c1 || c==c2);
  return first;
}


/* CONTROL_SLABEL returns a freshly allocated label name for the control
 *  node C in the module M. H maps controls to label names. Computes a new
 * label name if necessary.
 *
 * There is no guarantee that a label generated here appears eventually
 * in the text produced.
 *
 * There is no guarentee that a label generated here is jumped at.
 */

void control_slabel(m, c, h)
entity m;
control c;
hash_table h;
{
    string l;
    statement st = control_statement(c) ;

    if ((l = hash_get(h, (char *) c)) == HASH_UNDEFINED_VALUE) {
	string label = entity_name( statement_to_label( st )) ;

	l = empty_label_p( label ) ? new_label_name(m) : label ;
	debug(3, "control_slabel", "Associates label %s to stmt %s\n",
	      l, statement_identification(st));
	hash_put(h, (char *) c, strdup(l)) ;
    }
    else {
	debug(3, "control_slabel", "Retrieves label %s for stmt %s\n",
	      l, statement_identification(st));
    }

    pips_assert("control_slabel", strcmp(local_name(l), LABEL_PREFIX) != 0) ;
    pips_assert("control_slabel", strcmp(local_name(l), "") != 0) ;
    pips_assert("control_slabel", strcmp(local_name(l), "=") != 0) ;

    return;
}

string
control_has_label(c, h)
control c;
hash_table h;
{
    string l;
    statement st = control_statement(c) ;

    if ((l = hash_get(h, (char *) c)) == HASH_UNDEFINED_VALUE) {
	debug(3, "control_has_label", "Retrieves no label for stmt %s\n",
	      statement_identification(st));
      l = string_undefined;
    }
    else {
	debug(3, "control_has_label", "Retrieves label %s for stmt %s\n",
	      l, statement_identification(st));
	l = strdup(l);
    }

    return l;
}

/* ADD_CONTROL_GOTO adds to the text R a goto statement to the control
node SUCC from the current one OBJ in the MODULE and with a MARGIN.
LABELS maps control nodes to label names and SEENS (that links the
already prettyprinted node) is used to see whether a simple fall-through
wouldn't do. If a go to is effectively added, TRUE is returned. */

static bool add_control_goto(module, margin, r, obj, 
			     succ, labels, seens, cexit )
entity module;
int margin;
text r ;
control obj, succ, cexit ;
hash_table labels;
set seens ;
{
    string label ;
    bool added = FALSE;

    if( succ == (control)NULL ) {
	return added;
    }

    if(!statement_does_return(control_statement(obj))) {
      return added;
    }

    label = local_name(control_slabel(module, succ, labels))+
	    strlen(LABEL_PREFIX);

    /* FI: I broke a large conjunction and duplicated statements to
     * simplify debugging
     */
    if ((strcmp(label, RETURN_LABEL_NAME) == 0 && 
	 return_statement_p(control_statement(obj)))) {
      ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module, margin, label));
      added = TRUE;
    }
    else if(seens == (set)NULL) {
      ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module, margin, label));
      added = TRUE;
    }
    else if((get_bool_property("PRETTYPRINT_INTERNAL_RETURN") &&
	     succ == cexit)) {
      ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module, margin, label));
      added = TRUE;
    }
    else if(set_belong_p(seens, (char *)succ)) {
      ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module, margin, label));
      added = TRUE;
    }
    return added;
}

/* TEXT_CONTROL prettyprints the control node OBJ in the MODULE with a
MARGIN. SEENS is a trail that keeps track of already printed nodes and
LABELS maps control nodes to label names. The previously printed control
is in PREVIOUS. */

text text_control(module, margin, obj, previous, seens, labels, cexit)
entity module;
int margin;
control obj, *previous, cexit ;
set seens ;
hash_table labels;
{
    text r = make_text(NIL);
    sentence s = sentence_undefined;
    unformatted u = unformatted_undefined;
    statement st = control_statement(obj) ;
    list succs = NIL;
    list preds = NIL ;
    list pc = NIL;
    string label = NULL;
    string label_name = NULL;
    string comments = statement_comments(st);
    int npreds = 0;
    bool reachable = FALSE;

    debug(2, "text_control", "Begin for statement %s\n",
	  statement_identification(st));

    label = control_slabel(module, obj, labels);
    label_name = strdup(local_name(label)+strlen(LABEL_PREFIX)) ;

    npreds = gen_length(preds=control_predecessors(obj));

    switch(npreds) {
    case 0:
      /* Should only happen for the entry node 
       * Or for an unconnected exit node, since the exit node is explictly
       * prettyprinted.
       */
	break ;
    case 1: 
      if(check_io_statement_p(control_statement(CONTROL(CAR(preds)))))
	break;
      else if (*previous == CONTROL(CAR(preds)) &&
	    (obj != cexit || 
	     !get_bool_property("PRETTYPRINT_INTERNAL_RETURN"))) {
	  /* It is assumed that no GO TO has been generated because
	   * it was not necessary; it's up to add_control_goto().
	   * Note that the predecessor may have two successors... and
	   * that only the first successor can fall through...
	   */
	  if(CONTROL(CAR(control_successors(*previous)))==obj)
	    break ;
	  /* Unless the first successor has been seen before, in which case
	   * a GO TO was generated to reach it.
	   * OK, but a *forward* GO TO may have been generated...
	   */
	  else if(set_belong_p(seens,
			       (char *)CONTROL(CAR(control_successors(*previous)))))
	    break;
	  /* Unless the first successor of the previous control is a RETURN */
	  else if(entity_return_label_p
		  (statement_label
		   (control_statement
		    (CONTROL(CAR(control_successors(*previous)))))))
	    break;
	  /* Unless the predecessor does not return
	   * should encompass previous case and be encompassed
	   * by reachability test below
	   */
	  else if(!statement_does_return
		  (control_statement
		   (CONTROL(CAR(preds)))))
	    break;
	}
    default:
      /* The number of precedessors is not bounded but greater than one
       * when this point is reached
       */

      /* break if no predecessor "returns" (i.e. continues) */
      /* Useless, because one predecessor at least continues. */
      /* More subtle information is needed to avoid the landing label 
	 generation */
      /*
      reachable = FALSE;
      MAPL(lc, {
	statement ps = control_statement(CONTROL(CAR(lc)));
	if(reachable = statement_does_return(ps)) {
	  break;
	}
      }, preds);
      if(!reachable)
	break;
	*/

      /* Generate a new landing label if none is already available */
      if( empty_label_p( entity_name( statement_to_label( st )))) {
	    pc = CHAIN_SWORD(NIL,"CONTINUE") ;
	    s = make_sentence(is_sentence_unformatted,
			      make_unformatted(NULL, 0, margin, pc)) ;
	    unformatted_label(sentence_unformatted(s)) = label_name ;
	    ADD_SENTENCE_TO_TEXT(r, s);    
	    debug(3, "text_control", "Label %s generated for stmt %s\n",
		  label_name, statement_identification(st));
	}
    }

    switch(gen_length(succs=control_successors(obj))) {
    case 0:
	MERGE_TEXTS(r, text_statement(module, margin, st));
	(void) add_control_goto(module, margin, r, obj, 
			 cexit, labels, seens, (control)NULL ) ;
	break ;
    case 1:
	MERGE_TEXTS(r, text_statement(module, margin, st));
	(void) add_control_goto(module, margin, r, obj, 
			 CONTROL(CAR(succs)), labels, seens, cexit) ;
	break;
    case 2: {
	instruction i = statement_instruction(st);
	test t = test_undefined;
	bool added = FALSE;

	pips_assert("text_control", instruction_test_p(i));

	MERGE_TEXTS(r, init_text_statement(module, margin, st)) ;
	if (! string_undefined_p(comments)) {
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
						  comments));
	}
	pc = CHAIN_SWORD(NIL, "IF (");
	t = instruction_test(i);
	pc = gen_nconc(pc, words_expression(test_condition(t)));
	pc = CHAIN_SWORD(pc, ") THEN");
	u = make_unformatted(NULL, statement_number(st), margin, pc) ;

	if( !empty_label_p( entity_name( statement_label( st )))) {
	    unformatted_label(u) = strdup(label_name) ;
	}
	s = make_sentence(is_sentence_unformatted,u) ;
	ADD_SENTENCE_TO_TEXT(r, s);

	/* FI: PJ seems to assume that the true successors will be processed
	 * first and that a GOTO may not be needed. But the first successor
	 * may be the exit node which is not processed and the second
	 * successor is going to believe that it does not need a label
	 * because it is a direct successor.
	 */
	added = add_control_goto(module, margin+INDENTATION, r, obj, 
			 CONTROL(CAR(succs)), labels, seens, cexit) ;
	/* PJ forces the generation of a GOTO for the ELSE branch because
	 * he does not remember if the first branch fell thru or not
	 */
	if(added) {
	  text g = make_text(NIL);
	  bool else_goto_added =
	    add_control_goto(module, margin+INDENTATION, g, obj, 
			     CONTROL(CAR(CDR(succs))), labels, seens, cexit) ;
	  if(else_goto_added) {
	    ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	  }
	  MERGE_TEXTS(r, g);
	}
	else {
	  ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	  (void) add_control_goto(module, margin+INDENTATION, r, obj, 
				  CONTROL(CAR(CDR(succs))), labels, (set)NULL, cexit) ;
	}
	ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDIF"));
	break;
    }
    default:
	pips_error("text_control", "incorrect number of successors\n");
    }

    debug(2, "text_control", "End for statement %s\n",
	  statement_identification(st));

    return( r ) ;
}

static text
text_trail(entity module, int margin, list trail, hash_table labels)
{
  list cc = NIL;
  text r = make_text(NIL);

  pips_assert("trail cannot be empty", !ENDP(trail));

  for(cc=trail; !ENDP(cc); POP(cc)) {
    control c = CONTROL(CAR(cc));
    string l = string_undefined;
    int nsucc = gen_length(control_successors(c));
    statement st = control_statement(c);

    /* Is a label needed? */
    if((l=control_has_label(c, labels))!=string_undefined) {
      if(strcmp(l, label_local_name(statement_to_label(control_statement(c))))
	 != 0) {
	list pc = CHAIN_SWORD(NIL,"CONTINUE") ;
	sentence s = make_sentence(is_sentence_unformatted,
				   make_unformatted(NULL, 0, margin, pc)) ;
	unformatted_label(sentence_unformatted(s)) = l ;
	ADD_SENTENCE_TO_TEXT(r, s);    
	debug(3, "text_trail", "Label %s generated for stmt %s\n",
	      label_name, statement_identification(control_statement(c)));
      }
    }

    /* Perform the very same tests as in decorate_trail()
     * The successor may have a label but not need a GOTO to be reached.
     */

    switch(nsucc) {
    case 0:
      MERGE_TEXTS(r, text_statement(module, margin, st));
      break;
    case 1: {
      control succ = CONTROL(CAR(control_successors(c)));

      MERGE_TEXTS(r, text_statement(module, margin, st));

      /* If the statement "really" has a continuation (e.g. not a STOP
       * or a RETURN
       */
      if(statement_does_return(control_statement(c))) {
	if(!ENDP(cc)) {
	  control tsucc = CONTROL(CAR(CDR(cc)));
	  if(tsucc==succ) {
	    /* the control successor is the textual successor */
	    break;
	  }
	}
	/* A GOTO must be generated to reach the control successor */
	l = control_has_label(succ, labels);
	pips_assert("Must be labelled", l!= string_undefined);
	ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module,
						    margin+INDENTATION,
						    l));
      }
      break;
    }
    case 2: {
      control succ1 = CONTROL(CAR(control_successors(c)));
      control succ2 = CONTROL(CAR(CDR(control_successors(c))));
      instruction i = statement_instruction(st);
      test t = test_undefined;

      pips_assert("text_control", instruction_test_p(i));

      MERGE_TEXTS(r, init_text_statement(module, margin, st)) ;
      if (! string_undefined_p(comments)) {
	ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
					      comments));
      }

      pc = CHAIN_SWORD(NIL, "IF (");
      t = instruction_test(i);
      pc = gen_nconc(pc, words_expression(test_condition(t)));
      pc = CHAIN_SWORD(pc, ") THEN");
      u = make_unformatted(NULL, statement_number(st), margin, pc) ;

      if( !empty_label_p( entity_name( statement_label( st )))) {
	unformatted_label(u) = strdup(label_name) ;
      }
      s = make_sentence(is_sentence_unformatted,u) ;
      ADD_SENTENCE_TO_TEXT(r, s);

      /* Is there a textual successor? */
      if(!ENDP(cc)) {
	control tsucc = CONTROL(CAR(CDR(cc)));
	if(tsucc==succ1) {
	  if(tsucc==succ2) {
	    /* This may happen after restructuring */
	    ;
	  }
	  else {
	    /* succ2 must be reached by GOTO */
	    control_slabel(module, succ2, labels);
	    l = control_has_label(succ2, labels);
	    pips_assert("Must be labelled", l!= string_undefined);
	    ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	    ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module,
							margin+INDENTATION,
							l));
	  }
	  if(tsucc==succ2) {
	    /* succ1 must be reached by GOTO */
	    l = control_has_label(succ1, labels);
	    pips_assert("Must be labelled", l!= string_undefined);
	    ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module,
							margin+INDENTATION,
							l));
	  }
	  else {
	    /* Both successors must be labelled */
	    l = control_has_label(succ1, labels);
	    pips_assert("Must be labelled", l!= string_undefined);
	    ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module,
							margin+INDENTATION,
							l));
	    ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	    l = control_has_label(succ2, labels);
	    pips_assert("Must be labelled", l!= string_undefined);
	    ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module,
							margin+INDENTATION,
							l));
	  }
	}
      }
      else {
	/* Both successors must be textual predecessors */
	pips_assert("succ1 before c", appears_first_in_trail(trail, succ1, c));
	pips_assert("succ1 before c", appears_first_in_trail(trail, succ2, c));
	l = control_has_label(succ1, labels);
	pips_assert("Must be labelled", l!= string_undefined);
	ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module,
						    margin+INDENTATION,
						    l));
	    ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
	l = control_has_label(succ2, labels);
	pips_assert("Must be labelled", l!= string_undefined);
	ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module,
						    margin+INDENTATION,
						    l));
      }
      ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"ENDIF"));
      break;
    }
    default:
      pips_internal_error("Too many successors for a control node\n");
    }
  }

  return r;
}
