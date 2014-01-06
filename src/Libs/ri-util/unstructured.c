/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
  * Prettyprint unstructured
  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

static void decorate_trail(entity module, list trail, hash_table labels);
static text text_trail(entity module, int margin, list trail, hash_table labels);
static bool control_in_trail_p(list l, control c);
static bool appears_first_in_trail(list l, control c1, control c2);
static void set_control_to_label(entity m, control c, hash_table h);
static string control_to_label_name(control c, hash_table h);

text
text_unstructured(entity module,
                  const char* label,
                  int margin,
                  unstructured u,
		  int num)
{
    text r = make_text(NIL);
    hash_table labels = hash_table_make(hash_pointer, 0) ;

    debug(2, "text_unstructured", "Begin for unstructured %p\n", u);
   
    ifdebug(3) {
	list blocks = NIL;
	control cexit = unstructured_exit(u) ;
	control centry = unstructured_control(u) ;

	fprintf(stderr,"Unstructured %p (%p, %p)\n", u, centry, cexit) ;
	CONTROL_MAP( n, {
	    statement st = control_statement(n) ;

	    /*
	      fprintf(stderr, "\n%*sNode %p (%s)\n--\n", margin, "",
	      (unsigned int) n, control_to_label_name(n, labels)) ;
	      */
	    fprintf(stderr, "\n%*sNode %p (%s)\n--\n", margin, "",
		    n, statement_identification(st));
	    ifdebug(9)
	      print_text(stderr, text_statement(module,margin,st,NIL));
	    fprintf(stderr, "--\n%*sPreds:", margin, "");
	    MAPL(ps,{
		int so = statement_ordering(control_statement(CONTROL(CAR(ps))));
		fprintf(stderr,"%p (%d,%d), ", CONTROL(CAR(ps)),
			ORDERING_NUMBER(so), ORDERING_STATEMENT(so));
	    }, control_predecessors(n));
	    fprintf(stderr, "\n%*sSuccs:", margin, "") ;
	    MAPL(ss,{
		int so = statement_ordering(control_statement(CONTROL(CAR(ss))));
		fprintf(stderr,"%p (%d,%d), ", CONTROL(CAR(ss)),
			ORDERING_NUMBER(so), ORDERING_STATEMENT(so));
	    }, control_successors(n));
	    fprintf(stderr, "\n\n") ;
	}, centry , blocks) ;
	gen_free_list(blocks);
    }

    if (get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH"))
    {
	output_a_graph_view_of_the_unstructured
	    (r, module, label, margin, u, num);
    }
    else {
	list trail = NIL;

	if(get_bool_property("PRETTYPRINT_UNSTRUCTURED")) {
	    list pbeg = CHAIN_SWORD(NIL, "BEGIN UNSTRUCTURED");
	    unformatted unf = make_unformatted(strdup(get_comment_sentinel()), num, margin, pbeg);

	    ADD_SENTENCE_TO_TEXT(r, 
				 make_sentence(is_sentence_unformatted, unf));
	}

	/* build an arbitrary reverse trail of control nodes */
	trail = unstructured_to_trail(u);
	debug(3, "text_unstructured", "Trail length: %d\n", gen_length(trail));

	trail = gen_nreverse(trail);

	ifdebug(3)
	    dump_trail(trail);

	/* decorate control nodes with labels when necessary */
	decorate_trail(module, trail, labels);

	ifdebug(3)
	    dump_control_to_label_name(labels);

	/* generate text with labels and goto's */

	MERGE_TEXTS(r, text_trail(module, margin, trail, labels));

	if(get_bool_property("PRETTYPRINT_UNSTRUCTURED")) {
	    list pend = CHAIN_SWORD(NIL, "END UNSTRUCTURED");
	    unformatted unf = make_unformatted(strdup(get_comment_sentinel()), num, margin, pend);
	    ADD_SENTENCE_TO_TEXT(r, 
				 make_sentence(is_sentence_unformatted, unf));
	}
	gen_free_list(trail);
    }

    hash_table_free(labels) ;

    debug(2, "text_unstructured", "End for unstructured %p\n", u);

    return(r) ;
}

/* Any heuristics can be used to build the trail, depth or width first,
 * true or false branch first, lower statement numbering first or not,
 * sorted by statement numbering (but some statements have no number...).
 * No simple heuristics seems to be bullet proof.
 *
 * The exit node must be last in the trace, or an extra node has to be added
 * to reach the continuation of the unstructured (see text_trail()).
 *
 * For CONS convenience, the list is built in reverse order (and reversed
 * by the caller).
 */
static list
build_trail(list l, control c)
{
    control succ = control_undefined;
    int nsucc = 0;

    if(check_io_statement_p(control_statement(c)) &&
       !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) {
	/* Do not add this artificial node to the trail, follow the left
	 * successors only
	 */
	pips_assert("Must be a test statement",
		    instruction_test_p(statement_instruction(control_statement(c))));
	debug(3, "build_trail", "Skipping IO check %s",
	      statement_identification(control_statement(c)));
	succ = CONTROL(CAR(CDR(control_successors(c))));
	debug(3, "build_trail", "False Successor: %s",
	      statement_identification(control_statement(succ)));
	l = build_trail(l, succ);
	succ = CONTROL(CAR(control_successors(c)));
	debug(3, "build_trail", "True Successor: %s",
	      statement_identification(control_statement(succ)));
	l = build_trail(l, succ);
    }
    else {

	nsucc = gen_length(control_successors(c));

	debug(3, "build_trail", "for %s with %d successors\n",
	      statement_identification(control_statement(c)),
	      nsucc);
	ifdebug(3) {
	    int i = 1;
	    MAPL(cs, {
		statement ss = control_statement(CONTROL(CAR(cs)));
		debug(3, "build_trail", "Successor %d: %s",
		      i++, statement_identification(ss));
	    }, control_successors(c));
	}
    
	/* Add c to the trail if not already in */
	if(!control_in_trail_p(l, c)) {
	    debug(3, "build_trail", "Add to trail %s",
		  statement_identification(control_statement(c)));
	    l = CONS(CONTROL,c,l);
	    switch(nsucc) {
	    case 0:
		break;
	    case 1:
		succ = CONTROL(CAR(control_successors(c)));
		l = build_trail(l, succ);
		break;
	    case 2:
		/* Follow the false branch in depth first, assuming that IF GOTO's 
		 * mainly handle exceptions */
		succ = CONTROL(CAR(CDR(control_successors(c))));
		l = build_trail(l, succ);
		succ = CONTROL(CAR(control_successors(c)));
		l = build_trail(l, succ);
		break;
	    default:
		pips_internal_error("Too many successors (%d) for a control node",
			   nsucc);
	    }
	}
	else {
	    debug(3, "build_trail", "Already in trail %s",
		  statement_identification(control_statement(c)));
	}
    }
    return l;
}

list 
unstructured_to_trail(unstructured u)
{
    list trail = NIL;
    control centry = unstructured_control(u) ;
    control cexit = unstructured_exit(u) ;

    trail = build_trail(trail, centry);

	/* The exit node *must* be first (i.e. last) to reach the continuation
	 * of the unstructured, or never reached (e.g. because the program loops
	 * forever or stops in the unstructured).
	 */
    if(control_in_trail_p(trail, cexit)) {
	if(cexit!=CONTROL(CAR(trail))) {
	    gen_remove(&trail, cexit);
	    trail = CONS(CONTROL, cexit, trail);
	}
    }

    return trail;
}

void 
dump_trail(list trail)
{
    if(ENDP(trail)) {
	fprintf(stderr, "[dump_trail] trail is empty\n");
    }
    else {
	fprintf(stderr, "[dump_trail] begin\n");
	MAPL(cc, {
	    statement s = control_statement(CONTROL(CAR(cc)));
	    fprintf(stderr, "[dump_trail] %s", statement_identification(s));
	}, trail);
	fprintf(stderr, "[dump_trail] end\n");
    }
}

/* OK, a hash table could be used, as Pierre used too... but the order
 * is lost. You need both ordered and direct accesses. Easy to add later if too
 * slow.
 */
static bool
control_in_trail_p(list l, control c)
{
    bool found = gen_find_eq(c, l) != gen_chunk_undefined;
    return found;
}

static void
decorate_trail(entity module, list trail, hash_table labels)
{
    list cc = NIL;

    pips_assert("trail cannot be empty", !ENDP(trail));

    for(cc=trail; !ENDP(cc); POP(cc)) {
	control c = CONTROL(CAR(cc));
	int nsucc = gen_length(control_successors(c));

	debug(3, "decorate_trail", "Processing statement %s with %d successors\n",
	      statement_identification(control_statement(c)), nsucc);

	switch(nsucc) {
	case 0:
	    /* No need for a label, it must be the exit node... Should be asserted
	     * The exit node may have two successors
	     */
	    break;
	case 1: {
	    control succ = CONTROL(CAR(control_successors(c)));

	    if(check_io_statement_p(control_statement(succ)) &&
		 !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) {
	      /* The real successor is the false successor of the IO check */
	      succ = CONTROL(CAR(CDR(control_successors(succ))));
	      if(check_io_statement_p(control_statement(succ)) &&
		 !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) {
		/* The real successor is the false successor of the IO check */
		succ = CONTROL(CAR(CDR(control_successors(succ))));
	      }
	      pips_assert("The successor is not a check io statement",
			  !check_io_statement_p(control_statement(succ)));
	    }

	    /* If the statement "really" has a continuation (e.g. not a STOP
	     * or a RETURN)
	     */
	    /* if(statement_does_return(control_statement(c)) &&
	       !(check_io_statement_p(control_statement(succ)) &&
	       !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS"))) { */
	    if(statement_does_return(control_statement(c))) {
		if(!ENDP(CDR(cc))) {
		    control tsucc = CONTROL(CAR(CDR(cc)));
		    if(tsucc==succ) {
			/* the control successor is the textual successor */
			break;
		    }
		}
		/* A label must be associated with the control successor */
		pips_assert("Successor must be in trail",
			    control_in_trail_p(trail, succ));
		set_control_to_label(module, succ, labels);
	    }
	    break;
	}
	case 2: {
	    control succ1 = CONTROL(CAR(control_successors(c)));
	    control succ2 = CONTROL(CAR(CDR(control_successors(c))));

	    debug(3, "decorate_trail", "Successor 1 %s",
		  statement_identification(control_statement(succ1)));
	    debug(3, "decorate_trail", "Successor 2 %s",
		  statement_identification(control_statement(succ2)));

	    /* Is there a textual successor? */
	    if(!ENDP(CDR(cc))) {
		control tsucc = CONTROL(CAR(CDR(cc)));
		if(tsucc==succ1) {
		    if(tsucc==succ2) {
			/* This may happen after restructuring */
			;
		    }
		    else {
			/* succ2 must be labelled */
			pips_assert("Successor 2 must be in trail",
				    control_in_trail_p(trail, succ2));
			set_control_to_label(module, succ2, labels);
		    }
		}
		else {
		    if(tsucc==succ2) {
			/* succ1 must be labelled */
			pips_assert("Successor 1 must be in trail",
				    control_in_trail_p(trail, succ1));
			set_control_to_label(module, succ1, labels);
		    }
		    else {
			/* Both successors must be labelled */
			pips_assert("Successor 1 must be in trail",
				    control_in_trail_p(trail, succ1));
			set_control_to_label(module, succ1, labels);
			pips_assert("Successor 2 must be in trail",
				    control_in_trail_p(trail, succ2));
			set_control_to_label(module, succ2, labels);
		    }
		}
	    }
	    else {
		/* Both successors must be textual predecessors */
		pips_assert("succ1 before c", appears_first_in_trail(trail, succ1, c));
		pips_assert("succ2 before c", appears_first_in_trail(trail, succ2, c));
		set_control_to_label(module, succ1, labels);
		set_control_to_label(module, succ2, labels);
	    }
	    break;
	}
	default:
	    pips_internal_error("Too many successors for a control node");
	}
    }
}

/* returns true if c1 is encountered before c2 in trail, or if c1 == c2
 * is encountered in trail, false is c2 is encountered first and c2 != c1,
 * core dumps otherwise, if neither c1 nor c2 is in trail
 */
static bool
appears_first_in_trail(list trail, control c1, control c2)
{
    bool first = false;
    control c = control_undefined;

    MAPL(cc, {
	c = CONTROL(CAR(cc));

	if(c==c1) {
	    first = true;
	    break;
	}
	else if(c==c2) {
	    first = false;
	    break;
	}
    }, trail);
    pips_assert("At least one control should appear in trail", c==c1 || c==c2);
    return first;
}


/* set_control_to_label allocates label for the control
 * node c in the module m. h maps controls to label names. Computes a new
 * label name if necessary.
 *
 * There is no guarantee that a label generated here appears eventually
 * in the text produced.
 *
 * There is no guarantee that a label generated here is jumped at.
 */

static void 
set_control_to_label(entity m, control c, hash_table h)
{
    char* l;
    statement st = control_statement(c) ;

    if ((l = hash_get(h, (char *) c)) == HASH_UNDEFINED_VALUE) {
        const char* label = entity_name( statement_to_label( st )) ;
        if(empty_global_label_p( label )) {
            char * lname = new_label_local_name(m);
            const char * module_name = 
                entity_undefined_p(m) ?
                GENERATED_LABEL_MODULE_NAME:
                module_local_name(m);
                
            asprintf(&l,"%s" MODULE_SEP_STRING "%s",module_name,lname);
            free(lname);
        }
        else {
            l=strdup(label);
        }

        /* memory leak in debug code: statement_identification(). */
        pips_debug(3, "Associates label %s to stmt %s\n",
                l, statement_identification(st));
        hash_put(h, (char *) c, l) ;
    }
    else {
	debug(3, "set_control_to_label", "Retrieves label %s for stmt %s\n",
	      l, statement_identification(st));
    }

    pips_assert("set_control_to_label", strcmp(local_name(l), LABEL_PREFIX) != 0) ;
    pips_assert("set_control_to_label", strcmp(local_name(l), "") != 0) ;
    pips_assert("set_control_to_label", strcmp(local_name(l), "=") != 0) ;

    return;
}

static string
control_to_label_name(control c, hash_table h)
{
    char* l;
    statement st = control_statement(c) ;

    if ((l = hash_get(h, (char *) c)) == HASH_UNDEFINED_VALUE) {
	debug(3, "control_to_label_name", "Retrieves no label for stmt %s\n",
	      statement_identification(st));
	l = string_undefined;
    }
    else {
	debug(3, "control_to_label_name", "Retrieves label %s for stmt %s\n",
	      l, statement_identification(st));
	l = strdup(local_name(l)+sizeof(LABEL_PREFIX)-1);
    }

    return l;
}

void
dump_control_to_label_name(hash_table h)
{
    int i = 0;

    fprintf(stderr,"[dump_control_to_label_name] Begin\n");
    HASH_MAP(c,l,{
	fprintf(stderr, "Label %s -> %s", (char *) l,
		(char *) statement_identification(control_statement((control) c)));
	i++;
    }, h);
    fprintf(stderr,"[dump_control_to_label_name] %d labels, end\n", i);
}

static text text_trail(entity module, int margin, list trail, hash_table labels) {
  list cc = NIL;
  text r = make_text(NIL);

  pips_assert("trail cannot be empty", !ENDP(trail));

  for (cc = trail; !ENDP(cc); POP(cc)) {
    control c = CONTROL(CAR(cc));
    string l = string_undefined;
    int nsucc = gen_length(control_successors(c));
    statement st = control_statement(c);

    debug(3,
          "text_trail",
          "Processing statement %s",
          statement_identification(st));

    /* Is a label needed? */
    if ((l = control_to_label_name(c, labels)) != string_undefined) {
      if (strcmp(l, label_local_name(statement_to_label(control_statement(c))))
          != 0) {
        list pc = NIL;
        switch (get_prettyprint_language_tag()) {
          case is_language_fortran:
          case is_language_fortran95:
            pc = CHAIN_SWORD(pc, "CONTINUE");
            break;
          case is_language_c:
            pc = CHAIN_SWORD(pc, ";");
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        sentence s = make_sentence(is_sentence_unformatted,
                                   make_unformatted(NULL, 0, margin, pc));
        unformatted_label(sentence_unformatted(s)) = l;
        ADD_SENTENCE_TO_TEXT(r, s);
        debug(3,
              "text_trail",
              "Label %s generated for stmt %s\n",
              l,
              statement_identification(control_statement(c)));
      }
    }

    /* Perform the very same tests as in decorate_trail()
     * The successor may have a label but not need a GOTO to be reached.
     */

    switch(nsucc) {
      case 0:
	/* Build the statement text with enclosing braces if necessary in
	   C and skip parasitic continues */
        MERGE_TEXTS(r, text_statement_enclosed(module, margin, st, one_liner_p(st), true, NIL));
        break;
      case 1: {
        control succ = CONTROL(CAR(control_successors(c)));

        if (check_io_statement_p(control_statement(succ))
            && !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) {
          /* The real successor is the false successor of the IO check */
          succ = CONTROL(CAR(CDR(control_successors(succ))));
          if (check_io_statement_p(control_statement(succ))
              && !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) {
            /* The real successor is the false successor of the IO check */
            succ = CONTROL(CAR(CDR(control_successors(succ))));
          }
          pips_assert("The successor is not a check io statement",
              !check_io_statement_p(control_statement(succ)));
        }

	/* Build the statement text with enclosing braces if necessary in
	   C and skip parasitic continues */
        MERGE_TEXTS(r, text_statement_enclosed(module, margin, st, one_liner_p(st), true, NIL));

        /* If the statement "really" has a continuation (e.g. not a STOP
         * or a RETURN)
         */
        /* if(statement_does_return(st) &&
         !(check_io_statement_p(control_statement(succ)) &&
         !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) ) { */
        if (statement_does_return(st)) {
          if (!ENDP(CDR(cc))) {
            control tsucc = CONTROL(CAR(CDR(cc)));
            if (tsucc == succ) {
              /* the control successor is the textual successor */
              break;
            }
          }
          /* A GOTO must be generated to reach the control successor */
          l = control_to_label_name(succ, labels);
          pips_assert("Must be labelled", l!= string_undefined);
          ADD_SENTENCE_TO_TEXT(r, sentence_goto_label(module, NULL,
                  margin, l, 0));
        }
        break;
      }
      case 2: {
        control succ1 = CONTROL(CAR(control_successors(c)));
        control succ2 = CONTROL(CAR(CDR(control_successors(c))));
        instruction i = statement_instruction(st);
        test t = test_undefined;
        unformatted u = unformatted_undefined;
        list pc = NIL;
        sentence s = sentence_undefined;
        string comments = statement_comments(st);
        text r1 = make_text(NIL);
        bool no_endif = false;

        pips_assert("must be a test", instruction_test_p(i));

        MERGE_TEXTS(r, init_text_statement(module, margin, st));
        if (!string_undefined_p(comments)) {
          ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
                  strdup(comments)));
        }
        switch (get_prettyprint_language_tag()) {
          case is_language_fortran:
          case is_language_fortran95:
            pc = CHAIN_SWORD(NIL, "IF (");
            break;
          case is_language_c:
            pc = CHAIN_SWORD(NIL, "if (");
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        t = instruction_test(i);
        pc = gen_nconc(pc, words_expression(test_condition(t), NIL));

        /* Is there a textual successor? */
        if (!ENDP(CDR(cc))) {
          control tsucc = CONTROL(CAR(CDR(cc)));
          if (tsucc == succ1) {
            if (tsucc == succ2) {
              /* This may happen after restructuring */
              ;
            } else {
              /* succ2 must be reached by GOTO */
              l = control_to_label_name(succ2, labels);
              pips_assert("Must be labelled", l!= string_undefined);
              switch (get_prettyprint_language_tag()) {
                case is_language_fortran:
                case is_language_fortran95:
                  ADD_SENTENCE_TO_TEXT(r1,
                                       MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
                  break;
                case is_language_c:
                  ADD_SENTENCE_TO_TEXT(r1, MAKE_ONE_WORD_SENTENCE(margin,"}"));
                  ADD_SENTENCE_TO_TEXT(r1, MAKE_ONE_WORD_SENTENCE(margin,
                                                                  "else {" ));
                  break;
                default:
                  pips_internal_error("Language unknown !");
                  break;
              }
              ADD_SENTENCE_TO_TEXT(r1, sentence_goto_label(module, NULL,
                                                           margin+INDENTATION,
                                                           l, 0));
            }
          } else {
            if (tsucc == succ2) {
              /* succ1 must be reached by GOTO */
              l = control_to_label_name(succ1, labels);
              pips_assert("Must be labelled", l!= string_undefined);
              no_endif = true;
            } else {
              /* Both successors must be labelled */
              l = control_to_label_name(succ1, labels);
              pips_assert("Must be labelled", l!= string_undefined);
              ADD_SENTENCE_TO_TEXT(r1, sentence_goto_label(module, NULL,
                      margin+INDENTATION,
                      l, 0));
              switch (get_prettyprint_language_tag()) {
                case is_language_fortran:
                case is_language_fortran95:
                  ADD_SENTENCE_TO_TEXT(r1,
                                       MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
                  break;
                case is_language_c:
                  ADD_SENTENCE_TO_TEXT(r1, MAKE_ONE_WORD_SENTENCE(margin,"}"));
                  ADD_SENTENCE_TO_TEXT(r1, MAKE_ONE_WORD_SENTENCE(margin,
                                                                  "else {" ));
                  break;
                default:
                  pips_internal_error("Language unknown !");
                  break;
              }
              l = control_to_label_name(succ2, labels);
              pips_assert("Must be labelled", l!= string_undefined);
              ADD_SENTENCE_TO_TEXT(r1, sentence_goto_label(module, NULL,
                      margin+INDENTATION,
                      l, 0));
            }
          }
        } else {
          /* Both successors must be textual predecessors */
          pips_assert("succ1 before c", appears_first_in_trail(trail, succ1, c));
          pips_assert("succ1 before c", appears_first_in_trail(trail, succ2, c));
          l = control_to_label_name(succ1, labels);
          pips_assert("Must be labelled", l!= string_undefined);
          ADD_SENTENCE_TO_TEXT(r1, sentence_goto_label(module, NULL,
                  margin+INDENTATION,
                  l, 0));
          switch (get_prettyprint_language_tag()) {
            case is_language_fortran:
            case is_language_fortran95:
              ADD_SENTENCE_TO_TEXT(r1, MAKE_ONE_WORD_SENTENCE(margin,"ELSE"));
              break;
            case is_language_c:
              ADD_SENTENCE_TO_TEXT(r1, MAKE_ONE_WORD_SENTENCE(margin,"}"));
              ADD_SENTENCE_TO_TEXT(r1, MAKE_ONE_WORD_SENTENCE(margin,"else {" ));
              break;
            default:
              pips_internal_error("Language unknown !");
              break;
          }
          l = control_to_label_name(succ2, labels);
          pips_assert("Must be labelled", l!= string_undefined);
          ADD_SENTENCE_TO_TEXT(r1, sentence_goto_label(module, NULL,
                  margin+INDENTATION,
                  l, 0));
        }

        if (no_endif) {
          pc = CHAIN_SWORD(pc, ") ");
          pc = gen_nconc(pc, words_goto_label(l));
        } else {
          switch (get_prettyprint_language_tag()) {
            case is_language_fortran:
            case is_language_fortran95:
              pc = CHAIN_SWORD(pc, ") THEN");
              break;
            case is_language_c:
              pc = CHAIN_SWORD(pc, ") {");
              break;
            default:
              pips_internal_error("Language unknown !");
              break;
          }
        }
        u = make_unformatted(NULL, statement_number(st), margin, pc);

        if (!empty_global_label_p(entity_name( statement_label( st )))) {
          /*
           string ln = control_to_label_name(c, labels);
           if(string_undefined_p(ln)) {
           entity lab = statement_label(st);
           print_statement(st);
           pips_assert("c must have been encountered before", !string_undefined_p(ln));
           }
           unformatted_label(u) = strdup(ln);
           */
          unformatted_label(u) = strdup(label_local_name(statement_label(st)));
        }

        s = make_sentence(is_sentence_unformatted, u);
        ADD_SENTENCE_TO_TEXT(r, s);
        MERGE_TEXTS(r, r1);
        if (!no_endif) {
          switch (get_prettyprint_language_tag()) {
            case is_language_fortran:
            case is_language_fortran95:
              ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin, "ENDIF"));
              break;
            case is_language_c:
              ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin, "}"));
              break;
            default:
              pips_internal_error("Language unknown !");
              break;
          }
        }
        break;
      }
      default:
        pips_internal_error("Too many successors for a control node");
    }
  }

  return r;
}
