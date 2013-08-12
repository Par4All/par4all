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

#ifndef lint
char vcid_control_old_controlizer[] = "$Id$";
#endif /* lint */

/* \defgroup controlizer Controlizer phase to build the Hierarchical Control Flow Graph

   It computes the Hierarchical Control Flow Graph of a given statement
   according the control hierarchy.

   It is used in PIPS to transform the output of the parsers (the
   PARSED_CODE resource) into HCFG code (the PARSED_CODE resource).

   For example if there are some "goto" in a program, it will encapsulated
   the unstructured graph in an "unstructured" object covering all the
   goto and their label targets to localize the messy part and this object
   is put into a normal statement so that seen from above the code keep a
   good hierarchy.

   In PIPS the RI (Internal Representation or AST) is quite simple so that
   it is easy to deal with. But the counterpart is that some complex
   control structures need to be "unsugared". For example
   switch/case/break/default are transformed into tests, goto and label,
   for(;;) with break or continue are transformed into while() loops with
   goto/label, and so on.

   There are other phases in PIPS that can be used later to operate on the
   CODE to optimize it further.


   WARNINGS:

   . Temporary locations malloc()ed while recursing in the process are
     often not freed (to be done latter ... if required)

   . The desugaring of DO loops is not perfect (in case of side-effects
     inside loop ranges.

   Pierre Jouvelot (27/5/89) <- this is a French date :-)

   MODIFICATIONS (historian fun):

   . hash_get interface modification: in one hash table an undefined key
   meant an error; in another one an undefined key was associated to
   the default value empty_list; this worked as long as NULL was returned
   as NOT_FOUND value (i.e. HASH_UNDEFINED_VALUE); this would work again
   if HASH_UNDEFINED_VALUE can be user definable; Francois Irigoin, 7 Sept. 90

   @{
*/

/*
 * $Id$
 */

#include <stdio.h>
#include <strings.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "control.h"
#include "properties.h"

#include "misc.h"

#include "constants.h"


#define LABEL_TABLES_SIZE 10

/* UNREACHABLE is the hook used as a predecessor of statements that are
   following a goto. The list of unreachable statements is kept in the
   successors list. */

static list Unreachable;

/* LABEL_STATEMENTS maps label names to the list of statements where
   they appear (either as definition or reference). */

static hash_table Label_statements;

/* LABEL_CONTROL maps label names to their (possible forward) control
   nodes. */

static hash_table Label_control;


/* In C, we can have some "goto" inside a block from outside, that
   translate as any complex control graph into an "unstructured" in the
   PIPS jargon.

   Unfortunately, that means it break the structured block nesting that
   may carry declarations with the scoping information.

   So we need to track this scope information independently of the control
   graph. This is the aim of this declaration scope stack that is used to
   track scoping during visiting the RI.
*/
DEFINE_LOCAL_STACK(scoping_statement, statement)


/* FI -> PJ:
 *
 * The naming for ADD_PRED et ADD_SUCC is misleading. ADD_SUCC is in
 * fact a SET_SUCC.  ADD_PRED is UNION_PRED. When used to Newgen but
 * not to control, ADD_SUCC reduces readability.
 *
 * Furthermore, ADD_SUCC() is dangerous when used on a control that is
 * a test since the position in the successor list is
 * significant. true successors are in the odd positions (the first
 * element is of rank one). false successors are in the odd position.
 */

/* Add control "pred" to the predecessor set of control c if not already
   here */
#define ADD_PRED_IF_NOT_ALREADY_HERE(pred,c) (gen_once(pred,control_predecessors(c)))
#define ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred,c) (gen_once(pred,gen_copy_seq(control_predecessors(c))))

/* Update control c by setting its statement to s, by unioning its predecessor
 * set with pd, and by setting its successor set to sc (i.e. previous successors
 * are lost, but not previous predecessors).
 *
 * Note: This macro does not preserve the consistency of the control
 * flow graph as pd's successor list and sc predecessor list are not
 * updated.
 */
#define UPDATE_CONTROL(c,s,pd,sc) { \
	control_statement(c)=s; \
	MAPL(preds, {control_predecessors(c) = \
			      ADD_PRED_IF_NOT_ALREADY_HERE(CONTROL(CAR(preds)), c);}, \
	      pd); \
    gen_free_list(pd);\
    gen_free_list( control_successors(c));\
	control_successors(c)=sc; \
	}

#define PREDS_OF_SUCCS 1
#define SUCCS_OF_PREDS 2

/* PATCH_REFERENCES replaces all occurrences of FNODE by TNODE in the
   predecessors or successors lists of its predecessors
   or successors list (according to HOW, PREDS_OF_SUCCS or
   SUCCS_OF_PREDS).

   Move all the connection of:

   - the predecessors of FNODE to point to TNODE

   - or the successors of FNODE to point from TNODE
 */
static void patch_references(how, fnode, tnode)
int how;
control fnode, tnode;
{
    MAPL(preds, {
	control pred = CONTROL(CAR(preds));

	MAPL(succs, {
	    if(CONTROL(CAR(succs)) == fnode)
		    CONTROL_(CAR(succs)) = tnode;
	}, (how == SUCCS_OF_PREDS) ?
	     control_successors(pred) :
	     control_predecessors(pred));
    }, (how == SUCCS_OF_PREDS) ?
	 control_predecessors(fnode) :
	 control_successors(fnode));
}


/* Make a control node from a statement if needed.

   It is like make_control() except when the statement @param st
   has a label and is thus already in Label_control

   @return the new (in the case of a statement without a label) or already
   associated (in the case of a statement with a label) control node with
   the statement

   It returns NULL if the statement has a label but it is not associated to
   any control node yet
 */
static control make_conditional_control(statement st) {
  string label = entity_name(statement_label(st));

  if (empty_global_label_p(label))
    /* No label, so there cannot be a control already associated by a
       label */
    return make_control(st, NIL, NIL);
  else
      /* Get back the control node associated with this statement
	 label. Since we store control object in this hash table, use
	 cast. We rely on the fact that NIL for a list is indeed
	 NULL... */
    return (control)hash_get_default_empty_list(Label_control, label);
}


/* Get the control node associated to a label name

   It looks for the label name into the Label_control table.

   The @p name must be the complete entity name, not a local or a user name.

   @param name is the string name of the label entity

   @return the associated control
*/
static control get_label_control(string name) {
    control c;

    pips_assert("label is not the empty label", !empty_global_label_p(name)) ;
    c = (control)hash_get(Label_control, name);
    pips_assert("c is defined", c != (control) HASH_UNDEFINED_VALUE);
    pips_assert("c is a control", check_control(c));
    ifdebug(2) {
      check_control_coherency(c);
    }
    return(c);
}


/* Add the reference to the label NAME in the
   statement ST. A used_label is a hash_table that maps the label
   name to the list of statements that references it.

   A statement can appear many times for a label

   @param used_labels is the hash table used to record the statements
   related to a label

   @param name is the label entity name

   @param st is the statement to be recorded as related to the label
*/
static void update_used_labels(hash_table used_labels,
			       string name,
			       statement st) {
  list sts ;

  /* Do something only of there is a label: */
  if (!empty_global_label_p(name)) {
    list new_sts;
    /* Get a previous list of statements related with this label: */
    sts = hash_get_default_empty_list(used_labels, name) ;
    /* Add the given statement to the list */
    new_sts = CONS(STATEMENT, st, sts);
    if (hash_defined_p(used_labels, name))
      /* If there was already something associated to the label, register
	 the new list: */
      hash_update(used_labels, name,  new_sts);
    else
      /* Or create a new entry: */
      hash_put(used_labels, name,  new_sts);
    debug(5, "update_used_labels", "Reference to statement %d seen\n",
	  statement_number( st )) ;
  }
}


/* Unions 2 used-label hash maps

   @param l1 is an hash map

   @param l2 is another hash map

   @returns the union of @p l1 and @p l2 interpreted as in the context of
   update_used_labels()
*/
static hash_table union_used_labels(hash_table l1,
				    hash_table l2) {
  HASH_MAP(name, sts, {
      FOREACH(STATEMENT, s, sts) {
	update_used_labels(l1, name, s);
      };
    }, l2);
  return l1;
}


/* Compute whether all the label references in a statement are in a given
   label name to statement list mapping.

   @param st is the statement we want to check if it owns all allusion to
   the given label name in the @p used_labels mapping

   @param used_labels is a hash table mapping a label name to a list of
   statement that use it, as their label or because it is a goto to it

   @return true if all the label allusion in @p st are covered by the @p
   used_labels mapping.
*/
static bool covers_labels_p(statement st,
			    hash_table used_labels) {
  if( get_debug_level() >= 5 ) {
    pips_debug(0, "Statement %td (%p): \n ", statement_number(st), st);
    print_statement(st);
  }
  /* For all the labels in used_labels: */
  HASH_MAP(name, sts, {
      /* The statements using label name in used_labels: */
      list stats = (list) sts;

      /* For all the statements associated to label name: */
      FOREACH(STATEMENT,
	      def,
	      (list) hash_get_default_empty_list(Label_statements, name)) {
	bool found = false;
	/* Verify that def is in all the statements associated to the
	   label name according to used_labels. */
	FOREACH(STATEMENT, st, stats) {
	  found |= st == def;
	}

	if (!found) {
	  pips_debug(5, "does not cover label %s\n", (char *) name);
	  /* Not useful to go on: */
	  return(false);
	}
      }
    }, used_labels);

  if (get_debug_level() >= 5)
    fprintf(stderr, "covers its label usage\n");

  return(true);
}


static void add_proper_successor_to_predecessor(control pred, control c_res)
{
  /* Replaces the following statement: */
  /* control_successors(pred) = ADD_SUCC(c_res, pred); */

  /* Usually, too much of a mess: do not try to be too strict! */
  /*
  if(statement_test_p(control_statement(pred))) {
    control_successors(pred) = gen_nconc(control_successors(pred),
					 CONS(CONTROL, c_res, NIL));
    pips_assert("While building the CFG, "
		"a test control node may have one or two successors",
		gen_length(control_successors(pred))<=2);
  }
  else {
    if(gen_length(control_successors(pred))==0) {
      control_successors(pred) = CONS(CONTROL, c_res, NIL);
    }
    else if(gen_length(control_successors(pred))==1) {
      if(gen_in_list_p(succ,control_successors(pred))) {
	;
      }
      else {
	pips_internal_error("Two or more candidate successors "
			    "for a standard statement: %p and %p\n",
			    succ, CONTROL(CAR(control_successors(pred))));
      }
    }
    else {
      pips_internal_error("Two or more successors for non-test node %p",
			  pred);
    }
  }
  */

  if(statement_test_p(control_statement(pred))) {
    if(!gen_in_list_p(c_res, control_successors(pred))) {
      control_successors(pred) = gen_nconc(control_successors(pred),
					   CONS(CONTROL, c_res, NIL));
    }
    pips_assert("While building the CFG, "
		"a test control node may have one or two successors",
		gen_length(control_successors(pred))<=2);
  }
  else {
    /* Do whatever was done before and let memory leak! */
    gen_free_list(control_successors(pred));
    control_successors(pred) = CONS(CONTROL, c_res, NIL);
  }
}


/* CONTROLIZE_CALL controlizes the call C of statement ST in C_RES. The deal
   is to correctly manage STOP; since we don't know how to do it, so we
   assume this is a usual call with a continuation !!

   To avoid non-standard successors, IO statement with multiple
   continuations are not dealt with here. The END= and ERR= clauses are
   simulated by hidden tests. */

static bool controlize_call(statement st,
		     control pred,
		     control succ,
		     control c_res)
{
  pips_debug(5, "(st = %p, pred = %p, succ = %p, c_res = %p)\n",
	     st, pred, succ, c_res);

  UPDATE_CONTROL(c_res, st,ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		 CONS(CONTROL, succ, NIL));

  /* control_successors(pred) = ADD_SUCC(c_res, pred); */
  add_proper_successor_to_predecessor(pred, c_res);
  control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_res, succ);
  return(false);
}

/* LOOP_HEADER, LOOP_TEST and LOOP_INC build the desugaring phases of a
   do-loop L for the loop header (i=1), the test (i<10) and the increment
   (i=i+1). */

statement loop_header(statement sl)
{
  loop l = statement_loop(sl);
  statement hs = statement_undefined;

  expression i = make_entity_expression(loop_index(l), NIL);

  hs = make_assign_statement(i, range_lower(loop_range(l)));
  statement_number(hs) = statement_number(sl);

  return hs;
}

statement loop_test(statement sl)
{
  loop l = statement_loop(sl);
  statement ts = statement_undefined;
  string cs = string_undefined;
  call c = make_call(entity_intrinsic(GREATER_THAN_OPERATOR_NAME),
		     CONS(EXPRESSION,
			  make_entity_expression(loop_index(l), NIL),
			  CONS(EXPRESSION,
			       range_upper(loop_range(l)),
			       NIL)));
  test t = make_test(make_expression(make_syntax(is_syntax_call, c),
				     normalized_undefined),
		     make_plain_continue_statement(),
		     make_plain_continue_statement());
  string csl = statement_comments(sl);
  string prev_comm = empty_comments_p(csl)? /* empty_comments */ strdup("")  : strdup(csl);
  const char* lab;

  if(entity_empty_label_p(loop_label(l)))
    lab = ""; // FI: to be replaced by a symbolic constant
  else
    lab = label_local_name(loop_label(l));

  switch(get_prettyprint_language_tag()) {
    case is_language_fortran:
    case is_language_fortran95:
      cs = strdup(concatenate(prev_comm,
                              get_comment_sentinel(),
                              "     DO loop ",
                              lab,
                              " with exit had to be desugared\n",
                              NULL));
      break;
    case is_language_c:
      cs = prev_comm;
      break;
    default:
      pips_internal_error("This language is not handled !");
      break;
  }

  ts = make_statement(entity_empty_label(),
		      statement_number(sl),
		      STATEMENT_ORDERING_UNDEFINED,
		      cs,
		      make_instruction(is_instruction_test, t),NIL,NULL,
		      copy_extensions (statement_extensions(sl)), make_synchronization_none());
  return ts;
}

statement loop_inc(statement sl)
{
  loop l = statement_loop(sl);
  expression I = make_entity_expression(loop_index(l), NIL);
  expression II = make_entity_expression(loop_index(l), NIL);
  call c = make_call(entity_intrinsic(PLUS_OPERATOR_NAME), // Even for C code?
		     CONS(EXPRESSION,
			  I,
			  CONS(EXPRESSION,
			       range_increment(loop_range(l)),
			       NIL)));
  expression I_plus_one =
    make_expression(make_syntax(is_syntax_call, c),
		    normalized_undefined);
  statement is = statement_undefined;

  is = make_assign_statement(II, I_plus_one);
  statement_number(is) = statement_number(sl);

  return is;
}

/* CONTROLIZE_LOOP computes in C_RES the control graph of the loop L (of
   statement ST) with PREDecessor and SUCCessor. */

static bool controlize_loop(st, l, pred, succ, c_res, used_labels)
statement st;
loop l;
control pred, succ;
control c_res;
hash_table used_labels;
{
    hash_table loop_used_labels = hash_table_make(hash_string, 0);
    control c_body = make_conditional_control(loop_body(l));
    control c_inc = make_control(make_plain_continue_statement(), NIL, NIL);
    control c_test = make_control(make_plain_continue_statement(), NIL, NIL);
    bool controlized;

    pips_debug(5, "(st = %p, pred = %p, succ = %p, c_res = %p)\n",
	       st, pred, succ, c_res);

    controlize(loop_body(l), c_test, c_inc, c_body, loop_used_labels);

    if(covers_labels_p(loop_body(l),loop_used_labels)) {
	loop new_l = make_loop(loop_index(l),
			       loop_range(l),
			       control_statement(c_body),
			       loop_label(l),
			       loop_execution(l),
			       loop_locals(l));

	st = normalize_statement(st);
	UPDATE_CONTROL(c_res,
		       make_statement(statement_label(st),
				      statement_number(st),
				      STATEMENT_ORDERING_UNDEFINED,
				      statement_comments(st),
				      make_instruction(is_instruction_loop, new_l),
				      gen_copy_seq(statement_declarations(st)),
				      strdup(statement_decls_text(st)),
				      copy_extensions(statement_extensions(st)), make_synchronization_none()),
		       ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		       CONS(CONTROL, succ, NIL)) ;
	controlized = false;
	control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_res, succ);
    }
    else {
	control_statement(c_test) = loop_test(st);
	control_predecessors(c_test) =
		CONS(CONTROL, c_res, CONS(CONTROL, c_inc, NIL)),
	control_successors(c_test) =
		CONS(CONTROL, succ, CONS(CONTROL, c_body, NIL));
	control_statement(c_inc) = loop_inc(st);
	control_successors(c_inc) = CONS(CONTROL, c_test, NIL);
	UPDATE_CONTROL(c_res,
		       loop_header(st),
		       ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		       CONS(CONTROL, c_test, NIL));
	controlized = true ;
	control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_test, succ);
    }
    add_proper_successor_to_predecessor(pred, c_res);
    /* control_successors(pred) = ADD_SUCC(c_res, pred); */

    union_used_labels( used_labels, loop_used_labels);
    hash_table_free(loop_used_labels);

    pips_debug(5, "Exiting\n");

    return(controlized);
}


/* Generate a test statement ts for exiting loop sl.
 * There should be no sharing between sl and ts.
 */
static statement whileloop_test(statement sl)
{
    whileloop l = instruction_whileloop(statement_instruction(sl));
    statement ts = statement_undefined;
    string cs = string_undefined;
    call c = call_undefined;

    switch (get_prettyprint_language_tag()) {
      case is_language_fortran:
        c = make_call(entity_intrinsic(NOT_OPERATOR_NAME),
          CONS(EXPRESSION,
         copy_expression(whileloop_condition(l)),
         NIL));
        break;
      case is_language_c:
        c = make_call(entity_intrinsic(C_NOT_OPERATOR_NAME),
          CONS(EXPRESSION,
         copy_expression(whileloop_condition(l)),
         NIL));
        break;
      case is_language_fortran95:
        pips_internal_error("Need to update F95 case");
        break;
      default:
        pips_internal_error("Language unknown !");
        break;
    }

    test t = make_test(make_expression(make_syntax(is_syntax_call, c),
                                     normalized_undefined),
                     make_plain_continue_statement(),
                     make_plain_continue_statement());
    string csl = statement_comments(sl);
    /* string prev_comm = empty_comments_p(csl)? "" : strdup(csl); */
    string prev_comm = empty_comments_p(csl)? empty_comments /* strdup("") */ : strdup(csl);
    const char* lab ;

    switch (get_prettyprint_language_tag()) {
      case is_language_fortran:
      case is_language_fortran95:
        if(entity_empty_label_p(whileloop_label(l))) {
          cs = strdup(concatenate(prev_comm,
                                  get_comment_sentinel(),
                                  "     DO WHILE loop ",
                                  "with GO TO exit had to be desugared\n",
                                  NULL));
        } else {
          lab = label_local_name(whileloop_label(l));
          cs = strdup(concatenate(prev_comm,
                                  get_comment_sentinel(),
                                  "     DO WHILE loop ",
                                  lab,
                                  " with GO TO exit had to be desugared\n",
                                  NULL));
        }
        break;
      case is_language_c:
        cs = prev_comm;
        break;
      default:
        pips_internal_error("Language unknown !");
        break;
    }


    ts = make_statement(entity_empty_label(),
			statement_number(sl),
			STATEMENT_ORDERING_UNDEFINED,
			cs,
			make_instruction(is_instruction_test, t),NIL,NULL,
			copy_extensions (statement_extensions(sl)), make_synchronization_none());

    return ts;
}


/* CONTROLIZE_WHILELOOP computes in C_RES the control graph of the loop L (of
 *  statement ST) with PREDecessor and SUCCessor
 *
 * Derived by FI from controlize_loop()
 */

/* NN : What about other kind of whileloop, evaluation = after ? TO BE IMPLEMENTED   */

static bool controlize_whileloop(st, l, pred, succ, c_res, used_labels)
statement st;
whileloop l;
control pred, succ;
control c_res;
hash_table used_labels;
{
    hash_table loop_used_labels = hash_table_make(hash_string, 0);
    control c_body = make_conditional_control(whileloop_body(l));
    bool controlized;

    pips_debug(5, "(st = %p, pred = %p, succ = %p, c_res = %p)\n",
	       st, pred, succ, c_res);

    controlize(whileloop_body(l), c_res, c_res, c_body, loop_used_labels);

    if(covers_labels_p(whileloop_body(l),loop_used_labels)) {
	whileloop new_l = make_whileloop(whileloop_condition(l),
					 control_statement(c_body),
					 whileloop_label(l),
					 whileloop_evaluation(l));

	/* The edges between c_res and c_body, created by the above call to
	 * controlize are useless. The edge succ
	 * from c_res to c_body is erased by the UPDATE_CONTROL macro.
	 */
	gen_remove(&control_successors(c_body), c_res);
	gen_remove(&control_predecessors(c_body), c_res);
	gen_remove(&control_predecessors(c_res), c_body);

	st = normalize_statement(st);
	UPDATE_CONTROL(c_res,
		       make_statement(statement_label(st),
				      statement_number(st),
				      STATEMENT_ORDERING_UNDEFINED,
				      strdup(statement_comments(st)),
				      make_instruction(is_instruction_whileloop,
						       new_l),
				      gen_copy_seq(statement_declarations(st)),
				      strdup(statement_decls_text(st)),
				      copy_extensions(statement_extensions(st)), make_synchronization_none()),
		       ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		       CONS(CONTROL, succ, NIL));
	controlized = false;
    }
    else {
	control_statement(c_res) = whileloop_test(st);
	/* control_predecessors(c_res) =
	   CONS(CONTROL, pred, control_predecessors(c_res)); */
	/* ADD_PRED(pred, c_res); */
	control_predecessors(c_res) =
	   gen_once(pred, control_predecessors(c_res));
	control_successors(c_res) =
		CONS(CONTROL, succ, control_successors(c_res));
	controlized = true ;
	/* Cannot be consistent yet! */
	/* ifdebug(5) check_control_coherency(c_res); */
    }
    control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_res, succ);
    add_proper_successor_to_predecessor(pred, c_res);
    /* control_successors(pred) = ADD_SUCC(c_res, pred); */

    ifdebug(5) check_control_coherency(c_res);

    union_used_labels( used_labels, loop_used_labels);
    hash_table_free(loop_used_labels);

    pips_debug(5, "Exiting\n");

    return(controlized);
}


statement forloop_header(statement sl)
{
  forloop l = instruction_forloop(statement_instruction(sl));
  statement hs = instruction_to_statement(make_instruction_expression(forloop_initialization(l)));
  statement_number(hs) = statement_number(sl);

  return hs;
}


statement forloop_test(statement sl)
{
  forloop l = instruction_forloop(statement_instruction(sl));
  expression cond = forloop_condition(l);
  call c =  make_call(entity_intrinsic(C_NOT_OPERATOR_NAME),
		      CONS(EXPRESSION,
			   copy_expression(cond),
			   NIL));
  test t = make_test(make_expression(make_syntax(is_syntax_call, c),
				     normalized_undefined),
		     make_plain_continue_statement(),
		     make_plain_continue_statement());
  string csl = statement_comments(sl);
  string cs = empty_comments_p(csl)? empty_comments /* strdup("") */ : strdup(csl);

  statement ts = make_statement(entity_empty_label(),
				statement_number(sl),
				STATEMENT_ORDERING_UNDEFINED,
				cs,
				make_instruction(is_instruction_test, t),NIL,NULL,
				copy_extensions(statement_extensions(sl)), make_synchronization_none());

  ifdebug(8) {
    pips_debug(8, "Condition expression: ");
    print_expression(cond);
  }

  return ts;
}


statement forloop_inc(statement sl)
{
  forloop l = instruction_forloop(statement_instruction(sl));
  expression inc = forloop_increment(l);
  statement is = instruction_to_statement(make_instruction_expression(inc));
  statement_number(is) = statement_number(sl);

  ifdebug(8) {
    pips_debug(8, "Increment expression: ");
    print_expression(inc);
  }

  return is;
}


static bool controlize_forloop(st, l, pred, succ, c_res, used_labels)
statement st;
forloop l;
control pred, succ;
control c_res;
hash_table used_labels;
{
  hash_table loop_used_labels = hash_table_make(hash_string, 0);
  control c_body = make_conditional_control(forloop_body(l));
  control c_inc = make_control(make_plain_continue_statement(), NIL, NIL);
  control c_test = make_control(make_plain_continue_statement(), NIL, NIL);
  bool controlized = false; /* To avoid gcc warning about possible
			       non-initialization */

  pips_debug(5, "(st = %p, pred = %p, succ = %p, c_res = %p)\n",
	     st, pred, succ, c_res);
  ifdebug(1) {
    statement_consistent_p(st);
  }

  controlize(forloop_body(l), c_test, c_inc, c_body, loop_used_labels);

  ifdebug(1) {
    statement_consistent_p(st);
  }

  if (covers_labels_p(forloop_body(l),loop_used_labels)) {
    instruction ni = instruction_undefined;

    /* Try an unsafe conversion to a Fortran style DO loop: it assumes
       that the loop body does not define the loop index, but effects are
       not yet available when the controlizer is run. */
    if(get_bool_property("FOR_TO_DO_LOOP_IN_CONTROLIZER")) {
        forloop_body(l)= control_statement(c_body);
        sequence new_l = for_to_do_loop_conversion(l,st);

        if(!sequence_undefined_p(new_l)) {
            ni = make_instruction_sequence( new_l);
        }
    }

    /* If the DO conversion has failed, the WHILE conversion may be requested */
    if(instruction_undefined_p(ni)) {
        if(get_bool_property("FOR_TO_WHILE_LOOP_IN_CONTROLIZER")) {
            /* As a sequence cannot carry comments, the for loop comments
               are moved to the while loop */
            sequence wls = for_to_while_loop_conversion(forloop_initialization(l),
                    forloop_condition(l),
                    forloop_increment(l),
                    control_statement(c_body),
                    statement_extensions(st));

            /* These three fields have been re-used or freed by the previous call */
            forloop_initialization(l) = expression_undefined;
            forloop_condition(l) = expression_undefined;
            forloop_increment(l) = expression_undefined;

            ni = make_instruction_sequence(wls);
        }
        else {
            forloop new_l = make_forloop(forloop_initialization(l),
                    forloop_condition(l),
                    forloop_increment(l),
                    control_statement(c_body));

            ni = make_instruction_forloop(new_l);
        }
    }

    gen_remove(&control_successors(c_body), c_res);
    gen_remove(&control_predecessors(c_body), c_res);
    gen_remove(&control_predecessors(c_res), c_body);

    /* Quite a lot of sharing between st and d_st*/
    st = normalize_statement(st);
    statement d_st = make_statement(statement_label(st),
				    statement_number(st),
				    STATEMENT_ORDERING_UNDEFINED,
				    strdup(statement_comments(st)),
				    ni,
				    gen_copy_seq(statement_declarations(st)),
				    strdup(statement_decls_text(st)),
				    copy_extensions(statement_extensions(st)), make_synchronization_none());
    ifdebug(1) {
      statement_consistent_p(st);
      statement_consistent_p(d_st);
    }
    /* Since we may have replaced a statement that may have comments and
       labels by a sequence, do not forget to forward them where they can
       be: */
    fix_statement_attributes_if_sequence(d_st);
    ifdebug(1) {
      statement_consistent_p(d_st);
    }

    UPDATE_CONTROL(c_res,
		   d_st,
		   ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		   CONS(CONTROL, succ, NIL));
    controlized = false;
    control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_res, succ);
  }
  else /* The for loop cannot be preserved as a control structure*/
    {
      /* NN : I do not know how to deal with this, the following code does not always work

	 pips_internal_error("Forloop with goto not implemented yet");*/

      free_statement(control_statement(c_test));
      control_statement(c_test) = forloop_test(st);
      control_predecessors(c_test) =
	CONS(CONTROL, c_res, CONS(CONTROL, c_inc, NIL)),
	control_successors(c_test) =
	CONS(CONTROL, succ, CONS(CONTROL, c_body, NIL));
      free_statement(control_statement(c_inc));
      control_statement(c_inc) = forloop_inc(st);
      control_successors(c_inc) = CONS(CONTROL, c_test, NIL);
      UPDATE_CONTROL(c_res,
		     forloop_header(st),
		     ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		     CONS(CONTROL, c_test, NIL));
      controlized = true ;
      control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_test, succ);
    }
  add_proper_successor_to_predecessor(pred, c_res);
  /* control_successors(pred) = ADD_SUCC(c_res, pred); */

  ifdebug(5) check_control_coherency(c_res);

  union_used_labels( used_labels, loop_used_labels);
  hash_table_free(loop_used_labels);

  pips_debug(5, "Exiting\n");

  return(controlized);
}


/* Move all the declarations found in a list of control to a given
   statement

   @ctls is a list of control nodes

   It is useful in the controlizer to keep scoping of declarations even
   with unstructured that may destroy the variable scoping rules.

   If there are conflict names on declarations, they are renamed.

   It relies on correct calls to push_declarations()/push_declarations()
   before to track where to put the declarations.

   FI: let's assume they have become scoping_statement_push/pop...
*/
static void
move_declaration_control_node_declarations_to_statement(list ctls) {
  statement s = scoping_statement_head();
  list declarations = statement_declarations(s);
  statement s_above = scoping_statement_nth(2);
  pips_debug(2, "Dealing with block statement %p included into block"
	     " statement %p\n", s, s_above);
  if (s_above==NULL)
    /* No block statement above, so it is hard to move something there :-) */
    return;

  list declarations_above  = statement_declarations(s_above);
  list new_declarations = NIL;
  /* The variables created in case of name conflict*/
  list new_variables = NIL;
  hash_table old_to_new_variables = hash_table_make(hash_chunk, 0);

  /* Look for conflicting names: */
  FOREACH(ENTITY, e, declarations) {
    const char * name = entity_user_name(e);
    bool conflict = false;
    FOREACH(ENTITY, e_above, declarations_above) {
      const char * name_above = entity_user_name(e_above);
      pips_debug(2, "Comparing variables %s and %s\n",
		 entity_name(e), entity_name(e_above));

      if (strcmp(name, name_above) == 0) {
	/* There is a conflict name between a declaration in the current
	   statement block and one in the statement block above: */
	conflict = true;
	break;
      }
    }
    entity v;
    if (conflict) {
      pips_debug(2, "Conflict on variable %s\n", entity_name(e));

      /* Create a new variable with a non conflicting name: */
      v = clone_variable_with_unique_name(e,
					  s_above,
					  "",
					  "_",
					  entity_to_module_entity(e));
      new_variables = gen_entity_cons(v , new_variables);
      hash_put_or_update(old_to_new_variables, e, v);
    }
    else
      v = e;
    /* Add the inner declaration to the upper statement later */
    new_declarations = gen_entity_cons(v , new_declarations);
  }
  /* Remove the inner declaration from the inner statement block:
   */
  gen_free_list(statement_declarations(s));
  statement_declarations(s) = NIL;

  /* Add all the declarations to the statement block above and keep the
     same order: */
  statement_declarations(s_above) = gen_nconc(declarations_above,
					      gen_nreverse(new_declarations));

  /* Replace all the references on old variables to references to the new
     ones in all the corresponding control nodes by in the code */
  HASH_MAP(old, new, {
      FOREACH(CONTROL, c, ctls) {
	statement s = control_statement(c);
	replace_entity(s, old, new);
      }
      /* We should free in some way the old variable... */
    }, old_to_new_variables);
  hash_table_free(old_to_new_variables);
}


/* Take a list of controls @p ctls coming from a controlize_list() and
   compact the successive statements, i.e. concatenates (i=1) followed by
   (j=2) in a single control with a block statement (i=1;j=2).

   @return the last control node of the remaining control list. It may not
   be @p c_end if this one has been fused with a previous control node.

   Added a set to avoid investigating a removed node.  Many memory leaks
   removed. RK.

   This procedure cannot be replaced by fuse_sequences_in_unstructured()
   since the API is not the same and because of various goto from/to
   outside, this control list may belong to quite larger unstructured.
   */
static control compact_list(list ctls,
			    control c_end)
{
    control c_res;
    set processed_nodes;
    /* Pointer to the end of the current unstructured: */
    control c_last = c_end ;

    ifdebug(5) {
	pips_debug(0, "Begin with list c_end %p, ctls:", c_end);
	display_address_of_control_nodes(ctls);
	fprintf(stderr, "\n");
    }

    if (ENDP(ctls)) {
      /* Empty statement control list, clearly nothing to do: */
      return c_last;
    }

    processed_nodes = set_make(set_pointer);

    /* This is the iterator on the first element of the control list: */
    c_res = CONTROL(CAR(ctls));

    for(ctls = CDR(ctls); !ENDP(ctls); ctls = CDR(ctls)) {
	cons *succs, *succs_of_succ;
	instruction i, succ_i;
	statement st, succ_st;
	control succ;

	if (set_belong_p(processed_nodes, (char *) c_res)) {
	    /* Do not reprocess an already seen node. */
	    c_res = CONTROL(CAR(ctls));
	    continue;
	}

	if (gen_length(succs=control_successors(c_res)) != 1 ||
	    gen_length(control_predecessors(succ=CONTROL(CAR(succs)))) != 1 ||
	    gen_length(succs_of_succ=control_successors(succ)) != 1 ||
	    CONTROL(CAR(succs_of_succ)) == c_res ) {
	    /* We are at a non structured node or at the exit: keep
               the node and go on inspecting next node from the
               list. RK */
	    c_res = CONTROL(CAR(ctls));
	    continue;
	}

	st = control_statement(c_res) ;

  ifdebug(1) {
    statement_consistent_p(st);
    pips_assert("st is a block or a continue or st carries no delarations",
                statement_block_p(st)
                || continue_statement_p(st)
                || ENDP(statement_declarations(st)));
  }
  /* Ok, succ is defined. RK */
	succ_st = control_statement(succ);
	set_add_element(processed_nodes, processed_nodes, (char *) succ);

	if(!statement_undefined_p(control_statement(succ))
	   && !entity_empty_label_p(statement_label(control_statement(succ)))
	   && !return_label_p(entity_name(statement_label(control_statement(succ))))) {
	  /* Verify if the control node is not reachable on its label: */
	  entity l = statement_label(control_statement(succ));
	  string ln = entity_name(l);
	  control c = get_label_control(ln);
	  if(!control_undefined_p(c)) {
	    /* There is a label that may be used on this control node: do
	       not fuse */
	    if (c==succ) {
	      /* This happens quite often in Syntax with no
		 consequences; this leads to a core dump for
		 C_syntax/block_scope13.c */
	      pips_debug(2, "Do not fuse control %p since we will have a latent goto on it through label \"%s\"\n",
			 c_res, ln);
	      /* Do not fuse: go to next control node: */
	      c_res = CONTROL(CAR(ctls));
	      continue;
	    }
	    else
	      pips_internal_error("Inconsistent hash table Label_control: "
				  "same label points towards two different controls");
	  }
	}
	if(c_res != succ) {
	    /* If it is not a loop on c_res, fuse the nodes: */
	  if((statement_block_p(st) && !ENDP(statement_declarations(st)))
	     || (statement_block_p(succ_st) && !ENDP(statement_declarations(succ_st)))) {
	    /* You need a new block to fuse and to respect the scopes */
		i = make_instruction_block(CONS(STATEMENT, st,
						CONS(STATEMENT, succ_st, NIL)));
		control_statement(c_res) =
		    make_statement(entity_empty_label(),
				   STATEMENT_NUMBER_UNDEFINED,
				   STATEMENT_ORDERING_UNDEFINED,
				   string_undefined,
				   i,NIL,NULL,
				   empty_extensions (), make_synchronization_none());
	    ;
	  }
	  else {
	    if(!ENDP(statement_declarations(st))
	       && !continue_statement_p(st)) {
	      pips_user_warning("Declarations carried by a statement \"%s\""
				" which is not a block nor a continue!\n",
				statement_identification(st));
	    }
	    if(!instruction_block_p(i=statement_instruction(st))) {
		i = make_instruction_block(CONS(STATEMENT, st, NIL));
		control_statement(c_res) =
		    make_statement(entity_empty_label(),
				   STATEMENT_NUMBER_UNDEFINED,
				   STATEMENT_ORDERING_UNDEFINED,
				   string_undefined,
				   i,NIL,NULL,
				   empty_extensions (), make_synchronization_none());
	    }
	    if(instruction_block_p(succ_i=statement_instruction(succ_st))){
		instruction_block(i) =
		    gen_nconc(instruction_block(i),
			      instruction_block(succ_i));
		pips_debug(8, "Free statement %p with identification %s from control succ %p\n",
			   succ_st, statement_identification(succ_st), succ);
		statement_instruction(succ_st) = instruction_undefined;
		free_statement(succ_st);
		succ_st = statement_undefined;
		control_statement(succ) = statement_undefined;
	    }
	    else {
		instruction_block(i) =
		    gen_nconc(instruction_block(i),
			      CONS(STATEMENT, succ_st, NIL));
	    }
	  }
	  ifdebug(1) {
	    pips_assert("control succ and its statement are consistent",
	                control_consistent_p(succ));
	  }
	  /* Remove the useless control: */
	  control_statement(succ) = statement_undefined;
	  remove_a_control_from_an_unstructured(succ);
	}

	if(succ == c_last) {
	    /* We are at the end and the last node has
               disappeared... Update the pointer to the new one: */
	    c_last = c_res;
	    break;
	}
    ifdebug(1) {
      statement_consistent_p(st);
      statement_consistent_p(succ_st);
    }
  }
  set_free(processed_nodes);
  return c_last;
}


/* Do the equivalent of a mapcar of controlize on statement list @p sts.

   The trick is to keep a list of the controls to compact them later. Note
   that if a statement is controlized, then the predecessor has to be
   computed (i.e. is not the previous control of @p sts); this is the
   purpose of c_in.

   This function used to update its formal parameters pred and c_res, which
   makes stack visualization and debugging difficult.
*/
list controlize_list_1(list sts,
		       control i_pred,
		       control i_succ,
		       control i_c_res,
		       hash_table used_labels) {
  /* The list of all the control nodes associated to this statement
     list: */
  list ctls = NIL;
  control pred = i_pred;
  control succ = i_succ;
  control c_res = i_c_res;

  /* On all the statement list: */
  for(; !ENDP(sts); sts = CDR(sts)) {
    statement st = STATEMENT(CAR(sts));
    /* Create a control node for the successor of this statement if
       not the last one: */
    control c_next = ENDP(CDR(sts)) ? succ :
      make_conditional_control(STATEMENT(CAR(CDR(sts))));
    bool controlized;
    bool unreachable;

    ifdebug(5) {
      pips_debug(0, "Nodes linked with pred %p:\n", pred);
      display_linked_control_nodes(pred);
    }

    ifdebug(1) {
      check_control_coherency(pred);
      check_control_coherency(succ);
      check_control_coherency(c_next);
      check_control_coherency(c_res);
    }

    /* Controlize the current statement: */
    controlized = controlize(st, pred, c_next, c_res, used_labels);
    unreachable = ENDP(control_predecessors(c_next));

    /* Keep track of the control node associated to this statement: */
    ctls = CONS(CONTROL, c_res, ctls);

    if (unreachable) {
      /* Keep track globally of the unreachable code: */
      Unreachable = CONS(STATEMENT, st, Unreachable);
      ifdebug(2) {
	pips_debug(0, "There is a new unreachable statement:\n");
	print_statement(st);
      }
    }

    if (controlized) {
      /* The previous controlize() returned a non structured control */
      control c_in = make_control(make_plain_continue_statement(), NIL, NIL);

      ctls = CONS(CONTROL, c_in, ctls);
      /* Insert c_in as a predecessor of c_next

	 RK: I do not understand why this is needed...
      */
      control_predecessors(c_in) = control_predecessors(c_next);
      control_successors(c_in) = CONS(CONTROL, c_next, NIL);
      patch_references(SUCCS_OF_PREDS, c_next, c_in);
      control_predecessors(c_next) = CONS(CONTROL, c_in, NIL) ;
      pred = c_in;
    }
    else {
      /* If the next control node is unreachable, it will not be
	 connected to the previous predecessor, so allocate a new one
	 so that it has a predecessor that is completely unconnected of
	 previous control graph. */
      pred = (unreachable) ?
	make_control(make_plain_continue_statement(), NIL, NIL) :
	c_res;
    }
    /* The next control node is the control node of the next
       statement: */
    c_res = c_next ;
  }

  ifdebug(1) {
    /* The consistency check should be applied to all elements in
       ctls. Let's hope all controls in ctls are linked one way or
       the other. */
    control c = CONTROL(CAR(ctls));
    pips_debug(5, "Nodes from c %p\n", c);
    display_linked_control_nodes(c);
    check_control_coherency(c);

    check_control_coherency(pred);
    check_control_coherency(succ);
    check_control_coherency(c_res);
    pips_debug(5, "(pred = %p, succ = %p, c_res = %p)\n",
	       pred, succ, c_res);
    pips_debug(5, "Nodes from pred %p\n", pred);
    display_linked_control_nodes(pred);
    pips_debug(5, "Nodes from succ %p\n", succ);
    display_linked_control_nodes(succ);
    pips_debug(5, "Nodes from c_res %p\n", c_res);
    display_linked_control_nodes(c_res);
  }

  /* Since we built the list in reverse order to have a O(n)
     construction: */
  return gen_nreverse(ctls);
}


/* Computes in @p c_res the control graph of the list @p sts (of statement
   @p st) with @p pred predecessor and @p succ successor.

   We try to minimize the number of graphs by looking for graphs with one
   node only and picking the statement in that case.

   @return true if the code is not a structured control.
   */
static bool controlize_list(statement st,
			    list sts,
			    control pred,
			    control succ,
			    control c_res,
			    hash_table used_labels)
{
    hash_table block_used_labels = hash_table_make(hash_string, 0);
    control c_block = control_undefined;
    control c_end = make_control(make_plain_continue_statement(), NIL, NIL);
    control c_last = c_end;
    list ctls;
    bool controlized;
    bool hierarchized_labels;

    pips_debug(5, "Begin with (st = %p, pred = %p, succ = %p, c_res = %p)\n",
	       st, pred, succ, c_res);
    ifdebug(1) {
      ifdebug(8) {
	pips_debug(8, "\nControl nodes linked to pred = %p:\n", pred);
	display_linked_control_nodes(pred);
	pips_debug(8, "\n");
      }
      ifdebug(1) {
        statement_consistent_p(st);
        check_control_coherency(pred);
        check_control_coherency(succ);
        check_control_coherency(c_res);
      }
    }

    if(ENDP(sts)) {
      /* Empty statement list. It looks like from controlize() that we
	 cannot be called with an empty statement list... So I guess this
	 is dead code here. RK */
      list d = gen_copy_seq(statement_declarations(st));
      string dt
	= (statement_decls_text(st)==NULL || string_undefined_p(statement_decls_text(st))) ?
	strdup("")
	: strdup(statement_decls_text(st));
      string ct = string_undefined_p(statement_comments(st))?
	string_undefined /* Should be empty_comments ? */
	: strdup(statement_comments(st));

      /* FI: the statement extension of st is lost */
      c_block = make_control(make_empty_statement_with_declarations_and_comments(d, dt, ct),
			     NIL, NIL);
      /*pips_assert("declarations are preserved in control",
		  gen_length(statement_declarations(st))
		  ==gen_length(statement_declarations(control_statement(c_block))));*/

    }
    else {
      /* What happens to the declarations and comments attached to st? */
      /* Create a control node to hold what was the statement block, with
	 the first statement in it: */
      c_block = make_conditional_control(STATEMENT(CAR(sts)));
      /*pips_assert("declarations are preserved in conditional control",
		  gen_length(statement_declarations(st))
		  ==gen_length(statement_declarations(control_statement(c_block))));*/

    }

    ifdebug(1) {
      statement_consistent_p(st);
    }

    /* Do the real transformation of a statement list into a control
       graph: */
    ctls = controlize_list_1(sts, pred, c_end, c_block, block_used_labels);

    /* Compute if there are goto from/to the statements of the statement
       list: */
    hierarchized_labels = covers_labels_p(st, block_used_labels);

    if (!hierarchized_labels) {
      /* We are in trouble since we will have an unstructured with goto
	 from or to outside this statement sequence, but the statement
	 sequence that define the scoping rules is going to disappear...
	 So we gather all the declaration and push them up: */
      move_declaration_control_node_declarations_to_statement(ctls);
    }
    /* Since we have generated a big control graph from what could be a
       more structured statement block, try to restructure things a little
       bit, with c_last pointing to the last control node of the list: */
    c_last = compact_list(ctls, c_end);
    /* To avoid compact list: c_last = c_end; */
    //c_last = c_end;
    gen_free_list(ctls);
    ifdebug(5) {
	pips_debug(0, "Nodes from c_block %p\n", c_block);
	display_linked_control_nodes(c_block);
	pips_debug(0, "Nodes from c_last %p\n", c_last);
	display_linked_control_nodes(c_last);
    }
    /*    pips_assert("declarations are preserved in list",
		gen_length(statement_declarations(st))
		==gen_length(statement_declarations(control_statement(c_block))));*/

    if (hierarchized_labels) {
	/* There is no GOTO to/from  outside the statement list:
           hierarchize the control graph. */
	statement new_st = statement_undefined;

	/* Unlink the c_block from the unstructured. RK. */
	unlink_2_control_nodes(pred, c_block);
	unlink_2_control_nodes(c_block, c_end);

	if(ENDP(control_predecessors(c_block)) &&
	   ENDP(control_successors(c_block))) {
	  /* c_block is a lonely control node: */
	  new_st = control_statement(c_block);

	  /* FI: fragile attempt at keeping local declarations and their scope */
	  /* Does not work when st has been changed... */
	  if(/*!statement_block_p(new_st) &&*/ !ENDP(statement_declarations(st))) {
	    /* new_st = st*/;
	  }

	  /* PJ: fragile attempt at keeping local declarations and their scope */
	  ifdebug(1) {
	    statement_consistent_p(st);
	  }
	  if(!ENDP(statement_declarations(st))) {
	    pips_assert("the declarations are carried by a block",
			statement_block_p(st));
	    ifdebug(8) {
	      pips_debug(8, "Block declarations to copy: ");
	      print_entities(statement_declarations(st));
	      pips_debug(8, "End of declarations.\n");
	    }
	    if(statement_block_p(new_st)) {
	      if(ENDP(statement_declarations(new_st))) {
		statement_declarations(new_st) =
		  gen_copy_seq(statement_declarations(st));
	      }
	      else {
		new_st = make_block_statement(CONS(STATEMENT, new_st, NIL));
		statement_declarations(new_st) =
		  gen_copy_seq(statement_declarations(st));
	      }
	    }
	    else {
	      new_st = make_block_statement(CONS(STATEMENT, new_st, NIL));
	      statement_declarations(new_st) =
		gen_copy_seq(statement_declarations(st));
	    }
	    /* FI: Can't we remove the declarations in st? Why are
	       they gen_copy_seq? */
	  }

	  control_statement(c_block) = statement_undefined;
	    free_control(c_block);

	    /* FI: no need to update declarations as the code is structured */
	    ifdebug(1) {
	      statement_consistent_p(st);
	    }
	}
	else {
	    /* The control is kept in an unstructured: */
	    unstructured u = make_unstructured(c_block, c_last);
	    instruction i =
		make_instruction(is_instruction_unstructured, u);

	    ifdebug(1) {
		check_control_coherency(unstructured_control(u));
		check_control_coherency(unstructured_exit(u));
	    }
	    /* FI: So here you are putting declarations in an
	       unstructured? No surprise we end up in trouble
	       later. */
	    st = normalize_statement(st);
	    if(ENDP(statement_declarations(st))) {
	      new_st = make_statement(entity_empty_label(),
				      statement_number(st),
				      STATEMENT_ORDERING_UNDEFINED,
				      strdup(statement_comments(st)),
				      i,
				      gen_copy_seq(statement_declarations(st)),
				      strdup(statement_decls_text(st)),
				      copy_extensions(statement_extensions(st)), make_synchronization_none());
	    }
	    else {
	      statement us =
		make_statement(entity_empty_label(),
			       statement_number(st),
			       STATEMENT_ORDERING_UNDEFINED,
			       strdup(statement_comments(st)),
			       i,
			       NIL,
			       strdup(""),
			       empty_extensions(), make_synchronization_none());
	      new_st =
		make_empty_statement_with_declarations_and_comments(
								    gen_copy_seq(statement_declarations(st)),
								    strdup(statement_decls_text(st)),
				    empty_comments);
	      statement_extensions(new_st) = copy_extensions(statement_extensions(st));
	      statement_instruction(new_st) =
		make_instruction_block(CONS(STATEMENT, us, NIL));
	    }
	}

	/* Not a good idea from mine to add this free... RK
	   free_statement(control_statement(c_res)); */

	control_statement(c_res) = new_st;

	/* FI: when going down controlize list, these two nodes are
	   already linked, and they are relinked unconditionnally by
	   link_2_control_nodes() which does not check that its input
	   assumption is met. */
	unlink_2_control_nodes(pred, c_res);

	link_2_control_nodes(pred, c_res);
	link_2_control_nodes(c_res, succ);
	controlized = false;
    }
    else {
      pips_debug(2, "There are goto to/from outside this statement list\n");
	/* Update c_res to reflect c_block in fact: */
	/* We alredy have pred linked to c_block and the exit node
           linked to succ. RK */
	UPDATE_CONTROL(c_res,
		       control_statement(c_block),
		       gen_copy_seq(control_predecessors(c_block)),
		       control_successors(c_block));
	control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_end, succ);
	control_successors(c_end) = CONS(CONTROL, succ, NIL);
	patch_references(PREDS_OF_SUCCS, c_block, c_res);
	patch_references(SUCCS_OF_PREDS, c_block, c_res);
	controlized = true;
	/*	pips_assert("declarations are preserved",
		    gen_length(statement_declarations(st))
		    ==gen_length(statement_declarations(control_statement(c_res))));*/
    }

    ifdebug(1) {
      statement_consistent_p(st);
    }

    union_used_labels(used_labels, block_used_labels);

    hash_table_free(block_used_labels);

    pips_debug(5, "Exiting with controlized = %s\n",
	       bool_to_string(controlized));
    ifdebug(1) {
      ifdebug(8) {
	pips_debug(8, "\nNodes linked to pred %p\n", pred);
	display_linked_control_nodes(pred);
	pips_debug(8, "\n");
      }
	check_control_coherency(pred);
	check_control_coherency(succ);
	check_control_coherency(c_res);
	/*
	pips_assert("declarations are preserved",
		    gen_length(statement_declarations(st))
		    ==gen_length(statement_declarations(control_statement(c_res))));
	*/
    }

    ifdebug(1) {
      statement_consistent_p(st);
    }
    return(controlized);
}


/* Builds the control node of a statement @p st in @p c_res which is a
   test statement @p t.

   @return true if the control node generated is not structured_.
 */
static bool controlize_test(st, t, pred, succ, c_res, used_labels)
test t;
statement st;
control pred, succ;
control c_res;
hash_table used_labels;
{
  hash_table
    t_used_labels = hash_table_make(hash_string, 0),
    f_used_labels = hash_table_make(hash_string, 0);
  control c1 = make_conditional_control(test_true(t));
  control c2 = make_conditional_control(test_false(t));
  control c_join = make_control(make_plain_continue_statement(), NIL, NIL);
  statement s_t = test_true(t);
  statement s_f = test_false(t);
  bool controlized;

  pips_debug(5, "Entering (st = %p, pred = %p, succ = %p, c_res = %p)\n",
	     st, pred, succ, c_res);

  ifdebug(5) {
    pips_debug(1, "THEN at entry:\n");
    print_statement(s_t);
    pips_debug(1, "c1 at entry:\n");
    print_statement(control_statement(c1));
    pips_debug(1, "ELSE at entry:\n");
    print_statement(s_f);
    pips_debug(1, "c2 at entry:\n");
    print_statement(control_statement(c2));
    check_control_coherency(pred);
    check_control_coherency(succ);
    check_control_coherency(c_res);
  }

  controlize(s_t, c_res, c_join, c1, t_used_labels);
  /* Just put the IF statement in c_res so that
     add_proper_successor_to_predecessor() that may be called from the
     next controlize() here behave correctly: */
  control_statement(c_res) = st;
  controlize(s_f, c_res, c_join, c2, f_used_labels);

  if(covers_labels_p(s_t, t_used_labels) &&
     covers_labels_p(s_f, f_used_labels)) {
    /* If all the label jumped to from the THEN/ELSE statements are in
       their respecive statement, we can replace the unstructured test by
       a structured one: */
    test it = make_test(test_condition(t),
			control_statement(c1),
			control_statement(c2));
    /* c1 & c2 are no longer useful: */
    free_a_control_without_its_statement(c1);
    free_a_control_without_its_statement(c2);

    st = normalize_statement(st);
    UPDATE_CONTROL(c_res,
		   make_statement(statement_label(st),
				  statement_number(st),
				  STATEMENT_ORDERING_UNDEFINED,
				  strdup(statement_comments(st)),
				  make_instruction(is_instruction_test, it),
				  gen_copy_seq(statement_declarations(st)),
				  strdup(statement_decls_text(st)),
				  copy_extensions(statement_extensions(st)), make_synchronization_none()),
		   ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		   CONS(CONTROL, succ, NIL));
    control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_res, succ);
    controlized = false;
  }
  else {
    // Keep the unstructured test:
    UPDATE_CONTROL(c_res, st,
		   ADD_PRED_AND_COPY_IF_NOT_ALREADY_HERE(pred, c_res),
		   CONS(CONTROL, c1, CONS(CONTROL, c2, NIL)));
    test_true(t) = make_plain_continue_statement();
    test_false(t) = make_plain_continue_statement();
    control_predecessors(succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_join, succ);
    control_successors(c_join) = CONS(CONTROL, succ, NIL);
    controlized = true;
  }

  /* Be very careful when chaining c_res as successor of pred: if pred is
     associated to a test, the position of c_res as first or second
     successor of c_res is unknown. It should have been set by the
     caller.

     You might survive, using the fact that the true branch has been
     processed first because of the order of the two recursive calls to
     controlize(). The false branch whill be linked as second
     successor. But another controlize_xxx might have decided to link pred
     and c_res before calling controlize() and controlize_test(). Too much
     guessing IMHO (FI). */

  /* control_successors(pred) = ADD_SUCC(c_res, pred); */
  add_proper_successor_to_predecessor(pred, c_res);

  union_used_labels(used_labels,
		    union_used_labels(t_used_labels, f_used_labels));

  hash_table_free(t_used_labels);
  hash_table_free(f_used_labels);

  ifdebug(5) {
    pips_debug(1, "IF at exit:\n");
    print_statement(st);
    display_linked_control_nodes(c_res);
    check_control_coherency(pred);
    check_control_coherency(succ);
    check_control_coherency(c_res);
  }
  pips_debug(5, "Exiting\n");

  return(controlized);
}

/* INIT_LABEL puts the reference in the statement ST to the label NAME
   int the Label_statements table and allocate a slot in the Label_control
   table. */

static void init_label(name, st )
string name;
statement st;
{
    if(!empty_global_label_p(name)) {
	list used = (list) hash_get_default_empty_list(Label_statements, name);
	list sts = CONS(STATEMENT, st, used);
	/* Just append st to the list of statements pointing to
	   this label. */
	if (hash_defined_p(Label_statements, name))
	    hash_update(Label_statements, name,  sts);
	else
	    hash_put(Label_statements, name,  sts);

	if (! hash_defined_p(Label_control, name)) {
	    statement new_st = make_continue_statement(statement_label(st)) ;
	    control c = make_control( new_st, NIL, NIL);
	    pips_debug(8, "control %p allocated for label \"%s\"", c, name);
	    hash_put(Label_control, name, c);
	}
    }
}

/* CREATE_STATEMENTS_OF_LABELS gathers in the Label_statements table all
   the references to the useful label of the statement ST. Note that for
   loops, the label in the DO statement is NOT introduced. Label_control is
   also created. */

static void create_statements_of_label(st)
statement st;
{
    string name = entity_name(statement_label(st));
    instruction i;

    init_label(name, st);

    switch(instruction_tag(i = statement_instruction(st))) {
    case is_instruction_goto: {
	string where = entity_name(statement_label(instruction_goto(i)));

	init_label(where, st);
	break;
    }

    case is_instruction_unstructured:
	pips_internal_error("Found unstructured", "");

    default:
      ;
    }
}


static void create_statements_of_labels(st)
statement st ;
{
    gen_recurse(st,
		statement_domain,
		gen_true,
		create_statements_of_label);
}


/* SIMPLIFIED_UNSTRUCTURED tries to get rid of top-level and useless
   unstructured nodes.

   top is the entry node, bottom the exit node. result is ? (see
   assert below... Looks like it is the second node. RK)

   All the comments below are from RK who is reverse engineering all
   that stuff... :-(

   Looks like there are many memory leaks. */
static unstructured
simplified_unstructured(control top,
			control bottom,
			control res)
{
    list succs;
    statement st;
    unstructured u;
    instruction i;

    ifdebug(4) {
	pips_debug(0, "Accessible nodes from top:\n");
	display_linked_control_nodes(top);
	check_control_coherency(top);
	pips_debug(1, "Accessible nodes from bottom:\n");
	display_linked_control_nodes(bottom);
	check_control_coherency(bottom);
	pips_debug(1, "Accessible nodes from res:\n");
	display_linked_control_nodes(res);
	check_control_coherency(res);
    }

    ifdebug(1) {
      control_consistent_p(top);

      control_consistent_p(bottom);
    }

    u = make_unstructured(top, bottom);

    ifdebug(1) {
      unstructured_consistent_p(u);
    }

    if(!ENDP(control_predecessors(top))) {
    free_control(res);
	/* There are goto on the entry node: */
	return(u);
    }

    if(gen_length(succs=control_successors(top)) != 1) {
    free_control(res);
	/* The entry node is not a simple node sequence: */
	return(u);
    }

    pips_assert("The successor of \"top\" is \"res\"",
		CONTROL(CAR(succs)) == res);

    if(gen_length(control_predecessors(res)) != 1) {
    free_control(res);
	/* The second node has more than 1 goto on it: */
	return(u);
    }

    if(gen_length(succs=control_successors(res)) != 1) {
    free_control(res);
	/* Second node is not a simple node sequence: */
	return(u);
    }

    if(CONTROL(CAR(succs)) != bottom) {
	/* The third node is not the exit node: */
	return(u);
    }

    if(gen_length(control_predecessors(bottom)) != 1) {
    free_control(res);
	/* The exit node has more than 1 goto on it: */
	return(u);
    }

    if(!ENDP(control_successors(bottom))) {
    free_control(res);
	/* The exit node has a successor: */
	return(u);
    }

    /* Here we have a sequence of 3 control node: top, res and
       bottom. */
    gen_free_list(control_predecessors(res));
    gen_free_list(control_successors(res));
    control_predecessors(res) = control_successors(res) = NIL;
    st = control_statement(res);

    if(instruction_unstructured_p(i=statement_instruction(st))) {
	/* If the second node is an unstructured, just return it
           instead of top and bottom: (??? Lot of assumptions. RK) */
      ifdebug(1) {
        unstructured_consistent_p(instruction_unstructured(i));
      }
      unstructured iu = instruction_unstructured(i);
      instruction_unstructured(i)=unstructured_undefined;
      free_statement(st);
	return iu;
    }

    /* Just keep the second node as an unstructured with only 1
       control node: */
    unstructured_control(u) = unstructured_exit(u) = res;
    ifdebug(1) {
      unstructured_consistent_p(u);
    }
    return(u);
}


/* Computes in @p c_res the control node of the statement @p st whose
   predecessor control node is @p pred and successor @p succ.

   The @p used_LABELS is modified to deal with local use of labels.

   The invariant is that it links predecessors and successors of @p c_res,
   updates the successors of @p pred and the predecessors of @p succ.

   In fact, it cannot update the successors of @p pred because it cannot
   know which successor of @p pred @p c_RES is when @p pred is associated
   to a test. @p pred and @p c_res must be linked together when you enter
   controlize(), or they must be linked later by the caller. But they
   cannot be linked here thru the successor list of @p pred and, if the
   consistency is true here, they cannot be linked either by the
   predecessor list of @p succ. If they are linked later, it is useless to
   pass @p pred down. If they are linked earlier, they might have to be
   unlinked when structured code is found.

   @return true if the current statement isn't a structured control.
*/
bool controlize(statement st,
		control pred,
		control succ,
		control c_res,
		hash_table used_labels)
{
    instruction i = statement_instruction(st);
    entity elabel = statement_label(st);
    string label = entity_name(elabel);
    bool controlized = false;
    control n_succ = control_undefined; // To be used in case of goto

    ifdebug(5) {
	pips_debug(1,
		   "Begin with (st = %p, pred = %p, succ = %p, c_res = %p)\n"
		   "st at entry:\n",
		   st, pred, succ, c_res);
  ifdebug(1) {
    statement_consistent_p(st);
  }
	print_statement(st);
	/*
	pips_assert("pred is a predecessor of c_res",
		    gen_in_list_p(pred, control_predecessors(c_res)));
	pips_assert("c_res is a successor of pred",
		    gen_in_list_p(c_res, control_successors(pred)));
	*/
	pips_debug(1, "Begin with result c_res %p:\n", c_res);
	display_linked_control_nodes(c_res);
	check_control_coherency(pred);
	check_control_coherency(succ);
	check_control_coherency(c_res);
    }

    switch(instruction_tag(i)) {
    case is_instruction_block: {
      /* A C block may have a label and even goto from outside on it. */
      /* A block may only contain declarations with initializations
	 and side effects on the store */
      if(ENDP(instruction_block(i))) {
	/* Empty block */
	controlized = controlize_call(st, pred, succ, c_res);
	      ifdebug(1) {
	        statement_consistent_p(st);
	      }
      }
      else {
        ifdebug(1) {
          statement_consistent_p(st);
        }
	scoping_statement_push(st);
	controlized = controlize_list(st, instruction_block(i),
				      pred, succ, c_res, used_labels);
	      ifdebug(1) {
	        statement_consistent_p(st);
	      }

	ifdebug(5) {
	  pips_debug(1, "CFG consistency check before list6 controlization."
		     " Control \"pred\" %p:\n", pred);
	  display_linked_control_nodes(pred);
	  pips_debug(1, "Control \"succ\" %p:\n", succ);
	  display_linked_control_nodes(succ);

	  check_control_coherency(pred);
	  check_control_coherency(succ);
	  check_control_coherency(c_res);
	}
	/* If st carries local declarations, so should the statement
	   associated to c_res. */
	if(controlized && !ENDP(statement_declarations(st))
	   && ENDP(statement_declarations(control_statement(c_res)))) {
	  print_arguments(statement_declarations(st));
	  pips_user_warning("Some local declarations may have been lost\n");
	}
	scoping_statement_pop();
	      ifdebug(1) {
	        statement_consistent_p(st);
	      }
      }
      break;
    }
    case is_instruction_test:
	controlized = controlize_test(st, instruction_test(i),
				      pred, succ, c_res, used_labels);
	    ifdebug(1) {
	      statement_consistent_p(st);
	    }
	break;
    case is_instruction_loop:
	controlized = controlize_loop(st, instruction_loop(i),
				      pred, succ, c_res, used_labels);
	    ifdebug(1) {
	      statement_consistent_p(st);
	    }
	break;
    case is_instruction_whileloop:
	controlized = controlize_whileloop(st, instruction_whileloop(i),
					   pred, succ, c_res, used_labels);
	    ifdebug(1) {
	      statement_consistent_p(st);
	    }
	break;
    case is_instruction_goto: {
      /* Get the label name of the statement the goto point to: */
	string name = entity_name(statement_label(instruction_goto(i)));
	statement nop = make_continue_statement(statement_label(st));

	statement_number(nop) = statement_number(st);
	statement_comments(nop) = statement_comments(st);
	// Well, let's try this for the time being. What is the scope?!?
	statement_declarations(nop) = statement_declarations(st);
	statement_decls_text(nop) = statement_decls_text(st);

	n_succ = get_label_control(name);

	ifdebug(5) {
	  pips_debug(1, "CFG consistency check before goto controlization."
		     " Control \"pred\" %p:\n", pred);
	  display_linked_control_nodes(pred);
	  pips_debug(1, "Control \"n_succ\" %p:\n", n_succ);
	  display_linked_control_nodes(n_succ);
	  check_control_coherency(pred);
	  check_control_coherency(n_succ);
	  check_control_coherency(c_res);
	}

	/* Memory leak in CONS(CONTROL, pred, NIL). Also forgot to
           unlink the predecessor of the former successor of pred. RK */
	/* control_successors(pred) = ADD_SUCC(c_res, pred); */
	add_proper_successor_to_predecessor(pred, c_res);
	UPDATE_CONTROL(c_res, nop,
		       CONS(CONTROL, pred, NIL),
		       CONS(CONTROL, n_succ, NIL));
	control_predecessors(n_succ) = ADD_PRED_IF_NOT_ALREADY_HERE(c_res, n_succ);
	/* I do not know why, but my following code does not work. So
           I put back former one above... :-( RK. */
#if 0
	/* Use my procedures instead to set a GOTO from pred to
           c_res. RK */
	if (gen_length(control_successors(pred)) == 1)
	    unlink_2_control_nodes(pred, CONTROL(CAR(control_successors(pred))));
	link_2_control_nodes(pred, c_res);
	link_2_control_nodes(c_res, n_succ);
	/* Hmmm... A memory leak on the previous statement of c_res? */
	control_statement(c_res) = nop;
#endif
	update_used_labels(used_labels, name, st);
	controlized = true;
	break;
    }
    case is_instruction_call:
	/* FI: IO calls may have control effects; they should be handled here! */
	controlized = controlize_call(st, pred, succ, c_res);
	    ifdebug(1) {
	      statement_consistent_p(st);
	    }
	break;
    case is_instruction_forloop:
      pips_assert("We are really dealing with a for loop",
		  instruction_forloop_p(statement_instruction(st)));
      controlized = controlize_forloop(st, instruction_forloop(i),
				       pred, succ, c_res, used_labels);
      ifdebug(1) {
        statement_consistent_p(st);
      }
        /* SG+EC:some label may have been lost in the process
           fix it here instead of understanding why */
        if(!same_entity_p(statement_label(st),elabel)) {
            statement_label(st)=elabel;
        }
      break;
    case is_instruction_expression:
      /* PJ: controlize_call() controlize any "nice" statement */
      controlized = return_instruction_p(i) || controlize_call(st, pred, succ, c_res);

      ifdebug(1) {
        statement_consistent_p(st);
      }
      break;
    default:
	pips_internal_error("Unknown instruction tag %d", instruction_tag(i));
    }

    ifdebug(5) {
	statement_consistent_p(st);
	pips_debug(1, "st %p at exit:\n", st);
	print_statement(st);
	pips_debug(1, "Resulting Control c_res %p at exit:\n", c_res);
	display_linked_control_nodes(c_res);
	fprintf(stderr, "---\n");
	/* The declarations may be preserved at a lower level
	if(!ENDP(statement_declarations(st))
	   && ENDP(statement_declarations(control_statement(c_res)))) {
	  pips_internal_error("Lost local declarations");
	}
	*/
	check_control_coherency(pred);
	if(control_undefined_p(n_succ))
	  check_control_coherency(succ);
	else
	  check_control_coherency(n_succ);
	check_control_coherency(c_res);
    }

    /* Update the association between the current statement and its label:
    */
    update_used_labels(used_labels, label, st);

    return(controlized);
}


/* CONTROL_GRAPH returns the control graph of the statement ST. */
unstructured control_graph(st)
statement st;
{
    control result, top, bottom;
    hash_table used_labels = hash_table_make(hash_string, 0);
    unstructured u = unstructured_undefined;

    ifdebug(1) {
	pips_assert("Statement should be OK.", statement_consistent_p(st));
	set_bool_property("PRETTYPRINT_BLOCKS", true);
	set_bool_property("PRETTYPRINT_EMPTY_BLOCKS", true);
    }

    /* Since the controlizer does not seem to accept GOTO inside
       sequence from outside but it appears in the code with
       READ/WRITE with I/O exceptions (end=, etc), first remove
       useless blocks. RK */
    clean_up_sequences(st);

    ifdebug(1) {
      statement_consistent_p(st);
    }

    Label_statements = hash_table_make(hash_string, LABEL_TABLES_SIZE);
    Label_control = hash_table_make(hash_string, LABEL_TABLES_SIZE);
    create_statements_of_labels(st);

    result = make_conditional_control(st);
    top = make_control(make_plain_continue_statement(), NIL, NIL);
    bottom = make_control(make_plain_continue_statement(), NIL, NIL);
    Unreachable = NIL;

    ifdebug(1) {
      statement_consistent_p(st);
    }

    /* To track declaration scoping independently of control structure: */
    make_scoping_statement_stack();

    /* FI: structured or not, let's build an unstructured... */
    (void) controlize(st, top, bottom, result, used_labels);

    /* Clean up scoping stack: */
    free_scoping_statement_stack();

    /* The statement st is not consistent anymore here. */
    //statement_consistent_p(st);

    if(!ENDP(Unreachable)) {
	pips_user_warning("Some statements are unreachable\n");
	ifdebug(2) {
	    pips_debug(0, "Unreachable statements:\n");
	    MAP(STATEMENT, s, {
		pips_debug(0, "Statement %p:\n", s);
		print_statement(s);
	    }, Unreachable);
	}
    }
    hash_table_free(Label_statements);
    hash_table_free(Label_control);
    hash_table_free(used_labels);

    u = simplified_unstructured(top, bottom, result);

    ifdebug(5) {
	pips_debug(1,
	  "Nodes in unstructured %p (entry %p, exit %p) from entry:\n",
		   u, unstructured_control(u), unstructured_exit(u));
	display_linked_control_nodes(unstructured_control(u));
	pips_debug(1, "Accessible nodes from exit:\n");
	display_linked_control_nodes(unstructured_exit(u));
    }

    /* Since the controlizer is a sensitive pass, avoid leaking basic
       errors... */
    ifdebug(1) {
      unstructured_consistent_p(u);
    }

    reset_unstructured_number();
    unstructured_reorder(u);

    ifdebug(1) {
	check_control_coherency(unstructured_control(u));
	check_control_coherency(unstructured_exit(u));
	pips_assert("Unstructured should be OK.", unstructured_consistent_p(u));
    }

    return(u);
}

/*
  @}
*/
