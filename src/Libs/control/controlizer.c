/* There are some TODO !!! RK

 */








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
char vcid_control_controlizer[] = "$Id$";
#endif /* lint */

/* \defgroup controlizer Controlizer phase to build the Hierarchical Control Flow Graph

   It computes the Hierarchical Control Flow Graph of a given statement
   according the control hierarchy.

   It is used in PIPS to transform the output of the parsers (the
   PARSED_CODE resource) into HCFG code (the CODE resource).

   For example if there are some "goto" in a program, it will encapsulate
   the unstructured graph into an "unstructured" object covering all the
   gotos and their label targets to localize the messy part. Then this object
   is put into a normal statement so that, seen from above, the code keep a
   good hierarchy.

   In PIPS the RI (Internal Representation or AST) is quite simple so that
   it is easy to deal with. But the counterpart is that some complex
   control structures need to be "unsugared". For example
   switch/case/break/default are transformed into tests, goto and label,
   for(;;) with break or continue are transformed into while() loops with
   goto/label, and so on. It is up to the prettyprinter to recover the
   high level construction if needed (...and possible).

   In the case of C, it is far more complicated than with Fortran77 that
   was targeted by PIPS at the beginning because we need to deal with
   variable scoping according to their block definitions that may not be
   the same hierarchy that the HCFG, for example when you have a goto
   inside a block from outside and when you have local variables in the
   block. In C99 it is even worse since you can have executable
   declarations.

   There are other phases in PIPS that can be used later to operate on the
   CODE to optimize it further.

   Even if there is no fundamental reason that we cannot controlize an
   already controlized code, it is not implemented yet. For example, it
   could be useful to modify some CODE by adding some goto or complex
   control code and call again the controlizer on the CODE to have nice
   unstructured back instead of building correct unstructured statements
   from scratch or using some CODE dump-and-reparse as it is done in some
   PIPS phases (outliner...).

   TODO: a zen-er version of the recursion that avoid passing along the
   successor everywhere since we know at entry that it is the only
   successor of the main control node.

   This is the new version of the controlizer version rewritten by
   Ronan.Keryell@hpc-project.com

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


/* This maps label names to the list of statements where
   they appear (either as definition or reference).

   This is a global table to all the module.

   There are some other local tables used elsewhere in the code to have a
   more hierarchical owning information
*/
static hash_table Label_statements;


/* This maps label names to their (possible forward) control
   nodes.

   That is each statement with a label target is in a control that
   owns this statement.

   Since the code is analyzed from the beginning, we may have some labels
   that map to control node with no statement if their target statement
   has not been encountered yet, in the case of forward gotos for example.
*/
static hash_table Label_control;


/* In C, we can have some "goto" inside a block from outside, that
   translates as any complex control graph into an "unstructured" in the
   PIPS jargon.

   Unfortunately, that means it break the structured block nesting that
   may carry declarations with the scoping information.

   So we need to track this scope information independently of the control
   graph. This is the aim of this declaration scope stack that is used to
   track scoping during visiting the RI.
*/
DEFINE_LOCAL_STACK(scoping_statement, statement)


/* Make a control node around a statement if not already done

   It is like the make_control() constructor except when the statement @p
   st has a label and is already referenced in Label_control. Thus return
   the control node that already owns this statement.

   @param[in] st is the statement we want a control node around

   @return the new (in the case of a statement without a label) or already
   associated (in the case of a statement with a label) control node with
   the statement inside.

   It returns NULL if the statement has a label but it is not associated to
   any control node yet

   FI: the call sites do not seem to check if c==NULL...
 */
static control make_conditional_control(statement st) {
  string label = entity_name(statement_label(st));
  control c = control_undefined;

  if (empty_global_label_p(label))
    /* No label, so there cannot be a control already associated to a
       label */
  c = make_control(st, NIL, NIL);
  else
      /* Get back the control node associated with this statement
	 label. Since we store control object in this hash table, use
	 cast. We rely on the fact that NIL for a list is indeed
	 NULL... */
    c = (control) hash_get_default_empty_list(Label_control, label);

  pips_assert("c==0 || control_consistent_p(c)",
	      c==0 || control_consistent_p(c));

  return c;
}


/* Get the control node associated to a label name

   It looks for the label name into the Label_control table.

   The @p name must be the complete entity name, not a local or a user name.

   @param[in] name is the string name of the label entity

   The label must exist in the table.

   @return the associated control node.
*/
static control get_label_control(string name) {
    control c;

    pips_assert("label is not the empty label", !empty_global_label_p(name)) ;
    c = (control) hash_get(Label_control, name);
    pips_assert("c is defined", c != (control) HASH_UNDEFINED_VALUE);
    pips_assert("c is a control", check_control(c));
    ifdebug(2) {
      check_control_coherency(c);
    }
    return(c);
}


/* Mark a statement as related to a label.

   A @p used_label is a hash_table that maps the label name to the list of
   statements that reference it.

   A statement can appear many times for a given label

   @param used_labels[in,out] is the hash table used to record the statements
   related to a label. It is not updated for empty labels.

   @param name[in] is the label entity name. It may be the empty label.

   @param st[in] is the statement to be recorded as related to the label
*/
static void update_used_labels(hash_table used_labels,
			       string name,
			       statement st) {
  list sts ;

  /* Do something only if there is a label: */
  if (!empty_global_label_p(name)) {
    list new_sts;
    /* Get a previous list of statements related to this label: */
    sts = hash_get_default_empty_list(used_labels, name) ;
    /* Add the given statement to the list */
    new_sts = CONS(STATEMENT, st, sts);
    if (hash_defined_p(used_labels, name))
      /* If there was already something associated to the label, register
	 the new list: */
      hash_update(used_labels, name, new_sts);
    else
      /* Or create a new entry: */
      hash_put(used_labels, name, new_sts);
    /* FI: This debug message happens a lot after inlining? In spite
       of the source parsing, line numbers are not assigned? */
    pips_debug(5, "Reference to statement %d seen\n",
	       (int) statement_number( st )) ;
  }
}


/* Unions 2 hash maps that list statements referencing labels into one.

   @param[in,out] l1 is the hash map used as source and target

   @param[in] l2 is the other hash map

   @returns @p l1 that is the union of @p l1 and @p l2 interpreted as in
   the context of update_used_labels() */
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
   label name to statement list local mapping.

   @param st[in] is the statement we want to check if it owns all allusions to
   the given label name in the @p used_labels mapping

   @param[in] used_labels is a hash table mapping a label name to a list
   of statements that use it (à la Label_statements), as their own label
   or because it is a goto to it

   @return true if all the label allusions in @p st are covered by the @p
   used_labels mapping.
*/
static bool covers_labels_p(statement st,
			    hash_table used_labels) {
  ifdebug(5) {
    pips_debug(5, "Statement %td (%p): \n ", statement_number(st), st);
    print_statement(st);
  }
  /* For all the labels in used_labels: */
  void * cl = NULL;
  list stats = NIL;
  string name = string_undefined;
  while((cl=hash_table_scan(used_labels, cl, (void **) &name, (void **) &stats))) {
      // HASH_MAP(name, sts, {
      /* The statements using a label name in used_labels: */
      //list stats = (list) sts;

      /* For all the statements associated to this label name: */
    list sl = (list) hash_get_default_empty_list(Label_statements, (void *) name);
      FOREACH(STATEMENT, def, sl) {
	/* In one special case, def may have lost its use of a label:
	   if it is a goto to the next statement (see
	   controlize_goto()). Then it is a simple
	   continue. This special configuration never happens, except
	   in code generated by the inlining pass... or written by a
	   weird programmer. It seems easier to keep Label_statements
	   consistent rather than fix the problem here. */
	bool found = false;
	/* Verify that def is in all the statements associated globally to
	   the label name according to module-level Label_statements. */
	FOREACH(STATEMENT, st, stats) {
	  found |= st == def;
	}

	if (!found) {
	  pips_debug(5, "does not cover label %s\n", (char *) name);
	  /* Not useful to go on: */
	  return(false);
	}
      }
    }
    //}, used_labels);

  ifdebug(5)
    fprintf(stderr, "covers its label usage\n");

  return(true);
}


#if 0
/* Register globally a relation between a label and a statement.

   It marks the statement as related to a label at the global module level
   into global Label_statements table and create the control node to hold
   the statement.

   @param[in] name is the label of the statement

   @param[in] st

  byPut the reference in the statement ST to the label NAME
   int the Label_statements table and allocate a slot in the Label_control
   table. */
static void init_label(string name, statement st) {
  /* Do something only if there is a label: */
  if(!empty_global_label_p(name)) {
    /* Get the list of statements related to this label: */
    list used = (list) hash_get_default_empty_list(Label_statements, name);
    /* Append the given statement to the list of statements pointing to
       this label: */
    list sts = CONS(STATEMENT, st, used);
    /* Update or create the entry for this label according a previous
       existence or not: */
    if (hash_defined_p(Label_statements, name))
      hash_update(Label_statements, name, (char *) sts);
    else
      hash_put(Label_statements, name, (char *) sts);

    /* */
    if (! hash_defined_p(Label_control, name)) {
      statement new_st = make_continue_statement(statement_label(st)) ;
      control c = make_control( new_st, NIL, NIL);
      pips_debug(8, "control %p allocated for label \"%s\"", c, name);
      hash_put(Label_control, name, (char *)c);
    }
  }
}
#endif


/* Update the global module-level Label_statements table according to the
   labels a statement references.

   It also creates a control node for this statement if it has a label and
   register it to the module-global Label_control table.

   A statement can reference a label if it is its own label or if the
   statement is indeed a goto to this label.

   Label found in Fortran do-loop to set loop boundary or in Fortran IO
   using some FORMAT through its label are not considered here since it is
   not related to control flow.

   @param[in] st is the statement to look at for labels
*/
static void create_statements_of_label(statement st) {
  instruction i;

  /* Add the statement to its own label reference, if needed: */
  string name = entity_name(statement_label(st));
  if (!empty_global_label_p(name)) {
    /* Associate globally the statement to its own label: */
    update_used_labels(Label_statements, name, st);
    /* Create the control node with the statement inside: */
    control c = make_control(st, NIL, NIL);
    pips_debug(8, "control %p allocated for label \"%s\"", c, name);
    hash_put(Label_control, name, (char *)c);
  }

  switch(instruction_tag(i = statement_instruction(st))) {
    /* If the statement is a goto, add to the target label a reference to
       this statement: */
  case is_instruction_goto: {
    string where = entity_name(statement_label(instruction_goto(i)));
    /* Associate the statement to the target label: */
    update_used_labels(Label_statements, where, st);
    break;
  }

  case is_instruction_unstructured:
    pips_internal_error("Found unstructured", "");

  default:
    /* Do nothing special for other kind of instructions */
    ;
  }
}


/* Initialize the global Label_statements mapping for the module that
   associates for any label in the module the statements that refer to it.

   @param[in] st is the module (top) statement
*/
static void create_statements_of_labels(statement st) {
  /* Apply create_statements_of_label() on all the module statements */
  gen_recurse(st,
	      statement_domain,
	      gen_true,
	      create_statements_of_label);
}


/* @defgroup do_loop_desugaring Desugaring functions used to transform
   non well structured à la Fortran do-loop into an equivalent code with
   tests and gotos.

   @{
*/
/* LOOP_HEADER, LOOP_TEST and LOOP_INC build the desugaring phases of a
   do-loop L for the loop header (i=1), the test (i<10) and the increment
   (i=i+1). */

/* Make an index initialization header for an unsugared à la Fortran
   do-loop "do i = l,u,s"

   @param[in] sl is the do-loop statement

   @return an assignment statement "i = l"
 */
statement unsugared_loop_header(statement sl)
{
  loop l = statement_loop(sl);
  /* Build a reference expression to the loop index: */
  expression i = entity_to_expression(loop_index(l));
  /* Assign the lower range to it: */
  statement hs = make_assign_statement(i, copy_expression(range_lower(loop_range(l))));

  return hs;
}

statement unsugared_forloop_header(statement sl)
{
  forloop l = statement_forloop(sl);
  expression ie = copy_expression(forloop_initialization(l));
  instruction ii = make_instruction_expression(ie);
  statement is = instruction_to_statement(ii);

  return is;
}

statement unsugared_whileloop_header(statement sl __attribute__ ((__unused__)))
{
  statement hs = make_plain_continue_statement();

  return hs;
}


/* Do a crude test of end of do-loop for do-loop unsugaring.

   TODO : indeed the code is wrong since it only works for loops without
   side effects on the index inside the loop and if the stride is
   positive, and the upper bound greated than the lower bound

   @param[in] sl is a statement loop of the form "do i = l, u, s"

   @return a test statement "if (i < u)" with empty then and else branches.
*/
statement unsugared_loop_test(statement sl)
{
  loop l = statement_loop(sl);
  /* Build i < u */
  expression c = MakeBinaryCall(entity_intrinsic(GREATER_THAN_OPERATOR_NAME),
			  entity_to_expression(loop_index(l)),
			  copy_expression(range_upper(loop_range(l))));
  /* Build if (i < u) with empty branches: */
  test t = make_test(c,
		     make_plain_continue_statement(),
		     make_plain_continue_statement());

  statement ts = instruction_to_statement(make_instruction_test(t));
  return ts;
}

statement unsugared_forloop_test(statement sl)
{
  forloop l = statement_forloop(sl);
  expression c = copy_expression(forloop_condition(l));
  /* Build if (c) with empty branches: */
  test t = make_test(c,
		     make_plain_continue_statement(),
		     make_plain_continue_statement());

  statement ts = instruction_to_statement(make_instruction_test(t));
  return ts;
}

statement unsugared_whileloop_test(statement sl)
{
  whileloop wl = statement_whileloop(sl);
  expression c = copy_expression(whileloop_condition(wl));
  test t = make_test(c,
		     make_plain_continue_statement(),
		     make_plain_continue_statement());

  statement ts = instruction_to_statement(make_instruction_test(t));
  return ts;
}


/* Do an index increment instruction for do-loop unsugaring.

   TODO : indeed the code is wrong since it only works for loops without
   side effects on the index inside the loop

   @param[in] sl is a statement loop of the form "do i = l, u, s"

   @return a "i = i + s" statement test "if (i < u)" with empty then and
   else branches.
*/
statement unsugared_loop_inc(statement sl)
{
  loop l = statement_loop(sl);
  /* Build "i + s" */
  expression c = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			  // Even for C code? To verify for pointers...
			  entity_to_expression(loop_index(l)),
			  copy_expression(range_increment(loop_range(l))));
  /* Build "i = i +s" */
  statement s = make_assign_statement(entity_to_expression(loop_index(l)),
				      c);

  return s;
}

statement unsugared_forloop_inc(statement sl)
{
  forloop l = statement_forloop(sl);
  expression inc = copy_expression(forloop_increment(l));
  instruction i = make_instruction_expression(inc);
  statement s = instruction_to_statement(i);

  return s;
}

/* @} */

/* Computes the control graph of a Fortran do-loop statement

   @param[in,out] c_res is the entry control node with the do-loop
   statement to controlize. If the do-loop has complex control code such
   as some goto to outside, it is transformed into an equivalent control
   graph.

   @param[in,out] succ must be the control node successor of @p c_res that
   will be the current end of the control node sequence and an exit node

   @param[in,out] used_labels is a hash table mapping a label name to a
   list of statements that use it, as their label or because it is a goto
   to it

   @return true if the code is not a structured control.
*/
static bool controlize_loop(control c_res,
			    control succ,
			    hash_table used_labels)
{
  /* To track the statement related to labels inside the loop body: */
  hash_table loop_used_labels = hash_table_make(hash_string, 0);
  statement sl = control_statement(c_res);

  pips_debug(5, "(st = %p, c_res = %p, succ = %p)\n", sl, c_res, succ);

  loop l = statement_loop(sl);
  statement body_s = loop_body(l);

  /* Remove the loop body from the loop just in case we want to
     prettyprint our work in progress: */
  //loop_body(l) = statement_undefined;
  loop_body(l) = make_plain_continue_statement();
  /* Create a control node to host the loop body and insert it in the
     control graph: */
  control c_body = make_conditional_control(body_s);
  insert_control_in_arc(c_body, c_res, succ);
  /* We also insert a dummy node between the body and the exit that will
     be used for the incrementation because if the control body has goto
     to succ node, we will have trouble to insert it later: */
  control c_inc = make_control(make_plain_continue_statement(), NIL, NIL);
  insert_control_in_arc(c_inc, c_body, succ);
  /* TODO
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
  */
  /* Recurse by controlizing inside the loop: */
  // FI: this seems redundant with the above call to insert_control_in_arc()
  //link_2_control_nodes(c_body, c_inc);
  bool controlized = controlize_statement(c_body, c_inc, loop_used_labels);

  if (!controlized) {
    /* First the easy way. We have a kindly control-localized loop body,
       revert to the original code */
    pips_debug(6, "Since we can keep the do-loop, remove the useless control node %p that was allocated for the loop_body.\n", c_body);
    control_statement(c_body) = statement_undefined;
    /* Remove the control node from the control graph by carefully
       relinking around: */
    remove_a_control_from_an_unstructured(c_body);
    /* Remove also the dummy increment node that has not been used
       either: */
    remove_a_control_from_an_unstructured(c_inc);
    /* Move the loop body into its own loop: */
    loop_body(l) = body_s;
  }
  else {
    /* We are in trouble since the loop body is not locally structured,
       there are goto from inside or outside the loop body. So we
       replace the do-loop with a desugared version with an equivalent
       control graph. */
    /* Update the increment control node with the real computation: */
    /* First remove the dummy statement added above: */
    free_statement(control_statement(c_inc));
    /* And put "i = i + s" instead: */
    control_statement(c_inc) = unsugared_loop_inc(sl);
    /* Now build the desugared loop: */
    /* We can replace the former loop statement by the new header. That
       means that all pragma, comment, extensions, label on the previous
       loop stay on this. */
    control_statement(c_res) = unsugared_loop_header(sl);
    /* Add the continuation test between the header and the body that are
       already connected: */
    control c_test = make_control(unsugared_loop_test(sl), NIL, NIL);
    insert_control_in_arc(c_test, c_res, c_body);
    /* Detach the increment node from the loop exit */
    unlink_2_control_nodes(c_inc, succ);
    /* And reconnect it to the test node to make the loop: */
    link_2_control_nodes(c_inc, c_test);
    /* Add the else branch of the test toward the loop exit: */
    // FI: try to get the true and false successors at the right location...
    // link_2_control_nodes(c_test, succ);
    //control_successors(c_test) = gen_nconc(control_successors(c_test),
    //				   CONS(CONTROL, succ, NIL));
    //control_successors(c_test) = gen_nreverse(control_successors(c_test));
    control_successors(c_test) = CONS(CONTROL, succ, control_successors(c_test));
    control_predecessors(succ) = gen_nconc(control_predecessors(succ),
					   CONS(CONTROL, c_test, NIL));
    /* Detach the succ node from the body node */
    //unlink_2_control_nodes(c_body, succ);
    /* We can remove  */
  }

  /* Keep track of labels that were used by the statements of the loop: */
  union_used_labels( used_labels, loop_used_labels);
  hash_table_free(loop_used_labels);

  pips_debug(5, "Exiting\n");

  return controlized;
}

/* Computes the control graph of a C  for loop statement

   @param[in,out] c_res is the entry control node with the for loop
   statement to controlize. If the for loop has complex control code such
   as some goto to outside, it is transformed into an equivalent control
   graph.

   @param[in,out] succ must be the control node successor of @p c_res that
   will be the current end of the control node sequence and an exit node

   @param[in,out] used_labels is a hash table mapping a label name to a
   list of statements that use it, as their label or because it is a goto
   to it

   @return true if the code is not a structured control.
*/
static bool controlize_forloop(control c_res,
			       control succ,
			       hash_table used_labels)
{
  /* To track the statement related to labels inside the loop body: */
  hash_table loop_used_labels = hash_table_make(hash_string, 0);
  statement sl = control_statement(c_res);

  pips_debug(5, "(st = %p, c_res = %p, succ = %p)\n", sl, c_res, succ);

  forloop l = statement_forloop(sl);
  statement body_s = forloop_body(l);

  /* Remove the loop body from the loop just in case we want to
     prettyprint our work in progress: */
  //loop_body(l) = statement_undefined;
  forloop_body(l) = make_plain_continue_statement();
  /* Create a control node to host the loop body and insert it in the
     control graph: */
  control c_body = make_conditional_control(body_s);
  insert_control_in_arc(c_body, c_res, succ);
  /* We also insert a dummy node between the body and the exit that will
     be used for the incrementation because if the control body has goto
     to succ node, we will have trouble to insert it later: */
  control c_inc = make_control(make_plain_continue_statement(), NIL, NIL);
  insert_control_in_arc(c_inc, c_body, succ);
  /* TODO
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
  */
  /* Recurse by controlizing inside the loop: */
  // link_2_control_nodes(c_body, c_inc); already done by insert_control_in_arc
  bool controlized = controlize_statement(c_body, c_inc, loop_used_labels);

  if (!controlized) {
    /* First the easy way. We have a kindly control-localized loop body,
       revert to the original code */
    pips_debug(6, "Since we can keep the do-loop, remove the useless control node %p that was allocated for the loop_body.\n", c_body);
    control_statement(c_body) = statement_undefined;
    /* Remove the control node from the control graph by carefully
       relinking around: */
    remove_a_control_from_an_unstructured(c_body);
    /* Remove also the dummy increment node that has not been used
       either: */
    remove_a_control_from_an_unstructured(c_inc);
    /* Move the loop body into its own loop: */
    forloop_body(l) = body_s;
  }
  else {
    /* We are in trouble since the loop body is not locally structured,
       there are goto from inside or outside the loop body. So we
       replace the do-loop with a desugared version with an equivalent
       control graph. */
    /* Update the increment control node with the real computation: */
    /* First remove the dummy statement added above: */
    free_statement(control_statement(c_inc));
    /* And put "i = i + s" instead: */
    control_statement(c_inc) = unsugared_forloop_inc(sl);
    /* Now build the desugared loop: */
    /* We can replace the former loop statement by the new header. That
       means that all pragma, comment, extensions, label on the previous
       loop stay on this. */
    control_statement(c_res) = unsugared_forloop_header(sl);
    /* Add the continuation test between the header and the body that are
       already connected: */
    control c_test = make_control(unsugared_forloop_test(sl), NIL, NIL);
    insert_control_in_arc(c_test, c_res, c_body);
    /* Detach the increment node from the loop exit */
    unlink_2_control_nodes(c_inc, succ);
    /* And reconnect it to the test node to make the loop: */
    link_2_control_nodes(c_inc, c_test);
    /* Add the else branch of the test toward the loop exit: */
    // link_2_control_nodes(c_test, succ) does not support distinction
    // between true and false branch as first and second successors
    // FI: I hesitated to define a new loe level procedure in ri-util/control.c
    control_successors(c_test) = gen_nconc(control_successors(c_test),
					   CONS(CONTROL, succ, NIL));
    control_predecessors(succ) = gen_nconc(control_predecessors(succ),
					   CONS(CONTROL, c_test, NIL));
    /* Detach the succ node from the body node */
    //unlink_2_control_nodes(c_body, succ);
    /* We can remove  */
  }

  /* Keep track of labels that were used by the statements of the loop: */
  union_used_labels( used_labels, loop_used_labels);
  hash_table_free(loop_used_labels);

  pips_debug(5, "Exiting\n");

  return controlized;
}

/* Computes the control graph of a Fortran or C  while loop statement

   @param[in,out] c_res is the entry control node with the do-loop
   statement to controlize. If the do-loop has complex control code such
   as some goto to outside, it is transformed into an equivalent control
   graph.

   @param[in,out] succ must be the control node successor of @p c_res that
   will be the current end of the control node sequence and an exit node

   @param[in,out] used_labels is a hash table mapping a label name to a
   list of statements that use it, as their label or because it is a goto
   to it

   @return true if the code is not a structured control.
*/
static bool controlize_whileloop(control c_res,
				 control succ,
				 hash_table used_labels)
{
  /* To track the statement related to labels inside the loop body: */
  hash_table loop_used_labels = hash_table_make(hash_string, 0);
  statement sl = control_statement(c_res);

  pips_debug(5, "(st = %p, c_res = %p, succ = %p)\n", sl, c_res, succ);

  whileloop wl = statement_whileloop(sl);
  statement body_s = whileloop_body(wl);

  /* Remove the loop body from the loop just in case we want to
     prettyprint our work in progress: */
  // incompatible with debugging code
  //whileloop_body(wl) = statement_undefined;
  whileloop_body(wl) = make_plain_continue_statement();

  /* Create a control node to host the loop body and insert it in the
     control graph: */
  control c_body = make_conditional_control(body_s);
  // FI: if c_test were already available, it should be used instead
  // of succ
  // insert_control_in_arc(c_body, c_res, succ);

  // FI: this should be language neutral. The prettyprinter is
  // supposed to fix comments according to language rules...
  /* TODO
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
  */

  control c_test = make_control(unsugared_whileloop_test(sl), NIL, NIL);
  /* Recurse by controlizing inside the loop: */
  link_2_control_nodes(c_body, c_test);
  bool controlized = controlize_statement(c_body, c_test, loop_used_labels);

  if (!controlized) {
    /* First the easy way. We have a kindly control-localized loop body,
       revert to the original code */
    pips_debug(6, "Since we can keep the  whileloop, remove the useless control node %p that was allocated for the loop_body.\n", c_body);
    control_statement(c_body) = statement_undefined;
    /* Remove the control node from the control graph by carefully
       relinking around: */
    remove_a_control_from_an_unstructured(c_body);
    /* Remove also the dummy increment node that has not been used
       either: */
    //remove_a_control_from_an_unstructured(c_inc);
    /* Move the loop body into its own loop: */
    whileloop_body(wl) = body_s;
  }
  else {
    /* We are in trouble since the loop body is not locally structured,
       there are goto from inside or outside the loop body. So we
       replace the  while loop with a desugared version with an equivalent
       control graph. */
    /* Update the increment control node with the real computation: */
    /* First remove the dummy statement added above: */
    //free_statement(control_statement(c_inc));
    /* And put "i = i + s" instead: */
    //control_statement(c_inc) = unsugared_loop_inc(sl);
    /* Now build the desugared loop: */
    /* We can replace the former loop statement by the new header. That
       means that all pragma, comment, extensions, label on the previous
       loop stay on this. */
    //control_statement(c_res) = unsugared_loop_header(sl);
    // FI: c_res is useless and should be identified with c_test
    control_statement(c_res) = unsugared_whileloop_header(sl);
    /* Add the continuation test between the header and the body that are
       already connected: */
    //control c_test = make_control(unsugared_loop_test(sl), NIL, NIL);
    unlink_2_control_nodes(c_res, succ);
    link_2_control_nodes(c_res, c_test);
    //insert_control_in_arc(c_test, c_res, c_body);
    /* Detach succ from the loop body exit */
    //unlink_2_control_nodes(c_body, succ);
    /* And reconnect it to the test node to make the loop: */
    //link_2_control_nodes(c_body, c_test);
    /* Add the else branch of the test toward the loop exit: arc
       ordering matters */
    unlink_2_control_nodes(c_test, c_body);
    //link_2_control_nodes(c_test, succ);
    //link_2_control_nodes(c_test, c_body);
    link_3_control_nodes(c_test, c_body, succ);
    // link_2_control_nodes(c_test, c_res);
    //unlink_2_control_nodes(c_res, succ);

    pips_assert("c_test is a test with two successors",
		gen_length(control_successors(c_test))==2
		&& statement_test_p(control_statement(c_test)));
    pips_assert("c_body may have two successors if it is a test",
		( gen_length(control_successors(c_body))==2
		  && statement_test_p(control_statement(c_body)) )
		||
		( gen_length(control_successors(c_body))==1
		  && !statement_test_p(control_statement(c_body)) )
		);
    pips_assert("c_res should not be a test",
		gen_length(control_successors(c_res))==1
		&& !statement_test_p(control_statement(c_res)) );
  }

  /* Keep track of labels that were used by the statements of the loop: */
  union_used_labels( used_labels, loop_used_labels);
  hash_table_free(loop_used_labels);

  pips_debug(5, "Exiting\n");

  return controlized;
}

/* Computes the control graph of a C repeat until loop statement

   @param[in,out] c_res is the entry control node with the do-loop
   statement to controlize. If the do-loop has complex control code such
   as some goto to outside, it is transformed into an equivalent control
   graph.

   @param[in,out] succ must be the control node successor of @p c_res that
   will be the current end of the control node sequence and an exit node

   @param[in,out] used_labels is a hash table mapping a label name to a
   list of statements that use it, as their label or because it is a goto
   to it

   @return true if the code is not a structured control.
*/
static bool controlize_repeatloop(control c_res,
				 control succ,
				 hash_table used_labels)
{
  /* To track the statement related to labels inside the loop body: */
  hash_table loop_used_labels = hash_table_make(hash_string, 0);
  statement sl = control_statement(c_res);

  pips_debug(5, "(st = %p, c_res = %p, succ = %p)\n", sl, c_res, succ);

  whileloop wl = statement_whileloop(sl);
  statement body_s = whileloop_body(wl);

  /* Remove the loop body from the loop just in case we want to
     prettyprint our work in progress: */
  whileloop_body(wl) = make_plain_continue_statement();
  /* Create a control node to host the loop body and insert it in the
     control graph: */
  control c_body = make_conditional_control(body_s);
  //insert_control_in_arc(c_body, c_res, succ);
  /* We also insert a dummy node between the body and the exit that will
     be used for the incrementation because if the control body has goto
     to succ node, we will have trouble to insert it later: */
  //control c_inc = make_control(make_plain_continue_statement(), NIL, NIL);
  //insert_control_in_arc(c_inc, c_body, succ);
  /* TODO
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
  */
  control c_test = make_control(unsugared_whileloop_test(sl), NIL, NIL);
  /* Recurse by controlizing inside the loop: */
  link_2_control_nodes(c_body, c_test);
  bool controlized = controlize_statement(c_body, c_test, loop_used_labels);

  if (!controlized) {
    /* First the easy way. We have a kindly control-localized loop body,
       revert to the original code */
    pips_debug(6, "Since we can keep the do-loop, remove the useless control node %p that was allocated for the loop_body.\n", c_body);
    control_statement(c_body) = statement_undefined;
    /* Remove the control node from the control graph by carefully
       relinking around: */
    remove_a_control_from_an_unstructured(c_body);
    /* Remove also the dummy increment node that has not been used
       either: */
    //remove_a_control_from_an_unstructured(c_inc);
    /* Move the loop body into its own loop: */
    whileloop_body(wl) = body_s;
  }
  else {
    /* We are in trouble since the loop body is not locally structured,
       there are goto from inside or outside the loop body. So we
       replace the do-loop with a desugared version with an equivalent
       control graph. */
    /* Update the increment control node with the real computation: */
    /* First remove the dummy statement added above: */
    //free_statement(control_statement(c_inc));
    /* And put "i = i + s" instead: */
    //control_statement(c_inc) = unsugared_loop_inc(sl);
    /* Now build the desugared loop: */
    /* We can replace the former loop statement by the new header. That
       means that all pragma, comment, extensions, label on the previous
       loop stay on this. */
    control_statement(c_res) = unsugared_whileloop_header(sl);
    /* Add the continuation test between the header and the body that are
       already connected: */
    //control c_test = make_control(unsugared_loop_test(sl), NIL, NIL);
    // insert_control_in_arc(c_test, c_res, c_body);
    /* Detach the increment node from the loop exit */
    //unlink_2_control_nodes(c_inc, succ);
    /* And reconnect it to the test node to make the loop: */
    //link_2_control_nodes(c_inc, c_test);
    //link_2_control_nodes(c_body, c_test);
    //link_2_control_nodes(c_test, c_res);
    /* Add the else branch of the test toward the loop exit: */
    //link_2_control_nodes(c_test, succ);
    /* We can remove  */
    unlink_2_control_nodes(c_res, succ);
    link_2_control_nodes(c_res, c_body);
    /* Add the else branch of the test toward the loop exit: arc
       ordering matters */
    unlink_2_control_nodes(c_test, c_body);
    //link_2_control_nodes(c_test, succ);
    //link_2_control_nodes(c_test, c_body);
    link_3_control_nodes(c_test, c_body, succ);

    pips_assert("c_test is a test with two successors",
		gen_length(control_successors(c_test))==2
		&& statement_test_p(control_statement(c_test)));
    pips_assert("c_body may have two successors if it is a test",
		( gen_length(control_successors(c_body))==2
		  && statement_test_p(control_statement(c_body)) )
		||
		( gen_length(control_successors(c_body))==1
		  && !statement_test_p(control_statement(c_body)) )
		);
    pips_assert("c_res should not be a test",
		gen_length(control_successors(c_res))==1
		&& !statement_test_p(control_statement(c_res)) );
  }

  /* Keep track of labels that were used by the statements of the loop: */
  union_used_labels( used_labels, loop_used_labels);
  hash_table_free(loop_used_labels);

  pips_debug(5, "Exiting\n");

  return controlized;
}


#if 0
/* Generate a test statement ts for exiting loop sl.
 * There should be no sharing between sl and ts.
 */
statement whileloop_test(statement sl)
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
    string lab = string_undefined;

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

    abort();
    if (get_bool_property("PRETTYPRINT_C_CODE"))
      cs = prev_comm;
    else
      {
	if(entity_empty_label_p(whileloop_label(l))) {
	  cs = strdup(concatenate(prev_comm,
				  "C     DO WHILE loop ",
				  "with GO TO exit had to be desugared\n",
				  NULL));
	}
	else {
	  string lab = label_local_name(whileloop_label(l));
	  cs = strdup(concatenate(prev_comm,
				  "C     DO WHILE loop ",
				  lab,
				  " with GO TO exit had to be desugared\n",
				  NULL));
	}
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

bool old_controlize_whileloop(st, l, pred, succ, c_res, used_labels)
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

    abort();
    controlize_statement(whileloop_body(l), c_res, c_res, c_body, loop_used_labels);

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


bool controlize_forloop(st, l, pred, succ, c_res, used_labels)
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
  abort();
  ifdebug(1)
    statement_consistent_p(st);

  controlize_statement(forloop_body(l), c_test, c_inc, c_body, loop_used_labels);

  ifdebug(1)
    statement_consistent_p(st);

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
#endif

/* Move all the declarations found in a list of control to a given
   statement. Maintain the initializations where they belong. See
   Control/block_scope5n.

   @ctls is a list of control nodes

   It is useful in the controlizer to keep scoping of declarations even
   with unstructured that may destroy the variable scoping rules.

   If there are conflict names on declarations, they are renamed.

   It relies on correct calls to push_declarations()/pop_declarations()
   before to track where to put the declarations.
*/
static void
move_declaration_control_node_declarations_to_statement(list ctls) {
  statement s = scoping_statement_head();
  list declarations = statement_declarations(s);
  statement s_above = scoping_statement_nth(2);
  list nctls = NIL; // build a new list of controls to include the
		    // initialization statements that are derived from
		    // the declaration statements

  pips_debug(2, "Dealing with block statement %p included into block"
	     " statement %p\n", s, s_above);

  if (s_above == NULL)
    /* No block statement above, so it is hard to move something there :-) */
    return;

  list declarations_above  = statement_declarations(s_above);
  list new_declarations = NIL;
  /* The variables created in case of name conflict */
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

      /* Create a new variable with a non conflicting name without
	 inserting a new statement declaration which is likely to be
	 lost because s_above is currently represented as a list of
	 control nodes not as a standard sequence. */
      v = generic_clone_variable_with_unique_name(e,
						  s_above,
						  "",
						  "_",
						  entity_to_module_entity(e),
						  false);
      new_variables = gen_entity_cons(v , new_variables);
      hash_put_or_update(old_to_new_variables, e, v);
      // FI: I do not understand how v can already be in
      // new_declarations since v has just been cloned!
      // if(!gen_in_list_p(v, new_declarations))
      // new_declarations = gen_entity_cons(v , new_declarations);
      // The new variable v has already been added to the declarations
      // of s_above. The code would be easuer to understand if e were
      // added to these declarations in the else clause.
    }
    else {
      v = e;
      /* Add the inner declaration to the upper statement later */
      new_declarations = gen_entity_cons(v , new_declarations);
    }
  }

  /* Remove the inner declaration from the inner statement block:
   */
  gen_free_list(statement_declarations(s));
  statement_declarations(s) = NIL;

  /* Add all the declarations to the statement block above and
      preserve the order: */
  new_declarations = gen_nreverse(new_declarations);
  statement_declarations(s_above) = gen_nconc(declarations_above,
					      new_declarations);
  FOREACH(ENTITY, dv, new_variables) {
    // This does not seem to work because the above statement is being
    // controlized. Hence, the new declaration statements are simply
    // ignored... It might be better to rely on the block declaration
    // list and to fix the declarations a posteriori, maybe at the end
    // of controlize_statement(). The other option is to generate C99
    // code with declarations anywhere in the execution flow
    //add_declaration_statement(s_above, dv);
    ;
  }

  // Even with C99 code, the initializations should not be moved,
  // even when the initial value is numerically known. See
  // Control/block_scope5n. Unless the variable is static. See
  // Control/block_scope13.c
  if(true || get_bool_property("C89_CODE_GENERATION")) {
    /* Replace initializations in declarations by assignment
       statements, when possible; see split_initializations(); do not
       worry about variable renaming yet */
    FOREACH(CONTROL, c, ctls) {
      statement s = control_statement(c);
      nctls = gen_nconc(nctls, CONS(CONTROL, c, NIL));
      // FI: the entity must also be substituted in the
      // initializations contained by the declarations. Also old
      // declarations must be transformed into assignments.
      if(declaration_statement_p(s)) {
	int sn = statement_number(s);
	list icl = NIL;
	/* If old is declared in s, its declaration must be removed
	   and replaced if necessary by an assignment with its
	   initial value... It seems tricky at first if many
	   variables are declared simultaneously but is does not
	   matter if all have to be substituted. Oops for
	   functional declarations... */
	list dvl = statement_declarations(s);
	//list il = NIL; // initialization list
	FOREACH(ENTITY, dv, dvl) {
	  if(!entity_static_variable_p(dv)) {
	    value iv = entity_initial(dv);
	    if(!value_unknown_p(iv)) {
	      expression ie = variable_initial_expression(dv);
	      expression lhs= entity_to_expression(dv);
	      statement is = make_assign_statement(lhs, ie);
	      statement_number(is) = sn;
	      control ic = make_control(is, NIL, NIL);
	      /* FI: it would be better to use a comma expression in
		 order to replace a statement by a unique statement */
	      nctls = gen_nconc(nctls, CONS(CONTROL, ic, NIL));
	      icl = gen_nconc(icl, CONS(CONTROL, ic, NIL));
	      entity n = (entity) hash_get(old_to_new_variables, (void *) dv);
	      n = HASH_UNDEFINED_VALUE==n? dv : n;
	      free_value(entity_initial(n));
	      entity_initial(n) = make_value_unknown();
	    }
	  }
	}
	/* chain icl to c, assume ctls is a list over a control sequence... */
	if(!ENDP(icl)) {
	  pips_assert("c has one successor (but may be zero with"
		      " dead code behind declarations:-(",
		      gen_length(control_successors(c))==1);
	  control succ = CONTROL(CAR(control_successors(c)));
	  control fic = CONTROL(CAR(icl));
	  control cic = fic;
	  control lic = CONTROL(CAR(gen_last(icl)));
	  FOREACH(CONTROL, nc, CDR(icl)) {
	    /* The nodes in icl must be linked together */
	    link_2_control_nodes(cic, nc);
	    cic = nc;
	  }
	  unlink_2_control_nodes(c, succ);
	  link_2_control_nodes(c, fic);
	  link_2_control_nodes(lic, succ);
	  /* They should be added into ctls too... because the
	     initialization expressions may require some
	     renaming... but nctls probably takes care of that. */
	  gen_free_list(icl);
	}
	statement_declarations(s) = NIL;
      }
    }
  }
  else
    nctls = gen_copy_seq(ctls);

  /* Replace all references to old variables to references to the new
     ones in all the corresponding control nodes by in the code */
  void * ccp = NULL; // current couple pointer
  entity old, new;
  while((ccp = hash_table_scan(old_to_new_variables, ccp, (void *) &old, (void *) &new))) {
  //HASH_MAP(old, new, {
      FOREACH(CONTROL, c, nctls) {
	statement s = control_statement(c);
	if(true || !get_bool_property("C89_CODE_GENERATION")) { // C99 assumed
	  if(declaration_statement_p(s)) {
	    list dl = statement_declarations(s);
	    list cl;
	    for(cl=dl; !ENDP(cl); POP(cl)) {
	      entity dv = ENTITY(CAR(cl));
	      if(dv==old)
		ENTITY_(CAR(cl)) = new;
	    }
	  }
	}
	replace_entity(s, old, new);
      }
      /* We should free in some way the old variable... */
      //}, old_to_new_variables);
  }
  hash_table_free(old_to_new_variables);

  return;
}


/* Find the exit node of a sub-CFG defined by the list of nodes ctls
 * and by the first node to execute when finished. The entry node of
 * the sub-CFG is the last node of the ctls list, because it is built
 * backwards.
 *
 * RK: ctls is built backwards: hence its exit node is the first node in
 * the list.
 *
 * FI: in fact, it's a bit more complicated...
 *
 * The underlying purpose of this function only called from
 * controlize_sequence(() is to help separate a sub CFG with only one
 * entry and one exit point from the global graph. The entry node is
 * easy to find as CONTROL(CAR(gen_last(ctls))). The exit node may be
 * anywhere because of the goto statements. The succ node is the first
 * node executed after the CFG defined by ctls. But it may have no
 * predecessor or only some of its predecessors belong to the global
 * graph but not to the local graph. So we are interested in
 * predecessors of succ that are reachable from the entry node. Unless
 * we are lucky because succ has only one predecessor and this
 * predecessor has only one successor, succ. In that case, the
 * predecessor of succ is the exit node.
 *
 * The control graph may be altered with a newly allocated node or
 * not.
 *
 * @param ctls: list of the control nodes belonging to the subgraph to
 * isolate
 *
 * @param: succ is the first control node not in ctls to be executed
 * after the nodes in ctls.
 *
 * @return exit, the existing or newly allocated exit node. The
 * subgraph defined by ctls is updated when a new node is allocated.
 *
 * Two algorithms at least are possible: we can either start from the
 * predecessors of succ and check that a backward control path to the
 * entry of ctls exists, or we can start from entry and look for
 * control paths reaching succ. The second approach is implemented here.
 */
/*static*/ control find_exit_control_node(list ctls, control succ)
{
  control exit = CONTROL(CAR(ctls));
  control entry = CONTROL(CAR(gen_last(ctls)));

  ifdebug(8) {
    FOREACH(CONTROL, c, ctls) {
      check_control_coherency(c);
    }
    check_control_coherency(succ);
  }

  if(!(ENDP(control_successors(exit))
       || (gen_length(control_successors(exit))==1
	   && CONTROL(CAR(control_successors(exit)))==succ))) {
    /* look for a successor of entry that is a predecessor of succ */
    list visited = NIL;
    list to_be_visited = gen_copy_seq(control_successors(entry));
    bool found_p = false;
    exit = control_undefined;

    while(!ENDP(to_be_visited)) {
      FOREACH(CONTROL, c, to_be_visited) {
	if(gen_in_list_p(succ, control_successors(c))) {
	  if(control_undefined_p(exit))
	    exit = make_control(make_plain_continue_statement(), NIL, NIL);

	  insert_control_in_arc(exit, c, succ);
	  found_p = true;
	}
	if(c!=succ) { // Should always be true...
	  visited = CONS(CONTROL, c, visited);
	  gen_remove(&to_be_visited, c);
	  FOREACH(CONTROL, s, control_successors(c)) {
	    if(!gen_in_list_p(s, visited)
	       && !gen_in_list_p(s, to_be_visited)
	       && s!=succ && s!=exit)
	      // update the loop variable... within the two nested loops
	      to_be_visited = CONS(CONTROL, s, to_be_visited);
	  }
	}
	else {
	  pips_internal_error("design error\n.");
	}
      }
    }

    gen_free_list(visited);
    gen_free_list(to_be_visited);

    if(!found_p) {
    /* succ may be unreachable because ctls loops on itself*/
      exit = make_control(make_plain_continue_statement(), NIL, NIL);
    }
  }

  ifdebug(8) {
    FOREACH(CONTROL, c, ctls) {
      check_control_coherency(c);
    }
    check_control_coherency(exit);
  }

return exit;
}

/* FI: remake of function above, incomplete backward approach, now
   obsolete because the forward approach works. But who knows? The
   complexity of the backward approach might be lower than the
   complexity of the forward approach? */
control find_or_create_exit_control_node(list ctls, control succ)
{
  control exit = control_undefined;
  //control entry = CONTROL(CAR(gen_last(ctls)));

  ifdebug(8) {
    FOREACH(CONTROL, c, ctls) {
      check_control_coherency(c);
    }
    check_control_coherency(succ);
  }

  /* Because of the recursive descent in the AST... */
  pips_assert("succ has only one successor or none",
	      gen_length(control_successors(succ))<=1);

  /* Let's try to use an existing node as exit node */
  if(gen_length(control_predecessors(succ))==1) {
    control c = CONTROL(CAR(control_predecessors(succ)));
    if(gen_length(control_successors(c))==1) {
      pips_debug(8, "succ has a fitting predecessor as exit.\n");
      exit = c;
    }
  }

  if(control_undefined_p(exit)) {
    /* A new node is needed as exit node */
    exit = make_control(make_plain_continue_statement(), NIL, NIL);
    if(gen_length(control_predecessors(succ))==0) {
      /* The CFG contains an infinite loop and succ is never reached? */
      /* FI: we might need to check that the predecessor is not still
	 c_res? */
      /* Why would we link it to succ to have it unlinked by the
	 caller? */
      pips_debug(8, "succ is unreachable and a new exit node %p is created.\n",
		 exit);
    }
    else {
      /* succ must be removed from the current CFG. Its predecessors within
	 the current CFG must be moved onto the new exit node */
      FOREACH(CONTROL, c, control_predecessors(succ)) {
	/* Is c in the current CFG or in anoter one at a higher
	   level? Typical case: the current sequence is a test branch
	   and hence succ has two predecessors at the higher level */
	if(1 /*forward_control_path_p(entry, c)*/) {
	  // FI: could we use link and unlink of control nodes?
	  list pcl = list_undefined;
	  for(pcl = control_successors(c); !ENDP(pcl); POP(pcl)) {
	    control cc = CONTROL(CAR(pcl));
	    if(cc==succ)
	      CONTROL_(CAR(pcl))=exit;
	  }
	  control_predecessors(exit) = gen_nconc(control_predecessors(exit),
						 CONS(CONTROL, c, NIL));
	}
      }
      //control_predecessors(exit) = control_predecessors(succ);
      //control_predecessors(succ) = CONS(CONTROL, exit, NIL);
      /* hpftest62b.f: the sequence below pulls an extra statement in
	 the unstructured that is being built */
      /*
      statement fs = control_statement(succ);
      statement es = control_statement(exit);
      control_statement(exit) = fs;
      control_statement(succ) = es;
      */
      pips_debug(8, "succ is reachable thru %d control paths "
		 "but a new exit node %p is created.\n",
		 (int) gen_length(control_predecessors(exit)), exit);
    }
  }

  ifdebug(8) {
    FOREACH(CONTROL, c, ctls) {
      check_control_coherency(c);
    }
    check_control_coherency(exit);
  }

return exit;
}

/* Computes the control graph of a sequence statement

   We try to minimize the number of graphs by looking for graphs with one
   node only and picking the statement in that case.

   @param[in,out] c_res is the entry control node with the sequence statement to
   controlize. It may be at exit the entry control node of a potential
   unstructured

   @param[in,out] succ must be the control node successor of @p c_res that
   will be the current end of the control node sequence and an exit node

   @param[in,out] used_labels is a hash table mapping a label name to a
   list of statements that use it, as their label or because it is a goto
   to it

   @return true if the code is not a structured control, i.e. it has
   to be "controlized", i.e. transformed into an "unstructured".

   Also, side effect on hash table Label_statements if ever a goto is
   replaced by a continue because it points to the very next
   statement, via an indirect call to controlize_goto().
*/
static bool controlize_sequence(control c_res,
				control succ,
				hash_table used_labels) {
  statement st = control_statement(c_res);
  /* To see if the control graph will stay local or not: */
  hash_table block_used_labels = hash_table_make(hash_string, 0);
  bool controlized = false;

  pips_assert("st it a statement sequence", statement_sequence_p(st));

  ifdebug(5) {
    pips_debug(5, "Entering with nodes linked with c_res %p:\n", c_res);
    display_linked_control_nodes(c_res);
  }
  ifdebug(1) {
    check_control_coherency(c_res);
    check_control_coherency(succ);
  }

  scoping_statement_push(st);

  /* A C block may have a label and even goto from outside on it. */
  /* To track variable scoping, track this statement */
  // TODO scoping_statement_push(st);

  /* Get the list of statements in the block statement: */
  list sts = statement_block(st);
  /* The list of all the control nodes associated to this statement
     list: */
  list ctls = NIL;

  /* To track where to insert the current control node of a sequence
     element: */
  control pred = c_res;
  /* We first transform a structured block of statement in a thread of
     control node with all the statements that we insert from c_res up to
     succ: */
  bool must_be_controlized_p = false;
  //bool previous_control_preexisting_p = false;

  FOREACH(STATEMENT, s, sts) {
    /* Create a new control node for this statement, or retrieve one
       if it as a label: */
    control c = make_conditional_control(s);
    //bool control_preexisting_p = false;

    // FI: I'm not sure this is enough to detect that
    // make_conditional_control() has not made a new control but
    // retrieved a control by its label...
    //if(!ENDP(control_successors(c)) || !ENDP(control_predecessors(c)) ) {
    // FI: too bad if the label is unused... or used locally... or is
    // useless as happens after inlining for FREIA: "goto l1;l1: ;"
    if(!unlabelled_statement_p(s)) {
      pips_debug(8, "This control %p pre-existed. "
		 "The sequence cannot be controlized.\n", c);
      must_be_controlized_p = true;
      //control_preexisting_p = true;
    }

    // "insert_control_in_arc(c, pred, succ);" with additional checks
    unlink_2_control_nodes(pred,succ); // whether they are linked or not
    if(ENDP(control_successors(pred)))
      link_2_control_nodes(pred, c);
    if(ENDP(control_successors(c)))
      link_2_control_nodes(c, succ);


    /* Keep track of the control node associated to this statement. Note
       that the list is built in reverse order: */
    ctls = CONS(CONTROL, c, ctls);
    /* The next control node will be inserted after the new created node: */
    pred = c;
    //previous_control_preexisting_p = control_preexisting_p;
  }

  // FI: check that this is a neat sequence if it does not must be controlized
  if(!must_be_controlized_p) {
    FOREACH(CONTROL, c, ctls) {
      pips_assert("c may have only one successor even if it is a test "
		  "a this point",
		  ( gen_length(control_successors(c))==1
		    && statement_test_p(control_statement(c)) )
		  ||
		  ( gen_length(control_successors(c))==1
		    && !statement_test_p(control_statement(c)) )
		  );
    }
  }

  /* Now do the real controlizer job on the previously inserted thread of
     control nodes. */
  /* Note that we iterate in the reverse order of the
     statements since the control list was built up in reverse
     order. Indeed doing this in reverse order is simpler to write with
     this "next" stuff because we already have the current element and the
     successor "succ". */
  control next = succ;
  pips_debug(5, "Controlize each statement sequence node in reverse order:\n");
  FOREACH(CONTROL, c, ctls) {
    /* Recurse on each statement: controlized has been initialized to
       false */
    controlized |= controlize_statement(c, next, block_used_labels);
    /* The currently processed element will be the successor of the one to
       be processed: */
    next = c;
  }

  if (!controlized && !must_be_controlized_p) {

    // FI: check that this is a neat sequence
    FOREACH(CONTROL, c, ctls) {
      if(controlized) // unstructured case: impossible
	pips_assert("c may have two successors only if it is a test",
		    ( gen_length(control_successors(c))==2
		      && statement_test_p(control_statement(c)) )
		    ||
		    ( gen_length(control_successors(c))<=1
		      && !statement_test_p(control_statement(c)) )
		    );
      else // the sequence is structured: always
	pips_assert("c may have two successors only if it is a test",
		    ( gen_length(control_successors(c))==2
		      && statement_test_p(control_statement(c)) )
		    ||
		    ( gen_length(control_successors(c))==1
		      // FI: the test may not be unstructured; a
		      // structured test has only one successor
		      /* && !statement_test_p(control_statement(c))*/ )
		    );
    }
    /* Each control node of the sequence is indeed well structured, that
       each control node is without any goto from or to outside itself. So
       we can keep the original sequence back! */
    pips_debug(5, "Keep a statement sequence and thus remove"
	       " previously allocated control nodes for the sequence.\n");
    /* Easy, just remove all the control nodes of the sequence and relink
       around the control graph: */
    FOREACH(CONTROL, c, ctls) {
      statement s = control_statement(c);
      int nsucc = (int) gen_length(control_successors(c));
      /* Do not forget to detach the statement of its control node since
	 we do not want the statement to be freed at the same time: */
      pips_debug(6, "Removing useless control node %p.\n", c);
      control_statement(c) = statement_undefined;

      if(statement_test_p(s)) {
	// FI: this had not been planned by Ronan
	if(nsucc==1) {
	  // FI: might this happen when a test is found out well-structured?
	  remove_a_control_from_an_unstructured(c);
	}
	else {
	  pips_assert("a test has two successors\n", nsucc==2);
	  remove_a_control_from_an_unstructured_without_relinking(c);
	}
      }
      else {
	if(nsucc<=1) {
	  pips_assert("a non test has one successor at most\n", nsucc<=1);
	  /* Remove the control node from the control graph by carefully
	     relinking around: */
	  remove_a_control_from_an_unstructured(c);
	}
	else {
	  pips_debug(1, "Abnormal control: not a test, two successors.\n");
	  remove_a_control_from_an_unstructured_without_relinking(c);
	}
      }
    }
    // You may have to fix C89 declarations if some unstructured has
    // been created below in the recursion
    if(get_bool_property("C89_CODE_GENERATION")) {
      fix_block_statement_declarations(st);
    }
  }
  else {
    /* So, there is some unstructured stuff. We can consider 2 cases to
       simplify the generated code:

       - If the unstructured control code is local to the sequence
         statements, we can keep some locality by encapsulated this mess
         into an unstructured so that from outside this unstructured the
         code is view as structured (the H in HCFG!).

       - If not, there are some goto to or from control nodes outside of
         the sequence statement and then we cannot do anything and return
         the control graph marked as with control side effect.
    */
    bool covers_p = covers_labels_p(st, block_used_labels);
    /* Remove the sequence list but not the statements themselves since
       each one has been moved into a control node: */
    gen_free_list(sequence_statements(statement_sequence(st)));
    sequence_statements(statement_sequence(st)) = NIL;

    // FI: this fails because st has been gutted out... covers_p must
    // be computed earlier
    if (covers_p) {
      /* There are no goto from/to the statements of the statement list,
	 so we can encapsulate all this local control graph in an
	 "unstructured" statement: */
      /* Get the local exit node that is the head of the control list
	 since it was built in reverse order. Note that ctls cannot be
	 empty here because it an empty statement sequence should be
	 structured by definition and caught by the previous test. */
      /* FI: when the last statement is controlized, there is no
	 guarantee any longer that the control corrresponding to the
	 last statement of the sequence is the exit node. Also, an
	 exit node should have no successor by definition. To avoid
	 the issue, we could systematically add a nop statement to the
	 sequence. Or we could check that the last control node has no
	 successors, else look for the node connected to succ (might
	 be dangerous because succ may not really be part of the code)
	 and, if no such node is found because the sequence loops
	 forever, create a new unreachable control node with a NOP
	 statement. */
      // control exit = CONTROL(CAR(ctls));
      control exit = find_exit_control_node(ctls, succ);
      //control exit = find_or_create_exit_control_node(ctls, succ);
      control entry = CONTROL(CAR(gen_last(ctls)));
      pips_debug(5, "Create a local unstructured with entry node %p and exit node %p\n", entry, exit);

      /* First detach the local graph: */
      unlink_2_control_nodes(c_res, entry);
      /* FI: Not such a good idea to return succ as exit control node... */
      unlink_2_control_nodes(exit, succ);
      /* Reconnect around the unstructured: */
      link_2_control_nodes(c_res, succ);
      /* And create the local "unstructured" statement to take this local
	 graph: */
      unstructured u = make_unstructured(entry, exit);
      ifdebug(1) {
	check_control_coherency(entry);
	check_control_coherency(exit);

	/* Make sure that the new unstructured u is not linked to code
	   sitting above */
	list linked_nodes = NIL;
	control_map_get_blocs(entry, &linked_nodes);
	if(gen_in_list_p(c_res, linked_nodes)) {
	  list cp = NIL;
	  list vp = NIL;
	  find_a_control_path(entry, c_res, &cp, &vp, 0);
	  print_control_nodes(cp);
	  pips_internal_error("Some issue due to \"c_res\" with covers_p\n");
	}
	else if(gen_in_list_p(succ, linked_nodes)) {
	  list cp = NIL;
	  list vp = NIL;
	  find_a_control_path(entry, c_res, &cp, &vp, 0);
	  print_control_nodes(cp);
	  pips_internal_error("Some issue due to \"succ\" with covers_p\n");
	}
      }

      statement u_s =
	instruction_to_statement(make_instruction_unstructured(u));
      /* We keep the old block statement since it may old extensions,
	 declarations... If useless, it should be removed later by another
	 phase. So, move the unstructured statement as the only statement
	 of the block: */
      sequence_statements(instruction_sequence(statement_instruction(st))) =
	CONS(STATEMENT, u_s, NIL);
      // You may have to fix C89 declarations if some unstructured has
      // been created below in the recursion
      //if(get_bool_property("C89_CODE_GENERATION")) {
      fix_block_statement_declarations(st);
      //}
      /* From outside of this block statement, everything is hierarchized,
	 so we claim it: */
      controlized = false;
    }
    else {
      /* There are some goto from or to external control nodes, so we
	 cannot localize stuff. */
      pips_debug(5, "There are goto to/from outside this statement list"
		 " so keep control nodes without any hierarchy here.\n");
      /* Keep the empty block statement for extensions and declarations: */
      // FI; maybe for extensions, but certainly not for declarations;
      // similar code to flatten_code must be used to rename the local
      // variables and to move them up the AST; see sequence04.c for instance
      sequence_statements(instruction_sequence(statement_instruction(st))) =
	NIL;
      // TODO move declarations up, keep extensions & here
      move_declaration_control_node_declarations_to_statement(ctls);
      controlized = true;
    }
  }
  /* Integrate the statements related to the labels inside its statement
     to the current statement itself: */
  union_used_labels(used_labels, block_used_labels);
  /* Remove local label association hash map: */
  hash_table_free(block_used_labels);

#if 0
//  TODO
    if (!hierarchized_labels) {
    /* We are in trouble since we will have an unstructured with goto
       from or to outside this statement sequence, but the statement
       sequence that define the scoping rules is going to disappear...
       So we gather all the declaration and push them up: */
    move_declaration_control_node_declarations_to_statement(ctls);
  }
#endif
    ///* Revert to the variable scope of the outer block statement: */
  // TODO scoping_statement_pop();
  scoping_statement_pop();
  statement_consistent_p(st);

  return controlized;
}


/* Builds the control node of a test statement

   @param[in,out] c_res is the control node with the test statement
   (if/then/else)

   @param[in,out] succ is the control node successor of @p c_res

   @param[in,out] used_labels is a hash table mapping a label name to a list of
   statements that use it, as their label or because it is a goto to it

   @return true if the control node generated is not structured.
*/
static bool controlize_test(control c_res,
			    control succ,
			    hash_table used_labels)
{
  bool controlized;
  statement st = control_statement(c_res);
  test t = statement_test(st);
  /* The statements of each branch: */
  statement s_t = test_true(t);
  statement s_f = test_false(t);
  /* Create a control node for each branch of the test: */
  control c_then = make_conditional_control(s_t);
  control c_else = make_conditional_control(s_f);

  pips_debug(5, "Entering (st = %p, c_res = %p, succ = %p)\n",
	     st, c_res, succ);

  ifdebug(5) {
    pips_debug(1, "THEN at entry:\n");
    print_statement(s_t);
    pips_debug(1, "c_then at entry:\n");
    print_statement(control_statement(c_then));
    pips_debug(1, "ELSE at entry:\n");
    print_statement(s_f);
    pips_debug(1, "c_else at entry:\n");
    print_statement(control_statement(c_else));
    check_control_coherency(succ);
    check_control_coherency(c_res);
  }

  /* Use 2 hash table to figure out the label references in each branch to
     know if we will able to restructure them later: */
  hash_table t_used_labels = hash_table_make(hash_string, 0);
  hash_table f_used_labels = hash_table_make(hash_string, 0);

  /* Insert the control nodes for the branch into the current control
     sequence: */
  /* First disconnect the sequence: */
  unlink_2_control_nodes(c_res, succ);

  /* Then insert the 2 nodes for each branch, in the correct order since
     the "then" branch is the first successor of the test and the "else"
     branch is the second one: */
  // Correct order: link_2_control_nodes add the new arc in the
  // first slot; so reverse linking of c_else and c_then
  link_3_control_nodes(c_res, c_then, c_else);
  link_2_control_nodes(c_else, succ);
  //link_2_control_nodes(c_res, c_then);
  link_2_control_nodes(c_then, succ);

  /* Now we can controlize each branch statement to deal with some control
     flow fun: */
  controlize_statement(c_then, succ, t_used_labels);
  controlize_statement(c_else, succ, f_used_labels);

  /* If all the label jumped to from the THEN/ELSE statements are in their
     respective statement, we can replace the unstructured test by a
     structured one back: */
  /* FI: this test returns a wrong result for if02.c. The reason
     might be again controlize_goto() or a consequence of the two
     calls to controlize_statement() above. */
  if(covers_labels_p(s_t, t_used_labels)
     && covers_labels_p(s_f, f_used_labels)) {
    pips_debug(5, "Restructure the IF at control %p\n", c_res);
    test_true(t) = control_statement(c_then);
    test_false(t) = control_statement(c_else);
    /* Remove the old unstructured control graph: */
    control_statement(c_then) = statement_undefined;
    control_statement(c_else) = statement_undefined;
    /* c_then & c_else are no longer useful: */
    remove_a_control_from_an_unstructured_without_relinking(c_then);
    remove_a_control_from_an_unstructured_without_relinking(c_else);
    // You do not want to relink too much, but you should relink a minimum
    link_2_control_nodes(c_res, succ);

    /* The test statement is a structured test: */
    controlized = false;
  }
  else {
    pips_debug(5, "Destructure the IF at control %p\n", c_res);
    /* Keep the unstructured test. Normalize the unstructured test where
       in the control node of the test, the branch statements must be
       empty since the real code is in the 2 control node successors: */
    test_true(t) = make_plain_continue_statement();
    test_false(t) = make_plain_continue_statement();
    /* Warn we have an unstructured test: */
    controlized = true;
  }

  /* Update the used labels hash map from the 2 test branches */
  union_used_labels(used_labels,
		    union_used_labels(t_used_labels, f_used_labels));

  /* The local hash tables are no longer useful: */
  hash_table_free(t_used_labels);
  hash_table_free(f_used_labels);

  ifdebug(5) {
    pips_debug(1, "IF at exit:\n");
    print_statement(st);
    display_linked_control_nodes(c_res);
    check_control_coherency(succ);
    check_control_coherency(c_res);
  }
  pips_debug(5, "Exiting\n");

  return controlized;
}


/* Deal with "goto" when building the HCFG

   @param[in,out] c_res is the control node with the "goto" statement
   (that is an empty one in the unstructured since the "goto" in the HCFG
   is an arc, no longer a statement) that will point to a new control node
   with the given label

   @param[in,out] succ is the control node successor of @p c_res. Except
   if it is also the target of the "goto" by chance, it will become
   unreachable.

   @param[in,out] used_labels is a hash table mapping a label name to a
   list of statements that use it, as their label or because they are a
   goto to it */
static bool controlize_goto(control c_res,
			    control succ,
			    hash_table used_labels)
{
  bool controlized;
  statement st = control_statement(c_res);
  statement go_to = statement_goto(st);
  /* Get the label name of the statement the goto points to: */
  string name = entity_name(statement_label(go_to));
  /* Since the goto by itself is transformed into an arc in the control
     graph, the "goto" statement is no longer used. But some informations
     associated to the "goto" statement such as a comment or an extension
     must survive, so we keep them in a nop statement that will receive
     its attribute and will be the source arc representing the goto: */
  instruction i = statement_instruction(st);
  /* Disconnect the target statement: */
  instruction_goto(i) = statement_undefined;
  free_instruction(i);
  statement_instruction(st) = make_continue_instruction();

  /* Get the control node associated with the label we want to go to: */
  control n_succ = get_label_control(name);

  ifdebug(5) {
    pips_debug(5, "After freeing the goto, from c_res = %p:\n", c_res);
    display_linked_control_nodes(c_res);
    check_control_coherency(c_res);
    pips_debug(5, "From n_succ = %p:\n", n_succ);
    display_linked_control_nodes(n_succ);
    check_control_coherency(n_succ);
  }
  if (succ ==  n_succ) {
    /* Since it is a goto just on the next statement, nothing to do and it
       is not unstructured. But st must no longer be associated to name
       in hash table Label_statements. */
    list sl = (list) hash_get_default_empty_list(Label_statements, (void *) name);
    if(gen_in_list_p((void *) st, sl)) {
      gen_remove(&sl, (void *) st);
      hash_update(Label_statements, (void *) name, (void *) sl);
    }
    else {
      pips_internal_error("Label %s associated to statement %p is not associated"
			  " to statement %p in hash table Label_statements\n");
    }
    controlized = false;
  }
  else {
    /* The goto target is somewhere else, so it is clearly
       unstructured: */
    controlized = true;
    pips_debug(5, "Unstructured goto to label %s: control n_succ %p\n"
	       "\tSo statement in control %p is now unreachable from this way\n",
	       name, n_succ, succ);
    /* Disconnect the nop from the successor:*/
    unlink_2_control_nodes(c_res, succ);
    /* And add the edge in place of the goto from c_res to n_succ: */
    link_2_control_nodes(c_res, n_succ);

    /* Add st as locally related to this label name but do not add the
       target since it may be non local: */
    update_used_labels(used_labels, name, st);

    ifdebug(5) {
      pips_debug(5, "After freeing the goto, from c_res = %p:\n", c_res);
      display_linked_control_nodes(c_res);
      check_control_coherency(c_res);
      pips_debug(5, "From n_succ = %p:\n", succ);
      display_linked_control_nodes(succ);
      check_control_coherency(succ);
    }
    ifdebug(1)
      check_control_coherency(n_succ);
  }
  return controlized;
}


/* Controlize a call statement

   The deal is to correctly manage STOP; since we don't know how to do it
   yet, we assume this is a usual call with a continuation !

   To avoid non-standard successors, IO statement with multiple
   continuations are not dealt with here. The END= and ERR= clauses are
   simulated by hidden tests.

   @param[in] c_res is the control node with a call statement to controlize

   @param[in] succ is the control node successor of @p c_res

   @return false since we do nothing, so no non-structured graph
   generation yet...
*/
static bool controlize_call(control c_res,
			    control succ)
{
  statement st = control_statement(c_res);
  pips_debug(5, "(st = %p, c_res = %p, succ = %p)\n",
	     st, c_res, succ);

  return false;
}


/* Controlize a statement that is in a control node, that is restructure
   it to have a HCFG recursively (a hierarchical control flow graph).

   @param[in,out] c_res is the control node with the main statement to
   controlize. @p c_res should already own the statement at entry that
   will be recursively controlized. @p c_res can be seen as a potential
   unstructured entry node.

   @param[in,out] succ is the successor control node of @p c_res at
   entry. It may be no longer the case at exit if for example the code is
   unreachable or there is a goto to elsewhere in @p_res. @p succ can be
   seen as a potential unstructured exit node.

   @param[in,out] used_labels is a way to define a community of label we
   encounter in a top-down approac during the recursion with their related
   statements. With it, during the bottom-up approach, we can use it to
   know if a statement can be control-hierarchized or not. More
   concretely, it is a hash table mapping a label name to a list of
   statements that reference it, as their label or because it is a goto to
   it. This hash map is modified to represent local labels with their
   local related statements. This table is used later to know if all the
   statements associated to local labels are local or not. If they are
   local, the statement can be hierarchized since there is no goto from or
   to outside location.

   @return true if the current statement isn't a structured control.
*/
bool controlize_statement(control c_res,
			  control succ,
			  hash_table used_labels)
{
  /* Get the statement to controlized from its control node owner. The
     invariant is that this statement remains in its control node through
     the controlizer process. Only the content of the statement may
     change. */
  statement st = control_statement(c_res);
  instruction i = statement_instruction(st);
  bool controlized = false;

  ifdebug(5) {
    pips_debug(1,
	       "Begin with (st = %p, c_res = %p, succ = %p)\n"
	       "st at entry:\n",
	       st, c_res, succ);
    statement_consistent_p(st);
    print_statement(st);
    pips_debug(1, "Control list from c_res %p:\n", c_res);
    display_linked_control_nodes(c_res);
    check_control_coherency(succ);
    check_control_coherency(c_res);
  }

  switch(instruction_tag(i)) {

  case is_instruction_block:
    /* Apply the controlizer on a statement sequence, basically by
       considering it as a control node sequence */
    controlized = controlize_sequence(c_res, succ, used_labels);
    break;

  case is_instruction_test:
    /* Go on with a test: */
    controlized = controlize_test(c_res, succ, used_labels);
    break;

  case is_instruction_loop:
    /* Controlize a DO-loop à la Fortran: */
    controlized = controlize_loop(c_res, succ, used_labels);
    break;

  case is_instruction_whileloop: {
    /* Controlize a while() or do { } while() loop: */
    whileloop wl = instruction_whileloop(i);
    if(evaluation_before_p(whileloop_evaluation(wl)))
      controlized = controlize_whileloop(c_res, succ, used_labels);
    else
      controlized = controlize_repeatloop(c_res, succ, used_labels);
    statement_consistent_p(st);
    break;
  }

  case is_instruction_goto: {
    /* The hard case, the goto, that will add some trouble in this well
       structured world... */
    controlized = controlize_goto(c_res, succ, used_labels);

    break;
  }

  case is_instruction_call:
    /* Controlize some function call (that hides many things in PIPS) */
    /* FI: IO calls may have control effects; they should be handled here! */
    // FI: no specific handling of return? controlized = return_instruction_p(i)
    controlized = controlize_call(c_res, succ);
    statement_consistent_p(st);
    break;

  case is_instruction_forloop:
    pips_assert("We are really dealing with a for loop",
		instruction_forloop_p(statement_instruction(st)));
    controlized = controlize_forloop(c_res, succ, used_labels);
    statement_consistent_p(st);
    break;

  case is_instruction_expression: {
    expression e = instruction_expression(i);
    if(expression_reference_p(e)) {
      controlized = false;
    }
    else if(expression_call_p(e))
    /* PJ: controlize_call() controlize any "nice" statement, so even a C
       expression used as an instruction: */
      controlized = controlize_call(c_res, succ);
    statement_consistent_p(st);
    break;
  }

  default:
    pips_internal_error("Unknown instruction tag %d", instruction_tag(i));
  }

  statement_consistent_p(st);
  ifdebug(5) {
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

    check_control_coherency(succ);
    check_control_coherency(c_res);
  }

  /* Update the association between the current statement and its
     label (the empty label is ignored by update_used_labels()):
   */
  string label = entity_name(statement_label(st));
  update_used_labels(used_labels, label, st);

  return controlized;
}


/* Compute the hierarchical control flow graph (HCFG) of a statement

   @param st is the statement we want to "controlize"

   @return the unstructured with the control graph
*/
statement hcfg(statement st)
{
  statement result;

  pips_assert("Statement should be OK.", statement_consistent_p(st));
  ifdebug(1) {
    /* To help debugging, force some properties: */
    set_bool_property("PRETTYPRINT_BLOCKS", true);
    set_bool_property("PRETTYPRINT_EMPTY_BLOCKS", true);
  }

  /* Initialize an empty association from label to the statements using
     them: */
  hash_table used_labels = hash_table_make(hash_string, 0);

  /* Initialize global tables: */
  Label_statements = hash_table_make(hash_string, LABEL_TABLES_SIZE);
  Label_control = hash_table_make(hash_string, LABEL_TABLES_SIZE);
  /* Build the global table associating for any label all the statements
     that refer to it: */
  create_statements_of_labels(st);

  /* Construct the entry and exit control node of the unstructured to
     generate. First get the control node for all the code: */
  control entry = make_conditional_control(st);
  /* And make a successor node that is an empty instruction at first: */
  control exit = make_control(make_plain_continue_statement(), NIL, NIL);
  /* By default the entry is just connected to the exit that is the
     successor: */
  link_2_control_nodes(entry, exit);

  /* To track declaration scoping independently of control structure: */
  make_scoping_statement_stack();

  /* Build the HCFG from the module statement: */
  (void) controlize_statement(entry, exit, used_labels);

  /* Since this HCFG computation is quite general, we may have really an
     unstructured at the top level, which is not possible when dealing
     from the parser output. */
  if (gen_length(control_successors(entry)) == 1
      && CONTROL(CAR(control_successors(entry))) == exit) {
    /* The 2 control nodes are indeed still a simple sequence, so it is
       a structured statement at the top level and it is useless to
       build an unstructured to store it. */
    result = control_statement(entry);
    free_a_control_without_its_statement(entry);
    /* Really remove the useless by contruction statement too: */
    free_a_control_without_its_statement(exit);
  }
  else {
    /* For all the other case, it is not structured code at top level, so
       build an unstructured statement to represent it: */
    unstructured u = make_unstructured(entry, exit);
    result = instruction_to_statement(make_instruction_unstructured(u));
  }

  /* Clean up scoping stack: */
  free_scoping_statement_stack();

  /* Reset the tables used */
  hash_table_free(Label_statements);
  hash_table_free(Label_control);
  hash_table_free(used_labels);

  /* Since the controlizer is a sensitive pass, avoid leaking basic
     errors... */
  pips_assert("Statement should be OK.", statement_consistent_p(result));

  return result;
}

/*
  @}
*/
