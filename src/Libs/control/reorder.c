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
#include <stdio.h>
#include <strings.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "control.h"

/** @file reorder.c

    @brief These functions compute the statement_ordering of their arguments.

    To ease referencing statements, statements are numbered with an
    ordering that is made of 2 parts, an unstructured order that is the
    occurence order of unstructured control node in a depth first visit of
    the RI, and a local order that corresponds to appearance order in a
    depth first visit of the statements inside a control node. This last
    number in reset to one when encountering a control node.
*/

/* The current unstructured number, that is the number of control node
   encountered during depth first visiting  */
static int u_number;


/* Reset the unstructured number for a new module reordering. */
void reset_unstructured_number() {
  u_number = 0;
}


/* Compute the next unstructured order */
static int get_unstructured_number() {
  return u_number;
}


/* Compute the next unstructured order */
static int get_next_unstructured_number() {
  return ++u_number;
}


/* Compute the statement ordering of a statement and all its components

   This function should be rewritten with a gen_multi_recurse_context() on
   statements and controls...

   @param st is the statement which we want to compute the ordering
   @param un is the unstructured number before statement entry
   @param st is the statement number before statement entry

   @return the statement number after the end of the given statement
*/
static int statement_reorder(statement st, int un, int sn)
{
  instruction i = statement_instruction(st);
  pips_assert("instruction is defined", i!=instruction_undefined);

  // temporary, just to avoid rebooting...
  static int check_depth_hack = 0;
  check_depth_hack++;
  pips_assert("not too deep", check_depth_hack<10000);

  pips_debug(5, "entering for %"_intFMT" : (%d,%d)\n",
             statement_number(st), un, sn);

  statement_ordering(st) = MAKE_ORDERING(un, sn);

  sn += 1;

  switch (instruction_tag(i))
  {
  case is_instruction_sequence:
    pips_debug(5, "sequence\n");
    FOREACH (statement, s, sequence_statements(instruction_sequence(i)))
    {
      sn = statement_reorder(s, un, sn);
    }
    break;
  case is_instruction_test:
    pips_debug(5, "test\n");
    sn = statement_reorder(test_true(instruction_test(i)), un, sn);
    sn = statement_reorder(test_false(instruction_test(i)), un, sn);
    break;
  case is_instruction_loop:
    pips_debug(5, "loop\n");
    sn = statement_reorder(loop_body(instruction_loop(i)), un, sn);
    break;
  case is_instruction_whileloop:
    pips_debug(5, "whileloop\n");
    sn = statement_reorder(whileloop_body(instruction_whileloop(i)), un, sn);
    break;
  case is_instruction_forloop:
    pips_debug(5, "forloop\n");
    sn = statement_reorder(forloop_body(instruction_forloop(i)), un, sn);
    break;
  case is_instruction_goto:
  case is_instruction_call:
    pips_debug(5, "goto or call\n");
    break;
  case is_instruction_expression:
    pips_debug(5, "expression\n");
    break;
  case is_instruction_unstructured:
    pips_debug(5, "unstructured\n");
    unstructured_reorder(instruction_unstructured(i));
    break;
  default:
    pips_internal_error("Unknown tag %d", instruction_tag(i));
  }
  pips_debug(5, "exiting %d\n", sn);
  check_depth_hack--;
  return sn;
}


void control_node_reorder(__attribute__((unused)) control c,__attribute__((unused))  set visited_control) {
}

/* Reorder an unstructured

   All the nodes of the unstructured are numbered, first the reachable one
   and the the unreachable ones if any.

   @param u the unstructured to reorder.
*/
void unstructured_reorder(unstructured u) {
  /* To store the visited nodes by CONTROL_MAP and avoid visiting twice a
     control node: */
  list blocs = NIL;
  /* To avoid renaming twice a control node statement, keep track of the
     visited statements: */
  // set visited_control = 

  debug(2, "unstructured_reorder", "entering\n");

  /* First iterate on the reachable control nodes from the entry node of
     the unstructured and then from the exit node: */
  UNSTRUCTURED_CONTROL_MAP(c, u, blocs, {
      statement st = control_statement(c);
      /* Since we enter a control node, increase the unstructured
	 order: */
      int un =  get_next_unstructured_number();

      debug(3, "unstructured_reorder", "will reorder %d %d\n",
	    statement_number(st), un);
      ifdebug(3)
	print_statement(st);

      /* Since we enter a control node, number the statements inside this
	   control node with a statement number starting from 1: */
      statement_reorder(st, un, 1);
    });

  /* Free the list build up during the visit: */
  gen_free_list(blocs);

  debug(3, "unstructured_reorder", "exiting\n");
}


/* Reorder a module

   This recompute the ordering of a module, that is the ordering number of
   all the statements in the module..

   @param body is the top-level statement of the module to reorder
*/
void module_body_reorder(statement body) {
  /* If a module_body_reorder() is required, ordering_to_statement must be
     recomputed if any. So use module_reorder() instead of the low-level
     module_body_reorder(): */
  pips_assert("ordering to statement is not initialized",
	      !ordering_to_statement_initialized_p());

  debug_on("CONTROL_DEBUG_LEVEL");

  reset_unstructured_number();

  /* Reorder the module statements by beginning with unstructured number
     and statement number at 1 */
  statement_reorder(body, get_unstructured_number(), 1);

  debug_off();
}


/* Reorder a module and recompute order to statement if any.

   This recompute the ordering of a module, that is the ordering number of
   all the statements in the module..

   If there is also already a mapping from the ordering to the statements,
   it is reset and recompute after reordering.

   @param body is the top-level statement of the module to reorder
 */
void module_reorder(statement body) {
  bool ordering_mapping_used = ordering_to_statement_initialized_p();
  if(ordering_mapping_used)
    /* There was a mapping to associate a statement to a given ordering,
       so clean it: */
    reset_ordering_to_statement();

  module_body_reorder(body);

  if (ordering_mapping_used) {
    /* There was a mapping to associate a statement to a given ordering,
       so we recompute it: */
    set_ordering_to_statement(body);
    /* FI: I'd rather use set_ordering_to_statement() so that reset are
       properly called and no outdated ots hash table remains for ever in
       the background, but I do not want to break PIPS right now.

       How do you know ordering to statement to be useful in the future?

       May be, we are going to work on a different module very soon..
    */
  }
}
