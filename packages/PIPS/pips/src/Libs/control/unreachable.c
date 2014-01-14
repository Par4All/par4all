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
 * Detection of unreachable code, from the control flow point of view.
 */

#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"

/******************************************************** REACHED STATEMENTS */

/* A mapping to store if a given statement is reachable from the control
   flow point of view: */
GENERIC_LOCAL_FUNCTION(reached, persistant_statement_to_int)


GENERIC_LOCAL_FUNCTION(continued, persistant_statement_to_int)

#define reached_p(s)   (bound_reached_p(s))
#define continued_p(s) (load_continued(s))

static bool propagate(statement);

#define check_recursion(s)						\
  do {									\
    if (reached_p(s)) {							\
      if (bound_continued_p(s)) {					\
         pips_debug(5, "Statement %p already seen, thus stop recursion.\n", s); \
	 return continued_p(s); /* already computed */			\
      }									\
      else {								\
        pips_debug(5, "Statement %p marked as reached but the execution does not continue on afterwards.\n\tSo return FALSE.\n", s); \
        return false; /* avoids an infinite recursion... */		\
      }									\
    }									\
    else {								\
      pips_debug(5, "New statement %p marked as reached.\n", s);	\
      store_reached(s, true);						\
    }									\
  } while (false) /* Pattern to be able to use this macro like an instruction. */

static bool
control_propagate(control c)
{
  list lc = control_successors(c);
  int len = gen_length(lc);
  statement s = control_statement(c);
  pips_assert("max 2 successors", len<=2);
  pips_debug(1, "Entering control_propagate for control %p\n", c);

  /* If we already have dealt with the current control node, stop the
     recursion: */
  if (reached_p(s) && bound_continued_p(s)) {
    /* already computed */
    pips_debug(5, "Statement %p already seen, thus stop recursion.\n", s);
    return continued_p(s);
  }									\
  /* Investigate about the continuation status of the statement: */
  bool continued = propagate(s);

  if (continued) {
    /* Investigate the control successors only if the current one
       continues afterwards */
    if (len==2) {
      bool ctrue, cfalse;
      ctrue = control_propagate(CONTROL(CAR(lc)));
      cfalse = control_propagate(CONTROL(CAR(CDR(lc))));
      // Continue after the test only if at leat one branch goes on:
      continued = ctrue || cfalse;
    }
    else if (len==1) {
      control cn = CONTROL(CAR(lc));
      if (cn!=c) continued = control_propagate(cn);
      /* It seems like a fallacy here. It is not semantically different if
	 we have a cycle with more than 1 goto... It does not work for
	 irreductible graphs either but it is detected anyway now in
	 proagate(). RK */
      else continued = false; /* 1 GO TO 1 */
    }
  }
  pips_debug(1, "Ending control_propagate for control %p and statement %p returning continued %d\n",
	     c, s, continued);
  //  store_continued(control_statement(c), continued);
  return continued;
}

/* returns whether propagation is continued after s.
 * (that is no STOP or infinite loop encountered ??? ).
 * It is a MAY information. If there is no propagation, it
 * is a MUST information.
 */
static bool
propagate(statement s) {
    bool continued = true;
    instruction i;

    pips_assert("defined statement", !statement_undefined_p(s));
    ifdebug(5) {
      pips_debug(1, "Dealing with statement %p\n", s);
      print_statement(s);
    }

    check_recursion(s);

    i = statement_instruction(s);
    switch (instruction_tag(i))
    {
    case is_instruction_sequence:
    {
	list l = sequence_statements(instruction_sequence(i));
	while (l && (continued=propagate(STATEMENT(CAR(l)))))
	    POP(l);
	break;
    }
    case is_instruction_loop:
    {
	propagate(loop_body(instruction_loop(i)));
	break;
    }
    case is_instruction_whileloop:
    {
	/* Undecidable implies "propagate" by default */
	propagate(whileloop_body(instruction_whileloop(i)));
	break;
    }
    case is_instruction_forloop:
    {
	/* Undecidable implies "propagate" by default */
	propagate(forloop_body(instruction_forloop(i)));
	break;
    }
    case is_instruction_test:
    {
	test t = instruction_test(i);
	bool ctrue, cfalse;
	ctrue = propagate(test_true(t));
	cfalse = propagate(test_false(t));
	continued = ctrue || cfalse;
	break;
    }
    case is_instruction_unstructured:
    {
      unstructured u = instruction_unstructured(i);
      control c = unstructured_control(u);
      // Investigate inside the unstructured control graph:
      continued = control_propagate(c);
      if (continued) {
	/* If the unstructured is seen as going on, test if the exit node
	   as been marked as reachable */
	statement exit = control_statement(unstructured_exit(u));
	if (!reached_p(exit))
	  /* Since the exit node is not reached, that means that there is
	     an infinite loop is the unstructured, so globally it is not
	     continued: */
	  continued = false;
      }
      break;
    }
    case is_instruction_call:
    {
      /* FI: not satisfying; interprocedural control effects required here */
      entity f = call_function(instruction_call(i));
      continued = !ENTITY_STOP_P(f) && !ENTITY_ABORT_SYSTEM_P(f) && !ENTITY_EXIT_SYSTEM_P(f);
      break;
    }
    case is_instruction_expression:
    {
      continued = true; /* FI: interprocedural exit possible:-( */
      break;
    }
    case is_instruction_multitest:
    {
      pips_internal_error("Not implemented yet"); /* FI: undone by the controlizer? */
      break;
    }
    case is_instruction_goto:
	pips_internal_error("GOTO should have been eliminated by the controlizer");
	break;
    default:
	pips_internal_error("unexpected instruction tag");
    }

    pips_debug(1, "Continued for statement %p = %d\n", s, continued);
    store_continued(s, continued);
    return continued;
}


/***************************************************************** INTERFACE */

/* Compute reachable infomation from the @param start statement. */
void init_reachable(statement start) {
  debug_on("REACHABLE_DEBUG_LEVEL");
  init_reached();
  init_continued();
  propagate(start);
  debug_off();
}

/* Test if the given statement is reachable from some statements given at
   init_reachable(start) */
bool
statement_reachable_p(statement s)
{
    return reached_p(s);
}

/* Test if the execution goes on after the given statement. For example it
   is false after a "return" in C */
bool
statement_continued_p(statement s)
{
    if (bound_continued_p(s))
	return continued_p(s);
    else
	return false;
}

/* Remove reachability information about previously checked statements */
void
close_reachable(void)
{
    close_reached();
    close_continued();
}
