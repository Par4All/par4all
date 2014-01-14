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

#include <stdio.h>
#include <strings.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "text.h"
#include "text-util.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"
#include "control.h"

/* global mapping from statements to their control in the full control graph
 */
GENERIC_GLOBAL_FUNCTION(ctrl_graph, controlmap)

/* the crtl_graph is freed by hand, because the default behavior is
 * not convenient for my purpose. I would have needed a persistant
 * statement in the control, but it is not desired in pips.
 */
void clean_ctrl_graph()
{
    CONTROLMAP_MAP(s, c,
      {
	  pips_debug(7, "statement (%td,%td)\n",
		     ORDERING_NUMBER(statement_ordering(s)),
		     ORDERING_STATEMENT(statement_ordering(s)));

	  control_statement(c) = statement_undefined;
	  gen_free_list(control_successors(c));
	  control_successors(c) = NIL;
	  gen_free_list(control_predecessors(c));
	  control_predecessors(c) = NIL;
      },
	  get_ctrl_graph());

    close_ctrl_graph(); /* now it can be freed safely */
}

/*  add (s1) --> (s2),
 *  that is s2 as successor of s1 and s1 as predecessor of s2.
 *  last added put in first place.
 *  this property is used for a depth first enumeration.
 */
static void
add_arrow_in_ctrl_graph(statement s1, statement s2)
{
    control
	c1 = load_ctrl_graph(s1),
	c2 = load_ctrl_graph(s2);

    pips_debug(7, "(%td,%td:%td) -> (%td,%td;%td)\n",
	       ORDERING_NUMBER(statement_ordering(s1)),
	       ORDERING_STATEMENT(statement_ordering(s1)),
	       statement_number(s1),
	       ORDERING_NUMBER(statement_ordering(s2)),
	       ORDERING_STATEMENT(statement_ordering(s2)),
	       statement_number(s2));

    control_successors(c1) = gen_once(c2, control_successors(c1));
    control_predecessors(c2) = gen_once(c1, control_predecessors(c2));
}

static void add_arrows_in_ctrl_graph(statement s, /* statement */ list l)
{
    for(; !ENDP(l); l=CDR(l))
	add_arrow_in_ctrl_graph(s, STATEMENT(CAR(l)));
}

list /* of statement */
control_list_to_statement_list(/* control */ list lc)
{
    list /* of statements */ ls = NIL;
    MAP(CONTROL, c, ls = CONS(STATEMENT, control_statement(c), ls), lc);
    return ls;
}

static void statement_arrows(statement s, /* statement */ list next)
{
    instruction i = statement_instruction(s);
    tag t = instruction_tag(i);

    switch(t)
    {
    case is_instruction_block:
    {
	statement current, succ;
	list /* of statements */
	    l = instruction_block(i),
	    just_next;

	if (ENDP(l))
	{
	    add_arrows_in_ctrl_graph(s, next);
	    return;
	}
	/* else
	 */
	add_arrow_in_ctrl_graph(s, STATEMENT(CAR(l)));

	for(current = STATEMENT(CAR(l)),
	      l = CDR(l),
	      succ = ENDP(l) ? statement_undefined : STATEMENT(CAR(l)),
	      just_next = CONS(STATEMENT, succ, NIL);
	    !ENDP(l);
	    current = succ,
	      l = CDR(l),
	      succ = ENDP(l) ? statement_undefined : STATEMENT(CAR(l)),
	      STATEMENT_(CAR(just_next)) = succ)
	{
	    statement_arrows(current, just_next);
	}

	gen_free_list(just_next), just_next=NIL;

	statement_arrows(current, next);
	break;
    }
    case is_instruction_test:
    {
	test
	    x = instruction_test(i);
	statement
	    strue = test_true(x),
	    sfalse = test_false(x);

	add_arrow_in_ctrl_graph(s, sfalse),
	statement_arrows(sfalse, next);

	add_arrow_in_ctrl_graph(s, strue), /* true is before false */
	statement_arrows(strue, next);

	break;
    }
    case is_instruction_whileloop:
    case is_instruction_loop:
    {
	statement b;
	list /* of statements */ just_next =
	    gen_nconc(gen_copy_seq(next), CONS(STATEMENT, s, NIL));

	if(instruction_loop_p(i)) {
	    loop l = instruction_loop(i);
	    b = loop_body(l);
	}
	else {
	    whileloop l = instruction_whileloop(i);
	    b = whileloop_body(l);
	}

	add_arrows_in_ctrl_graph(s, next); /* no iteration */
	add_arrow_in_ctrl_graph(s, b);     /* some iterations, first */

	statement_arrows(b, just_next);

	gen_free_list(just_next), just_next=NIL;

	break;
    }
    case is_instruction_call:
	add_arrows_in_ctrl_graph(s, next);
	break;
    case is_instruction_unstructured:
    {
	unstructured u = instruction_unstructured(i);
	list /* of statements */
	    blocks = NIL,
	    lstat = NIL;
	statement x;
	control c_in = unstructured_control(u);

	add_arrow_in_ctrl_graph(s, control_statement(c_in));

	/* hmmm... I'm not too confident in this loop.
	 * ??? what should be done with next?
	 * ??? should I trust the graph? I hope I can.
	 */
	CONTROL_MAP(c,
		{
		    x = control_statement(c);
		    lstat = control_list_to_statement_list
			(control_successors(c));

		    statement_arrows(x, lstat);
		    gen_free_list(lstat);
		},
		    c_in,
		    blocks);

	gen_free_list(blocks);

	break;
    }
    case is_instruction_goto:
    default:
	pips_internal_error("unexpected instruction tag (%d)", t);
    }
}

static void stmt_rewrite(statement s)
{
    if (!bound_ctrl_graph_p(s))
	store_ctrl_graph(s, make_control(s, NIL, NIL));
}

void build_full_ctrl_graph(statement s)
{
    pips_debug(3, "statement (%td,%td:%td)\n",
	       ORDERING_NUMBER(statement_ordering(s)),
	       ORDERING_STATEMENT(statement_ordering(s)),
	       statement_number(s));

    /*  first pass to initialize the ctrl_graph controls
     */
    init_ctrl_graph();
    gen_multi_recurse(s,
		      statement_domain, gen_true, stmt_rewrite, /* STATEMENT */
		      NULL);

    /*  second pass to link the statements
     */
    statement_arrows(s, NIL);
}

/* FULL CONTROL GRAPH for module NAME
 */
void full_control_graph(string name)
{
    statement s = (statement) db_get_memory_resource(DBR_CODE, name, true);
    build_full_ctrl_graph(s);

    /*  should put something in the db if made as a pass
     */
}

/* TRAVELLING on the control graph
 *
 * init, next, close functions.
 * - the init function is given a starting point (statement) s and a
 *   decision function decision. The function should say whether to
 *   go on with the successors of its arguments.
 * - the next function gives the next visited statement.
 * - the close function frees the static data.
 */

#ifndef bool_undefined
    #define bool_undefined ((bool) (-15))
    #define bool_undefined_p(b) ((b)==bool_undefined)
#endif

/* Static data for the travel.
 * - the stack stores the statements to see
 * - the mapping stores the already stacked statements
 * - decision function...
 */
DEFINE_LOCAL_STACK(to_see, statement)
GENERIC_LOCAL_MAPPING(stacked, bool, statement)
static bool (*travel_decision)(statement);

static void push_if_necessary(statement s)
{
  if (load_statement_stacked(s)!=true)
  {
    to_see_push(s);
    store_statement_stacked(s, true);
  }
}

/* it is pushed in *reverse* order to preserve the depth first view.
 */
static void push(/* control */ list l)
{
    if (!ENDP(l))
	push(CDR(l)),
	push_if_necessary(control_statement(CONTROL(CAR(l))));
}

static void push_successors(statement s)
{
    push(control_successors(load_ctrl_graph(s)));
}

void init_ctrl_graph_travel(statement s, bool (*decision)(statement))
{
    make_to_see_stack();             /* initializations */
    make_stacked_map();
    travel_decision = decision;

    store_statement_stacked(s, true); /* no loop back */
    push_successors(s);
}

bool next_ctrl_graph_travel(statement *ps)
{
    while (!to_see_empty_p())
    {
	*ps = to_see_pop();
	if ((*travel_decision)(*ps))
	{
	    push_successors(*ps);
	    return true;
	}
    }

    return false;
}

void close_ctrl_graph_travel(void)
{
    free_to_see_stack();
    free_stacked_map();
}

/*  That's all
 */
