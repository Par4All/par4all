#include <stdio.h>
#include <strings.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
/*
#include "text.h"
#include "text-util.h"
#include "properties.h"
*/

#include "misc.h"
#include "ri-util.h"
#include "control.h"

/* These functions compute the statement_ordering of their arguments. 
   U_NUMBER is the current unstructured number and *S_NUMBER the current
   statement number, in the current U_NUMBER. */

static int u_number;
#define RESET_UNSTRUCTURED_NUMBER {u_number = 1;}
#define NEW_UNSTRUCTURED_NUMBER (u_number++)

void reset_unstructured_number()
{
    RESET_UNSTRUCTURED_NUMBER
}

static int statement_reorder(st, un, sn)
statement st;
int un, sn;
{
    instruction i = statement_instruction(st);
    void unstructured_reorder();

    debug(5, "statement_reorder", "entering for %d : (%d,%d)\n",
	  statement_number(st), un, sn);

    statement_ordering(st) = MAKE_ORDERING(un, sn);

    sn += 1;

    switch (instruction_tag(i)) {
      case is_instruction_block:
	debug(5, "statement_reorder", "block\n");
	MAPL(sts, {
	    sn = statement_reorder(STATEMENT(CAR(sts)), un, sn);
	}, instruction_block(i));
	break;
      case is_instruction_test:
	debug(5, "statement_reorder", "test\n");
	sn = statement_reorder(test_true(instruction_test(i)), un, sn);
	sn = statement_reorder(test_false(instruction_test(i)), un, sn);
	break;
      case is_instruction_loop:
	debug(5, "statement_reorder", "loop\n");
	sn = statement_reorder(loop_body(instruction_loop(i)), un, sn);
	break;
      case is_instruction_goto:
      case is_instruction_call:
	debug(5, "statement_reorder", "goto or call\n");
	break;
      case is_instruction_unstructured: 
	debug(5, "statement_reorder", "unstructured\n");
	unstructured_reorder(instruction_unstructured(i));
	break;
      default:
	pips_error("statement_reorder", "Unknown tag %d\n", 
		   instruction_tag(i));
    }

    debug(5, "statement_reorder", "exiting %d\n", sn);

    return(sn);
}

void unstructured_reorder(u)
unstructured u;
{
    cons *blocs = NIL;

    debug(5, "unstructured_reorder", "entering\n");

    CONTROL_MAP(ctl, {
	statement st = control_statement(ctl);
	int un = NEW_UNSTRUCTURED_NUMBER;

	debug(5, "unstructured_reorder", "will reorder %d %d\n",
	      statement_number(st), un);

	statement_reorder(st, un, 1);
    }, unstructured_control(u), blocs);

    gen_free_list(blocs);

    debug(5, "unstructured_reorder", "exiting\n");
}


void module_body_reorder(body)
statement body;
{
    /* If a module_body_reorder() is required, ordering_to_statement
       must be recomputed */
    pips_assert("module_body_reorder", !ordering_to_statement_initialized_p());

    debug_on("CONTROL_DEBUG_LEVEL");

    RESET_UNSTRUCTURED_NUMBER;

    /* FI: I do not understand why unstructured numbering is not
     * restarted from 0.
     * statement_reorder(body, NEW_UNSTRUCTURED_NUMBER, 1);
     */
    statement_reorder(body, 0, 1);

    debug_off();
}


void module_reorder(body)
statement body;
{
    if(ordering_to_statement_initialized_p()) {
	reset_ordering_to_statement();
    }
    module_body_reorder(body);
    initialize_ordering_to_statement(body);
}
