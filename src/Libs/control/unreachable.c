/*
 * Detection of unreachable code.
 * Should move formats if necessary.
 *
 * $Id$
 *
 * $Log: unreachable.c,v $
 * Revision 1.1  1997/11/10 18:12:15  coelho
 * Initial revision
 *
 */ 

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"

/******************************************************** REACHED STATEMENTS */

GENERIC_LOCAL_FUNCTION(reached, statement_int);
GENERIC_LOCAL_FUNCTION(continued, statement_int);

#defined reached_p(s)  (bound_reached_p(s))
#define continued_p(s) (load_continued(s))

static bool
control_propagate(control c)
{
    bool continued = propagate(control_statement(c));
    list lc = control_successors(c);
    int len = gen_length(lc);

    pips_assert("max 2 successors", len<=2);

    if (len==2)
    {
	bool ctrue, cfalse;
	ctrue = control_propagate(CONTROL(CAR(lc)));
	cfalse = control_propagate(CONTROL(CAR(CDR(lc))));
	continued = ctrue || cfalse;
    }
    else if (continued && len==1)
	continued = control_propagate(CONTROL(CAR(lc)));

    return continued;
}

/* returns whether propagation is continued after s.
 * (that is no STOP or infinite loop encountered ??? ).
 */
bool 
propagate(statement s)
{
    bool continued = TRUE;
    instruction i;
    pips_assert("defined statement", !statement_undefined_p(s));

    if (reached_p(s))
    {
	if (bound_continued_p(s))
	    return continued_p(s); /* already computed */
	else
	    return TRUE; /* avoids an infinite recursion */
    }
    else store_reached(s, TRUE);


    i = statement_instruction(s);
    switch (instruction_tag(i))
    {
    case is_instruction_sequence:
    {
	list l = instruction_sequence(i);
	while (l && (continued=propagate(STATEMENT(CAR(l)))))
	    POP(l);
	break;
    }
    case is_instruction_loop:
    {
	propagate(loop_body(instruction_loop(i)));
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
	control c = unstructured_control(instruction_unstructured(i));
	continued = control_propagate(c);
	break;
    }
    case is_instruction_call:
    {
	continued = !ENTITY_STOP_P(call_function(instruction_call(i)));
	break;
    }
    case is_instruction_whileloop:
    {
	propagate(whileloop_body(instruction_whileloop(i)));
	break;
    }
    case is_instruction_goto:
	pips_internal_error("GOTO not welcome\n");
	break;
    default:
	pips_internal_error("unexpected instruction tag\n");
    }

    store_continued(s, continued);
    return continued;
}

static void 
drop_code_rwt(statement s)
{
    if (!reached_p(s)) 
    {
	free_instruction(statement_instruction(s)); /* rough */
	statement_instruction(s) = make_continue_instruction();
    }
}

void 
drop_unreachable_code(statement start)
{
    init_reached();
    init_continued();

    propagate(start);
    /* ??? what about FORMAT? */
    gen_recurse(start, statement_domain, gen_true, drop_code_rwt);
    
    close_reached();
    close_continued();
}
