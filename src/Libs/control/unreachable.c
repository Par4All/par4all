/*
 * Detection of unreachable code, from the control flow point of view.
 *
 * $Id$
 *
 * $Log: unreachable.c,v $
 * Revision 1.6  1997/11/14 09:29:50  keryell
 * Removed 2 nasty ";".
 *
 * Revision 1.5  1997/11/13 08:07:05  coelho
 * statement_continued_p added...
 *
 * Revision 1.4  1997/11/12 17:06:51  coelho
 * 1 go to 1 fixed...
 *
 * Revision 1.3  1997/11/12 12:07:58  coelho
 * new interface for Ronan to use it...
 *
 * Revision 1.2  1997/11/10 18:19:52  coelho
 * typos fixed...
 *
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

GENERIC_LOCAL_FUNCTION(reached, persistant_statement_to_int)
GENERIC_LOCAL_FUNCTION(continued, persistant_statement_to_int)

#define reached_p(s)   (bound_reached_p(s))
#define continued_p(s) (load_continued(s))

static bool propagate(statement);

#define check_recursion(s)					\
    if (reached_p(s))						\
    {								\
	if (bound_continued_p(s))				\
	    return continued_p(s); /* already computed */	\
	else							\
	    return FALSE; /* avoids an infinite recursion... */	\
    }								\
    else store_reached(s, TRUE);


static bool
control_propagate(control c)
{
    bool continued = propagate(control_statement(c));
    list lc = control_successors(c);
    int len = gen_length(lc);
    pips_assert("max 2 successors", len<=2);

    check_recursion(c);

    if (len==2)
    {
	bool ctrue, cfalse;
	ctrue = control_propagate(CONTROL(CAR(lc)));
	cfalse = control_propagate(CONTROL(CAR(CDR(lc))));
	continued = ctrue || cfalse;
    }
    else if (continued && len==1)
    {
	control cn = CONTROL(CAR(lc));
	if (cn!=c) continued = control_propagate(cn);
	else continued = FALSE; /* 1 GO TO 1 */
    }

    store_continued(c, continued);
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


/***************************************************************** INTERFACE */

void
init_reachable(statement start)
{
    init_reached();
    init_continued();
    propagate(start);
}

bool
statement_reachable_p(statement s)
{
    return reached_p(s);
}

bool
statement_continued_p(statement s)
{
    if (bound_continued_p(s))
	return continued_p(s);
    else
	return FALSE;
}

void 
close_reachable(void)
{
    close_reached();
    close_continued();
}
