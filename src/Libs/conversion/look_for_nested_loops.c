/* Package generation (for the hyperplane transformation?)
 * $RCSfile: look_for_nested_loops.c,v $ version $Revision$, 
 * ($Date: 1998/10/13 07:13:58 $, ) 
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "misc.h"
#include "ri-util.h"
#include "control.h"
#include "text.h"
#include "text-util.h"
#include "prettyprint.h"
#include "constants.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "conversion.h"


/* void look_for_nested_loop_statements(statement s)
 * search  the  nested  loops  in the statement s
 *  
*/
void 
look_for_nested_loop_statements(s,loop_transformation, loop_predicate)
statement s;
statement (*loop_transformation)();
bool (*loop_predicate)();
{
    instruction i;
    cons *b, *b1;
    statement ss, new_s = statement_undefined;
    loop l;
    cons *list_loop_statement;
    test tt;
    statement true_s, false_s;
	
    i = statement_instruction(s);
    switch (instruction_tag(i)) {
    case is_instruction_loop:
	new_s = s;
	l = instruction_loop(i);

	if ((*loop_predicate)(l)) {
	    list_loop_statement = CONS (STATEMENT,s,NIL);

	    ifdebug(9) {
		debug(9, "look_for_nested_loop_statements",
		      "Before transformation:\n");
		debug_on("ZERO_DEBUG_LEVEL");
		print_text(stderr,text_statement(entity_undefined, 0, s));
		debug_off();
	    }

	    new_s = look_for_inner_loops(l,list_loop_statement,
					 loop_transformation, 
					 loop_predicate);
	    
	    ifdebug(9) {
		debug(9, "look_for_nested_loop_statements",
		      "After transformation:\n");
		pips_assert("look_for_nested_loop_statement", statement_consistent_p(new_s));
		debug_on("ZERO_DEBUG_LEVEL");
		print_text(stderr,text_statement(entity_undefined,0,new_s));
		debug_off();
	    }
	    if (new_s != statement_undefined) {
		i = statement_instruction(s);
		instruction_loop(i) =
		    instruction_loop(statement_instruction(new_s)); }
	}
	else look_for_nested_loop_statements(loop_body(l),loop_transformation, 
					     loop_predicate);
	break;

    case is_instruction_block:

	b= instruction_block(i);
	ss = STATEMENT(CAR(b));
	look_for_nested_loop_statements(ss,loop_transformation, loop_predicate);
	for(b1 = CDR(b); !ENDP(b1); b1 = CDR(b1)) {
	    ss = STATEMENT(CAR(b1));
	    look_for_nested_loop_statements(ss,loop_transformation, loop_predicate);
	}
	break;

    case is_instruction_call:
	break;

    case is_instruction_test:

	tt = instruction_test(i);
	true_s = test_true(tt);
	false_s= test_false(tt);
	look_for_nested_loop_statements(true_s,loop_transformation, loop_predicate);
	look_for_nested_loop_statements(false_s,loop_transformation, loop_predicate);
	break;

    case is_instruction_unstructured:
	look_for_nested_loops_unstructured(instruction_unstructured(i),loop_transformation, loop_predicate);
	break;

    case is_instruction_goto:
	pips_error("look_for_nested_loop_statements",
		   "unexpected goto in code");
    default:
	pips_error("look_for_nested_loop_statements",
		   "unexpected tag %d\n",instruction_tag(i));
    }
}

statement look_for_inner_loops(l,sl,loop_transformation, loop_predicate)
loop l;
cons *sl;
statement (*loop_transformation)();
bool (*loop_predicate)();

{
    statement lb = loop_body(l);
    instruction i = statement_instruction(lb);
    statement ss;
    statement new_s = statement_undefined;
    unstructured unst;
    loop li;
    cons *b, *b1;
    test tt;
    statement true_s, false_s;

    /* check that i is a block */
    switch (instruction_tag(i)) {

    case is_instruction_loop:

	li = instruction_loop(i);
	sl = CONS(STATEMENT,lb,sl);
	new_s = look_for_inner_loops(li,sl,loop_transformation, loop_predicate);
	return(new_s);
	break;

    case is_instruction_block:

	b = instruction_block(i);
	ss = STATEMENT(CAR(b));
	i = statement_instruction(ss);
	if (instruction_loop_p(i)){
	    /* i is an inner loop, append it to the list of loops */
	    li = instruction_loop(i);
	    sl = CONS(STATEMENT,ss,sl);
    	    new_s = look_for_inner_loops(li,sl,loop_transformation, loop_predicate);
	}
	else {				
	    /*there are no more nested loops */

	    look_for_nested_loop_statements(ss,loop_transformation, loop_predicate);
	    debug_on("ZERO_DEBUG_LEVEL");
	    new_s = (*loop_transformation)(sl,loop_predicate);
	    debug_off();
	}
	
	for( b1=CDR(b); !ENDP(b1); b1 = CDR(b1) ) {
	    ss = STATEMENT(CAR(b1));
	    look_for_nested_loop_statements(ss,loop_transformation, loop_predicate);
	}
	return(new_s);
	break;


	
    case is_instruction_test:

	tt = instruction_test(i);
	true_s = test_true(tt);
	false_s= test_false(tt);
	look_for_nested_loop_statements(true_s,loop_transformation, loop_predicate);
	look_for_nested_loop_statements(false_s,loop_transformation, loop_predicate);
	debug_on("ZERO_DEBUG_LEVEL");
	new_s = (*loop_transformation)(sl,loop_predicate);
	debug_off();
	return(new_s);
	break;

    case is_instruction_goto:

	pips_error("look_for_inner_loop","unexpected goto");

    case is_instruction_call:
	debug_on("ZERO_DEBUG_LEVEL");
	new_s = (*loop_transformation)(sl,loop_predicate);
	debug_off();

	return(new_s);
	break;

    case is_instruction_unstructured:

	unst =instruction_unstructured(i);
	ss=control_statement(unstructured_control(unst));
	i = statement_instruction(ss);
	if (instruction_loop_p(i)) {
	    li = instruction_loop(i);
	    sl = CONS(STATEMENT,ss,sl);
	    new_s = look_for_inner_loops(li,sl,loop_transformation, loop_predicate);
	}
	else {
	    debug_on("ZERO_DEBUG_LEVEL");
	    new_s = (*loop_transformation)(sl,loop_predicate);
	    debug_off();
	    break;
	}
	return(new_s);
	break;

    default:
	pips_error("look_for_inner_loop",
		   "unexpected tag %d\n",instruction_tag(i));
    }

    return(statement_undefined); /* just to avoid a gcc warning */
}


void print_loops_list(mod,sl)
entity mod;
cons *sl;
{
    MAPL(cl, {
	loop l = LOOP(CAR(cl));
	text t = text_loop(mod, entity_name(loop_label(l)), 2, l, 0);
	print_text(stderr,t); },
	 sl);
}

/*void look_for_nested_loops_unstructured(unstructured u)
 *  search the nested loops contained in the 
 * unstructured u
 */
void look_for_nested_loops_unstructured(u,loop_transformation, loop_predicate)
unstructured u;
statement (*loop_transformation) ();
bool (*loop_predicate)();
{
    cons *blocs = NIL;
    control ct = unstructured_control(u);

    debug_on("GENERATION_DEBUG_LEVEL");
    CONTROL_MAP(c, {
	statement st = control_statement(c) ;
	(void) look_for_nested_loop_statements(st,loop_transformation, loop_predicate);
    }, ct, blocs) ;
    gen_free_list(blocs);
    debug_off();
}
