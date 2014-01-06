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
/* Package generation (for the hyperplane transformation?)
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"
#include "text-util.h"
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
void look_for_nested_loop_statements(statement s,
                                     statement (*loop_transformation)(list, bool (*)(statement)),
                                     bool (*loop_predicate)(statement))
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

        if ((*loop_predicate)(s)) {
	    list_loop_statement = CONS (STATEMENT,s,NIL);

	    ifdebug(9) {
		pips_debug(9, "Before transformation:\n");
		debug_on("ZERO_DEBUG_LEVEL");
		print_text(stderr,text_statement(entity_undefined, 0, s, NIL));
		debug_off();
	    }

	    new_s = look_for_inner_loops(l,
					 list_loop_statement,
					 loop_transformation,
					 loop_predicate);

	    ifdebug(9) {
		pips_debug(9, "After transformation:\n");
		pips_assert("look_for_nested_loop_statement", statement_consistent_p(new_s));
		debug_on("ZERO_DEBUG_LEVEL");
		print_text(stderr,text_statement(entity_undefined,0,new_s, NIL));
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

	b = instruction_block(i);
	if(ENDP(b)) break;
	ss = STATEMENT(CAR(b));
    /* SG: added to cope with empty statements */
    while(empty_statement_or_continue_p(ss) && !ENDP(CDR(b))) {
        POP(b);
        ss=STATEMENT(CAR(b));
    }
	look_for_nested_loop_statements(ss, loop_transformation, loop_predicate);
	for(b1 = CDR(b); !ENDP(b1); b1 = CDR(b1)) {
	    ss = STATEMENT(CAR(b1));
	    look_for_nested_loop_statements(ss, loop_transformation, loop_predicate);
	}
	break;

    case is_instruction_call:
	break;

    case is_instruction_test:

	tt = instruction_test(i);
	true_s = test_true(tt);
	false_s= test_false(tt);
	look_for_nested_loop_statements(true_s, loop_transformation, loop_predicate);
	look_for_nested_loop_statements(false_s, loop_transformation, loop_predicate);
	break;

    case is_instruction_whileloop: {

	whileloop wl = instruction_whileloop(i);
	statement body = whileloop_body(wl);

	look_for_nested_loop_statements(body, loop_transformation, loop_predicate);
	break;
    }

    case is_instruction_forloop: {

	forloop fl = instruction_forloop(i);
	statement body = forloop_body(fl);

	look_for_nested_loop_statements(body, loop_transformation, loop_predicate);
	break;
    }

    case is_instruction_unstructured:
	look_for_nested_loops_unstructured(instruction_unstructured(i),
					   loop_transformation,
					   loop_predicate);
	break;

    case is_instruction_goto:
	pips_internal_error("unexpected goto in code");
    default:
	pips_internal_error("unexpected tag %d",instruction_tag(i));
    }
}

/* FI: I do not understand how debug levels are managed... They should be factored out. */
statement look_for_inner_loops(loop l,
			       list sl,
                               statement (*loop_transformation)(list, bool (*)(statement)),
                               bool (*loop_predicate)(statement))

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
	break;

    case is_instruction_block:
retry:
	b = instruction_block(i);
	ss = STATEMENT(CAR(b));
    /* SG: added to cope with empty statements */
    while(empty_statement_or_continue_p(ss) && !ENDP(CDR(b))) {
        POP(b);
        ss=STATEMENT(CAR(b));
    }
	i = statement_instruction(ss);
    if(instruction_block_p(i)) 
        goto retry;
	if (instruction_loop_p(i)){
	    /* i is an inner loop, append it to the list of loops */
	    li = instruction_loop(i);
	    sl = CONS(STATEMENT,ss,sl);
    	    new_s = look_for_inner_loops(li, sl, loop_transformation, loop_predicate);
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
	break;

    case is_instruction_whileloop: {
      statement body = whileloop_body(instruction_whileloop(i));

      look_for_nested_loop_statements(body, loop_transformation, loop_predicate);
      new_s = (*loop_transformation)(sl,loop_predicate);
      break;
    }

    case is_instruction_forloop: {
      statement body = forloop_body(instruction_forloop(i));

      look_for_nested_loop_statements(body, loop_transformation, loop_predicate);
      new_s = (*loop_transformation)(sl,loop_predicate);
      break;
    }
    case is_instruction_goto:

	pips_internal_error("unexpected goto");

    case is_instruction_call:
	debug_on("ZERO_DEBUG_LEVEL");
	new_s = (*loop_transformation)(sl,loop_predicate);
	debug_off();

	break;

    case is_instruction_unstructured:

      /* FI: I do not understand this piece of code. I expect a
	 look_for_nested_loop_statements(). */
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
	break;

    default:
	pips_internal_error("unexpected tag %d",instruction_tag(i));
    }

    return new_s;
}


void print_loops_list(mod,sl)
entity mod;
cons *sl;
{
    MAPL(cl, {
	loop l = LOOP(CAR(cl));
	text t = text_loop(mod, entity_name(loop_label(l)), 2, l, 0, NIL);
	print_text(stderr,t); },
	 sl);
}

/*void look_for_nested_loops_unstructured(unstructured u)
 *  search the nested loops contained in the 
 * unstructured u
 */
void look_for_nested_loops_unstructured(unstructured u,
                                        statement (*loop_transformation) (list, bool (*)(statement)),
                                        bool (*loop_predicate)(statement))
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
