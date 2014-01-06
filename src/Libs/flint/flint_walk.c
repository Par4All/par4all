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
/* This file contains a set of functions defined to walk through the
 * internal representation of a module and chek different things, as the
 * number of arguments, the compatibility of the types of these arguments,
 * ...  There is one function for each domain. Its basic principle is to
 * walk through the internal representation by recursive calls to
 * functions handling sub-domains.
 * 
 * Laurent Aniort & Fabien Coelho 1992
 */

#include "local.h"

/************************************************************/

/* Print message and exit violently from flint */
#define FATAL(msg,value) {fprintf(stderr,msg,value);exit(1);}

/************************************************************/
/* The only global variable : the name of the current module */
/* extern char    *current_module_name; unused and changed */

/************************************************************/
/* These two functions deal with the boundaries of finite   */
/* arrays, verifying their definition as expressions        */

void            flint_cons_dimension(pc)
    list            pc;
{
    flint_dimension(DIMENSION(CAR(pc)));
}

dimension       flint_dimension(d)
    dimension       d;
{
    expression      el = dimension_lower(d);
    expression      eu = dimension_upper(d);

    flint_expression(el);
    flint_expression(eu);

    return (d);
}

/************************************************************/
/* This function checks the syntax of a call and recursively */
/* verifies its arguments                                   */

call            flint_call(c)
    call            c;
{
    list            la = call_arguments(c);

    check_the_call(c);

    gen_mapl((gen_iter_func_t)flint_cons_actual_argument, la);

    return (c);
}
/************************************************************/
/* This function verifies that incremented or decremented   */
/* loop(for) indices do not overflow or underflow their     */
/* limits                                                   */

range           flint_range(r)
    range           r;
{
    expression      el = range_lower(r);
    expression      eu = range_upper(r);
    expression      ei = range_increment(r);

    flint_expression(el);
    flint_expression(eu);
    flint_expression(ei);

    return (r);
}
/************************************************************/
/* A reference is used when given a function the address of */
/* an element instead of its value.                         */

reference       flint_reference(r)
    reference       r;
{
    list            pc = reference_indices(r);

    (void) check_the_reference(r);

    gen_mapl((gen_iter_func_t)flint_cons_expression, pc);

    return (r);
}
/************************************************************/
/* verification of syntaxes                                 */

void            flint_syntax(s)
    syntax          s;
{
    reference       re;
    range           ra;
    call            c;

    /* branch according to the syntax subclass */
    switch (syntax_tag(s)) {
    case is_syntax_reference:
	re = syntax_reference(s);
	flint_reference(re);
	break;
    case is_syntax_range:
	ra = syntax_range(s);
	flint_range(ra);
	break;
    case is_syntax_call:
	c = syntax_call(s);
	flint_call(c);
	break;
    default:
	FATAL("flint_syntax: unexpected tag %u\n", syntax_tag(s));
    }
}

/************************************************************/
/* These two functions operate on the list of expressions   */

void            flint_cons_expression(pc)
    list            pc;
{
    flint_expression(EXPRESSION(CAR(pc)));
}


void            
flint_cons_actual_argument(list pc)
{
    expression e = EXPRESSION(CAR(pc));

    /* An array actual argument may have no indices */
    if(expression_reference_p(e)) {
	reference r = expression_reference(e);

	if(gen_length(reference_indices(r))!=0)
	    flint_expression(e);
    }
    else {
	flint_expression(e);
    }
}

expression      flint_expression(e)
    expression      e;
{
    syntax          s = expression_syntax(e);

    flint_syntax(s);

    return (e);
}
/************************************************************/
/* Recursive verification of a loop as (range)+(expression) */

loop            flint_loop(l)
    loop            l;
{
    range           r = loop_range(l);
    statement       s = loop_body(l);

    flint_range(r);
    flint_statement(s);

    return (l);
}
/************************************************************/
/* A test is taken as (expression)+(statement)+(statement)  */

test            flint_test(t)
    test            t;
{
    expression      e = test_condition(t);
    statement       st = test_true(t);
    statement       sf = test_false(t);

    flint_expression(e);
    flint_statement(st);
    flint_statement(sf);

    return (t);
}
/************************************************************/
/* Verification of an instruction with branching according  */
/* to its subclass as defined in the data structure         */

instruction flint_instruction(i)
instruction i;
{
    list            pc;
    test            t;
    loop            l;
    call            c;
    unstructured    u;

    switch (instruction_tag(i)) {
    case is_instruction_block:
	pc = instruction_block(i);
	gen_mapl((gen_iter_func_t)flint_cons_statement, pc);
	break;
    case is_instruction_test:
	t = instruction_test(i);
	flint_test(t);
	break;
    case is_instruction_loop:
	l = instruction_loop(i);
	flint_loop(l);
	break;
    case is_instruction_goto:
	break;
    case is_instruction_unstructured:
	u = instruction_unstructured(i);
	flint_unstructured(u);
	break;
    case is_instruction_call:
	c = instruction_call(i);
	(void) check_procedure(c);
	flint_call(c);
	break;
    default:
	FATAL("flint_instruction: unexpected tag %u\n", instruction_tag(i));
    }

    return (i);
}
/************************************************************/
void            flint_unstructured(u)
    unstructured    u;
{
    list            blocs = NIL;

    CONTROL_MAP(c, {
	flint_statement(control_statement(c));
    }, unstructured_control(u), blocs);

    gen_free_list(blocs);
}
/************************************************************/
void            flint_cons_statement(pc)
    list            pc;
{
    flint_statement(STATEMENT(CAR(pc)));
}

extern statement 
    flint_current_statement;

statement       flint_statement(s)
statement       s;
{
    instruction     
	i = statement_instruction(s);
    statement       
	saved = flint_current_statement;
    
    flint_current_statement = s;
    
    flint_instruction(i);
    
    flint_current_statement = saved;
    return (s);
}

/************************************************************/
/* End of File */
