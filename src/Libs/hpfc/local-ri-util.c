
/*
 * What could be in ri-util, or even in ri.h!
 *
 * Fabien Coelho May 1993
 */


#include <stdio.h>
#include <string.h>

extern int fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"

#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "hpfc.h"
#include "defines-local.h"

/*
 * type_variable_dup
 */
type type_variable_dup(t)
type t;
{
    if(type_variable_p(t))
    {
	variable
	    v = type_variable(t);

	return(MakeTypeVariable(variable_basic(v),
				ldimensions_dup(variable_dimensions(v))));
    }
    else
	return(t); /* !!! means sharing */
}

/*
 * perfectly_nested_parallel_loop_to_body
 *
 * FC 930722: modified from perfectly_nested_loop_to_body, ri-util,
 */
statement perfectly_nested_parallel_loop_to_body(loop_nest, plloop)
statement loop_nest;
list *plloop;
{
    instruction 
	ins = statement_instruction(loop_nest);

    switch(instruction_tag(ins)) 
    {
    case is_instruction_block: 
    {
	list 
	    lb = instruction_block(ins);
	statement 
	    first_s = STATEMENT(CAR(lb));
	instruction 
	    first_i = statement_instruction(first_s);
	
	if (instruction_loop_p(first_i) && (gen_length(lb)==1)) /* perfect ? */
	{
	    if (execution_parallel_p(loop_execution(instruction_loop(first_i))))
	    {
		*plloop = gen_nconc(*plloop, 
			 CONS(LOOP, instruction_loop(first_i), 
				    NIL));
		return(perfectly_nested_parallel_loop_to_body(first_s, plloop));
	    }
	    else 
		return(loop_nest);
	}
	else
	    return(loop_nest);

	break;
    }
    case is_instruction_loop: 
    {
	statement 
	    sbody = loop_body(instruction_loop(ins));

	if (execution_parallel_p(loop_execution(instruction_loop(ins))))
	{
	    *plloop = gen_nconc(*plloop, 
		     CONS(LOOP, instruction_loop(ins), 
			  NIL));
	    return(perfectly_nested_parallel_loop_to_body(sbody, plloop));
	}
	else
	    return(loop_nest);
	break;
    }
    case is_instruction_test:
    case is_instruction_call:
    case is_instruction_unstructured:
    case is_instruction_goto:
	return(loop_nest);
    default:
	pips_error("perfectly_nested_parallel_loop_to_body", "illegal tag\n");
	break;
    }
    
    return(loop_nest); /* just to avoid a gcc warning */
}

/*
 * that is all
 */
