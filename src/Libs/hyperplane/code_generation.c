/* package hyperplane
 *
 * $Id$
 *
 * $Log: code_generation.c,v $
 * Revision 1.5  1998/10/13 06:57:27  irigoin
 * Generate fresh new loops instead of creating meaningless sharing with
 * initial loops.
 *
 * Revision 1.4  1998/10/09 15:53:20  irigoin
 * Typing improved and better integration with RCS
 *
 *
 */

#include  <stdio.h>
/* #include <sys/stdtypes.h> */  /* for debug with dbmalloc */
/* #include "malloc.h" */

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "ri-util.h"
#include "conversion.h"

#include "hyperplane.h"

/* statement code_generation(cons *lls,Pvecteur pvg[], Pbase base_oldindex,
 * Pbase base_newindex)
 * generation of hyperplane code for the nested loops  (cons *lls).
 * the new nested loops will be ;
 *  DOSEQ Ip = ...
 *    DOALL Jp = ...
 *      DOALL Kp = ...
 *        ...
 *  ENDDO
 * modification : Keep the code in the sequential version. Let the parallelisation 
 * to generate the parallel code.
 * suggested by BB, modified by Yi-Qing. 17/05/92
 *  
 */
statement code_generation(lls, pvg, base_oldindex, base_newindex, sc_newbase)
cons *lls;
Pvecteur pvg[];
Pbase base_oldindex;
Pbase base_newindex;
Psysteme sc_newbase;
{
    statement state_lhyp = statement_undefined;
    instruction instr_lhyp = instruction_undefined;
    loop l_old = loop_undefined;
    loop l_hyp = loop_undefined;
    range rl = range_undefined;
    expression lower = expression_undefined;
    expression upper = expression_undefined;
    statement bl = statement_undefined; 
    statement s_loop = statement_undefined;
    Pbase pb = BASE_UNDEFINED;

    bl = loop_body(instruction_loop(statement_instruction(STATEMENT(CAR(lls)))));
    statement_newbase(bl,pvg,base_oldindex);
    /* make the parallel loops from inner loop to out loop*/
   
    for(pb=base_reversal(base_newindex);lls!=NIL; lls=CDR(lls), pb = pb->succ) {
	/* handling of current loop */
	s_loop = STATEMENT(CAR(lls));
	l_old = instruction_loop(statement_instruction(s_loop));

	/* new bounds for new index related to the old index of the old loop*/
	make_bound_expression(pb->var, base_newindex,sc_newbase, &lower, &upper);
	rl = make_range(lower, upper, make_integer_constant_expression(1));

	/*
	l_hyp = make_loop((entity) pb->var,
			  rl,
			  bl,
			  loop_label(l_old),
			  make_execution(is_execution_sequential,UU),
			  loop_locals(l_old));
	*/

	/* FI: I do not understand how you could keep a go to target (!)
	 * or a list of local variables
	 */
	l_hyp = make_loop((entity) pb->var,
			  rl,
			  bl,
			  entity_empty_label(),
			  make_execution(is_execution_sequential,UU),
			  NIL);

	bl = makeloopbody(l_hyp, s_loop);
    }
    
    instr_lhyp = make_instruction(is_instruction_loop,l_hyp);
    state_lhyp = make_statement(statement_label(s_loop),
				statement_number(s_loop),
				statement_ordering(s_loop),
				statement_comments(s_loop),
				instr_lhyp);
    return(state_lhyp);
}



