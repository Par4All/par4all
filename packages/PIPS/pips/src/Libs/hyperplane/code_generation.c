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
/* package hyperplane
 */

#include  <stdio.h>
/* #include <sys/stdtypes.h> */  /* for debug with dbmalloc */
/* #include "stdlib.h" */

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
 *
 * modification : Keep the code in the sequential version. Let the
 * parallelisation to generate the parallel code.  suggested by BB,
 * modified by Yi-Qing. 17/05/92
 *
 */
statement code_generation(list lls,
			  Pvecteur pvg[],
			  Pbase base_oldindex,
			  Pbase base_newindex,
			  Psysteme sc_newbase,
			  bool preserve_entry_label_p)
{
    statement state_lhyp = statement_undefined;
    instruction instr_lhyp = instruction_undefined;
    loop l_hyp = loop_undefined;
    range rl = range_undefined;
    expression lower = expression_undefined;
    expression upper = expression_undefined;
    statement bl = statement_undefined; 
    statement s_loop = statement_undefined;
    Pbase pb = BASE_UNDEFINED;

    bl = loop_body(instruction_loop(statement_instruction(STATEMENT(CAR(lls)))));
    statement_newbase(bl,pvg,base_oldindex);
    /* make the parallel loops from inner loop to outermost loop*/

    for(pb=base_reversal(base_newindex);lls!=NIL; lls=CDR(lls), pb = pb->succ) {
	/* handling of current loop */
	s_loop = STATEMENT(CAR(lls));

	/* new bounds for new index related to the old index of the old loop*/
	make_bound_expression(pb->var, base_newindex,sc_newbase, &lower, &upper);
	rl = make_range(lower, upper, int_to_expression(1));

	/*
    loop l_old = loop_undefined;
	l_old = instruction_loop(statement_instruction(s_loop));
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

	bl = makeloopbody(l_hyp, s_loop, true);
    }
    
    instr_lhyp = make_instruction(is_instruction_loop,l_hyp);
    state_lhyp = copy_statement(s_loop);
    if(!preserve_entry_label_p)
      clear_label(state_lhyp);
    free_instruction(statement_instruction(state_lhyp));
    statement_instruction(state_lhyp) = instr_lhyp;
    return(state_lhyp);
}



