#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"

#include "control.h"

bool blind_loop_distribution_filter(instruction l)
{
    blind_loop_distribution_rewrite(l);
    return TRUE;
}

void blind_loop_distribution_rewrite(instruction l)
{
    if(instruction_loop_p(l)) {
	instruction b = statement_instruction(loop_body(instruction_loop(l)));
	if(instruction_block_p(b) && gen_length(instruction_block(b)) > 1) {
	    list ls = instruction_block(b);
	    list lls = NIL;

	    loop_body(instruction_loop(l)) = statement_undefined;

	    MAP(STATEMENT, s, {
		instruction nli = copy_instruction(l);
		loop_body(instruction_loop(nli)) = s;
		lls = gen_nconc(lls, CONS(STATEMENT, instruction_to_statement(nli), NIL));
	    }, ls);

	    free_loop(instruction_loop(l));
	    instruction_tag(l) = is_instruction_sequence;
	    instruction_sequence(l) = make_sequence(lls);
	}
    }
}

bool
blind_loop_distribution(char * mod_name)  
{
    statement mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

    debug_on("BLIND_LOOP_DISTRIBUTION_LEVEL");

    ifdebug(1) {
	debug(1,"blind_loop_distribution", "Begin for %s\n", mod_name);
	pips_assert("Statements inconsistants...", statement_consistent_p(mod_stmt));
    }

    /* Loop distribution on the way up */
    /*
    gen_recurse(mod_stmt, instruction_domain,
		gen_true, blind_loop_distribution_rewrite);
    */

    /* Loop distribution on the way down */
    gen_recurse(mod_stmt, instruction_domain,
		blind_loop_distribution_filter, gen_null);

    /* Reorder the module because new statements have been generated. */
    module_reorder(mod_stmt);

    ifdebug(1)
	pips_assert("Statements inconsistants...", statement_consistent_p(mod_stmt));

    debug(1,"blind_loop_distribution", "End for %s\n", mod_name);

    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
  
    return TRUE;
}

bool
transformation_test(char * mod_name)  
{
    return blind_loop_distribution(mod_name);
}
