/*
  $Id$

  $Log: transformation_test.c,v $
  Revision 1.3  1998/12/03 15:52:01  coelho
  flattening added.

  Revision 1.2  1998/12/03 14:58:15  coelho
  cosmetics.

*/

#include <stdio.h> 

#include "genC.h"
#include "linear.h"
#include "resources.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "pipsdbm.h"
#include "control.h"

/* blindly distribute intruction l if it is a loop.
 */
static void blind_loop_distribute(instruction l)
{
  if(instruction_loop_p(l)) 
  {
    instruction b = statement_instruction(loop_body(instruction_loop(l)));
    flatten_block_if_necessary(b); /* avoid sequences of sequences. */

    if(instruction_block_p(b) && gen_length(instruction_block(b)) > 1) 
    {
      list /* of statements */ lls = NIL, ls = instruction_block(b);

      loop_body(instruction_loop(l)) = statement_undefined; /* unlink body */
      
      MAP(STATEMENT, s, {
	instruction nli = copy_instruction(l);
	loop_body(instruction_loop(nli)) = s;
	lls = gen_nconc(lls, 
		    CONS(STATEMENT, instruction_to_statement(nli), NIL));
      }, ls);
      
      free_loop(instruction_loop(l)); /* drop old loop. */
      instruction_tag(l) = is_instruction_sequence; /* new sequence */
      instruction_sequence(l) = make_sequence(lls);
    }
  }
}

/* distribute any loop in module mod_name.
   implemented top-down. could be done bottom-up.
 */
bool blind_loop_distribution(char * mod_name)  
{
  /* get code from dbm. */
  statement mod_stmt = (statement) 
    db_get_memory_resource(DBR_CODE, mod_name, TRUE);
  
  debug_on("BLIND_LOOP_DISTRIBUTION_LEVEL");
  
  pips_debug(1, "begin for %s\n", mod_name);
  pips_assert("statement is consistent", statement_consistent_p(mod_stmt));
  
  /* BOTTOM-UP. could be implemented TOP-DOWN. */
  gen_recurse(mod_stmt, 
	      instruction_domain, gen_true, blind_loop_distribute);

  /* Reorder the module because new statements have been generated. */
  module_reorder(mod_stmt);
  
  pips_assert("statement is consistent", statement_consistent_p(mod_stmt));
  pips_debug(1, "end for %s\n", mod_name);
  
  debug_off();

  /* return code to DBM. */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
  return TRUE; /* everything was fine. */
}

/* apply a transformation on mod_name.
   called automatically by pipsmake.
 */
bool transformation_test(char * mod_name)  
{
    return blind_loop_distribution(mod_name);
}
