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

#include <stdio.h> 

#include "genC.h"
#include "linear.h"
#include "resources.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
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
static bool blind_loop_distribution(char * mod_name)  
{
  /* get code from dbm. */
  statement mod_stmt = (statement) 
    db_get_memory_resource(DBR_CODE, mod_name, true);
  
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
  return true; /* everything was fine. */
}

/* apply a transformation on mod_name.
   called automatically by pipsmake.
 */
bool transformation_test(char * mod_name)  
{
    return blind_loop_distribution(mod_name);
}
