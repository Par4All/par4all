/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "control.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"

static
bool dowhile_to_while_walker(statement stmt)
{
	instruction instr = statement_instruction(stmt);
	if( instruction_whileloop_p(instr) )
	{
		whileloop wl = instruction_whileloop(instr);
		/* it's a do -while loop */
		if( evaluation_after_p(whileloop_evaluation(wl) ) )
		{
			/* do-while -> while-do */
			free_evaluation(whileloop_evaluation(wl));
			whileloop_evaluation(wl)=make_evaluation_before();
			/* push while-do instruction */
			list new_statements = CONS(STATEMENT,instruction_to_statement(instr),NIL);
			/* duplicate while-do statements and push it */
			statement duplicated_statement = make_empty_statement();
			clone_context cc = make_clone_context(
					get_current_module_entity(),
					get_current_module_entity(),
					duplicated_statement
			);
			instruction_block(statement_instruction(duplicated_statement)) = CONS(STATEMENT,clone_statement( whileloop_body(wl),cc ),NIL);
			free_clone_context(cc);
			new_statements=CONS(STATEMENT,duplicated_statement,new_statements);
			/* create new instruction sequence */
			instruction new_instr = make_instruction_sequence(make_sequence(new_statements));
			/* see how elegant is the patching ? */
			statement_instruction(stmt)=new_instr;
		}
	}
    return true;
}

bool
dowhile_to_while(char *module_name)
{
	/* prelude */
	set_current_module_entity( module_name_to_entity(module_name) );
	set_current_module_statement(
			(statement) db_get_memory_resource(DBR_CODE, module_name, TRUE)
			);

	/* transformation */
  	gen_recurse(get_current_module_statement(), statement_domain, gen_true, &dowhile_to_while_walker);

	/* postlude */
	module_reorder(get_current_module_statement());
	DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

	reset_current_module_statement();
	reset_current_module_entity();

	return true;
}
