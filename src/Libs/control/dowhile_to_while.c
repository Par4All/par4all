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
			clone_context cc = make_clone_context(
					get_current_module_entity(),
					get_current_module_entity(),
					get_current_module_statement()
			);
			statement duplicated_statement = clone_statement( whileloop_body(wl),cc );
			free_clone_context(cc);
			new_statements=CONS(STATEMENT,duplicated_statement,new_statements);
			/* create new instruction sequence */
			instruction new_instr = make_instruction_sequence(make_sequence(new_statements));
			/* see how elegant is the patching ? */
			statement_instruction(stmt)=new_instr;
		}
	}
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
