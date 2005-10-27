/* $Id$ */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"

#define DEBUG_NAME "INSTRUCTION_SELECTION_DEBUG_LEVEL"

typedef struct 
{
	/* input */
	entity bplus, bminus, uminus;

	/* output */
	entity imaop, imsop;
	
	/* stats */
	int n_ima, n_ims;
}
inst_sel_ctx;

static void select_op_rwt(call c, inst_sel_ctx * ctx)
{
	entity fun = call_function(c);
	list /* of expression */ args = call_arguments(c);
	int nargs = gen_length(args);
	
	if (fun==ctx->bplus && nargs==2)
	{
		/*  a * b + c   ->   ima(a,b,c) */
		/*  a * b + -c  ->   ims(a,b,c) */
		/*  a + b * c   ->   ima(b,c,a) */
		/* -a + b * c   ->   ims(b,c,a) */

	}
	else if (fun==ctx->bminus && nargs==2)
	{
		/*  a * b - c   ->   ims(a,b,c) */
		/*  a - b * c   ->  -ims(b,c,a) */
		/* -a - b * c   ->  -ima(b,c,a) */

	}
	return;
}

bool instruction_selection(string module_name)
{
	inst_sel_ctx ctx;
	statement stat;

	debug_on(DEBUG_NAME);

	/* get data from pipsdbm
	 */
	set_current_module_entity(local_name_to_top_level_entity(module_name));

    set_current_module_statement((statement)
	    db_get_memory_resource(DBR_CODE, module_name, TRUE));

	stat = get_current_module_statement();

	/* init gen_recurse context
	 */
	ctx.bplus = entity_intrinsic(PLUS_OPERATOR_NAME);
	ctx.bminus = entity_intrinsic(MINUS_OPERATOR_NAME);
	ctx.uminus = entity_intrinsic(UNARY_MINUS_OPERATOR_NAME);
	ctx.imaop = entity_intrinsic(IMA_OPERATOR_NAME);
	ctx.imsop = entity_intrinsic(IMS_OPERATOR_NAME);
	ctx.n_ima = 0;
	ctx.n_ims = 0;

	gen_context_multi_recurse(stat, &ctx,
							  call_domain, gen_true, select_op_rwt,
							  NULL);

	/* store statement back to pipsdbm
	 */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);

    reset_current_module_entity();
    reset_current_module_statement();

    debug_off();

	return TRUE;
}
