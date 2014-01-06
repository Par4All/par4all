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
#include <stdlib.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"
#include "effects.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "resources.h"
#include "pipsdbm.h"

#define DEBUG_NAME "INSTRUCTION_SELECTION_DEBUG_LEVEL"

typedef struct 
{
	/* input */
	entity bmult, bplus, bminus, uminus;

	/* output */
	entity imaop, imsop;
	
	/* stats */
	int n_ima, n_ims;
}
inst_sel_ctx;

/* whether e is a call to op with len parameters
 * if ok, the list of arguments is returned.
 * if not, NIL is returned.
 */
static list /* of expression */ 
is_this_op(expression e, entity op, size_t len)
{
	if (expression_call_p(e))
	{
		call c = syntax_call(expression_syntax(e));
		list args = call_arguments(c);
		if (call_function(c)==op && gen_length(args)==len)
			return args;
	}
	return NIL;
}

static void update_call(call c, entity op, expression e, list le)
{
	expression 
		a = EXPRESSION(gen_nth(0, le)),
		b = EXPRESSION(gen_nth(1, le));

	/* ??? MEMORY LEAK */
	call_function(c) = op;
	call_arguments(c) = gen_make_list(expression_domain, a, b, e, NIL);
}

static void select_op_rwt(call c, inst_sel_ctx * ctx)
{
	entity fun = call_function(c);
	list /* of expression */ args = call_arguments(c);
	int nargs = gen_length(args);
	
	if (fun==ctx->bplus && nargs==2)
	{
		list /* of expression */ lm;

		lm = is_this_op(EXPRESSION(gen_nth(0, args)), ctx->bmult, 2);
		if (lm)
		{
			/*  a * b + c   ->   ima(a,b,c) */
			update_call(c, ctx->imaop, EXPRESSION(gen_nth(1, args)), lm);
			ctx->n_ima++;
		}
		else
		{
			lm = is_this_op(EXPRESSION(gen_nth(1, args)), ctx->bmult, 2);
			if (lm)
			{
				/*  a + b * c   ->   ima(b,c,a) */
				update_call(c, ctx->imaop, EXPRESSION(gen_nth(0, args)), lm);
				ctx->n_ima++;
			}
		}
		/*  a * b + -c  ->   ims(a,b,c) */
		/* -a + b * c   ->   ims(b,c,a) */
	}
	else if (fun==ctx->bminus && nargs==2)
	{
		list /* of expression */ lm;

		lm = is_this_op(EXPRESSION(gen_nth(0, args)), ctx->bmult, 2);
		if (lm)
		{
			/*  a * b - c   ->   ims(a,b,c) */
			update_call(c, ctx->imsop, EXPRESSION(gen_nth(1, args)), lm);
			ctx->n_ims++;
		}
		else
		{
			/*  a - b * c   ->  -ims(b,c,a) */
			/* -a - b * c   ->  -ima(b,c,a) */
			/*
			lm = is_this_op(EXPRESSION(gen_nth(1, args)), ctx->bmult, 2);
			if (lm)
			{
				update_call(c, ctx->imaop, EXPRESSION(gen_nth(0, args)), lm);
				ctx->n_ima++;
			}
			*/
		}
	}
	return;
}

bool instruction_selection(const char* module_name)
{
	inst_sel_ctx ctx;
	statement stat;

	debug_on(DEBUG_NAME);

	/* get data from pipsdbm
	 */
	set_current_module_entity(local_name_to_top_level_entity(module_name));

    set_current_module_statement((statement)
	    db_get_memory_resource(DBR_CODE, module_name, true));

	stat = get_current_module_statement();

	/* init gen_recurse context
	 */
	ctx.bmult = entity_intrinsic(MULTIPLY_OPERATOR_NAME);
	ctx.bplus = entity_intrinsic(PLUS_OPERATOR_NAME);
	ctx.bminus = entity_intrinsic(MINUS_OPERATOR_NAME);
	ctx.uminus = entity_intrinsic(UNARY_MINUS_OPERATOR_NAME);
	ctx.imaop = entity_intrinsic(IMA_OPERATOR_NAME);
	ctx.imsop = entity_intrinsic(IMS_OPERATOR_NAME);
	ctx.n_ima = 0;
	ctx.n_ims = 0;

	gen_context_multi_recurse(stat, &ctx,
				  call_domain, gen_true, select_op_rwt,
				  /* Do not optimize subscript expressions */
				  reference_domain, gen_false, gen_null,
				  NULL);

	/* store statement back to pipsdbm
	 */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);

    reset_current_module_entity();
    reset_current_module_statement();

    debug_off();

	return true;
}
