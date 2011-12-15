/*
  $Id$

  Copyright 1989-2011 MINES ParisTech
  Copyright 2011 HPC Project

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
/*
 * This File contains the generic functions necessary for the computation of
 * live paths analyzes. Beatrice Creusillet.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "pipsdbm.h"
#include "resources.h"

#include "pointer_values.h"
#include "effects-generic.h"

/**************************** GENERIC INTERPROCEDURAL LIVE_IN PATHS ANALYSIS */

bool live_in_summary_paths_engine(const char* module_name)
{
  list l_glob = NIL, l_loc = NIL;
  statement module_stat;

  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true) );
  module_stat = get_current_module_statement();

  (*effects_computation_init_func)(module_name);

  set_live_in_paths((*db_get_live_in_paths_func)(module_name));

  l_loc = load_live_in_paths_list(module_stat);

  l_glob = (*effects_local_to_global_translation_op)(l_loc);

  (*db_put_live_in_summary_paths_func)(module_name, l_glob);

  reset_current_module_entity();
  reset_current_module_statement();
  reset_live_in_paths();

  (*effects_computation_reset_func)(module_name);

  return true;
}

/**************************** GENERIC INTERPROCEDURAL LIVE_OUT PATHS ANALYSIS */

typedef struct {
  entity current_callee;
  list l_current_paths;
} live_out_summary_engine_context;

static void
set_live_out_summary_engine_context(live_out_summary_engine_context *ctxt, entity callee)
{
  ctxt->l_current_paths = NIL;
}

static void
reset_live_out_summary_engine_context(live_out_summary_engine_context *ctxt)
{
  ctxt->current_callee = entity_undefined;
  ctxt->l_current_paths = NIL;
}

static void
update_live_out_summary_engine_context_paths(live_out_summary_engine_context *ctxt, list l_paths)
{
  pips_debug_effects(3, "adding paths\n", l_paths);
  ctxt->l_current_paths = (*effects_test_union_op)(ctxt->l_current_paths , l_paths,
					   effects_same_action_p);

  pips_debug_effects(3, "current paths\n", ctxt->l_current_paths);
}

static void
live_out_paths_from_call_site_to_callee(call c, live_out_summary_engine_context *ctxt)
{
    if (call_function(c) != ctxt->current_callee)
	return;

    statement current_stmt = (statement) gen_get_ancestor(statement_domain, c);

    /* there is a problem here because the context is in the store preceding the statement
       whereas live_out_paths hold after the statement.
       So I think I must apply the transformer to get the post-condition.
       That's ok for simple paths since they are store independent. BC.
    */
    transformer stmt_context = (*load_context_func)(current_stmt);

    /* There may also be a problem here in C, because calls may be
       anywhere inside an expression, and not solely as standalone
       statements. BC.
    */
    list l_out = load_live_out_paths_list(current_stmt);

    list l_paths = generic_effects_forward_translation(ctxt->current_callee,
						call_arguments(c), l_out,
						stmt_context);
    update_live_out_summary_engine_context_paths(ctxt, l_paths);
}

static bool
live_out_summary_paths_stmt_filter(statement s,
				   __attribute__((unused)) live_out_summary_engine_context *ctxt)
{
    pips_debug(1, "statement %03zd\n", statement_number(s));
    return(true);
}

static void
live_out_paths_from_caller_to_callee(entity caller, entity callee,
				     live_out_summary_engine_context *ctxt)
{
    const char *caller_name;
    statement caller_statement;

    reset_current_module_entity();
    set_current_module_entity(caller);
    caller_name = module_local_name(caller);
    pips_debug(2, "begin for caller: %s\n", caller_name);

    /* All we need to perform the translation */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, caller_name, true) );
    caller_statement = get_current_module_statement();

    (*effects_computation_init_func)(caller_name);

    set_live_out_paths( (*db_get_live_out_paths_func)(caller_name));

    ctxt->current_callee = callee;
    gen_context_multi_recurse(caller_statement, &ctxt,
		      statement_domain, live_out_summary_paths_stmt_filter, gen_null,
		      call_domain, live_out_paths_from_call_site_to_callee, gen_null,
		      NULL);

    reset_live_out_paths();
    reset_current_module_entity();
    set_current_module_entity(callee);
    reset_current_module_statement();

    (*effects_computation_reset_func)(caller_name);
}


bool live_out_summary_paths_engine(const char* module_name)
{
  /* Look for all call sites in the callers */
  callees callers = (callees) db_get_memory_resource(DBR_CALLERS,
						     module_name,
						     true);
  entity callee = module_name_to_entity(module_name);

  set_current_module_entity(callee);
  make_effects_private_current_context_stack();

  debug_on("LIVE_OUT_EFFECTS_DEBUG_LEVEL");
  ifdebug(1)
    {
      pips_debug(1, "begin for %s with %td callers\n",
		 module_name,
		 gen_length(callees_callees(callers)));
      FOREACH(STRING, caller_name, callees_callees(callers))
	{fprintf(stderr, "%s, ", caller_name);}
      fprintf(stderr, "\n");
    }

  live_out_summary_engine_context ctxt;
  set_live_out_summary_engine_context(&ctxt, callee);
  FOREACH(STRING, caller_name, callees_callees(callers))
    {
      entity caller = module_name_to_entity(caller_name);
      live_out_paths_from_caller_to_callee(caller, callee, &ctxt);
    }

  (*db_put_live_out_summary_paths_func)(module_name, ctxt.l_current_paths);

  ifdebug(1)
    {
      set_current_module_statement( (statement)
				    db_get_memory_resource(DBR_CODE,
							   module_local_name(callee),
							   true) );

      pips_debug(1, "live out paths for module %s:\n", module_name);
      (*effects_prettyprint_func)(ctxt.l_current_paths);
      pips_debug(1, "end\n");
      reset_current_module_statement();
    }

  reset_live_out_summary_engine_context(&ctxt);
  reset_current_module_entity();
  free_effects_private_current_context_stack();

  debug_off();
  return true;
}

/********************* GENERIC INTRAPROCEDURAL LIVE_IN AND LIVE_OUT PATHS ANALYSIS */


bool live_paths_engine(const char *module_name)
{
  return true;
}
