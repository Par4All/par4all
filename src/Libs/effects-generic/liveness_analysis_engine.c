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


typedef struct {
  bool memory_effects_only;
  bool memory_in_out_effects_only;
  effects_representation_val representation;
} live_paths_analysis_context;


static void init_live_paths_context(live_paths_analysis_context *ctxt,
				    effects_representation_val representation)
{
  ctxt->memory_effects_only = get_bool_property("MEMORY_EFFECTS_ONLY");
  ctxt->memory_in_out_effects_only = get_bool_property("MEMORY_IN_OUT_REGIONS_ONLY");
  ctxt->representation = representation;
}

static list __attribute__((unused)) convert_rw_effects(list lrw, live_paths_analysis_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    lrw = effects_store_effects(lrw);

  return lrw;
}


static void __attribute__((unused)) reset_converted_rw_effects(list *lrw, live_paths_analysis_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    gen_free_list(*lrw);
  *lrw = NIL;
}



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
       statements.  It may be better to call this function during the
       intraprocedural analysis of callers and not during a standalone
       phase which would have to deal with calls in sub-expressions,
       and thus also with store/pointers modifications.
       BC.
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

/**************** GENERIC INTRAPROCEDURAL LIVE_IN AND LIVE_OUT PATHS ANALYSIS */

static void
live_in_paths_of_statement(statement s,
			   live_paths_analysis_context *ctxt)
{
  /* live in paths may have already been computed,
     for instance for sequences
  */
  pips_debug(1,"begin for statement with ordering: %03zd and number: %03zd\n",
	     statement_ordering(s), statement_number(s));

  if ( !bound_live_in_paths_p(s) )
    {
      pips_debug(1, "not bound\n");
      /* First, we get the live out paths of the statement */
      list l_live_out = effects_dup(load_live_out_paths_list(s));

      /* Live_in(S) = (live_out(S) o T(S) - W(S)) U IN (S) */

      // first move the live_out effects of the statement in the store
      // before the statement
      transformer t = (*load_completed_transformer_func)(s);

      (*effects_transformer_composition_op)(l_live_out, t);

      // then remove the effects written by the statement
      // and add the in effects of the statement
      list l_write =  convert_rw_effects(load_rw_effects_list(s),ctxt);
      list l_in = load_in_effects_list(s);

      list l_live_in = (*effects_union_op)
	(effects_dup(l_in),
	 (*effects_sup_difference_op)(l_live_out, effects_dup(l_write),
				      r_w_combinable_p),
	 effects_same_action_p);

      store_live_in_paths_list(s, l_live_in);

      reset_converted_rw_effects(&l_write, ctxt);

    }
  pips_debug(1,"end\n");
  return;
}

static bool
live_paths_from_unstructured_to_nodes(unstructured unstr, live_paths_analysis_context *ctxt)
{
  return false;
}

static bool
live_paths_from_forloop_to_body(forloop l, live_paths_analysis_context *ctxt)
{
  return false;
}

static bool
live_paths_from_whileloop_to_body(whileloop l, live_paths_analysis_context *ctxt)
{
  return false;
}

static bool
live_paths_from_loop_to_body(loop l, live_paths_analysis_context *ctxt)
{
  return false;
}

static bool
live_out_paths_from_test_to_branches(test t, live_paths_analysis_context *ctxt)
{
  statement
    current_stmt = (statement) gen_get_ancestor(statement_domain, t);

  pips_debug(1,"begin\n");

  /* First, we get the live out paths of the statement corresponding to the test
   */
  list l_live_out_test = load_live_out_paths_list(current_stmt);

  /* The live out paths for each branch are the live_out_paths
     of the test
  */
  store_live_out_paths_list(test_true(t), effects_dup(l_live_out_test));
  store_live_out_paths_list(test_false(t), effects_dup(l_live_out_test));

  /* The live in paths of the test are computed with  the surrounding
     sequence live paths computation.
  */
  pips_debug(1,"end\n");

  return true;
}

static bool
live_paths_from_block_to_statements(sequence seq,
				    live_paths_analysis_context *ctxt)
{
  statement seq_stmt = (statement) gen_get_ancestor(statement_domain, seq);
  list l_stmt = sequence_statements(seq);

  pips_debug(1, "begin\n");

  if (ENDP(l_stmt))
    {
      if (get_bool_property("WARN_ABOUT_EMPTY_SEQUENCES"))
	pips_user_warning("empty sequence\n");
      store_live_in_paths_list(seq_stmt, NIL);
    }
  else
    {
      list l_rev_stmt = NIL; /* reversed version of the sequence to avoid recursion*/
      FOREACH(STATEMENT, stmt, l_stmt)
	{
	  l_rev_stmt = CONS(STATEMENT, stmt, l_rev_stmt);
	}

      list l_next_stmt_live_in = load_live_out_paths_list(seq_stmt);
      FOREACH(STATEMENT, stmt, l_rev_stmt)
	{
	  /* the live out paths of the current statement are the live in paths
	     of the next statement (or the live out of the while block
	     if last statement)
	  */
	  list l_live_out = l_next_stmt_live_in;
	  store_live_out_paths_list(stmt, effects_dup(l_live_out));

	  /* Now let us compute the live in paths */
	  live_in_paths_of_statement(stmt, ctxt);
	  l_next_stmt_live_in = load_live_in_paths_list(stmt);
	}

      /* The block live in paths are the live in paths of the first statement
         which is the last visited statement
      */
      store_live_in_paths_list(seq_stmt, effects_dup(l_next_stmt_live_in));

      gen_free_list(l_rev_stmt);
    }
  pips_debug(1, "end\n");

 return true;
}

static bool
__attribute__ ((__unused__)) live_paths_from_instruction_to_expression(instruction instr,
					  live_paths_analysis_context *ctxt)
{
  if (instruction_expression_p(instr))
    {
    }
  return false;
}


static void
live_paths_of_module_statement(statement stmt, live_paths_analysis_context *ctxt)
{

  gen_context_multi_recurse
    (stmt, ctxt,
     statement_domain, gen_true, live_in_paths_of_statement,
     //instruction_domain, live_paths_from_instruction_to_expression, gen_null,
     /* for expression instructions solely, equivalent to gen_true otherwise */
     sequence_domain, live_paths_from_block_to_statements, gen_null,
     test_domain, live_out_paths_from_test_to_branches, gen_null,
     loop_domain, live_paths_from_loop_to_body, gen_null,
     whileloop_domain, live_paths_from_whileloop_to_body, gen_null,
     forloop_domain, live_paths_from_forloop_to_body, gen_null,
     unstructured_domain, live_paths_from_unstructured_to_nodes, gen_null,

     /* Stop on these nodes: */
     call_domain, gen_false, gen_null, /* calls are treated in another phase*/
     expression_domain, gen_false, gen_null,
     NULL);

  return;
}

bool live_paths_engine(const char *module_name, effects_representation_val representation)
{
  debug_on("LIVE_PATHS_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  set_current_module_entity(module_name_to_entity(module_name));

  /* Get the code of the module. */
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true) );
  statement module_statement = get_current_module_statement();



  make_effects_private_current_context_stack();
  (*effects_computation_init_func)(module_name);

  /* Get the various effects and in_effects of the module. */

  set_rw_effects((*db_get_rw_effects_func)(module_name));
  set_invariant_rw_effects((*db_get_invariant_rw_effects_func)(module_name));

  set_cumulated_in_effects((*db_get_cumulated_in_effects_func)(module_name));
  set_in_effects((*db_get_in_effects_func)(module_name));
  set_invariant_in_effects((*db_get_invariant_in_effects_func)(module_name));


  /* initialise the map for live_out and live_int paths */
  init_live_in_paths();
  init_live_out_paths();

  /* Get the live_out_summary_paths of the current module */
  list l_sum_live_out = (*db_get_live_out_summary_paths_func)(module_name);
  /* And stores them as the live out paths of the module statement */
  store_live_out_paths_list(module_statement, effects_dup(l_sum_live_out));

  /* Compute the live paths of the module. */
  live_paths_analysis_context ctxt;
  init_live_paths_context(&ctxt, representation);
  live_paths_of_module_statement(module_statement, &ctxt);

  pips_debug(1, "end\n");

  (*db_put_live_out_paths_func)(module_name, get_live_out_paths());
  (*db_put_live_in_paths_func)(module_name, get_live_in_paths());

  reset_rw_effects();
  reset_invariant_rw_effects();
  reset_in_effects();
  reset_cumulated_in_effects();
  reset_invariant_in_effects();
  reset_live_in_paths();
  reset_live_out_paths();
  (*effects_computation_reset_func)(module_name);

  free_effects_private_current_context_stack();

  reset_current_module_entity();
  reset_current_module_statement();
  pips_debug(1, "end\n");

  debug_off();
  return true;
}
