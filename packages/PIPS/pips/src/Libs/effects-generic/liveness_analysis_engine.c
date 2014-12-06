/*
  $Id$

  Copyright 1989-2014 MINES ParisTech
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
  ctxt->memory_in_out_effects_only = get_bool_property("MEMORY_IN_OUT_EFFECTS_ONLY");
  if (!  ctxt->memory_effects_only)
    // if the MEMORY_EFFECTS_ONLY is FALSE on input,
    // we have to temporarily set MEMORY_EFFECTS_ONLY to
    // TRUE so that functions computing R/W effects
    // do not compute non memory effects which would have to be filtered.
    set_bool_property("MEMORY_EFFECTS_ONLY", true);
  ctxt->representation = representation;
}

static void reset_live_paths_context(live_paths_analysis_context  *ctxt)
{
  if (!  ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    // if the MEMORY_EFFECTS_ONLY is FALSE on input, and
    // we did not want to compute IN non-memory effects,
    // we have to reset MEMORY_EFFECTS_ONLY to FALSE.
    set_bool_property("MEMORY_EFFECTS_ONLY", false);
}


static list convert_rw_effects(list lrw, live_paths_analysis_context *ctxt)
{
  if (!ctxt->memory_effects_only)
    lrw = effects_store_effects(lrw);

  return lrw;
}


static void  reset_converted_rw_effects(list *lrw, live_paths_analysis_context *ctxt)
{
  if (!ctxt->memory_effects_only)
    gen_free_list(*lrw);
  *lrw = NIL;
}

static list convert_in_effects(list lin, live_paths_analysis_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && !ctxt->memory_in_out_effects_only)
    lin = effects_store_effects(lin);

  return lin;
}


static void reset_converted_in_effects(list *lin, live_paths_analysis_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && !ctxt->memory_in_out_effects_only)
    gen_free_list(*lin);
  *lin = NIL;
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

  // functions that can be pointed by effects_computation_init_func:
  // effects_computation_no_init
  // init_convex_in_out_regions
  // init_convex_rw_regions
  (*effects_computation_init_func)(module_name);

  set_live_in_paths((*db_get_live_in_paths_func)(module_name));

  l_loc = load_live_in_paths_list(module_stat);

  // functions that can be pointed by effects_local_to_global_translation_op:
  // effects_dynamic_elim
  // regions_dynamic_elim
  l_glob = (*effects_local_to_global_translation_op)(l_loc);

  (*db_put_live_in_summary_paths_func)(module_name, l_glob);

  reset_current_module_entity();
  reset_current_module_statement();
  reset_live_in_paths();

  // functions that can be pointed by effects_computation_reset_func:
  // effects_computation_no_reset
  // reset_convex_in_out_regions
  // reset_convex_rw_regions
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
  ctxt->current_callee = callee;
  ctxt->l_current_paths = list_undefined;
}

static void
reset_live_out_summary_engine_context(live_out_summary_engine_context *ctxt)
{
  ctxt->current_callee = entity_undefined;
  ctxt->l_current_paths = list_undefined;
}

static void
update_live_out_summary_engine_context_paths(live_out_summary_engine_context *ctxt, list l_paths)
{
  pips_debug_effects(3, "adding paths\n", l_paths);
  if (list_undefined_p(ctxt->l_current_paths))
    ctxt->l_current_paths = l_paths;
  else
    // functions that can be pointed by effects_test_union_op:
    // EffectsMayUnion
    // RegionsMayUnion
    // ReferenceTestUnion
    ctxt->l_current_paths = (*effects_test_union_op)(ctxt->l_current_paths , l_paths,
        effects_same_action_p);

  pips_debug_effects(3, "current paths\n", ctxt->l_current_paths);
}

static bool
live_out_paths_from_call_site_to_callee(call c, live_out_summary_engine_context *ctxt)
{
  pips_debug(1, "considering call to %s\n", entity_name(call_function(c)));
  pips_debug(2, "current context callee: %s\n",  entity_name(ctxt->current_callee));

  if (call_function(c) != ctxt->current_callee)
  {
    return (true);
  }
  pips_debug(1, "good candidate\n");

  statement current_stmt = (statement) gen_get_ancestor(statement_domain, c);

  /* there is a problem here because the context is in the store preceding the statement
   * whereas live_out_paths hold after the statement.
   * So I think I must apply the transformer to get the post-condition.
   * That's ok for simple paths since they are store independent. BC.
   */
  transformer stmt_context = (*load_context_func)(current_stmt);

  /* There may also be a problem here in C, because calls may be
   * anywhere inside an expression, and not solely as standalone
   * statements.  It may be better to call this function during the
   * intraprocedural analysis of callers and not during a standalone
   * phase which would have to deal with calls in sub-expressions,
   * and thus also with store/pointers modifications.
   * BC.
   */
  list l_out = load_live_out_paths_list(current_stmt);

  pips_debug_effects(3, "current statement live_out_paths:\n", l_out);
  list l_paths = generic_effects_forward_translation(ctxt->current_callee,
      call_arguments(c), l_out,
      stmt_context);
  update_live_out_summary_engine_context_paths(ctxt, l_paths);
  return true;
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
  pips_debug(2, "begin for caller: %s, and callee: %s\n", caller_name,
      entity_name(callee));
  pips_debug(2, "current context callee: %s\n",  entity_name(ctxt->current_callee));
  /* All we need to perform the translation */
  set_current_module_statement( (statement)
      db_get_memory_resource(DBR_CODE, caller_name, true) );
  caller_statement = get_current_module_statement();

  // functions that can be pointed by effects_computation_init_func:
  // effects_computation_no_init
  // init_convex_in_out_regions
  // init_convex_rw_regions
  (*effects_computation_init_func)(caller_name);

  set_live_out_paths( (*db_get_live_out_paths_func)(caller_name));

  //ctxt->current_callee = callee;
  gen_context_multi_recurse(caller_statement, ctxt,
      statement_domain, live_out_summary_paths_stmt_filter, gen_null,
      call_domain, live_out_paths_from_call_site_to_callee, gen_null,
      NULL);

  reset_live_out_paths();
  reset_current_module_entity();
  set_current_module_entity(callee);
  reset_current_module_statement();

  // functions that can be pointed by effects_computation_reset_func:
  // effects_computation_no_reset
  // reset_convex_in_out_regions
  // reset_convex_rw_regions
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
  debug_on("LIVE_PATHS_DEBUG_LEVEL");
  FOREACH(STRING, caller_name, callees_callees(callers))
    {
      entity caller = module_name_to_entity(caller_name);
      live_out_paths_from_caller_to_callee(caller, callee, &ctxt);
    }
  debug_off();
  if (list_undefined_p(ctxt.l_current_paths))
    ctxt.l_current_paths = NIL;
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

  return true;
}

/**************** GENERIC INTRAPROCEDURAL LIVE_IN AND LIVE_OUT PATHS ANALYSIS */

static void
live_in_paths_of_statement(statement s,
    live_paths_analysis_context *ctxt)
{
  /* live in paths may have already been computed,
   * for instance for sequences
   */
  pips_debug(1,"begin for statement with ordering: %03zd and number: %03zd\n",
      statement_ordering(s), statement_number(s));

  if ( !bound_live_in_paths_p(s) )
  {
    pips_debug(1, "not bound\n");
    /* First, we get the live out paths of the statement */
    list l_live_out = effects_dup(load_live_out_paths_list(s));

    /* Live_in(S) = (live_out(S) o T(S) - W(S)) U IN (S) */

    /* first take care of declarations */
    if (c_module_p(get_current_module_entity()) &&
        (declaration_statement_p(s) ))
    {
      FOREACH(ENTITY, decl, statement_declarations(s)) {
        storage decl_storage = entity_storage(decl);
        if (storage_ram_p(decl_storage)
            // keep paths on static variables
            && !static_area_p(ram_section(storage_ram(decl_storage))))
          l_live_out = filter_effects_with_declaration(l_live_out, decl);
      }
    }

    // then move the live_out effects of the statement in the store
    // before the statement
    transformer t = (*load_completed_transformer_func)(s);

    // functions that can be pointed by effects_transformer_composition_op:
    // effects_composition_with_transformer_nop
    // effects_undefined_composition_with_transformer
    // convex_regions_transformer_compose
    // simple_effects_composition_with_effect_transformer
    (*effects_transformer_composition_op)(l_live_out, t);

    // next, remove the effects written by the statement
    // and add the in effects of the statement
    list l_rw =  convert_rw_effects(load_rw_effects_list(s),ctxt);
    list l_write = effects_write_effects_dup(l_rw);
    pips_debug_effects(5, "write effects:\n", l_write);

    list l_in = convert_in_effects(load_in_effects_list(s), ctxt);
    pips_debug_effects(5, "in effects:\n", l_in);

    // functions that can be pointed by effects_union_op:
    // ProperEffectsMustUnion
    // RegionsMustUnion
    // ReferenceUnion
    // EffectsMustUnion
    // functions that can be pointed by effects_sup_difference_op:
    // effects_undefined_binary_operator
    // RegionsSupDifference
    // EffectsSupDifference
    list l_live_in = (*effects_union_op)(
        effects_dup(l_in),
        (*effects_sup_difference_op)(l_live_out, l_write,
            r_w_combinable_p),
            effects_same_action_p);
    pips_debug_effects(5, "resulting live in paths:\n", l_live_in);


    store_live_in_paths_list(s, l_live_in);

    reset_converted_rw_effects(&l_rw, ctxt);
    reset_converted_in_effects(&l_in, ctxt);

  }
  pips_debug(1,"end\n");
  return;
}

static bool
live_out_paths_from_unstructured_to_nodes(unstructured unstr,
    live_paths_analysis_context *ctxt)
{
  pips_debug(1,"begin\n");

  /* Adopt a very conservative strategy:
   * live_out(node)=  live_out(unstructured)*may U (U_{n \in nodes} IN(nodes))
   * If convex paths were to be computed, transformers should be taken into account.
   */
  /* The exit node is a particular case: its live_out paths are the live_out paths
   * of the unstructured.
   */

  statement
  current_stmt = (statement) gen_get_ancestor(statement_domain, unstr);
  control exit_ctrl = unstructured_exit( unstr );
  statement exit_node = control_statement(exit_ctrl);

  list l_live_out_unstr = effects_dup(load_live_out_paths_list(current_stmt));
  pips_debug_effects(3, "live out paths of whole unstructured:\n", l_live_out_unstr);
  store_live_out_paths_list(exit_node, l_live_out_unstr);

  if(control_predecessors(exit_ctrl) == NIL && control_successors(exit_ctrl) == NIL)
  {
    /* there is only one statement in u; */
    pips_debug(6, "unique node\n");
  }
  else
  {
    l_live_out_unstr = effects_dup(l_live_out_unstr);
    effects_to_may_effects(l_live_out_unstr); /* It may be executed... */

    list l_in = NIL;

    list l_nodes = NIL;
    UNSTRUCTURED_CONTROL_MAP
    (ctrl, unstr, l_nodes,
        {
            statement node_stmt = control_statement(ctrl);
            pips_debug(3, "visiting node statement with ordering: "
                "%03zd and number: %03zd\n",
                statement_ordering(node_stmt),
                statement_number(node_stmt));

            list l_in_node = convert_in_effects(load_in_effects_list(node_stmt), ctxt);
            list l_in_node_may = effects_dup(l_in_node);
            effects_to_may_effects(l_in_node_may);

            // functions that can be pointed by effects_union_op:
            // ProperEffectsMustUnion
            // RegionsMustUnion
            // ReferenceUnion
            // EffectsMustUnion
            l_in = (*effects_union_op)(l_in_node_may, l_in,
                effects_same_action_p);
            pips_debug_effects(5, "current in effects:\n", l_in);
            reset_converted_in_effects(&l_in_node, ctxt);
        });

    // functions that can be pointed by effects_union_op:
    // ProperEffectsMustUnion
    // RegionsMustUnion
    // ReferenceUnion
    // EffectsMustUnion
    list l_live_out_nodes = (*effects_union_op)(l_in, l_live_out_unstr,
        effects_same_action_p);
    pips_debug_effects(3, "l_live_out_nodes:\n", l_live_out_nodes);

    FOREACH(CONTROL, ctrl, l_nodes)
    {

      statement node_stmt = control_statement(ctrl);

      /* be sure live_out paths are not stored twice, in particular
	     for the exit node
       */
      if ( !bound_live_out_paths_p(node_stmt) )
      {
        pips_debug(3, "storing live out paths for node statement "
            "with ordering: %03zd and number: %03zd\n",
            statement_ordering(node_stmt),
            statement_number(node_stmt));
        store_live_out_paths_list(node_stmt, effects_dup(l_live_out_nodes));
      }
    }
  }

  pips_debug(1,"end\n");

  return true;
}

static bool
live_out_paths_from_forloop_to_body(forloop l,
    live_paths_analysis_context *ctxt)
{
  statement
  current_stmt = (statement) gen_get_ancestor(statement_domain, l);
  statement body = forloop_body(l);

  pips_debug(1,"begin\n");

  /* The live out paths of an iteration are the paths that
   *   - belong to the live in paths of the next iteration if it exists,
   *     that is to say if the condition evaluates to true;
   *     the write effects of the condition evaluation mask the
   *     live in paths of the next iteration; and the in effects
   *     of the condition evaluation must be added.
   *   - or belong to the live out of the whole loop if the next iteration
   *     does not exist. However, the condition is evaluated,
   *     and its write effects mask the live out paths at the end of the
   *     whole loop.
   */

  /* First, we get the live out paths of the whole loop
   */
  list l_live_out_loop = effects_dup(load_live_out_paths_list(current_stmt));
  pips_debug_effects(3, "live out paths of whole loop:\n", l_live_out_loop);
  effects_to_may_effects(l_live_out_loop); /* It may be executed... */

  /* ...or another next iteration of the loop body */
  list l_in_next_iter = convert_in_effects(load_invariant_in_effects_list(body), ctxt);
  list l_in_next_iter_may = effects_dup(l_in_next_iter);
  effects_to_may_effects(l_in_next_iter_may);
  pips_debug_effects(3, "(may) in effects of next iterations:\n",
      l_in_next_iter_may);
  reset_converted_in_effects(&l_in_next_iter, ctxt);

  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  list l_live_out_after_cond =
      (*effects_union_op)(l_live_out_loop,
          l_in_next_iter_may,
          effects_same_action_p);

  pips_debug_effects(3, "live out paths after header evaluation:\n",
      l_live_out_after_cond);

  /* Take into account the effects of the header */
  list l_prop_cond = convert_rw_effects(load_proper_rw_effects_list(current_stmt),
      ctxt);

  list l_write_cond =
      proper_to_summary_effects(effects_write_effects_dup(l_prop_cond));
  pips_debug_effects(3, "write effects of condition:\n", l_write_cond);


  /* We don't have the in effects of the header evaluation
   * we safely approximate them by its may read proper effects
   * if there are write effects during its evaluation
   * We could be much more precise here by distinguishing
   * the three different components and re-computing their effects.
   * However, for loops are generally converted to loops or while loops
   * by the controlizer, so I leave this as it is for the time being.
   */
  list l_in_cond =
      proper_to_summary_effects(effects_read_effects_dup(l_prop_cond));
  if (!ENDP(l_write_cond)) effects_to_may_effects(l_in_cond);
  pips_debug_effects(3, "approximation of in effects of condition:\n", l_in_cond);

  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  // functions that can be pointed by effects_sup_difference_op:
  // effects_undefined_binary_operator
  // RegionsSupDifference
  // EffectsSupDifference
  list l_live_out_body =
      (*effects_union_op)(l_in_cond,
          (*effects_sup_difference_op)(l_live_out_after_cond,
              l_write_cond,
              r_w_combinable_p),
              effects_same_action_p);


  reset_converted_rw_effects(&l_prop_cond, ctxt);

  pips_debug_effects(3, "live out paths of loop body:\n", l_live_out_body);

  store_live_out_paths_list(body, l_live_out_body);

  pips_debug(1,"end\n");

  return true;
}

static bool
live_out_paths_from_whileloop_to_body(whileloop l, live_paths_analysis_context *ctxt)
{
  statement
  current_stmt = (statement) gen_get_ancestor(statement_domain, l);
  statement body = whileloop_body(l);

  pips_debug(1,"begin\n");

  /* The live out paths of an iteration are the paths that
   *   - belong to the live in paths of the next iteration if it exists,
   *     that is to say if the condition evaluates to true;
   *     the write effects of the condition evaluation mask the
   *     live in paths of the next iteration; and the in effects
   *     of the condition evaluation must be added.
   *   - or belong to the live out of the whole loop if the next iteration
   *     does not exist. However, the condition is evaluated,
   *     and its write effects mask the live out paths at the end of the
   *     whole loop.
   */

  /* First, we get the live out paths of the whole loop
   */
  list l_live_out_loop = effects_dup(load_live_out_paths_list(current_stmt));
  pips_debug_effects(3, "live out paths of whole loop:\n", l_live_out_loop);
  effects_to_may_effects(l_live_out_loop); /* It may be executed... */

  /* ...or another next iteration of the loop body */
  list l_in_next_iter = convert_in_effects(load_invariant_in_effects_list(body), ctxt);
  list l_in_next_iter_may = effects_dup(l_in_next_iter);
  effects_to_may_effects(l_in_next_iter_may);
  pips_debug_effects(3, "(may) in effects of next iterations:\n",
      l_in_next_iter_may);
  reset_converted_in_effects(&l_in_next_iter, ctxt);

  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  list l_live_out_after_cond =
      (*effects_union_op)(l_live_out_loop,
          l_in_next_iter_may,
          effects_same_action_p);

  pips_debug_effects(3, "live out paths after any condition evaluation:\n",
      l_live_out_after_cond);

  /* Take into account the effects of the condition */
  list l_prop_cond = convert_rw_effects(load_proper_rw_effects_list(current_stmt),
      ctxt);

  list l_write_cond =
      proper_to_summary_effects(effects_write_effects_dup(l_prop_cond));
  pips_debug_effects(3, "write effects of condition:\n", l_write_cond);


  /* We don't have the in effects of the condition evaluation
   * we safely approximate them by its may read proper effects
   * if there are write effects during its evaluation
   */
  list l_in_cond =
      proper_to_summary_effects(effects_read_effects_dup(l_prop_cond));
  if (!ENDP(l_write_cond)) effects_to_may_effects(l_in_cond);
  pips_debug_effects(3, "approximation of in effects of condition:\n", l_in_cond);

  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  // functions that can be pointed by effects_sup_difference_op:
  // effects_undefined_binary_operator
  // RegionsSupDifference
  // EffectsSupDifference
  list l_live_out_body =
      (*effects_union_op)(l_in_cond,
          (*effects_sup_difference_op)(l_live_out_after_cond,
              l_write_cond,
              r_w_combinable_p),
              effects_same_action_p);


  reset_converted_rw_effects(&l_prop_cond, ctxt);

  pips_debug_effects(3, "live out paths of loop body:\n", l_live_out_body);

  store_live_out_paths_list(body, l_live_out_body);

  pips_debug(1,"end\n");

  return true;
}

static void
live_in_paths_of_whileloop(whileloop l, live_paths_analysis_context *ctxt)
{
  statement
  current_stmt = (statement) gen_get_ancestor(statement_domain, l);
  bool before_p = evaluation_before_p(whileloop_evaluation(l));
  statement body = whileloop_body(l);

  list l_live_in_loop = NIL;

  pips_debug(1,"begin\n");

  /* If the loop belongs to a sequence of statements, which is
   * generally the case, its live in paths have already been computed.
   * I don't yet know if they may be much more precisely computed here.
   * For performance reasons, I don't try to recompute them if they
   * are already available.
   * see also the comment inside the else branch of if (!before_p)
   * about the way of computing live in paths.
   * The code below is likely to be seldom executed;
   * however, it has been tested on small cases.
   */
  if ( !bound_live_in_paths_p(current_stmt) )
  {
    list l_live_in_body = effects_dup(load_live_in_paths_list(body));
    pips_debug_effects(3, "live in paths of loop body:\n", l_live_in_body);

    if (!before_p)
    {
      pips_debug(1, "do while loop: live in paths are those of the body\n");
      l_live_in_loop = l_live_in_body;
    }
    else
    {
      /* the live in paths of the loop are either those of the
       * first iteration if there is one, or the live out paths
       * of the whole loop is there is no iteration;
       * we must remove the write effects and add the in effects
       * of the condition evaluation
       */

      /* As we currently only compute simple live paths,
       * I wonder if there is a difference with the live out paths
       * of the loop body...
       */

      /* Get the live out paths of the whole loop
       */
      list l_live_out_loop = effects_dup(load_live_out_paths_list(current_stmt));
      pips_debug_effects(3, "live out paths of whole loop:\n", l_live_out_loop);
      effects_to_may_effects(l_live_out_loop); /* It may be executed... */

      /* ...or the first iteration of the loop body */
      effects_to_may_effects(l_live_in_body);

      // functions that can be pointed by effects_union_op:
      // ProperEffectsMustUnion
      // RegionsMustUnion
      // ReferenceUnion
      // EffectsMustUnion
      list l_live_out_after_cond =
          (*effects_union_op)(l_live_out_loop,
              l_live_in_body,
              effects_same_action_p);

      pips_debug_effects(3, "live out paths after first condition evaluation:\n",
          l_live_out_after_cond);

      /* Take into account the effects of the condition */
      list l_prop_cond =
          convert_rw_effects(load_proper_rw_effects_list(current_stmt), ctxt);

      list l_write_cond =
          proper_to_summary_effects(effects_write_effects_dup(l_prop_cond));
      pips_debug_effects(3, "write effects of condition:\n", l_write_cond);

      /* We don't have the in effects of the condition evaluation
       * we safely approximate them by its may read proper effects
       * if there are write effects during its evaluation
       */
      list l_in_cond =
          proper_to_summary_effects(effects_read_effects_dup(l_prop_cond));
      if (!ENDP(l_write_cond)) effects_to_may_effects(l_in_cond);
      pips_debug_effects(3, "approximation of in effects of condition:\n",
          l_in_cond);

      // functions that can be pointed by effects_union_op:
      // ProperEffectsMustUnion
      // RegionsMustUnion
      // ReferenceUnion
      // EffectsMustUnion
      // functions that can be pointed by effects_sup_difference_op:
      // effects_undefined_binary_operator
      // RegionsSupDifference
      // EffectsSupDifference
      l_live_in_loop =
          (*effects_union_op)(l_in_cond,
              (*effects_sup_difference_op)(l_live_out_after_cond,
                  l_write_cond,
                  r_w_combinable_p),
                  effects_same_action_p);

      reset_converted_rw_effects(&l_prop_cond, ctxt);
    }

    pips_debug_effects(3, "live in paths of loop:\n", l_live_in_loop);

    store_live_in_paths_list(current_stmt, l_live_in_loop);
  }
  else
  {
    pips_debug(1, "already bound\n");
  }

  pips_debug(1,"end\n");

  return;
}


static bool
live_out_paths_from_loop_to_body(loop l, live_paths_analysis_context *ctxt)
{
  statement
  current_stmt = (statement) gen_get_ancestor(statement_domain, l);
  statement
  body = loop_body(l);
  entity
  index = loop_index(l);

  pips_debug(1,"begin\n");

  /* First, we get the live out paths of the statement corresponding to the
   * whole loop
   */
  list l_live_out_loop = effects_dup(load_live_out_paths_list(current_stmt));
  pips_debug_effects(3, "live out paths of whole loop:\n", l_live_out_loop);

  list l_rw_body = convert_rw_effects(load_rw_effects_list(body),
      ctxt);
  list l_w_body = effects_write_effects_dup(l_rw_body);
  reset_converted_rw_effects(&l_rw_body, ctxt);

  if (!loop_executed_at_least_once_p(l))
    effects_to_may_effects(l_w_body);
  // functions that can be pointed by effects_sup_difference_op:
  // effects_undefined_binary_operator
  // RegionsSupDifference
  // EffectsSupDifference
  l_live_out_loop = (*effects_sup_difference_op)(l_live_out_loop, l_w_body,
      r_w_combinable_p);
  pips_debug_effects(3, "live out paths of an iteration due to live out paths of whole loop\n", l_live_out_loop);


  /* We then compute the live out paths of the loop body */
  /* The live out paths of an iteration are the paths that
   *   - belong to the in effects of the next iterations if there exist some,
   *     except for the loop locals if the loop is parallel;
   *   - or belong to the live out of the whole loop if the next iteration
   *     does not exist.
   *   - to these, for the C language, we must add the read effects resulting
   *     from the evaluation of the incremental expression and of the upper
   *     bound. This includes an exact path for the loop index.
   */
  list l_masked_vars = NIL;
  if(loop_parallel_p(l))
    l_masked_vars = loop_locals(l);

  list l_inv_in_body = convert_in_effects(load_invariant_in_effects_list(body), ctxt);
  list l_inv_in_body_may =
      effects_dup_without_variables(l_inv_in_body, l_masked_vars);
  effects_to_may_effects(l_inv_in_body_may);
  pips_debug_effects(3, "(may) live in paths of an iteration:\n", l_inv_in_body_may);

  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  list l_live_out_body = (*effects_union_op)(l_live_out_loop,
      l_inv_in_body_may,
      effects_same_action_p);
  reset_converted_in_effects(&l_inv_in_body, ctxt);

  list l_prop_header = convert_rw_effects(load_proper_rw_effects_list(current_stmt),
      ctxt);
  list l_live_in_header =
      proper_to_summary_effects(effects_read_effects_dup(l_prop_header));
  reset_converted_rw_effects(&l_prop_header, ctxt);
  // functions that can be pointed by reference_to_effect_func:
  // reference_to_simple_effect
  // reference_to_convex_region
  // reference_to_reference_effect
  list l_index = CONS(EFFECT,
      (*reference_to_effect_func)(make_reference(index, NIL), make_action_read_memory(), false),
      NIL);
  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  l_live_in_header = (*effects_union_op)(l_index, l_live_in_header, effects_same_action_p);

  pips_debug_effects(3, "live in paths of loop header:\n", l_live_in_header);

  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  l_live_out_body = (*effects_union_op)(l_live_out_body,
      l_live_in_header,
      effects_same_action_p);

  pips_debug_effects(3, "live out paths of loop body:\n", l_live_out_body);

  store_live_out_paths_list(body, l_live_out_body);

  pips_debug(1,"end\n");
  return true;
}

static void
live_in_paths_of_loop(loop l, live_paths_analysis_context *ctxt)
{
  /* The live in paths of a loop statement are the live in paths of its header
   * plus the live in paths of its first iteration if it is exectuted
   */

  statement
  current_stmt = (statement) gen_get_ancestor(statement_domain, l);
  statement
  body = loop_body(l);
  entity
  index = loop_index(l);

  pips_debug(1,"begin\n");

  /* If the loop belongs to a sequence of statements, which is
   * generally the case, its live in paths have already been computed.
   * I don't yet know if they may be much more precisely computed here.
   * For performance reasons, I don't try to recompute them if they
   * are already available.
   * The code below is likely to be seldom executed;
   * however, it has been tested on small cases.
   */
  if ( !bound_live_in_paths_p(current_stmt) )
  {

    /* Live in effects of header  */
    /* We assume that there is no side effect in the loop header;
     * thus the live in effects of the header are similar to its proper read effects
     */
    list l_prop_header = convert_rw_effects(load_proper_rw_effects_list(current_stmt),
        ctxt);
    list l_live_in_header =
        proper_to_summary_effects(effects_read_effects_dup(l_prop_header));
    reset_converted_rw_effects(&l_prop_header, ctxt);
    pips_debug_effects(3, "live in paths of loop header:\n", l_live_in_header);

    /* Live in effects of first iteration if it exists minus read effects
     * on the loop index, which are masked by its initialization in the header,
     * and read effects on loop locals if the loop is parallel;
     */
    list l_masked_vars = NIL;
    if(loop_parallel_p(l))
      l_masked_vars = loop_locals(l);
    /* beware: potential sharing with loop_locals(l) */
    l_masked_vars = CONS(ENTITY, index, l_masked_vars);

    list l_live_in_first_iter =
        effects_dup_without_variables(load_live_in_paths_list(body),
            l_masked_vars);

    /* free l_masked_vars: beware of the potential sharing with loop_locals */
    if (loop_parallel_p(l) && !ENDP(l_masked_vars))
      CDR(l_masked_vars) = NIL;
    gen_free_list(l_masked_vars);

    if (! normalizable_and_linear_loop_p(index, loop_range(l)))
    {
      pips_debug(7, "non linear loop range.\n");
      effects_to_may_effects(l_live_in_first_iter);
    }
    // functions that can be pointed by loop_descriptor_make_func:
    // loop_undefined_descriptor_make
    // loop_convex_descriptor_make
    else if(loop_descriptor_make_func == loop_undefined_descriptor_make)
    {
      if (!loop_executed_at_least_once_p(l))
        effects_to_may_effects(l_live_in_first_iter);
    }
    else
    {
      pips_internal_error("live paths of loop not implemented for convex paths\n");
    }

    pips_debug_effects(3, "live in paths of loop first iteration:\n", l_live_in_header);

    /* put them together */
    // functions that can be pointed by effects_union_op:
    // ProperEffectsMustUnion
    // RegionsMustUnion
    // ReferenceUnion
    // EffectsMustUnion
    list l_live_in_loop = (*effects_union_op)(l_live_in_header,
        l_live_in_first_iter,
        effects_same_action_p);

    pips_debug_effects(3, "live in paths of loop:\n", l_live_in_loop);

    store_live_in_paths_list(current_stmt, l_live_in_loop);
  }

  pips_debug(1,"end\n");
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
	  pips_debug(1,"dealing with statement with ordering: %03zd and number: %03zd\n",
		     statement_ordering(stmt), statement_number(stmt));

	  list l_live_out = l_next_stmt_live_in;
	  pips_debug_effects(2, "storing effects:\n", l_live_out);
	  store_live_out_paths_list(stmt, effects_dup(l_live_out));

	  /* Now let us compute the live in paths */
	  live_in_paths_of_statement(stmt, ctxt);
	  l_next_stmt_live_in = load_live_in_paths_list(stmt);
	}

      /* The block live in paths are the live in paths of the first statement
         which is the last visited statement.
	 However, in case of nested sequences, which may be the case when there are
	 macros for instance, the live_in paths may have already been computed.
	 It's not yet clear if the effects we have just computed are more precise
	 or not.
      */
      if ( !bound_live_in_paths_p(seq_stmt) )
	{
	  pips_debug_effects(2, "storing live_in paths of sequence statement:\n", l_next_stmt_live_in);
	  store_live_in_paths_list(seq_stmt, effects_dup(l_next_stmt_live_in));
	}

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
     loop_domain, live_out_paths_from_loop_to_body, live_in_paths_of_loop,
     whileloop_domain, live_out_paths_from_whileloop_to_body, live_in_paths_of_whileloop,
     forloop_domain, live_out_paths_from_forloop_to_body, gen_null,
     unstructured_domain, live_out_paths_from_unstructured_to_nodes, gen_null,

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
  // functions that can be pointed by effects_computation_init_func:
  // effects_computation_no_init
  // init_convex_in_out_regions
  // init_convex_rw_regions
  (*effects_computation_init_func)(module_name);

  /* Get the various effects and in_effects of the module. */

  set_proper_rw_effects((*db_get_proper_rw_effects_func)(module_name));
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
  reset_live_paths_context(&ctxt);

  pips_debug(1, "end\n");

  (*db_put_live_out_paths_func)(module_name, get_live_out_paths());
  (*db_put_live_in_paths_func)(module_name, get_live_in_paths());

  reset_proper_rw_effects();
  reset_rw_effects();
  reset_invariant_rw_effects();
  reset_in_effects();
  reset_cumulated_in_effects();
  reset_invariant_in_effects();
  reset_live_in_paths();
  reset_live_out_paths();
  // functions that can be pointed by effects_computation_reset_func:
  // effects_computation_no_reset
  // reset_convex_in_out_regions
  // reset_convex_rw_regions
  (*effects_computation_reset_func)(module_name);

  free_effects_private_current_context_stack();

  reset_current_module_entity();
  reset_current_module_statement();
  pips_debug(1, "end\n");

  debug_off();
  return true;
}
