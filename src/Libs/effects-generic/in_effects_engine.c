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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: in_effects_engine.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the computation of
 * all types of in effects.
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "pipsdbm.h"
#include "resources.h"
#include "preprocessor.h"


#include "effects-generic.h"
#include "effects-convex.h"



typedef struct {
  bool memory_effects_only;
  bool memory_in_out_effects_only;
  effects_representation_val representation;
} in_effects_context;


static void init_in_effects_context(in_effects_context *ctxt,
			     effects_representation_val representation)
{
  ctxt->memory_effects_only = get_bool_property("MEMORY_EFFECTS_ONLY");
  ctxt->memory_in_out_effects_only = get_bool_property("MEMORY_IN_OUT_EFFECTS_ONLY");

  if (!  ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    // if the MEMORY_EFFECTS_ONLY is FALSE on input, and
    // we don't want to compute IN non-memory effects,
    // we have to temporarily set MEMORY_EFFECTS_ONLY to
    // TRUE so that functions computing R/W effects
    // do not compute non memory effects which would have to be filtered.
    set_bool_property("MEMORY_EFFECTS_ONLY", true);
  ctxt->representation = representation;
}

static void reset_in_effects_context(in_effects_context *ctxt)
{
  if (!  ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    // if the MEMORY_EFFECTS_ONLY is FALSE on input, and
    // we did not want to compute IN non-memory effects,
    // we have to reset MEMORY_EFFECTS_ONLY to FALSE.
    set_bool_property("MEMORY_EFFECTS_ONLY", false);
}

static list convert_rw_effects(list lrw, in_effects_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    lrw = effects_store_effects(lrw);

  return lrw;
}


static void reset_converted_rw_effects(list *lrw, in_effects_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    gen_free_list(*lrw);
  *lrw = NIL;
}

/* =============================================================================
 *
 * GENERIC INTERPROCEDURAL IN EFFECTS ANALYSIS
 *
 * =============================================================================
 */


/* bool summary_in_effects_engine(const char* module_name)
 * input    : the name of the current module.
 * output   : the list of summary in effects
 * modifies : nothing.
 * comment  : computes the summary in effects of the current module, using the
 *	      in effects of its embedding statement.
 */
bool
summary_in_effects_engine(const char *module_name)
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

  set_in_effects((*db_get_in_effects_func)(module_name));

  l_loc = load_in_effects_list(module_stat);

  // functions that can be pointed by effects_local_to_global_translation_op:
  // effects_dynamic_elim
  // regions_dynamic_elim
  l_glob = (*effects_local_to_global_translation_op)(l_loc);

  (*db_put_summary_in_effects_func)(module_name, l_glob);


  reset_current_module_entity();
  reset_current_module_statement();
  reset_in_effects();

  // functions that can be pointed by effects_computation_reset_func:
  // effects_computation_no_reset
  // reset_convex_in_out_regions
  // reset_convex_rw_regions
  (*effects_computation_reset_func)(module_name);

  return true;
}


/***************************** GENERIC INTRAPROCEDURAL IN EFFECTS ANALYSIS */

/*
typedef void (*void_fun)();
static void list_gen_consistent_p(list l)
{
    pips_debug(1, "length is %d\n", gen_length(l));
    gen_map((void_fun)gen_consistent_p, l);
    gen_consistent_p(get_in_effects());
}
*/
#define debug_consistent(l) /* ifdebug(9) list_gen_consistent_p(l) */

static bool in_effects_stmt_filter(statement s, in_effects_context *ctxt)
{
  pips_debug(1, "Entering statement %03zd :\n", statement_ordering(s));
  effects_private_current_stmt_push(s);
  effects_private_current_context_push((*load_context_func)(s));
  return true;
}

static void in_effects_of_statement(statement s, in_effects_context *ctxt)
{
  store_invariant_in_effects_list(s, NIL);
  debug_consistent(NIL);
  effects_private_current_stmt_pop();
  effects_private_current_context_pop();
  pips_debug(1, "End statement %03zd :\n", statement_ordering(s));
}

/**
   computes the in effects of the declarations from the list of
   effects after the declaration

   @param[out] lin_after_decls is the list of effects in the store after the declarations;
   it is modified.
   @param[in] l_decl is the ordered list of declarations.

   usage: l = rw_effects_of_declarations(l, l_decl)
 */
static list in_effects_of_declarations(list lin_after_decls, list l_decl, in_effects_context *ctxt)
{
  list lin_before_decls = NIL; /* the returned list */
  list lin_after_first_decl = NIL; /* effects after first declaration */

  if (!ENDP(l_decl))
  {
    // treat last declarations first to compute the in effects in the store juste after
    // the first declaration
    if (!ENDP(CDR(l_decl)))
      lin_after_first_decl = in_effects_of_declarations(lin_after_decls, CDR(l_decl), ctxt);
    else
      lin_after_first_decl = lin_after_decls;

    // then handle top declaration
    entity decl = ENTITY(CAR(l_decl));
    storage decl_s = entity_storage(decl);

    ifdebug(8)
    {
      type ct = entity_basic_concrete_type(decl);
      pips_debug(8, "dealing with entity : %s with type %s\n",
          entity_local_name(decl),words_to_string(words_type(ct,NIL,false)));
    }

    if (storage_ram_p(decl_s)
        /* static variable declaration has no effect, even in case of initialization. */
        && !static_area_p(ram_section(storage_ram(decl_s)))
        && type_variable_p(entity_type(decl)))
    {
      value v_init = entity_initial(decl);
      expression exp_init = expression_undefined;
      if(value_expression_p(v_init))
        exp_init = value_expression(v_init);

      // filter l_rw_after_decls with the declaration
      lin_after_first_decl = filter_effects_with_declaration(lin_after_first_decl, decl);

      // and then add the in effects and remove the write effects
      // due to the initialization part
      // we should also apply the transformer of this expression
      // but this not currently implemented.
      if(!expression_undefined_p(exp_init))
      {
        /* Here we should compute the IN effects of the expression -
         * However, there is currently no interface to compute them -
         * So I take the read effects of the expression.
         * If the expression has write effects, I change the approximations
         * to may
         */
        list l_full_rw_exp_init = generic_proper_effects_of_expression(exp_init);
        l_full_rw_exp_init = proper_to_summary_effects(l_full_rw_exp_init);
        list l_rw_exp_init = convert_rw_effects(l_full_rw_exp_init, ctxt);

        list l_w_exp_init = effects_write_effects(l_rw_exp_init);
        list l_r_exp_init = effects_read_effects(l_rw_exp_init);

        if (!ENDP(l_w_exp_init))
          effects_to_may_effects(l_r_exp_init);

        // IN(before decls) = IN(first decl) U (IN(after first decl) - W(first decl))
        // functions that can be pointed by effects_union_op:
        // ProperEffectsMustUnion
        // RegionsMustUnion
        // ReferenceUnion
        // EffectsMustUnion
        // functions that can be pointed by effects_sup_difference_op:
        // effects_undefined_binary_operator
        // RegionsSupDifference
        // EffectsSupDifference
        lin_before_decls = (*effects_union_op)(
            effects_dup(l_r_exp_init),
            (*effects_sup_difference_op)(lin_after_first_decl,
                effects_dup(l_w_exp_init), r_w_combinable_p),
                effects_same_action_p);

        gen_full_free_list(l_full_rw_exp_init); /* free ell the expression effects */
        reset_converted_rw_effects(&l_rw_exp_init, ctxt);
        gen_free_list(l_r_exp_init);
        gen_free_list(l_w_exp_init);
      }
      else
        lin_before_decls = lin_after_first_decl;

    } /* if (storage_ram(decl_s) && !static_area_p(ram_section(storage_ram(decl_s)))) */
    else
    {
      lin_before_decls = lin_after_first_decl;
    }
  } /* if (!ENDP(CDR(l_decl))) */
  else
    lin_before_decls = lin_after_decls;
  // we should also do some kind of unioning...

  // this could maybe be placed before, only for generic_proper_effects_of_expression(exp_init) */
  if (get_constant_paths_p())
  {
    list l_tmp = lin_before_decls;
    lin_before_decls = pointer_effects_to_constant_path_effects(lin_before_decls);
    effects_free(l_tmp);
  }

  return lin_before_decls;
}


static list r_in_effects_of_sequence(list l_inst, in_effects_context *ctxt)
{
  statement first_statement;
  list remaining_block = NIL;

  list s1_lin; /* in effects of first statement */
  list rb_lin; /* in effects of remaining block */
  list l_in = NIL; /* resulting in effects */
  list s1_lrw; /* rw effects of first statement */
  transformer t1; /* transformer of first statement */

  pips_debug(3," begin\n");

  first_statement = STATEMENT(CAR(l_inst));
  ifdebug(3)
  {
    pips_debug(3," first statement is (ordering %03zd): \n",
        statement_ordering(first_statement));
    print_statement(first_statement);
  }
  remaining_block = CDR(l_inst);

  /* Is it the last instruction of the block */
  if (!ENDP(remaining_block))
  {
    // recursively call the function before doing anything else
    // to avoid large memory usage in case of long sequences
    rb_lin = r_in_effects_of_sequence(remaining_block, ctxt);

    ifdebug(6)
    {
      pips_debug(6," in effects for remaining block:\n");
      (*effects_prettyprint_func)(rb_lin);
    }

    // move the in effects of the remaining block in the store before the
    // first statement
    t1 = (*load_completed_transformer_func)(first_statement);

    /* Nga Nguyen, 25/04/2002. Bug found in fct.f about the transformer of loops
     * Preconditions added to regions take into account the loop exit preconditions
     * but the corresponding loop transformers do not. This may give false results
     * when applying transformers to regions. So we have to add loop exit conditions
     * to the transformers. */

    //	if (statement_loop_p(first_statement))
    // {
    //  loop l = statement_loop(first_statement);
    //   t1 = (*add_loop_exit_condition_func)(t1,l);
    // }

    // functions that can be pointed by effects_transformer_composition_op:
    // effects_composition_with_transformer_nop
    // effects_undefined_composition_with_transformer
    // convex_regions_transformer_compose
    // simple_effects_composition_with_effect_transformer
    (*effects_transformer_composition_op)(rb_lin, t1);

    if (c_module_p(get_current_module_entity()) &&
        (declaration_statement_p(first_statement) ))
    {
      // if it's a declaration statement, effects will be added on the fly
      // as declarations are handled.
      pips_debug(5, "first statement is a declaration statement\n");
      list l_decl = statement_declarations(first_statement);
      l_in = in_effects_of_declarations(rb_lin, l_decl, ctxt);
      rb_lin = NIL; /* the list has been freed by the previous function */
    }
    else // not a declaration statement
    {
      s1_lrw = convert_rw_effects(load_rw_effects_list(first_statement),ctxt);
      ifdebug(6)
      {
        pips_debug(6," rw effects for first statement:\n");
        (*effects_prettyprint_func)(s1_lrw);
      }
      s1_lin = effects_dup(load_in_effects_list(first_statement));
      ifdebug(6)
      {
        pips_debug(6," in effects for first statement:\n");
        (*effects_prettyprint_func)(s1_lin);
      }

      /* Nga Nguyen, 25/04/2002.rb_lin may contain regions with infeasible system {0==-1}
       *     => remove them from rb_lin*/

      // rb_lin = (*remove_effects_with_infeasible_system_func)(rb_lin);

      /* IN(block) = (IN(rest_of_block) - W(S1)) U IN(S1) */
      // functions that can be pointed by effects_union_op:
      // ProperEffectsMustUnion
      // RegionsMustUnion
      // ReferenceUnion
      // EffectsMustUnion
      // functions that can be pointed by effects_sup_difference_op:
      // effects_undefined_binary_operator
      // RegionsSupDifference
      // EffectsSupDifference
      l_in = (*effects_union_op)(
          s1_lin,
          (*effects_sup_difference_op)(rb_lin, effects_dup(s1_lrw),
              r_w_combinable_p),
              effects_same_action_p);

      // no leak
      reset_converted_rw_effects(&s1_lrw,ctxt);
    }
  } // if (!ENDP(remaining_block))
  else
  {
    if (c_module_p(get_current_module_entity()) &&
        (declaration_statement_p(first_statement) ))
    {
      // if it's a declaration statement, effects will be added on the fly
      // as declarations are handled.
      pips_debug(5, "first statement is a declaration statement\n");
      list l_decl = statement_declarations(first_statement);
      l_in = in_effects_of_declarations(NIL, l_decl, ctxt);
    }
    else
    {
      l_in = effects_dup(load_in_effects_list(first_statement));
    }
  }

  ifdebug(6)
  {
    pips_debug(6,"cumulated_in_effects:\n");
    (*effects_prettyprint_func)(l_in);
  }
  store_cumulated_in_effects_list(first_statement, effects_dup(l_in));

  debug_consistent(l_in);
  return l_in;
}

static void
in_effects_of_sequence(sequence block, in_effects_context *ctxt)
{
  list l_in = NIL;
  statement current_stat = effects_private_current_stmt_head();
  list l_inst = sequence_statements(block);

  pips_debug(2, "Effects for statement %03zd:\n",
      statement_ordering(current_stat));

  if (ENDP(l_inst))
  {
    if (get_bool_property("WARN_ABOUT_EMPTY_SEQUENCES"))
      pips_user_warning("empty sequence\n");
  }
  else
  {
    l_in = r_in_effects_of_sequence(l_inst, ctxt);
  }

  ifdebug(2)
  {
    pips_debug(2, "IN effects for statement%03zd:\n",
        statement_ordering(current_stat));
    (*effects_prettyprint_func)(l_in);
    pips_debug(2, "end\n");
  }

  store_in_effects_list(current_stat, l_in);
  debug_consistent(l_in);
}



static void
in_effects_of_test(test t, in_effects_context *ctxt)
{
  statement current_stat = effects_private_current_stmt_head();
  list lt, lf, lc_in;
  list l_in = NIL;

  pips_debug(2, "Effects for statement %03zd:\n",
      statement_ordering(current_stat));

  /* IN effects of the true branch */
  lt = effects_dup(load_in_effects_list(test_true(t))); /* FC: dup */
  /* IN effects of the false branch */
  lf = effects_dup(load_in_effects_list(test_false(t))); /* FC: dup */

  /* IN effects of the combination of both */
  // functions that can be pointed by effects_test_union_op:
  // EffectsMayUnion
  // RegionsMayUnion
  // ReferenceTestUnion
  l_in = (*effects_test_union_op)(lt, lf, effects_same_action_p);

  /* IN effects of the condition */
  /* They are equal to the proper read effects of the statement if
   * there are no side-effects. When there are write effects, we
   * should compute the in effects of the expression, but we still
   * use the read effects and turn them to may effects as an
   * over-approximation.
   */

  lc_in = convert_rw_effects(load_proper_rw_effects_list(current_stat),ctxt);
  list lc_in_r = effects_read_effects_dup(lc_in);
  if (gen_length(lc_in_r) != gen_length(lc_in))
    // there are write effects -> conservatively switch to may for the moment
    effects_to_may_effects(lc_in_r);

  /* in regions of the test */
  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  l_in = (*effects_union_op)(l_in, lc_in_r, effects_same_action_p);

  reset_converted_rw_effects(&lc_in,ctxt);

  ifdebug(2)
  {
    pips_debug(2, "IN effects for statement %03zd:\n",
        statement_ordering(current_stat));
    (*effects_prettyprint_func)(l_in);
    pips_debug(2, "end\n");
  }

  store_in_effects_list(current_stat, l_in);
  debug_consistent(l_in);
}

/* Rin[for (c) s] = Rr[for (c) s] * may
 */
static void in_effects_of_forloop(forloop f, in_effects_context *ctxt)
{
  statement current;

  pips_debug(1, "considering for loop 0x%p\n", (void *) f);

  current = effects_private_current_stmt_head();
  list lrw = convert_rw_effects(load_rw_effects_list(current),ctxt);
  list lin = effects_read_effects_dup(lrw);
  /* switch to MAY... */
  effects_to_may_effects(lin);
  store_in_effects_list(current, lin);
  store_invariant_in_effects_list(current, NIL);
  reset_converted_rw_effects(&lrw, ctxt);
}

/* Rin[while (c) s] = Rin[c] U (Rr[while (c) s] * may)
 */
static void in_effects_of_whileloop(whileloop w, in_effects_context *ctxt)
{
  statement current;
  bool before_p = evaluation_before_p(whileloop_evaluation(w));

  pips_debug(1, "considering while loop 0x%p\n", (void *) w);

  current = effects_private_current_stmt_head();
  list l_rw = convert_rw_effects(load_rw_effects_list(current), ctxt);
  list l_in = effects_read_effects_dup(l_rw);
  /* switch to MAY... */
  effects_to_may_effects(l_in);

  /* add the loop condition evaluation effects */
  if (before_p)
  {
    list l_prop_cond = convert_rw_effects(load_proper_rw_effects_list(current),
        ctxt);

    pips_debug_effects(3, "effects of condition:\n", l_prop_cond);

    /* We don't have the in effects of the condition evaluation
     * we safely approximate them by its may read proper effects
     * if there are write effects during its evaluation
     */
    list l_in_cond = effects_read_effects_dup(l_prop_cond);

    if ( gen_length(l_in_cond) != gen_length(l_prop_cond))
      /* there are write effects in the condition */
      effects_to_may_effects(l_in_cond);
    else
      l_in_cond = proper_to_summary_effects(l_in_cond);

    pips_debug_effects(3, "approximation of in effects of condition:\n", l_in_cond);

    // functions that can be pointed by effects_union_op:
    // ProperEffectsMustUnion
    // RegionsMustUnion
    // ReferenceUnion
    // EffectsMustUnion
    l_in = (*effects_union_op)(l_in,
        l_in_cond,
        effects_same_action_p);

    reset_converted_rw_effects(&l_prop_cond, ctxt);
  }

  store_in_effects_list(current, l_in);
  store_invariant_in_effects_list(current, NIL);
  reset_converted_rw_effects(&l_rw, ctxt);
}

/* list in_effects_of_loop(loop l)
 * input    : a loop, its transformer and its context.
 * output   : the corresponding list of in regions.
 * modifies : in_regions_map.
 * comment  : IN(loop) = proj[i] (proj[i'] (IN(i) - W(i', i'<i))) U IN(i=1)
 */
static void in_effects_of_loop(loop l, in_effects_context *ctxt)
{
  statement current_stat = effects_private_current_stmt_head();

  range r;
  statement b;
  entity i, i_prime = entity_undefined, new_i = entity_undefined;

  list lbody_in; /* in regions of the loop body */
  list global_in, global_in_read_only;/* in regions of non local variables */
  list global_write; /* invariant write regions of non local variables */
  list l_prop, l_prop_read, l_prop_write; /* proper effects of header */
  list l_in = NIL;
  transformer loop_trans = transformer_undefined;

  pips_debug(1, "begin\n");

  i = loop_index(l);
  r = loop_range(l);
  b = loop_body(l);

  pips_debug(1, "loop index %s.\n", entity_minimal_name(i));

  /* IN EFFECTS OF HEADER */
  /* We assume that there is no side effect in the loop header;
   * thus the IN effects of the header are similar to its proper read effects
   * when there are no side effects.
   */
  list l_prop_init = convert_rw_effects(load_proper_rw_effects_list(current_stat), ctxt);
  l_prop = proper_to_summary_effects(effects_dup(l_prop_init));
  reset_converted_rw_effects(&l_prop_init, ctxt);
  l_prop_read = effects_read_effects(l_prop);
  l_prop_write = effects_write_effects(l_prop);
  gen_free_list(l_prop);
  /* END - IN EFFECTS OF HEADER */

  /* INVARIANT WRITE EFFECTS OF LOOP BODY STATEMENT. */
  list l_inv = convert_rw_effects(load_invariant_rw_effects_list(b), ctxt);
  global_write = effects_write_effects_dup(l_inv);
  reset_converted_rw_effects(&l_inv, ctxt);

  pips_debug_effects(4, "W(i)= \n", global_write);

  /* IN EFFECTS OF LOOP BODY STATEMENT. */
  lbody_in = load_in_effects_list(b);

  store_cumulated_in_effects_list(b, effects_dup(lbody_in));
  debug_consistent(lbody_in);

  /* Effects on locals are masked if the loop is parallel */
  if(loop_parallel_p(l))
    global_in = effects_dup_without_variables(lbody_in, loop_locals(l));
  else
    global_in = effects_dup(lbody_in);

  pips_debug_effects(4, "initial IN(i)= \n", global_in);

  /* COMPUTATION OF INVARIANT IN EFFECTS */

  /* We get the loop transformer, which gives the loop invariant */
  /* We must remove the loop index from the list of modified variables */
  loop_trans = (*load_transformer_func)(current_stat);
  loop_trans = transformer_remove_variable_and_dup(loop_trans, i);

  /* And we compute the invariant IN effects. */
  // functions that can be pointed by effects_transformer_composition_op:
  // effects_composition_with_transformer_nop
  // effects_undefined_composition_with_transformer
  // convex_regions_transformer_compose
  // simple_effects_composition_with_effect_transformer
  (*effects_transformer_composition_op)(global_in, loop_trans);
  update_invariant_in_effects_list(b, effects_dup(global_in));

  pips_debug_effects(4, "invariant IN(i)= \n", global_in);

  /* OPTIMIZATION : */
  /* If there is no write effect on a variable imported by the loop body,
   * then, the same effect is imported by the whole loop.
   */
  /* Fix me: This may not be valid if there are effects on abstract locations.
   */
  global_in_read_only =
      effects_entities_inf_difference(effects_dup(global_in),
          effects_dup(global_write),
          r_w_combinable_p);
  global_in =
      effects_entities_intersection(global_in,
          effects_dup(global_write),
          r_w_combinable_p);

  pips_debug_effects(4, "reduced IN(i)= \n", global_in);

  if (!ENDP(global_in))
  {
    /* If the loop range cannot be represented in the chosen representation
     * then, no useful computation can be performed.
     */

    if (! normalizable_and_linear_loop_p(i, r))
    {
      pips_debug(7, "non linear loop range.\n");
      effects_to_may_effects(global_in);
    }
    // functions that can be pointed by loop_descriptor_make_func:
    // loop_undefined_descriptor_make
    // loop_convex_descriptor_make
    else if(loop_descriptor_make_func == loop_undefined_descriptor_make)
    {
      if (!loop_executed_at_least_once_p(l))
        effects_to_may_effects(global_in);
      /* else global_in is still valid because store independent */
    }
    else
    {
      descriptor range_descriptor	 = descriptor_undefined;
      Value incr;
      Pvecteur v_i_i_prime = VECTEUR_UNDEFINED;
      bool saved_add_precondition_to_scalar_convex_regions =
          add_precondition_to_scalar_convex_regions;

      pips_debug(7, "linear loop range.\n");

      add_precondition_to_scalar_convex_regions = true;

      /* OPTIMIZATION: */
      /* keep only in global_write the write regions corresponding to
       * regions in global_in. */
      global_write =
          effects_entities_intersection(global_write,
              effects_dup(global_in),
              w_r_combinable_p);
      pips_debug_effects(4, "reduced W(i)= \n", global_write);

      /* VIRTUAL NORMALIZATION OF LOOP (the new increment is equal
       * to +/-1).
       * This may result in a new loop index, new_i, with an updated
       * range descriptor. Effects are updated at the same time.
       */
      // functions that can be pointed by loop_descriptor_make_func:
      // loop_undefined_descriptor_make
      // loop_convex_descriptor_make
      range_descriptor = (*loop_descriptor_make_func)(l);

      /* first work around the fact that loop preconditions have not been
       * added to scalar regions
       */
      global_write = scalar_regions_sc_append(global_write,
          descriptor_convex(range_descriptor),
          true);
      global_in = scalar_regions_sc_append(global_in,
          descriptor_convex(range_descriptor),
          true);

      // functions that can be pointed by effects_loop_normalize_func:
      // effects_loop_normalize_nop
      // effects_undefined_loop_normalize
      // convex_regions_loop_normalize
      (*effects_loop_normalize_func)(global_write, i, r,
          &new_i, range_descriptor, true);
      // functions that can be pointed by effects_loop_normalize_func:
      // effects_loop_normalize_nop
      // effects_undefined_loop_normalize
      // convex_regions_loop_normalize
      (*effects_loop_normalize_func)(global_in, i, r,
          &new_i, range_descriptor, false);

      if (!entity_undefined_p(new_i) && !same_entity_p(i,new_i)) {
        add_intermediate_value(new_i);
        i = new_i;
      }

      /* COMPUTATION OF IN EFFECTS. We must remove the effects written
       * in previous iterations i.e. IN(i) - U_i'(i'<i)[W(i')] for a
       * positive increment, and	IN(i) - U_i'(i < i')[W(i')]
       * for a negative one.
       */

      /* computation of W(i') */
      /* i' is here an integer scalar value */
      if (get_descriptor_range_p()) {
        add_intermediate_value(i);
        i_prime = entity_to_intermediate_value(i);
        // functions that can be pointed by effects_descriptors_variable_change_func:
        // effects_descriptors_variable_change_nop
        // effects_undefined_descriptors_variable_change
        // convex_regions_descriptor_variable_rename
        (*effects_descriptors_variable_change_func)(
            global_write, i, i_prime);
      }

      ifdebug(4){
        pips_debug(4, "W(i')= \n");
        (*effects_prettyprint_func)(global_write);
      }


      /* We must take into account the fact that i<i' or i'<i. */
      /* This is somewhat implementation dependent. BC. */

      if (get_descriptor_range_p())
      {
        incr = vect_coeff
            (TCST, (Pvecteur) normalized_linear(
                NORMALIZE_EXPRESSION(range_increment(r))));
        v_i_i_prime = vect_make(
            VECTEUR_NUL,
            (Variable) (value_pos_p(incr) ? i_prime : i), VALUE_ONE,
            (Variable) (value_pos_p(incr) ? i : i_prime), VALUE_MONE,
            TCST, VALUE_ONE);
        range_descriptor =
            descriptor_inequality_add(range_descriptor, v_i_i_prime);
      }

      // functions that can be pointed by effects_union_over_range_op:
      // effects_union_over_range_nop
      // simple_effects_union_over_range
      // convex_regions_union_over_range
      global_write = (*effects_union_over_range_op)(
          global_write, i_prime, r, range_descriptor);
      free_descriptor(range_descriptor);

      pips_debug_effects(4, "U_i'[W(i')] = \n", global_write);

      /* IN = IN(i) - U_i'[W(i')] */
      // functions that can be pointed by effects_sup_difference_op:
      // effects_undefined_binary_operator
      // RegionsSupDifference
      // EffectsSupDifference
      global_in = (*effects_sup_difference_op)(global_in, global_write,
          r_w_combinable_p);
      pips_debug_effects(4, "IN(i) - U_i'[W(i')] = \n", global_in);

      /* We eliminate the loop index */
      // functions that can be pointed by effects_union_over_range_op:
      // effects_union_over_range_nop
      // simple_effects_union_over_range
      // convex_regions_union_over_range
      (*effects_union_over_range_op)(global_in, i, range_undefined,
          descriptor_undefined);
      add_precondition_to_scalar_convex_regions =
          saved_add_precondition_to_scalar_convex_regions;
    }
  }

  /* we project the read_only regions along the actual loop index i */
  // functions that can be pointed by effects_union_over_range_op:
  // effects_union_over_range_nop
  // simple_effects_union_over_range
  // convex_regions_union_over_range
  (*effects_union_over_range_op)(global_in_read_only, loop_index(l),
      r, descriptor_undefined);

  global_in = gen_nconc(global_in, global_in_read_only);

  /* we remove the write effects from the proper regions of the loop */
  // functions that can be pointed by effects_sup_difference_op:
  // effects_undefined_binary_operator
  // RegionsSupDifference
  // EffectsSupDifference
  l_in = (*effects_sup_difference_op)(
      global_in, l_prop_write, r_w_combinable_p);

  /* we merge these regions with the proper in regions of the loop */
  // functions that can be pointed by effects_union_op:
  // ProperEffectsMustUnion
  // RegionsMustUnion
  // ReferenceUnion
  // EffectsMustUnion
  l_in = (*effects_union_op)(l_in, l_prop_read, effects_same_action_p);

  store_in_effects_list(current_stat,l_in);
  debug_consistent(l_in);

  pips_debug(1, "end\n");
}


static list
in_effects_of_external(entity func, list real_args)
{
  list le = NIL;
  const char *func_name = module_local_name(func);
  statement current_stat = effects_private_current_stmt_head();

  pips_debug(4, "translating effects for %s\n", func_name);

  if (! entity_module_p(func))
  {
    pips_internal_error("%s: bad function", func_name);
  }
  else
  {
    list func_eff;
    transformer context;

    /* Get the in summary effects of "func". */
    func_eff = (*db_get_summary_in_effects_func)(func_name);
    /* Translate them using context information. */
    context = (*load_context_func)(current_stat);
    le = generic_effects_backward_translation
        (func, real_args, func_eff, context);
  }
  return le;
}

static bool written_before_read_p(entity ent,list args)
{
  MAP(EXPRESSION,exp,
  {
    /* an argument in a READ statement can be a reference or an implied-DO call*/
    list l = expression_to_reference_list(exp,NIL);
    MAP(REFERENCE,ref,
    {
      entity e = reference_variable(ref);
      if (same_entity_p(e,ent))
	{
	  /* the variable is in the reference list, check if it is written or read*/
	  if (expression_reference_p(exp))
	    {
	      reference r = expression_reference(exp);
	      if (reference_scalar_p(r))
		return true;
	      return false;
	    }
	  return false;
	}
    },l);
  },args);

  return false;
}


static void
in_effects_of_call(call c, in_effects_context *ctxt)
{
  statement current_stat = effects_private_current_stmt_head();

  list l_in = NIL;
  entity e = call_function(c);
  tag t = value_tag(entity_initial(e));
  const char* n = module_local_name(e);

  list pc = call_arguments(c);

  pips_debug(1, "begin\n");
  ifdebug(3)
  {
    pips_debug(3, "for current statement \n");
    print_statement(current_stat);
  }
  switch (t) {
  case is_value_constant:
    pips_debug(5, "constant %s\n", n);
    /* consnant have no in effects, do they ? */
    break;
  case is_value_code:
    pips_debug(5, "external function %s\n", n);
    l_in = in_effects_of_external(e, pc);
    debug_consistent(l_in);
    l_in = proper_to_summary_effects(l_in);
    debug_consistent(l_in);
    break;

  case is_value_intrinsic:
    pips_debug(5, "intrinsic function %s\n", n);
    debug_consistent(l_in);
    list l_rw = convert_rw_effects(load_rw_effects_list(current_stat), ctxt);
    debug_consistent(l_rw);
    ifdebug(2){
      pips_debug(2, "R/W effects: \n");
      (*effects_prettyprint_func)(l_rw);
    }

    l_in = effects_read_effects_dup(l_rw);
    reset_converted_rw_effects(&l_rw, ctxt);

    /* Nga Nguyen 25/04/2002 : what about READ *,NDIM,(LSIZE(N), N=1,NDIM) ?
     *   Since NDIM is written before read => no IN effect on NDIM.
     *
     *   So I add tests for the READ statement case. But this is only true for scalar variables :-)
     *
     *   READ *,M,N,L,A(M*N),B(A(L)) ==> MAY-IN A ???*/

    if (strcmp(entity_local_name(e),READ_FUNCTION_NAME)==0)
    {
      list args = call_arguments(c);
      pips_debug(5, " READ function\n");
      MAP(EFFECT,reg,
          {
              entity ent = effect_entity(reg);
              if (entity_scalar_p(ent) && written_before_read_p(ent,args))
                gen_remove(&l_in,reg);
          },l_in);
    }

    debug_consistent(l_in);
    break;

  default:
    pips_internal_error("unknown tag %d", t);
    break;
  }

  ifdebug(2){
    pips_debug(2, "IN effects: \n");
    (*effects_prettyprint_func)(l_in);
  }

  store_in_effects_list(current_stat,l_in);
  debug_consistent(l_in);

  pips_debug(1, "end\n");
}

/* Just to handle one kind of instruction, expressions which are not
   calls.  As we do not distinguish between Fortran and C, this
   function is called for Fortran module although it does not have any
   effect.
 */
static void in_effects_of_expression_instruction(instruction i, in_effects_context *ctxt)
{
  //list l_proper = NIL;
  statement current_stat = effects_private_current_stmt_head();
  //instruction inst = statement_instruction(current_stat);
  pips_debug(2, "begin for expression instruction in statement%03zd\n",
	     statement_ordering(current_stat));

  /* Is the call an instruction, or a sub-expression? */
  if (instruction_expression_p(i))
    {
      list lrw = convert_rw_effects(load_rw_effects_list(current_stat), ctxt);
      list lin = effects_read_effects_dup(lrw);
      reset_converted_rw_effects(&lrw, ctxt);
      store_in_effects_list(current_stat, lin);
    }

  pips_debug(2, "end for expression instruction in statement%03zd\n",
	     statement_ordering(current_stat));

}


static void
in_effects_of_unstructured(unstructured u, in_effects_context *ctxt)
{
  statement current_stat = effects_private_current_stmt_head();
  list blocs = NIL ;
  control ct;
  list l_in = NIL, l_tmp = NIL;

  pips_debug(1, "begin\n");

  ct = unstructured_control( u );

  if(control_predecessors(ct) == NIL && control_successors(ct) == NIL)
  {
    /* there is only one statement in u; */
    pips_debug(6, "unique node\n");
    l_in = effects_dup(load_in_effects_list(control_statement(ct)));
  }
  else
  {
    transformer t_unst = (*load_transformer_func)(current_stat);
    CONTROL_MAP(c,
        {
            l_tmp = load_in_effects_list(control_statement(c));
            // functions that can be pointed by effects_test_union_op:
            // EffectsMayUnion
            // RegionsMayUnion
            // ReferenceTestUnion
            l_in = (*effects_test_union_op) (l_in, effects_dup(l_tmp),
                effects_same_action_p);
        },
        ct, blocs) ;
    // functions that can be pointed by effects_transformer_composition_op:
    // effects_composition_with_transformer_nop
    // effects_undefined_composition_with_transformer
    // convex_regions_transformer_compose
    // simple_effects_composition_with_effect_transformer
    (*effects_transformer_composition_op)(l_in, t_unst);
    effects_to_may_effects(l_in);

    gen_free_list(blocs) ;
  }

  store_in_effects_list(current_stat, l_in);
  debug_consistent(l_in);
  pips_debug(1, "end\n");
}


static void
in_effects_of_module_statement(statement module_stat, in_effects_context *ctxt)
{
  make_effects_private_current_stmt_stack();

  pips_debug(1,"begin\n");

  gen_context_multi_recurse(
      module_stat, ctxt,
      statement_domain, in_effects_stmt_filter, in_effects_of_statement,
      sequence_domain, gen_true, in_effects_of_sequence,
      test_domain, gen_true, in_effects_of_test,
      call_domain, gen_true, in_effects_of_call,
      loop_domain, gen_true, in_effects_of_loop,
      whileloop_domain, gen_true, in_effects_of_whileloop,
      forloop_domain, gen_true, in_effects_of_forloop,
      unstructured_domain, gen_true, in_effects_of_unstructured,
      /* Just to retrieve effects of instructions with kind
       * expression since they are ruled out by the next clause */
      instruction_domain, gen_true, in_effects_of_expression_instruction,
      expression_domain, gen_false, gen_null, /* NOT THESE CALLS */
      NULL);

  pips_debug(1,"end\n");
  free_effects_private_current_stmt_stack();
}


/* bool in_regions(const char* module_name):
 * input    : the name of the current module.
 * requires : that transformers and precondition maps be set if needed.
 *            (it depends on the chosen instanciation of *load_context_func
 *            and *load_transformer_func).
 * output   : nothing !
 * modifies :
 * comment  : computes the in effects of the current module.
 */
bool in_effects_engine(const char * module_name, effects_representation_val representation)
{
  statement module_stat;
  make_effects_private_current_context_stack();
  set_current_module_entity(module_name_to_entity(module_name));

  /* Get the code of the module. */
  set_current_module_statement( (statement)
      db_get_memory_resource(DBR_CODE, module_name, true) );
  module_stat = get_current_module_statement();

  // functions that can be pointed by effects_computation_init_func:
  // effects_computation_no_init
  // init_convex_in_out_regions
  // init_convex_rw_regions
  (*effects_computation_init_func)(module_name);

  debug_on("IN_EFFECTS_DEBUG_LEVEL");
  pips_debug(1, "begin for module %s\n", module_name);

  /* set necessary effects maps */
  set_rw_effects((*db_get_rw_effects_func)(module_name));
  set_invariant_rw_effects((*db_get_invariant_rw_effects_func)(module_name));
  set_proper_rw_effects((*db_get_proper_rw_effects_func)(module_name));

  /* initialise the maps for in regions */
  init_in_effects();
  init_invariant_in_effects();
  init_cumulated_in_effects();

  /* Compute the effects of the module. */
  in_effects_context ctxt;
  init_in_effects_context(&ctxt, representation);
  in_effects_of_module_statement(module_stat, &ctxt);
  reset_in_effects_context(&ctxt);

  /* Put computed resources in DB. */
  (*db_put_in_effects_func)
        (module_name, get_in_effects());
  (*db_put_invariant_in_effects_func)
        (module_name, get_invariant_in_effects());
  (*db_put_cumulated_in_effects_func)
        (module_name, get_cumulated_in_effects());

  pips_debug(1, "end\n");

  debug_off();

  reset_current_module_entity();
  reset_current_module_statement();

  reset_rw_effects();
  reset_invariant_rw_effects();
  reset_proper_rw_effects();

  reset_in_effects();
  reset_invariant_in_effects();
  reset_cumulated_in_effects();

  // functions that can be pointed by effects_computation_reset_func:
  // effects_computation_no_reset
  // reset_convex_in_out_regions
  // reset_convex_rw_regions
  (*effects_computation_reset_func)(module_name);

  free_effects_private_current_context_stack();

  return true;
}
