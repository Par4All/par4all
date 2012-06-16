/*

  $Id$

  Copyright 1989-2011 MINES ParisTech

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

#include "genC.h"
#include "linear.h"

// newgen
#include "ri.h"
#include "effects.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "callgraph.h"
#include "properties.h"

#include "resources.h"
#include "pipsdbm.h"

#include "control.h" // for clean_up_sequences
#include "effects-generic.h" // {set,reset}_proper_rw_effects

#include "freia.h"
#include "hwac.h"

/********************************************************************* UTILS */

/* remove "register" qualifier from qualifier list
 */
static list remove_register(list lq)
{
  FOREACH(qualifier, q, lq)
  {
    if (qualifier_register_p(q))
    {
      gen_remove_once(&lq, q);
      // register may be there once
      return lq;
    }
  }
  return lq;
}

// this should exists somewhere???
// top-down so that it is in ascending order?
// I'm not sure that it makes much sense, as
// the statement number is expected to be the source code line number?

static bool sr_flt(statement s, int * number)
{
  instruction i = statement_instruction(s);
  if (instruction_sequence_p(i))
    // it seems that sequences are expected not to have a number
    // just in case, force the property
    statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
  else if (statement_number(s)!=STATEMENT_NUMBER_UNDEFINED)
    // just add a number if none are available?
    // otherwise, the initial number is kept
    statement_number(s) = (*number)++;
  return true;
}

static void stmt_renumber(statement s)
{
  int number=1;
  gen_context_recurse(s, &number, statement_domain, sr_flt, gen_null);
}

/* look for a variable
 */
typedef struct
{
  bool used;
  entity var;
} vis_ctx;

static void vis_rwt(const reference r,  vis_ctx * ctx)
{
  if (reference_variable(r)==ctx->var)
  {
    ctx->used = true;
    gen_recurse_stop(NULL);
  }
}

static bool variable_is_used(statement s, entity var)
{
  vis_ctx ctx = { false, var };
  gen_context_recurse(s, &ctx, reference_domain, gen_true, vis_rwt);
  return ctx.used;
}

/*********************************** UNROLL FREIA CONVERGENCE LOOPS FOR SPOC */

/*
  The purpose of this transformation is to unroll a special case of
  while convergence loop by an appropriate amount in order to fill-in
  the spoc pipeline.

  The implementation is quick and dirty, could be improved if needed.

  More could be done on these line if the pipeline before the while is not
  full, some iterations could be moved there as well. However, the information
  available when this transformation is applied is not sufficient to decide
  to do so.

  The two volumes to be compared are taken from the two last stages.
 */

typedef struct {
  int morpho;
  int comp;
  int vol;
  bool bad_stuff;
} fcs_ctx;

static bool fcs_count(statement s, fcs_ctx * ctx)
{
  if (!statement_call_p(s))
  {
    ctx->bad_stuff = true;
    gen_recurse_stop(NULL);
  }
  else
  {
    call c = freia_statement_to_call(s);
    if (c)
    {
      const char* called = entity_local_name(call_function(c));
      // could be: dilate + inf + vol or erode + sup + vol
      if (same_string_p(called, AIPO "erode_8c") ||
          same_string_p(called, AIPO "erode_6c") ||
          same_string_p(called, AIPO "dilate_8c") ||
          same_string_p(called, AIPO "dilate_6c"))
        ctx->morpho++;
      else if (same_string_p(called, AIPO "inf") ||
               same_string_p(called, AIPO "sup"))
        ctx->comp++;
      else if (same_string_p(called, AIPO "global_vol"))
        ctx->vol++;
      // else { // should do more filtering!
      // ctx->bad_stuff = true;
      // gen_recurse_stop(NULL);
      // }
    }
  }
  return false;
}

/* @return whether sq contains freia simple calls to E/D, </>, vol, =
 * this is rather an over approximation of whether it is a convergence
 * sequence, should be refined...
 */
static bool freia_convergence_sequence_p(sequence sq)
{
  fcs_ctx ctx = { 0, 0, 0, false };
  gen_context_recurse(sq, &ctx, statement_domain, fcs_count, gen_null);
  return ctx.morpho==1 && ctx.comp==1 && ctx.vol==1 && !ctx.bad_stuff;
}

/* util defined somewhere ?????? */
typedef struct { entity ovar, nvar; } subs_ctx;

static void ref_rwt(reference r, subs_ctx * ctx)
{
  if (reference_variable(r)==ctx->ovar)
    reference_variable(r)=ctx->nvar;
}

static void substitute_reference_variable(statement s, entity ovar, entity nvar)
{
  subs_ctx ctx = { ovar, nvar };
  gen_context_recurse(s, &ctx, reference_domain, gen_true, ref_rwt);
}

static void maybe_unroll_while_rwt(whileloop wl, bool * changed)
{
  if (statement_sequence_p(whileloop_body(wl)))
  {
    sequence sq = statement_sequence(whileloop_body(wl));
    if (freia_convergence_sequence_p(sq))
    {
      // unroll!
      int factor = get_int_property(spoc_depth_prop);
      pips_assert("some unrolling factor", factor>0);
      list ol = sequence_statements(sq);

      // sorry, just a hack to avoid a precise pattern matching
      statement first = STATEMENT(CAR(ol));
      pips_assert("s is an assignement", statement_call_p(first));
      gen_remove_once(&ol, first);
      list col = gen_copy_seq(ol), nl = NIL;
      list lass = call_arguments(statement_call(first));
      entity
        previous = expression_variable(EXPRESSION(CAR(lass))),
        current = expression_variable(EXPRESSION(CAR(CDR(lass))));

      // cleanup vol from col
      statement vols = statement_undefined;
      FOREACH(statement, s, col)
      {
        call c = freia_statement_to_call(s);
        if (c && same_string_p(entity_local_name(call_function(c)),
                               AIPO "global_vol"))
        {
          gen_remove_once(&col, s);
          vols = copy_statement(s);
          break;
        }
      }
      pips_assert("vol statement found", vols!=statement_undefined);

      // do factor-1
      while (--factor)
        nl = gen_nconc(gen_full_copy_list(col), nl);

      // compute "previous" volume for comparison
      substitute_reference_variable(vols, current, previous);

      // swith off "register" on previous, if need be...
      variable v = type_variable(entity_type(previous));
      variable_qualifiers(v) = remove_register(variable_qualifiers(v));

      // then reuse initial list for the last one
      sequence_statements(sq) = gen_nconc(nl, CONS(statement, vols, ol));

      // cleanup temporary list
      gen_free_list(col);

      // must renumber...
      *changed = true;
    }
  }
}

static bool freia_unroll_while_for_spoc(statement s)
{
  bool changed = false;
  gen_context_recurse(s, &changed,
                      whileloop_domain, gen_true, maybe_unroll_while_rwt);
  return changed;
}

bool freia_unroll_while(const string module)
{
  // sanity check
  int factor = get_int_property(spoc_depth_prop);
  if (factor<=1)
  {
    pips_user_warning("cannot unroll with factor=%d\n", factor);
    return true;
  }

  // else, do the job
  debug_on("PIPS_HWAC_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module);

  // else do the stuff
  statement mod_stat =
    (statement) db_get_memory_resource(DBR_CODE, module, true);
  set_current_module_statement(mod_stat);
  set_current_module_entity(module_name_to_entity(module));

  // do the job
  bool changed = freia_unroll_while_for_spoc(mod_stat);

  // update if changed
  if (changed)
  {
    stmt_renumber(mod_stat);
    module_reorder(mod_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, mod_stat);
  }

  // cleanup
  reset_current_module_statement();
  reset_current_module_entity();
  debug_off();
  return true;
}

/********************** SPECIAL CASE SCALAR WW DEPENDENCY HANDLING FOR FREIA */
/*
  When switching do-while to while on FREIA convergence loops, the code
  generates a scalar WW dependency within the sequence which prevents an
  operation to be merged. Pips is designed to be intelligent about loops
  for parallelization, whereas I really need some cleverness in sequences
  for FREIA compilation.

  The following code, where the reduction is a volume computation in our case:

    reductionl(image1, &something);
    somethingelse = something;
    reduction(image2, &something);

  should be replaced by:

    reduction(image1, &somethingelse);
    reduction(image2, &something);

  I'm not sure how this transformation could be generalized, possibly
  based on chains...
  Moreover, I'm not really sure how much I can rely on effects
  because of the pointers involved.
  So this is just a quick hack for my particular case.
*/

/* return the freia reduction entity, or NULL if it does not apply
 */
static entity freia_reduction_variable(const statement s)
{
  if (freia_statement_aipo_call_p(s))
  {
    const call c = freia_statement_to_call(s);
    const string called = (const string) entity_local_name(call_function(c));
    if (same_string_p(called, AIPO "global_min") ||
        same_string_p(called, AIPO "global_max") ||
        same_string_p(called, AIPO "global_min_coord") ||
        same_string_p(called, AIPO "global_max_coord") ||
        same_string_p(called, AIPO "global_vol"))
    {
      const expression last = EXPRESSION(CAR(gen_last(call_arguments(c))));
      // should be a call to "&" operator
      if (expression_address_of_p(last))
      {
        const expression arg =
          EXPRESSION(CAR(call_arguments(expression_call(last))));
        if (syntax_reference_p(expression_syntax(arg)))
          return expression_variable(arg);
      }
    }
  }
  // not found!
  return NULL;
}

static void sww_seq_rwt(const sequence sq, bool * changed)
{
  // current status of search
  statement previous_reduction = NULL;
  entity previous_variable = NULL;
  statement assignment = NULL;
  entity assigned = NULL;

  // this is a partial O(n) implementation
  // beware that we may have VERY LONG sequences...
  FOREACH(statement, s, sequence_statements(sq))
  {
    bool tocheck = true;
    entity red = freia_reduction_variable(s);
    if (red)
    {
      if (previous_reduction && assignment && red==previous_variable)
      {
        // matched!
        // replace variable in first reduction
        hash_table replacements = hash_table_make(hash_pointer, 0);
        hash_put(replacements, previous_variable, assigned);
        replace_entities(previous_reduction, replacements);
        hash_table_free(replacements);
        // and now useless remove assignment
        free_instruction(statement_instruction(assignment));
        statement_instruction(assignment) = make_continue_instruction();
        // make assigned NOT a register... just in case
        variable v = type_variable(entity_type(assigned));
        variable_qualifiers(v) = remove_register(variable_qualifiers(v));
        // we did something
        *changed = true;
      }

      // restart current status with new found reduction
      previous_reduction = s;
      previous_variable = red;
      assignment = NULL;
      assigned = NULL;
      tocheck = false;
    }

    // detect reduction variable assignment
    if (previous_reduction && !assignment &&
        instruction_assign_p(statement_instruction(s)))
    {
      list args = call_arguments(instruction_call(statement_instruction(s)));
      expression e1 = EXPRESSION(CAR(args)), e2 = EXPRESSION(CAR(CDR(args)));
      if (expression_reference_p(e1) && expression_reference_p(e2) &&
          reference_variable(expression_reference(e2))==previous_variable)
      {
        assignment = s;
        assigned = reference_variable(expression_reference(e1));
        tocheck = false;
      }
    }

    if (tocheck && previous_variable && variable_is_used(s, previous_variable))
    {
      // full cleanup
      previous_reduction = NULL;
      previous_variable = NULL;
      assignment = NULL;
      assigned = NULL;
    }
  }
}

bool freia_remove_scalar_ww_deps(const string module)
{
  debug_on("PIPS_HWAC_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module);

  // else do the stuff
  statement mod_stat =
    (statement) db_get_memory_resource(DBR_CODE, module, true);
  set_current_module_statement(mod_stat);
  set_current_module_entity(module_name_to_entity(module));

  // do the job
  bool changed = false;
  gen_context_recurse(mod_stat, &changed,
                      sequence_domain, gen_true, sww_seq_rwt);

  // update database if changed
  if (changed)
    // stmt_renumber(mod_stat); module_reorder(mod_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, mod_stat);

  reset_current_module_statement();
  reset_current_module_entity();
  debug_off();
  return true;
}
