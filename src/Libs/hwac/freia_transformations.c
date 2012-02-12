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

/*********************************** UNROLL FREIA CONVERGENCE LOOPS FOR SPOC */

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

      // cleanup vol from col
      FOREACH(statement, s, col)
      {
        call c = freia_statement_to_call(s);
        if (c && same_string_p(entity_local_name(call_function(c)),
                               AIPO "global_vol"))
        {
          gen_remove_once(&col, s);
          break;
        }
      }

      // do factor-1
      while (--factor)
        nl = gen_nconc(gen_full_copy_list(col), nl);

      // then reuse initial list for the last one
      sequence_statements(sq) = CONS(statement, first, gen_nconc(nl, ol));

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

  reset_current_module_statement();
  reset_current_module_entity();
  debug_off();
  return true;
}
