/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
  transformations package

  coarse_grain_parallelization.c :  Beatrice Creusillet, october 1996
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  This file contains the functions to parallelize loops by using the
  regions of their bodies. This enables the parallelization of loops
  containing "if" constructs and unstructured parts of code. But it may
  be less aggressive that Allen and Kennedy on other loop nests, because
  the regions of the loop bodies may be imprecise.
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
#include "parser_private.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
#include "reduction.h"
#include "transformations.h"

/* includes pour system generateur */
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "ricedg.h"

#include "reductions_private.h"
#include "reductions.h"


static bool local_use_reductions_p;
GENERIC_LOCAL_FUNCTION(statement_reductions, pstatement_reductions)


/**
 * Reduction context
 * Carry the depth of the loop nest and the reduced loop (if applicable)
 */
typedef struct coarse_grain_ctx {
  int depth;
  list reduced_loops;
  bool parallelized_at_least_one_loop;
} coarse_grain_ctx;


/** Parallelize a loop by using region informations to prove iteration
    independance.

    @param l is the loop to parallelize

    @return true to ask the calling NewGen iterator to go on recursion on
    further loops in this loop.
 */
static bool whole_loop_parallelize(loop l, coarse_grain_ctx *ctx)
{
  /* Get the statement owning this loop: */
  statement loop_stat = (statement) gen_get_ancestor(statement_domain, l);
  statement inner_stat = loop_body(l);

  if (!get_bool_property("PARALLELIZE_AGAIN_PARALLEL_CODE")
      && loop_parallel_p(l)) {
    return false;
  }

  if (statement_may_contain_exiting_intrinsic_call_p(inner_stat))
    return false;

  /* get the loop body preconditions */
  transformer body_prec = load_statement_precondition(inner_stat);

  /* do not declare as parallel a loop which is never executed */
  if (transformer_empty_p(body_prec)) {
      pips_debug(1, "non feasible inner statement -> SEQUENTIAL LOOP\n");
      execution_tag(loop_execution(l)) = is_execution_sequential;
      return false;
  }

  /* ...needed by TestCoupleOfReferences(): */
  list l_enclosing_loops = CONS(STATEMENT, loop_stat, NIL);

  /* Get the loop invariant regions for the loop body: */
  list l_reg = load_invariant_rw_effects_list(inner_stat);

  /* To store potential conflicts to study: */
  list l_conflicts = NIL;

  /* To keep track of a current conflict disabling the parallelization: */
  bool may_conflicts_p = false;

  /* Can we discard conflicts due to thread-safe variables? */
  bool thread_safe_p = get_bool_property("PARALLELIZATION_IGNORE_THREAD_SAFE_VARIABLES");

  pips_debug(1,"begin\n");

  /**
   *  Reduction handling, if a reference is present in the summary it means
   *  That we can safely ignore conflict that involved it
   *
   *  MA: seems buggy since it relies here on entity instead of reference
   */
  set lreductions;
  if(local_use_reductions_p) {
    lreductions = set_make(set_pointer);
    pips_debug(1,"Fetching reductions for this loop\n");
    reductions rs = (reductions)load_statement_reductions(loop_stat);
    FOREACH(REDUCTION,r,reductions_list(rs)) {
      entity e = reference_variable(reduction_reference(r));
      pips_debug(1,"Ignoring dependences on %s for this loop\n",entity_local_name(e));
      set_add_element(lreductions,lreductions,e);
    }
  }

  pips_debug(1,"building conflicts\n");
  ifdebug(2) {
    fprintf(stderr, "original invariant regions:\n");
    print_regions(l_reg);
  }

  /* First, builds list of conflicts: */
  FOREACH(EFFECT, reg, l_reg) {
    entity e = region_entity(reg);

    if (gen_chunk_undefined_p(gen_find_eq(effect_entity(reg),loop_locals(l)))
        && region_write_p(reg)
        && store_effect_p(reg)
        && !(thread_safe_p && thread_safe_variable_p(e))
        && !(local_use_reductions_p && set_belong_p(lreductions,e))
        ) {
      reference r = effect_any_reference(reg);
      int d = gen_length(reference_indices(r));
      conflict conf = conflict_undefined;

      /* Add a write-write conflict to the list: */
      conf = make_conflict(reg, reg, cone_undefined);
      l_conflicts = gen_nconc(l_conflicts, CONS(CONFLICT, conf, NIL));

      /* Search for a write-read/read-write conflict */
      FOREACH(EFFECT, reg2, l_reg) {
        reference r2 = effect_any_reference(reg2);
        int d2 = gen_length(reference_indices(r2));

        /**
         * FI->RK: Careful, you are replicating code of chains.c,
         * add_conflicts(). Why cannot you use region_chains?
         *
         * The test below must evolve with Beatrice's work on memory
         * access paths. d<=d2 is a very preliminary test for memory
         * access paths.
         */
        if (same_entity_p(e,region_entity(reg2)) && store_effect_p(reg2) && region_read_p(reg2) && d<=d2) {
          /* Add a write-read conflict */
          conf = make_conflict(reg, reg2, cone_undefined);
          l_conflicts = gen_nconc(l_conflicts, CONS(CONFLICT, conf, NIL));
          /* There is at most one read region for entity e by definition
              of the regions, so it's useless to go on interating: */
          break;
        }
      }
    }
  }

  /* THEN, TESTS CONFLICTS */
  pips_debug(1,"testing conflicts\n");
  /* We want to test for write/read and read/write dependences at the same
   * time. */
  Finds2s1 = true;
  FOREACH(CONFLICT, conf, l_conflicts) {
    effect reg1 = conflict_source(conf);
    effect reg2 = conflict_sink(conf);
    list levels = NIL;
    list levelsop = NIL;
    Ptsg gs = SG_UNDEFINED;
    Ptsg gsop = SG_UNDEFINED;

    ifdebug(2) {
      fprintf(stderr, "testing conflict from:\n");
      print_region(reg1);
      fprintf(stderr, "\tto:\n");
      print_region(reg2);
    }

    /* Use the function TestCoupleOfReferences from ricedg. */
    /* We only consider one loop at a time, disconnected from
     * the other enclosing and inner loops. Thus l_enclosing_loops
     * only contains the current loop statement.
     * The list of loop variants is empty, because we use loop invariant
     * regions (they have been composed by the loop transformer).
     */
    levels = TestCoupleOfReferences(l_enclosing_loops, region_system(reg1),
                                    inner_stat, reg1, effect_any_reference(reg1),
                                    l_enclosing_loops, region_system(reg2),
                                    inner_stat, reg2, effect_any_reference(reg2),
                                    NIL, &gs, &levelsop, &gsop);
    ifdebug(2) {
      fprintf(stderr, "result:\n");
      if (ENDP(levels) && ENDP(levelsop))
        fprintf(stderr, "\tno dependence\n");

      if (!ENDP(levels)) {
        fprintf(stderr, "\tdependence at levels: ");
        FOREACH(INT, l, levels)
        fprintf(stderr, " %d", l);
        fprintf(stderr, "\n");

        if (!SG_UNDEFINED_P(gs)) {
          Psysteme sc = SC_UNDEFINED;
          fprintf(stderr, "\tdependence cone:\n");
          sg_fprint_as_dense(stderr, gs, gs->base);
          sc = sg_to_sc_chernikova(gs);
          fprintf(stderr,"\tcorresponding linear system:\n");
          sc_fprint(stderr,sc,(get_variable_name_t)entity_local_name);
          sc_rm(sc);
        }
      }
      if (!ENDP(levelsop)) {
        fprintf(stderr, "\topposite dependence at levels: ");
        FOREACH(INT, l, levelsop)
        fprintf(stderr, " %d", l);
        fprintf(stderr, "\n");

        if (!SG_UNDEFINED_P(gsop)) {
          Psysteme sc = SC_UNDEFINED;
          fprintf(stderr, "\tdependence cone:\n");
          sg_fprint_as_dense(stderr, gsop, gsop->base);
          sc = sg_to_sc_chernikova(gsop);
          fprintf(stderr,"\tcorresponding linear system:\n");
          sc_fprint(stderr,sc,(get_variable_name_t)entity_local_name);
          sc_rm(sc);
        }
      }
    }
    /* If the dependence cannot be disproved, add it to the list of
     * assumed dependences. */
    if (!ENDP(levels) || !ENDP(levelsop))
      may_conflicts_p = true;

    gen_free_list(levels);
    gen_free_list(levelsop);
    if (!SG_UNDEFINED_P(gs))
      sg_rm(gs);
    if (!SG_UNDEFINED_P(gsop))
      sg_rm(gsop);
  }

  /* Was there any conflict? */
  if (may_conflicts_p)
    /* Do not change the loop since it is sequential: */
    pips_debug(1, "SEQUENTIAL LOOP\n");
  else {
    /* Mark the loop as parallel since we did not notice any conflict: */
    if(local_use_reductions_p) {
      pips_debug(1, "PARALLEL LOOP WITH REDUCTIONS\n");
      ctx->reduced_loops = CONS(int,statement_ordering(loop_stat),ctx->reduced_loops);
    } else {
      pips_debug(1, "PARALLEL LOOP\n");
      // If the loop was sequential, we mark it as parallel and register
      // that we parallelized at least one loop
      if(loop_sequential_p(l)) {
        ctx->parallelized_at_least_one_loop = true;
        execution_tag(loop_execution(l)) = is_execution_parallel;
      }
    }
  }

  /* Finally, free conflicts */
  pips_debug(1,"freeing conflicts\n");
  FOREACH(CONFLICT, c, l_conflicts) {
    conflict_source(c) = effect_undefined;
    conflict_sink(c) = effect_undefined;
    free_conflict(c);
  }
  gen_free_list(l_conflicts);

  gen_free_list(l_enclosing_loops);

  if(local_use_reductions_p) {
    set_free(lreductions);
  }

  pips_debug(1,"end\n");

  return(true);
}


/** Parallelize a code statement by using region informations to prove
    iteration independance.

    @param module_stat is the module statement to parallelize as a code
    reference

    @param ctx is the context for keeping informations about what was done
 */
static void coarse_grain_loop_parallelization(statement module_stat, coarse_grain_ctx *ctx) {
  pips_debug(1,"begin\n");


  // Iterate on the loops to try parallelizing them:
  gen_context_recurse(module_stat,
                      ctx,
                      loop_domain,
                      whole_loop_parallelize,
                      gen_true
                      );

  pips_debug(1,"end\n");
}


/** Parallelize code by using region informations to prove iteration
    independence.

    @param module_name is the name of the module to parallelize
    @return true in case of success. Indeed, return alway true. :-)
 */
static bool coarse_grain_parallelization_main(const char* module_name,
                                       bool use_reductions_p)
{
    statement module_stat;
    entity module;

    /* Warn everybody here if we use or not reductions: */
    local_use_reductions_p = use_reductions_p;
    if(local_use_reductions_p) {
      set_statement_reductions(
          (pstatement_reductions)db_get_memory_resource(DBR_CUMULATED_REDUCTIONS,
                                                        module_name,
                                                        true));
    }

    /* Get the code of the module. */
    /* Build a copy of the CODE since we rewrite it on the fly to build a
       PARALLELIZED_CODE with it. Do not use
       db_get_memory_resource(,,false) since it is more efficient to use a
       gen_copy_statement() and it has nasty effects on informations
       attached on the DBR_CODE with the statement addresses (see Trac
       ticket #159 in 2009).

       Well, indeed even this does not work. So this phase changes the code
       resource... */
    module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);

    set_current_module_statement(module_stat);
    module = module_name_to_entity(module_name);
    set_current_module_entity(module);

    /* Get and use cumulated_effects: */
    set_cumulated_rw_effects((statement_effects)
           db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));

    /* Build mapping between variables and semantics informations: */
    module_to_value_mappings(module);

    /* use preconditions to check that loop bodies are not dead code */
    set_precondition_map((statement_mapping)
            db_get_memory_resource(DBR_PRECONDITIONS, module_name, true));

    /* Get and use invariant read/write regions */
    set_invariant_rw_effects((statement_effects)
        db_get_memory_resource(DBR_INV_REGIONS, module_name, true));

    print_parallelization_statistics(module_name, "ante", module_stat);

    ResetLoopCounter();

    debug_on("COARSE_GRAIN_PARALLELIZATION_DEBUG_LEVEL");

    // Initialize a context to keep track of what is done
    coarse_grain_ctx ctx = { 0, NIL, false };
    coarse_grain_loop_parallelization(module_stat, &ctx);

    debug_off();

    print_parallelization_statistics
        (module_name, "post", module_stat);
    reset_cumulated_rw_effects();
    reset_invariant_rw_effects();
    reset_precondition_map();
    reset_current_module_entity();
    reset_current_module_statement();
    free_value_mappings();

    if(local_use_reductions_p) {
      reset_statement_reductions();
      DB_PUT_MEMORY_RESOURCE(DBR_REDUCTION_PARALLEL_LOOPS,
           module_name,
           (char*) make_reduced_loops(ctx.reduced_loops));
    } else {
      /* update loop_locals according to exhibited parallel loops */
      /* this require recomputing effects */
      /* This is only one strategy among others, and has been discussed in ticket #538 */
      /* The advantage of this one is to be compatible with Fortran 77 and C */
      /* To declare private variables at innermost levels even when loops are not
       * parallel, use another phase prior to this one. BC - 07/2011
       */
      bool locals_changed = update_loops_locals(module_name, module_stat);
      if(ctx.parallelized_at_least_one_loop || locals_changed) {
        DB_PUT_MEMORY_RESOURCE(DBR_CODE,
                               module_name,
                               (char*) module_stat);
      }
    }

    return true;
}

/** Parallelize code by using region informations to prove iteration
    independance.

    @param module_name is the name of the module to parallelize
    @return true in case of success. Indeed, return always true. :-)
 */
bool coarse_grain_parallelization(const char* module_name) {
  /* Do not use reductions: */
  return coarse_grain_parallelization_main(module_name, false);
}

/** Parallelize code by using region informations to prove iteration
    independence. Use reduction information to filter out false
    dependencies.

    This is an independent phase and not only a property since we need
    more resources.

    @param module_name is the name of the module to parallelize
    @return true in case of success. Indeed, return always true. :-)
 */
bool coarse_grain_parallelization_with_reduction(const char* module_name) {
  /* Use reductions: */
  return coarse_grain_parallelization_main(module_name, true);
}
