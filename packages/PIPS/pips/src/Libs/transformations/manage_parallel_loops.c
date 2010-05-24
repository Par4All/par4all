/*
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
#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "control.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"


static list current = NIL;
static list next = NIL;
static list privates = NIL;

/// @brief the fonction aims at identifing the parallel loops and queues them
/// in the next list.
/// @return FALSE when a parallel loop is found
/// @param l, the loop to process
static bool identify_outer_loops (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    next = gen_loop_cons (l, next);
    return FALSE;
  }
  return TRUE;
}

/// @brief collect the privates variables of inner loops
/// @return TRUE
/// @param l, the loop to process
static bool collect_privates (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    list var = loop_private_variables_as_entites (l, TRUE, TRUE);
    privates = gen_nconc (privates, var);
  }
  return TRUE;
}

/// @brief make the inner loops sequential
/// @param l, the loop to process
static void  process_loop (loop l) {
  if (execution_parallel_p (loop_execution (l))) {
    // this loop is an innner loop -> make it sequential
    execution_tag (loop_execution (l)) = is_execution_sequential;
  }
}

/**
**/
bool manage_nested_parallelism (const string module_name) {

  // Use this module name and this environment variable to set
  statement mod_stmt = PIPS_PHASE_PRELUDE(module_name,
					  "MANAGE_PARALLEL_LOOPS_DEBUG_LEVEL");

  int threshold = get_int_property ("NESTED_PARALLELISM_THRESHOLD");
  if (threshold > 0) {
    // initialize the next list with all outer parallel loops
    gen_recurse(mod_stmt, loop_domain, identify_outer_loops, gen_identity);
    current = next;
    next = NIL;
    for (int i = 2; i <= threshold; i++) {
      // mark the nested loop at level i
      FOREACH (LOOP, l, current) {
	gen_recurse(loop_body (l), loop_domain, identify_outer_loops, gen_identity);
      }
      gen_free_list (current);
      current = next;
      next = NIL;
    }
  }

  // Targeted outer loops have been identified. They need to be processed
  // inne loops are marked sequential and local variables are moved
  // at the outer loop level
  FOREACH (LOOP, l, current) {
    gen_recurse(loop_body (l), loop_domain, collect_privates, process_loop);
    // need to merge entity one by one otherwise a newgen assertion
    // (about "no sharing of cons") raises
      list locals = loop_locals (l);
      FOREACH (ENTITY, e, privates) {
	if (gen_in_list_p (e, locals) == FALSE) {
	  locals = gen_entity_cons (e, locals);
	}
      }
      loop_locals (l) = locals;
      gen_free_list (privates);
      privates = NIL;
  }

  gen_free_list (current);
  current = NIL;

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(mod_stmt);

  return TRUE;
}
