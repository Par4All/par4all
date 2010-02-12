/* A simple phase that outline for loops for scalopes project

   pierre.villalon@hpc-project.com
*/


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "control.h"
#include "callgraph.h"
#include "pipsdbm.h"
#include "transformations.h"
#include "resources.h"
#include "properties.h"

/// the list to store the loop to outline
list loops_to_outline;

/// @brief build the list of loop to outline. Only first level loops are
/// targeted.
/// @return TRUE if the loop is a first level loop.
/// @param l, the loop to test.
static bool build_loop_list (loop l) {
  pips_debug (5, "outer loop to be outlined\n");
  statement stmt = (statement) gen_get_ancestor(statement_domain, l);
  loops_to_outline = CONS (STATEMENT, stmt, loops_to_outline);
  pips_debug (5, "statement %p added\n", (void*) stmt);
  return FALSE;
}

bool scalopify (const string module_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "SCALOPIFY_DEBUG_LEVEL");
  // Get the effects to be used by the outliner
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,
								     module_name,
								     TRUE));

  // build the list of loops to outline
  loops_to_outline = NIL;
  gen_recurse (module_statement,
	       loop_domain, build_loop_list, gen_identity);
  pips_debug (5, "%d loops to outline found\n", gen_length (loops_to_outline));

  // revert the list and apply outlining
  loops_to_outline = gen_nreverse (loops_to_outline);

  FOREACH (STATEMENT, s, loops_to_outline) {
    pips_debug (5, "outlining statement %p\n", (void*) s);
    list local = CONS (STATEMENT, s, NIL);
    (void) outliner (build_new_top_level_module_name ("task"), local);
    gen_free_list(local);
  }
  gen_free_list(loops_to_outline);

  // No longer use effects:
  reset_cumulated_rw_effects();

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
			 compute_callees(get_current_module_statement()));

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);

  return TRUE;
}
