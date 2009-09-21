/* A simple phase that outline parallel loops onto GPU

   Ronan.Keryell@hpc-project.com
*/

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "pipsdbm.h"
#include "resources.h"

/** Store the loop nests found that meet the spec to be executed on a GPU,
    with its depth */
hash_table loop_nests_to_outline;


static bool
mark_loop_to_outline(const statement s) {
  /* An interesting loop must be parallel first...

     We recurse on statements instead of loops in order to pick
     informations on the statement itself, such as pragmas
  */
  int parallel_loop_nest_depth = depth_of_parallel_perfect_loop_nest(s);
  ifdebug(2) {
    pips_debug(1, "Statement with // dephth %d\n", parallel_loop_nest_depth);
    print_statement(s);
  }
  if (parallel_loop_nest_depth > 0) {
    // Register the loop-nest with its depth:
    hash_put(loop_nests_to_outline, s, (void *)parallel_loop_nest_depth);
    /* Since we only outline outermost loop-nest, stop digging further in
       this statement: */
    return FALSE;
  }
  // This statement is not a parallel loop, go on digging:
  return TRUE;
}


static void
gpu_ify_statement(statement s, int depth) {
  ifdebug(1) {
    pips_debug(1, "Parallel loop-nest of dephth %d\n", depth);
    print_statement(s);
    outliner("kernel", build_new_top_level_module_name(), s);
  }
}


bool gpu_ify(const char * module_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "GPU_IFY_DEBUG_LEVEL");

  // Initialize the loop nest set to outline to the empty set yet:
  loop_nests_to_outline = hash_table_make(hash_chunk, 100);

  // Mark interesting loops:
  gen_recurse(module_statement,
	      statement_domain, mark_loop_to_outline, gen_identity);

  // Outline the previous marked loop nests:
  HASH_MAP(s, depth, {
      gpu_ify_statement(s, (_int) depth);
    }, loop_nests_to_outline);

  hash_table_free(loop_nests_to_outline);

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
			 compute_callees(get_current_module_statement()));

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
}
