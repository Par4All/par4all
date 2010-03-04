/* A simple phase that outline parallel loops onto GPU

   Ronan.Keryell@hpc-project.com
*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

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

/** Store the loop nests found that meet the spec to be executed on a
    GPU. Use a list and not a set or hash_map to have always the same
    order */
list loop_nests_to_outline;


static bool
mark_loop_to_outline(const statement s) {
  /* An interesting loop must be parallel first...

     We recurse on statements instead of loops in order to pick
     informations on the statement itself, such as pragmas
  */
  int parallel_loop_nest_depth = depth_of_parallel_perfect_loop_nest(s);
  ifdebug(3) {
    pips_debug(1, "Statement %td with // depth %d\n", statement_number(s),
	       parallel_loop_nest_depth);
    print_statement(s);
  }
  if (parallel_loop_nest_depth > 0) {
    // Register the loop-nest (note the list is in the reverse order):
    loop_nests_to_outline = CONS(STATEMENT, s, loop_nests_to_outline);
    /* Since we only outline outermost loop-nest, stop digging further in
       this statement: */
    pips_debug(1, "Statement %td marked to be outlined\n", statement_number(s));
    return FALSE;
  }
  // This statement is not a parallel loop, go on digging:
  return TRUE;
}


static void
gpu_ify_statement(statement s, int depth) {
  ifdebug(1) {
    pips_debug(1, "Parallel loop-nest of depth %d\n", depth);
    print_statement(s);
  }
  /* First outline the innermost code (the kernel itself) to avoid
     spoiling its memory effects if we start with the outermost code
     first: */
  list sk = CONS(STATEMENT,
		 perfectly_nested_loop_to_body_at_depth(s, depth),
		 NIL);
  outliner(build_new_top_level_module_name("kernel_wrapper"), sk);

  // Outline the kernel launcher:
  list sl = CONS(STATEMENT, s, NIL);
  outliner(build_new_top_level_module_name("kernel_launcher"), sl);
}


bool gpu_ify(const char * module_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "GPU_IFY_DEBUG_LEVEL");

  // Get the effects and use them:
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE));

  // Initialize the loop nest set to outline to the empty set yet:
  loop_nests_to_outline = NIL;

  // Mark interesting loops:
  gen_recurse(module_statement,
	      statement_domain, mark_loop_to_outline, gen_identity);

  /* Outline the previous marked loop nests.
     First put the statements to outline in the good order: */
  loop_nests_to_outline = gen_nreverse(loop_nests_to_outline);
  FOREACH(STATEMENT, s, loop_nests_to_outline) {
    // We could have stored the depth, but it complexify the code...
    gpu_ify_statement(s, depth_of_parallel_perfect_loop_nest(s));
  }

  gen_free_list(loop_nests_to_outline);

  // No longer use effects:
  reset_cumulated_rw_effects();

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
			 compute_callees(get_current_module_statement()));

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
}
