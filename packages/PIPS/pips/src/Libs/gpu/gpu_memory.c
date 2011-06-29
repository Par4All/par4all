/* A simple phase that allocate memory on an accelerator and add memory
   transfer around a kernel

   Ronan.Keryell@hpc-project.com
*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "control.h"
#include "callgraph.h"
#include "pipsdbm.h"
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
    return false;
  }
  // This statement is not a parallel loop, go on digging:
  return true;
}


static void
gpu_memory_apply(statement s, int depth) {
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
  outliner(build_new_top_level_module_name("kernel_wrapper",false), sk);

  // Outline the kernel launcher:
  list sl = CONS(STATEMENT, s, NIL);
  outliner(build_new_top_level_module_name("kernel_launcher",false), sl);
}


bool gpu_memory(const char * module_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "GPU_IFY_DEBUG_LEVEL");

  // Get the effects and use them:
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS,module_name,true));

  // Apply the transformation on the kernel calls:
  gen_recurse(module_statement, call_domain, gpu_memory_apply, gen_identity);

  // No longer use effects:
  reset_cumulated_rw_effects();

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
			 compute_callees(get_current_module_statement()));

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
}
