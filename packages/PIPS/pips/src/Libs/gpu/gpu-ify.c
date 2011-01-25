/* A simple phase that outline parallel loops onto GPU

   Ronan.Keryell@hpc-project.com
*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

// To have asprintf:
#include <stdio.h>

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
#include "accel-util.h"
#include "resources.h"
#include "properties.h"

/** Store the loop nests found that meet the spec to be executed on a
    GPU. Use a list and not a set or hash_map to have always the same
    order */
static list loop_nests_to_outline;


/* These are the possibles prefixes for outline stuff, they are computed from a
 * property and the current module name
 */
static string kernel_prefix   = 0;
static string wrapper_prefix  = 0;
static string launcher_prefix = 0;
static string fwrapper_prefix = 0;


/* Return a pointer on the first char after the bad_prefix */
static const char* clean_prefix(const char* full_name, string bad_prefix) {
  int len = strlen(bad_prefix);
  if(strncasecmp(full_name,bad_prefix,len)==0) {
    full_name = full_name+len;
  }
  // Jump over separator
  if(*full_name=='_') full_name++;
  return full_name;
}

/**
 * Trying to get only the original function name without prefix
 *
 */
static const char* get_clean_mod_name(const char *mod_name) {

  kernel_prefix   = get_string_property("GPU_KERNEL_PREFIX");
  launcher_prefix = get_string_property("GPU_LAUNCHER_PREFIX");
  wrapper_prefix  = get_string_property("GPU_WRAPPER_PREFIX");
  fwrapper_prefix = get_string_property("GPU_FORTRAN_WRAPPER_PREFIX");

  const char * clean_mod_name = mod_name;

  clean_mod_name = clean_prefix(clean_mod_name,launcher_prefix);
  clean_mod_name = clean_prefix(clean_mod_name,fwrapper_prefix);
  clean_mod_name = clean_prefix(clean_mod_name,wrapper_prefix);
  clean_mod_name = clean_prefix(clean_mod_name,kernel_prefix);
  return clean_mod_name;
}

/**
 * Build the outline function name
 */
static string build_outline_name(const char *base_prefix,
                                      const char *mod_name) {
  bool name_suffix_p = get_bool_property("GPU_OUTLINE_SUFFIX_WITH_OWNER_NAME");

  char *prefix;
  if(name_suffix_p) {
    // strdup because concatenate is used during build_new_top_level_module_name
    prefix = strdup(concatenate(base_prefix,"_",mod_name,NULL));
  } else {
    prefix = strdup(base_prefix);
  }

  string outline_name = build_new_top_level_module_name(prefix,true);

  free(prefix);

  return outline_name;
}



#if 0
/* Get the intrinsic function to get iteration coordinate

   @param coordinate is the coordinate number (0, 1...)

   @return the entity of the intrinsics
 */
static entity get_coordinate_intrinsic(int coordinate) {
  // Get the iteration coordinate intrinsic, for example P4A_vp_1:
  string coord_name;
  asprintf(&coord_name,
	   get_string_property("GPU_COORDINATE_INTRINSICS_FORMAT"),
	   coordinate);
  entity coord_intrinsic = FindOrMakeDefaultIntrinsic(coord_name, 1);
  free(coord_name);
  return  coord_intrinsic;
}
#endif


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




/* Transform a loop nest into a GPU kernel

   @param s is the loop-nest statement

   @param depth is the number of loop in the loop nest to be taken out as
   the GPU iterators

   Several properties can be used to change the behviour of this function,
   as explained in pipsmake-rc

   For example is depth = 2 and s is:
   for(i = 1; i <= 499; i += 1)
      for(j = 1; j <= 499; j += 1)
         save[i][j] = 0.25*(space[i-1][j]+space[i+1][j]+space[i][j-1]+space[i][j+1]);

   it generates something like:
   [...]
   If the GPU_USE_LAUNCHER property is true, this kind of function is generated:
   void p4a_kernel_launcher_0(float_t save[501][501], float_t space[501][501])
   {
     int i;
     int j;
   for(i = 1; i <= 499; i += 1)
      for(j = 1; j <= 499; j += 1)

         p4a_kernel_wrapper_0(save, space, i, j);
   }

   If the GPU_USE_WRAPPER property is true, this kind of function is generated:
   void p4a_kernel_wrapper_0(float_t save[501][501], float_t space[501][501], int i, int j)
   {
     // To be assigned to a call to P4A_vp_0: i
     // To be assigned to a call to P4A_vp_1: j
     p4a_kernel_0(save, space, i, j);
   }

   If the GPU_USE_KERNEL property is true, this kind of function is generated:
   void p4a_kernel_0(float_t save[501][501], float_t space[501][501], int
                     i, int j) {
     save[i][j] = 0.25*(space[i-1][j]+space[i+1][j]+space[i][j-1]+space[i][j+1]);
   }

*/
static void
gpu_ify_statement(statement s, int depth, const char* mod_name) {
  ifdebug(1) {
    pips_debug(1, "Parallel loop-nest of depth %d\n", depth);
    print_statement(s);
  }
  // Get the statement inside the loop-nest:
  statement inner = perfectly_nested_loop_to_body_at_depth(s, depth);

  /* If we want to oultine a kernel: */
  if (get_bool_property("GPU_USE_KERNEL")) {
    /* First outline the innermost code (the kernel itself) to avoid
       spoiling its memory effects if we start with the outermost code
       first. The kernel name with a prefix defined in the
       GPU_KERNEL_PREFIX property: */
    list sk = CONS(STATEMENT, inner, NIL);
    outliner(build_outline_name(kernel_prefix, mod_name),sk);
    //insert_comments_to_statement(inner, "// Call the compute kernel:");
  }

  /* Do we need to insert a wrapper phase to reconstruct iteration
     coordinates from hardware intrinsics? */
  if (get_bool_property("GPU_USE_WRAPPER")) {
    /* Add index initialization from GPU coordinates, in the reverse order
       since we use insert_comments_to_statement() to avoid furthering the
       first statement from its original comment: */
    for(int i = depth - 1; i >= 0; i--) {
      entity index = perfectly_nested_loop_index_at_depth(s, i);
      // Get the iteration coordinate intrinsic, for example P4A_vp_1:
      /*
	This code makes a
resource SUMMARY_EFFECTS[p4a_kernel_launcher_1] is in 'required' status since 149
resource CUMULATED_EFFECTS[p4a_kernel_launcher_1] is in 'required' status since 152
resource PROPER_EFFECTS[p4a_kernel_launcher_1] is in 'required' status since 152
resource SUMMARY_EFFECTS[p4a_kernel_wrapper_1] is in 'required' status since 152
resource CUMULATED_EFFECTS[p4a_kernel_wrapper_1] is in 'required' status since 155
resource PROPER_EFFECTS[p4a_kernel_wrapper_1] is in 'required' status since 155
user error in rmake: recursion on resource SUMMARY_EFFECTS of p4a_kernel_wrapper_1
      statement assign = make_assign_statement(entity_to_expression(index),
					       MakeUnaryCall(get_coordinate_intrinsic(i),
     entity_to_expression(index)));
     So keep simple right now
      */

      /* Add a comment to know what to do later: */
      string comment;
      string intrinsic_name;
      /* Map the inner loop index (numbered i) with the lower GPU
	 coordinate (numbered depth - 1 - i)). In this way, if the code
	 was cache-friendly, it should remain GPU-memory friendly

	 Build the intrinsics of this form: P4A_vp_<depth - 1 - i>
      */
      asprintf(&intrinsic_name,
	       get_string_property("GPU_COORDINATE_INTRINSICS_FORMAT"),
	       depth - 1 - i);
      /* Add a comment in the form of

	 To be replaced with a call to P4A_vp_1: j

	 that may replaced by a post-processor later by

	 j = P4A_vp_1();
	 or whatever according to the target accelerator
      */
      asprintf(&comment, "%s To be assigned to a call to %s: %s\n",
	       c_module_p(get_current_module_entity()) ? "//" : "C",
	       intrinsic_name,
	       entity_user_name(index));
      free(intrinsic_name);
      insert_comments_to_statement(inner, comment);
    }

    /* Then outline the innermost code again (the kernel wrapper) that owns
       the kernel call. The kernel wrapper name with a prefix defined in the
       GPU_WRAPPER_PREFIX property: */
    list sk = CONS(STATEMENT, inner, NIL);
    outliner(build_outline_name(wrapper_prefix, mod_name), sk);
    //insert_comments_to_statement(inner, "// Call the compute kernel wrapper:");
  }

  if (get_bool_property("GPU_USE_LAUNCHER")) {
    /* Outline the kernel launcher with a prefix defined in the
       GPU_LAUNCHER_PREFIX property: */
    list sl = CONS(STATEMENT, s, NIL);
    statement st;
    st = outliner(build_outline_name(launcher_prefix, mod_name), sl);
    if (get_bool_property("GPU_USE_FORTRAN_WRAPPER")) {
      string fwp = strdup(concatenate(fwrapper_prefix,"_",mod_name,NULL));
      ifdebug(3) {
        pips_debug(1, "Outline Fortan_wrapper with prefix %s\n", fwp);
      }
      outliner (build_new_top_level_module_name(fwp, true),CONS(STATEMENT,st,NIL));
	  free(fwp);
    }
    //insert_comments_to_statement(inner, "// Call the compute kernel launcher:");
  }
}

/* Transform all the parallel loop nests of a module into smaller
   independent functions suitable for GPU-style accelerators.

   What can be done is more detailed in gpu_ify_statement().  The various
   functions are generated or not according to different properties.

   @param module_name is the name of the module to work on.

   @return TRUE since it should succeed...
*/
bool gpu_ify(const string mod_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(mod_name,
						  "GPU_IFY_DEBUG_LEVEL");

  // Get the effects and use them:
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,mod_name,TRUE));

  // Initialize the loop nest set to outline to the empty set yet:
  loop_nests_to_outline = NIL;

  // Mark interesting loops:
  gen_recurse(module_statement,
	      statement_domain, mark_loop_to_outline, gen_identity);

  /* Outline the previous marked loop nests.
     First put the statements to outline in the good order: */
  loop_nests_to_outline = gen_nreverse(loop_nests_to_outline);

  /* Clean module name from prefix */
  const char* clean_mod_name=get_clean_mod_name(global_name_to_user_name(entity_name(get_current_module_entity())));

  FOREACH(STATEMENT, s, loop_nests_to_outline) {
    // We could have stored the depth, but it complexifies the code...
    gpu_ify_statement(s, depth_of_parallel_perfect_loop_nest(s),clean_mod_name);
  }

  gen_free_list(loop_nests_to_outline);

  // No longer use effects:
  reset_cumulated_rw_effects();

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name,
			 compute_callees(get_current_module_statement()));

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
  // The macro above does a "return TRUE" indeed.
}
