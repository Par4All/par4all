/* Simple phases to show how to add stuff to a module: a comment, a call,...

 Pedagogical, but also useful for RK for a project...

 Ronan.Keryell@hpc-project.com
 François Irigoin
*/

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"       // for debug_on()
#include "control.h"    // for module_reorder()
#include "callgraph.h"  // for compute_callees()
#include "properties.h" // for get_string_property()
#include "pipsdbm.h"
#include "resources.h"


/** @defgroup Pedagogical_phases Pedagogical phases in PIPS

    This module introduces some pedagogical minimal phases to dive into
    PIPS development, with some examples of programming and documentation
    style (such as this module itself!)

    @{
*/

/** Add a comment at the begin of a module code.

    It is a pedagogical phase but is also used for CUDA generation.

    @param module_name is the name of the module to apply this phase on.

    @return a boolean which is TRUE if it works. Well, indeed this phase
    assumes to always work...
*/
bool prepend_comment(char * module_name) {
  /* Use this module name and this environment variable to set */

  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "PREPEND_COMMENT_DEBUG_LEVEL");

  // Get the value of the property containing the comment to be prepended
  string comment = get_string_property("PREPEND_COMMENT");

  insert_comments_to_statement(module_statement, comment);

  /* Put back the new statement for the module */
  PIPS_PHASE_POSTLUDE(module_statement);
}


/** This function adds a call to ficticious global function "MY_TRACK" as
    first executabble statement of module "mn" in C89 style. It is not
    supposed to be of any use beyond training new comers to PIPS.

    Function "MY_TRACK" is assumed to take no arguments and to return
    no results. No source code is available for it.

    It is written to show how to use Newgen primitives in PIPS PPoPP
    2010 tutorial, not to be used. See make_empty_subroutine() in
    ri-util/entity.c for a similar function which might be moved into
    ri-util/module.c.

    @param mn is the name of the module to apply this phase on.

    @return TRUE as we assume it always works, although repetitive
    calls to make_entity() with the same global name might
    fail:-). Not tested on Fortran code. No non-regression tests for
    it either.
 */
bool prepend_call(string mn) {
  type rt = make_type_void();
  functional ft = make_functional(NIL, rt);
  type t = make_type_functional(ft);
  storage st = make_storage_rom();
  code co = make_code(NIL, strdup(""), make_sequence(NIL),NIL,
		      make_language_unknown());
  value v = make_value_code(co);
  string name = "MY_TRACK";
  string ffn = strdup(concatenate(TOP_LEVEL_MODULE_NAME,
				  MODULE_SEP_STRING,
				  name,
				  NULL));
  /* This works only once. So use FindOrCreateEntity() instead... or
     rely on functions in library ri-util. */
  entity f = make_entity(ffn, t, st, v);
  call ca = make_call(f, NIL);
  instruction i = make_instruction_call(ca);
  statement ns = instruction_to_statement(i);
  statement module_statement = PIPS_PHASE_PRELUDE(mn,
						  "PREPEND_CALL_DEBUG_LEVEL");
  pips_assert("f really is an entity", check_entity(f));
  insert_statement(module_statement, ns, TRUE);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mn, compute_callees(module_statement));
  PIPS_PHASE_POSTLUDE(module_statement);
}

/** End of this group
    @} */
