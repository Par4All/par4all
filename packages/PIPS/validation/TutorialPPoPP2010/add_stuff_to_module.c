/* A simple phase to demonstrate how to add a comment to a module.

 Pedagogical, but also useful for a project...

 Ronan.Keryell@hpc-project.com

*/

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "control.h"
#include "pipsdbm.h"
#include "resources.h"

bool prepend_comment(char * module_name) {
  /* Use this module name and this environment variable to set */

  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "PREPEND_COMMENT_DEBUG_LEVEL");

  // Get the value of the property containing the comment to be prepended
  string comment = get_string_property("PREPEND_COMMENT");

  insert_comments_to_statement(module_statement, comment);

  /* Put back the new statement module */
  PIPS_PHASE_POSTLUDE(module_statement);
}
