/*
 * $Id$
 *
 * Typecheck Fortran code.
 * by Son PhamDinh 03-05/2000
 */

#include <stdio.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"

#include "bootstrap.h" /* type of intrinsics stuff... */

bool type_checker(string name)
{
  statement stat;
  debug_on("TYPE_CHECKER_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);

  stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE);
  set_current_module_statement(stat);

  /* do the job. */

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, name, stat);
  reset_current_module_statement();

  pips_debug(1, "done");
  debug_off();
  return TRUE;
}
