/* A simple phase to show up how to use Xpath to select objects in PIPS */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "control.h"
#include "pipsdbm.h"
#include "resources.h"

bool simple_xpath_test(char * module_name) {
  /* Use this module name and this environment variable to set */

  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "SXT_DEBUG_LEVEL");
  /* Put back the new statement module */
  PIPS_PHASE_POSTLUDE(module_statement);
}
