/*
 * $Id$
 *
 * Unimodular loop transformations with guard elimination.
 */

#include <stdio.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"
#include "database.h"
#include "resources.h"
#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

bool guard_elimination(string module)
{
  statement stat;
                                  
  debug_on("GUARD_ELIMINATION_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module);

  /* get code from pips DBM.
   */
  set_current_module_entity(local_name_to_top_level_entity(module));
  stat = (statement) db_get_memory_resource(DBR_CODE, module, TRUE); 
  set_current_module_statement(stat);

  /* do the job
   */
  

  /* return result to pipsdbm
   */    
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, stat);
  
  reset_current_module_entity();
  reset_current_module_statement();
  
  debug_off();
  
  return TRUE; /* okay ! */
}
