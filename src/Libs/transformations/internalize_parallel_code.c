/*
  
  For historical reasons, a parallelized program in PIPS is no longer a
  classical PIPS program we can work on later because sequential code and
  parallelized code do not have the same resource type. Unfortunately,
  improving the parallelized code with some other transformations
  (dead-code elimination, etc) is also useful. Thus we add a new
  pseudo-transformation that coerce a parallel code into a classical
  (sequential) one to consider parallelization as an internal
  transformation in PIPS.

  Note this transformation may no be usable with some special
  parallelization in PIPS such as WP65 or HPFC that generate other resource
  types.

  Ronan.Keryell@enstb.org
 */

#ifndef lint
static char vcid[] = "$Id$";
#endif /* lint */


#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "text.h"
#include "ri.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "transformations.h"


/* I love writing so simple passes... :-)

  Basically do only a
  DBR_CODE(mod_name) = (DBR_CODE) DBR_PARALLELIZED_CODE(mod_name) */
bool internalize_parallel_code(char mod_name[])
{
    statement mod_stmt;
    
    debug_on("INTERNALIZE_PARALLEL_CODE_DEBUG_LEVEL");

    /* Get the parallelized code and tell PIPS_DBM we do not want to
       modify it: */
    mod_stmt = (statement) db_get_memory_resource(DBR_PARALLELIZED_CODE,
						  mod_name,
						  FALSE);

    /* Reorder the module, because new statements have been generated. */
    /* module_reorder(mod_stmt); */

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stmt);

    debug(2,"internalize_parallel_code","done for %s\n", mod_name);
    debug_off();

    return TRUE;
}
