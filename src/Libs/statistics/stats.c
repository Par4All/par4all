/* 
 * $Id$
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

#include "statistics.h"

static bool loop_flt(loop l)
{
  fprintf(stderr, "IN loop on %s\n", entity_name(loop_index(l)));
  return TRUE;
}

static void loop_rwt(loop l)
{
  fprintf(stderr, "OUT loop on %s\n", entity_name(loop_index(l)));
}


int loop_statistics(string name)
{
  statement stat;

  debug_on("STATISTICS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);

  set_current_module_entity(local_name_to_top_level_entity(name));
 
  stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE); /* ??? */

  gen_context_multi_recurse
    (stat, NULL,
     loop_domain, loop_flt, loop_rwt,
     NULL);

  DB_PUT_FILE_RESOURCE(DBR_STATS_FILE, strdup(name), NULL);

  pips_debug(1, "done.\n");
  debug_off();
  return TRUE;
}
