
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"

#include "sac.h"

bool simdizer(char * module_name)
{
   /* get the resources */
   statement module_stmt = (statement)
      db_get_memory_resource(DBR_CODE, module_name, TRUE);

   set_current_module_statement(module_stmt);
   set_current_module_entity(local_name_to_top_level_entity(module_name));
   set_proper_rw_effects((statement_effects)
      db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));
   set_precondition_map((statement_mapping)
      db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE));

   debug_on("VARIABLE_WIDTH_DEBUG_LEVEL");
   /* Now do the job */
   printf("My code is called !!! Sooo sweet....\n");
   debug_off();

   /* update/release resources */
   reset_current_module_statement();
   reset_current_module_entity();
   reset_proper_rw_effects();
   reset_precondition_map();

   return TRUE;
}

