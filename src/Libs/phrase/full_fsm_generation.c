#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "text-util.h"

#include "dg.h"
#include "transformations.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "ricedg.h"
#include "semantics.h"
#include "control.h"

#include "fsm_generation.h"

static graph dependence_graph;

/*********************************************************
 * Phase main
 *********************************************************/

bool full_fsm_generation(string module_name)
{
  entity module;
  entity state_variable;

   /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      TRUE);

  module = local_name_to_top_level_entity(module_name);
  
  set_current_module_statement(stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  dependence_graph = 
    (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);
  
  debug_on("FSM_GENERATION_DEBUG_LEVEL");
  /* Now do the job */
  
  pips_assert("Statement is consistent after FULL_FSM_GENERATION", 
	       statement_consistent_p(stat));
  
  /* Reorder the module, because new statements have been added */  
  module_reorder(stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, 
			 compute_callees(stat));
  
  /* update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
  
  debug_off();
  
  return TRUE;
}
