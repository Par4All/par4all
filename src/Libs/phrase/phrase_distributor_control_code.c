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

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"

/*********************************************************
 * Phase main for PHRASE_DISTRIBUTOR_CONTROL_CODE
 *********************************************************/

bool phrase_distributor_control_code(string module_name)
{
  entity module;
  
  /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      TRUE);
  
  module = local_name_to_top_level_entity(module_name);
  
  set_current_module_statement(stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  
  debug_on("PHRASE_DISTRIBUTOR_DEBUG_LEVEL");

  /* Now do the job */

  pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR_CONTROL_CODE\n");
  pips_debug(2, "END of PHRASE_DISTRIBUTOR_CONTROL_CODE\n");

  /* Check the coherency of data */

  pips_assert("Statement structure is consistent after PHRASE_DISTRIBUTOR_CONTROL_CODE", 
	      gen_consistent_p((gen_chunk*)stat));
	      
  pips_assert("Statement is consistent after PHRASE_DISTRIBUTOR_CONTROL_CODE", 
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

