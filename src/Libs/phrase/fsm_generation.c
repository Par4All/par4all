/* 
 *
 * This phase is used for PHRASE project. 
 *
 * NB: The PHRASE project is an attempt to automatically (or
 * semi-automatically) transform high-level language for partial
 * evaluation in reconfigurable logic (such as FPGAs or DataPaths).

* This library provides phases allowing to build and modify "Finite State
* Machine"-like code portions which will be later synthetized in
* reconfigurable units.

* This phase tries to generate finite state machine from arbitrary code by
* applying rules numeroting branches of the syntax tree and using it as
* state variable for the finite state machine.

* This phase recursively transform each UNSTRUCTURED statement in a
* WHILE-LOOP statement controlled by a state variable, whose different
* values are associated to the different statements.

* To add flexibility, the behavior of FSM_GENERATION is controlled by the
* property FSMIZE_WITH_GLOBAL_VARIABLE which control the fact that the
* same global variable (global to the current module) must be used for
* each FSMized statements.

 * alias fsm_generation 'FSM Generation'
 *
 * fsm_generation        > MODULE.code
 *                       > PROGRAM.entities
 *       < PROGRAM.entities
 *       < MODULE.code
 *
 */ 

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
#include "properties.h"

#include "phrase_tools.h"
#include "fsm_generation.h"

//static graph dependence_graph;


/*********************************************************
 * Phase main
 *********************************************************/

bool fsm_generation(string module_name)
{
  entity module;
  entity state_variable = NULL;

   /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      TRUE);

  module = local_name_to_top_level_entity(module_name);
  
  set_current_module_statement(stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  /* dependence_graph = 
     (graph) db_get_memory_resource(DBR_DG, module_name, TRUE); */
  
  debug_on("FSM_GENERATION_DEBUG_LEVEL");

  /* Now do the job */

  /* If property FSMIZE_WITH_GLOBAL_VARIABLE is set to true, we
   * declare here the state variable which will be used in the whole
   * module */
  if (get_bool_property("FSMIZE_WITH_GLOBAL_VARIABLE")) {
    state_variable = create_state_variable(module_name, 0);
  }

  stat = fsmize_statement(stat, state_variable, module_name);
  
  pips_assert("Statement is consistent after FSM_GENERATION", 
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
