/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
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
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"

#include "text-util.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "control.h"
#include "callgraph.h"
#include "properties.h"

#include "phrase_tools.h"
#include "fsm_generation.h"

//static graph dependence_graph;


/*********************************************************
 * Phase main
 *********************************************************/

bool fsm_generation(const char* module_name)
{
  entity state_variable = NULL;

   /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      true);

  
  set_current_module_statement(stat);
  set_current_module_entity(module_name_to_entity(module_name)); // FI: redundant
  /* dependence_graph = 
     (graph) db_get_memory_resource(DBR_DG, module_name, true); */
  
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
  
  return true;
}
