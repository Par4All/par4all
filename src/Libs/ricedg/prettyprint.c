/*********************************************************************************/
/* DG PRINTING FUNCTIONS                                                         */
/*********************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"
#include "text.h"
#include "ri.h"
#include "graph.h"
#include "dg.h"
#include "database.h"

#include "misc.h"
#include "text-util.h"

#include "ri-util.h" /* linear.h is included in */
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "pipsdbm.h"
#include "semantics.h"

#include "constants.h"
#include "properties.h"
#include "resources.h"

/* includes for generating systems, needed by ricedg.h */
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ricedg.h"

bool 
print_whole_dependence_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      FALSE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      FALSE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      TRUE);
    return print_dependence_graph(mod_name);
}

bool 
print_effective_dependence_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      TRUE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      FALSE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES", 
		      FALSE);
    return print_dependence_graph(mod_name);
}

bool 
print_loop_carried_dependence_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      TRUE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      TRUE);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      TRUE);
    return print_dependence_graph(mod_name);
}

bool 
print_dependence_graph(mod_name)
string mod_name;
{
    string dg_name = NULL;
    string local_dg_name = NULL;
    FILE *fp;
    graph dg;
    statement mod_stat;

    set_current_module_entity(local_name_to_top_level_entity(mod_name));
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, mod_name, TRUE) );
    mod_stat = get_current_module_statement();
    initialize_ordering_to_statement(mod_stat);

    dg = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);
    local_dg_name = db_build_file_resource_name(DBR_DG, mod_name, ".dg");
    dg_name = strdup(concatenate(db_get_current_workspace_directory(), 
				 "/", local_dg_name, NULL));
    fp = safe_fopen(dg_name, "w");
    
    debug_on("RICEDG_DEBUG_LEVEL");

    if (get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS") || 
	get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS"))
	prettyprint_dependence_graph_view(fp, mod_stat, dg);
    else  
	prettyprint_dependence_graph(fp, mod_stat, dg);
    
    debug_off();
    
    safe_fclose(fp, dg_name);
    free(dg_name);
    
    DB_PUT_FILE_RESOURCE(DBR_DG_FILE, strdup(mod_name), 
 			 local_dg_name);
    
    reset_current_module_statement();
    reset_current_module_entity();
    reset_ordering_to_statement();

    return TRUE;
}

/* That's all */
