/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/*****************************************************************************/
/* DG PRINTING FUNCTIONS                                                     */
/*****************************************************************************/

#include "local.h"

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
print_filtered_dependence_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      TRUE);
    return print_filtered_dg_or_dvdg(mod_name, FALSE);
}

bool
print_filtered_dependence_daVinci_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      TRUE);
    return print_filtered_dg_or_dvdg(mod_name, TRUE);
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

static bool 
print_dependence_or_chains_graph(string mod_name, bool with_dg)
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
    set_ordering_to_statement(mod_stat);

    /* get the dg or chains... */
    dg = (graph) db_get_memory_resource(
	with_dg? DBR_DG: DBR_CHAINS, mod_name, TRUE);

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
    
    DB_PUT_FILE_RESOURCE(DBR_DG_FILE, strdup(mod_name), local_dg_name);
    
    reset_current_module_statement();
    reset_current_module_entity();
    reset_ordering_to_statement();

    return TRUE;
}

bool print_dependence_graph(string name)
{
    return print_dependence_or_chains_graph(name, TRUE);
}

bool print_chains_graph(string name)
{
    return print_dependence_or_chains_graph(name, FALSE);
}

/* That's all */
