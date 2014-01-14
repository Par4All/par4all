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
/*****************************************************************************/
/* DG PRINTING FUNCTIONS                                                     */
/*****************************************************************************/

#include "local.h"
#include "genC.h"

bool print_whole_dependence_graph(string mod_name)
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      false);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      false);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      true);
    return print_dependence_graph(mod_name);
}

bool print_filtered_dependence_graph(string mod_name)
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      true);
    return print_filtered_dg_or_dvdg(mod_name, false);
}

bool
print_filtered_dependence_daVinci_graph(mod_name)
string mod_name;
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      true);
    return print_filtered_dg_or_dvdg(mod_name, true);
}

bool print_effective_dependence_graph(string mod_name)
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      true);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      false);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      false);
    return print_dependence_graph(mod_name);
}

bool print_loop_carried_dependence_graph(string mod_name)
{
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS",
		      true);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS",
		      true);
    set_bool_property("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES",
		      true);
    return print_dependence_graph(mod_name);
}

static bool print_dependence_or_chains_graph(string mod_name, bool with_dg)
{
    string dg_name = NULL;
    string local_dg_name = NULL;
    FILE *fp;
    graph dg;
    statement mod_stat;

    set_current_module_entity(local_name_to_top_level_entity(mod_name));
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, mod_name, true) );
    mod_stat = get_current_module_statement();
    set_ordering_to_statement(mod_stat);

    /* get the dg or chains... */
    dg = (graph) db_get_memory_resource(
	with_dg? DBR_DG: DBR_CHAINS, mod_name, true);

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

    return true;
}

bool print_dependence_graph(string name)
{
    return print_dependence_or_chains_graph(name, true);
}

bool print_chains_graph(string name)
{
    return print_dependence_or_chains_graph(name, false);
}


/** \fn static bool print_dot_dependence_or_chains_graph( string mod_name,
 *                                                        bool with_dg )
 *  \brief Output dependence graph in a file in graphviz dot format
 *  \param mod_name is the name of the module
 *  \param with_dg is a flag indicating if it should use the chains or the dg
 *  \return always true (pipsmake will be happy)
 */
static bool print_dot_dependence_or_chains_graph(string mod_name,
						 bool with_dg )
{
  string dg_name = NULL;
  string local_dg_name = NULL;
  FILE *fp;
  graph dg;
  statement mod_stat;

  set_current_module_entity( local_name_to_top_level_entity( mod_name ) );
  set_current_module_statement( (statement) db_get_memory_resource( DBR_CODE,
								    mod_name,
								    true ) );
  mod_stat = get_current_module_statement( );
  set_ordering_to_statement( mod_stat );

  // get the dg or chains...
  dg = (graph) db_get_memory_resource( with_dg ? DBR_DG : DBR_CHAINS,
				       mod_name,
				       true );

  // Prepare the output file
  local_dg_name = db_build_file_resource_name( DBR_DG, mod_name, ".dot" );
  dg_name = strdup( concatenate( db_get_current_workspace_directory( ),
				 "/",
				 local_dg_name,
				 NULL ) );
  fp = safe_fopen( dg_name, "w" );


  debug_on( "RICEDG_DEBUG_LEVEL" );

  // Print the graph in the file
  prettyprint_dot_dependence_graph( fp, mod_stat, dg );

  debug_off( );

  safe_fclose( fp, dg_name );
  free( dg_name );

  // FIXME strdup result should be freed,
  DB_PUT_FILE_RESOURCE( DBR_DOTDG_FILE, strdup( mod_name ), local_dg_name );

  reset_current_module_statement( );
  reset_current_module_entity( );
  reset_ordering_to_statement( );

  return true;
}

/** \fn bool print_dot_chains_graph(string name)
 *  \brief This pipmake pass output the chains in
 *  a file usable by the graphviz tool dot.
 *  \param name is the name of the module, given by pipsmake
 *  \return always true ? see print_dot_dependence_graph
 */
bool print_dot_chains_graph( string name ) {
  return print_dot_dependence_or_chains_graph( name, false );
}

/** \fn bool print_dot_dependence_graph(string name)
 *  \brief This pipmake pass output the DG in
 *  a file usable by graphviz tool dot.
 *  \param name is the name of the module, given by pipsmake
 *  \return always true ? see print_dot_dependence_graph
 */
bool print_dot_dependence_graph( string name ) {
  return print_dot_dependence_or_chains_graph( name, true );
}
