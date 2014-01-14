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

#ifndef lint
char vcid_scheduling_scheduling[] = "$Id$";
#endif /* lint */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>

#include "genC.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"

/* C3 includes          */
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "union.h"
#include "matrix.h"


/* Pips includes        */

#include "complexity_ri.h"
#include "database.h"
#include "dg.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "tiling.h"

#include "text.h"
#include "text-util.h"
#include "graph.h"
#include "paf_ri.h"
#include "paf-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "scheduling.h"
#include "array_dfg.h"

/* Local defines */
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;

#define BDT_EXT ".bdt_file"

/*==================================================================*/
/* void print_bdt(module_name): print the bdt of module name
 *
 * AC 94/03/30
 */

bool print_bdt(module_name)

 char     *module_name;
{
  char *localfilename;
  FILE        *fd;
  char        *filename;
  bdt the_bdt;

  debug_on( "PRINT_BDT_DEBUG_LEVEL" );

  if (get_debug_level() > 1)
    user_log("\n\n *** PRINTING BDT for %s\n", module_name);

  the_bdt = (bdt) db_get_memory_resource(DBR_BDT, module_name, true);

  localfilename = strdup(concatenate(module_name, BDT_EXT, NULL));
  filename = strdup(concatenate(db_get_current_workspace_directory(), 
				"/", localfilename, NULL));
  
  fd = safe_fopen(filename, "w");
  fprint_bdt(fd, the_bdt);
  safe_fclose(fd, filename);
  
  DB_PUT_FILE_RESOURCE(DBR_BDT_FILE, strdup(module_name), localfilename);
  
  free(filename);
  
  if(get_debug_level() > 0)
    fprintf(stderr, "\n\n *** PRINT_BDT DONE\n");
  
  debug_off();
  
  return(true);
}

/*==================================================================*/
/* void scheduling(mod_name ): this is the main function to calculate
 * the schedules of the node of a dfg. It first reverse the graph to
 * have each node in function of its predecessors, then calculates
 * the strongly connected components by the Trajan algorithm, then
 * calls the function that really find the schedules.
 *
 * AC 93/10/30
 */

bool scheduling(mod_name)
char            *mod_name;
{
  graph           dfg, rdfg;
  sccs            rgraph;
  bdt             bdt_list;
  entity          ent;
  statement       mod_stat;
  static_control  stco;
  statement_mapping STS;
  
  debug_on("SCHEDULING_DEBUG_LEVEL");
  if (get_debug_level() > 0)
  {
    fprintf(stderr,"\n\nBegin scheduling\n");
    fprintf(stderr,"DEBUT DU PROGRAMME\n");
    fprintf(stderr,"==================\n\n");
  }
  
  /* We get the required data: module entity, code, static_control, dataflow
   * graph, timing function.
   */
  ent = local_name_to_top_level_entity(mod_name);
  
  set_current_module_entity(ent);
  
  mod_stat = (statement)db_get_memory_resource(DBR_CODE, mod_name, true);
  STS = (statement_mapping) db_get_memory_resource(DBR_STATIC_CONTROL,
					    mod_name, true);
  stco = (static_control) GET_STATEMENT_MAPPING(STS, mod_stat);
  
  set_current_stco_map(STS);

  if (stco == static_control_undefined) 
    pips_internal_error("This is an undefined static control !");
  
  if (!static_control_yes(stco)) 
    pips_internal_error("This is not a static control program !");
  
  /* Read the DFG */
  dfg = (graph)db_get_memory_resource(DBR_ADFG, mod_name, true);
  dfg = adg_pure_dfg(dfg);
  rdfg = dfg_reverse_graph(dfg);
  if (get_debug_level() > 5) fprint_dfg(stderr, rdfg);
  rgraph = dfg_find_sccs(rdfg);
  
  if (get_debug_level() > 0) fprint_sccs(stderr, rgraph);
  
  bdt_list = search_graph_bdt(rgraph);
  
  if (get_debug_level() > 0)
  {
    fprintf(stderr,"\n==============================================");
    fprintf(stderr,"\nBase de temps trouvee :\n");
    fprint_bdt_with_stat(stderr, bdt_list);
    fprintf(stderr,"\nEnd of scheduling\n");
  }
  
  DB_PUT_MEMORY_RESOURCE(DBR_BDT, strdup(mod_name), bdt_list);
  
  reset_current_stco_map();
  reset_current_module_entity();

  debug_off();

  return(true);
}
