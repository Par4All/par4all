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

/* Name     : print.c
 * Package  : prgm_mapping
 * Author   : Alexis Platonoff
 * Date     : septembre 1993
 *
 * Historic :
 * - 23 sept 93, creation, AP
 *
 * Documents:
 * Comments : This file contains the functions used for printing the data
 * structures used for prgm_mapping.
 */

/* Ansi includes 	*/
#include <stdio.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
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
#include "matrice.h"
#include "matrix.h"

/* Pips includes 	*/
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "graph.h"
#include "paf_ri.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"
#include "paf-util.h"

/* Macro functions  	*/

/* Global variables 	*/

/* Internal variables 	*/

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;


/* ======================================================================== */
void fprint_plc(fp, obj)
FILE *fp;
plc obj;
{
  placement func;
  list funcs, dims, d;
  int stmt;

  fprintf(fp, "\nPROGRAM MAPPING :\n================= \n");

  for(funcs = plc_placements(obj); funcs != NIL; funcs = CDR(funcs)) {
    func = PLACEMENT(CAR(funcs));
    stmt = placement_statement(func);
    dims = placement_dims(func);

    /* Mod by AP, oct 6th 95: the number of the instruction is the
       vertex number minus BASE_NODE_NUMBER. */
    fprintf(fp, "Ins_%d :", stmt-BASE_NODE_NUMBER);

    for(d = dims; d != NIL; d = CDR(d)) {
      expression exp = EXPRESSION(CAR(d));
      fprintf(fp, "%s", words_to_string(words_expression(exp)));
      if(CDR(d) == NIL)
        fprintf(fp, "\n");
      else
        fprintf(fp, " , ");
    }
  }
  fprintf(fp, "\n");
}


/* ======================================================================== */
void plc_fprint_proto(fp, g, StmtToProto)
FILE *fp;
graph g;
hash_table StmtToProto;
{

 list l;

 /* For each node of the data flow graph we print its prototype. */
 for(l = graph_vertices(g); l != NIL; l = CDR(l))
   {
    vertex v = VERTEX(CAR(l));
    int stmt = dfg_vertex_label_statement((dfg_vertex_label) vertex_vertex_label(v));

    Ppolynome pp_proto = (Ppolynome) hash_get(StmtToProto, (char *) stmt);

    /* Print */
    fprintf(fp, "ins_%d: ",  stmt);
    polynome_fprint(fp, pp_proto, entity_local_name, pu_is_inferior_var);
    fprintf(fp, "\n");
   }
}


/* ======================================================================== */
void plc_fprint_distance(fp, g, DtfToDist)
FILE *fp;
graph g;
hash_table DtfToDist;
{
  Ppolynome pp_dist;
  list l, su_l, df_l;

  for(l = graph_vertices(g); l != NIL; l = CDR(l)) {
    vertex v = VERTEX(CAR(l));

    fprintf(fp, "source stmt: %d\n", vertex_int_stmt(v));

    su_l = vertex_successors(v);

    for( ; su_l != NIL; su_l = CDR(su_l)) {
      successor su = SUCCESSOR(CAR(su_l));
      int sink_stmt = vertex_int_stmt(successor_vertex(su));

      df_l = dfg_arc_label_dataflows((dfg_arc_label) successor_arc_label(su));

      for( ; df_l != NIL; df_l = CDR(df_l)) {
	dataflow df = DATAFLOW(CAR(df_l));

	fprint_dataflow(fp, sink_stmt, df);

	pp_dist = (Ppolynome) hash_get(DtfToDist, (char *) df);

        fprintf(fp, "DF Dist:");
        polynome_fprint(fp, pp_dist, pu_variable_name, pu_is_inferior_var);
        fprintf(fp, "\n");
      }
    }
  }
}


/* ======================================================================== */
void plc_fprint_dfs(fp, df_l, DtfToStmt, DtfToWgh)
FILE *fp;
list df_l;
hash_table DtfToWgh, DtfToStmt;
{
 list l = df_l;

 fprintf(fp, "Listes triee des flots de donnees :\n");

 for( ; l != NIL; l = CDR(l))
   {
    dataflow df = DATAFLOW(CAR(l));
    fprintf(fp, "Poids %d ***", (int) hash_get(DtfToWgh, (char *) df));
    fprint_dataflow(fp, (int) hash_get(DtfToStmt, (char *) df), df);
   }
  fprintf(fp, "\n");
}


/* ======================================================================== */
/*
 * void fprint_pla_pp_dims(fp, one_placement)
 */
void fprint_pla_pp_dims(fp, one_placement)
FILE *fp;
placement one_placement;
{
  list plc_dims =  placement_dims(one_placement);
  int count = 1;

  fprintf(fp, "\nBroadcast Mapping for %d :\n",
	  placement_statement(one_placement));

  for(; !ENDP(plc_dims); POP(plc_dims), count++) {
    Ppolynome crt_pp = (Ppolynome) CHUNK(CAR(plc_dims));

    fprintf(fp, "Dim %d :", count);
    polynome_fprint(stderr, crt_pp, pu_variable_name, pu_is_inferior_var);
    fprintf(fp, "\n");
  }
}


/* ======================================================================== */
/*
 * void fprint_plc_pp_dims(fp, one_plc)
 */
void fprint_plc_pp_dims(fp, one_plc)
FILE *fp;
plc one_plc;
{
  list pla_l = plc_placements(one_plc);

  fprintf(fp, "\nBROADCAST PROGRAM MAPPING :\n===========================\n");

  for(; !ENDP(pla_l); POP(pla_l)) {
    placement crt_pla = PLACEMENT(CAR(pla_l));
    fprint_pla_pp_dims(fp, crt_pla);
  }
}

