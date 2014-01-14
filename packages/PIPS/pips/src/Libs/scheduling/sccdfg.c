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
char vcid_scheduling_sccdfg[] = "$Id$";
#endif /* lint */
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "genC.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "union.h"
#include "matrix.h"

#include "ri.h"
#include "graph.h"
#include "dg.h"

#include "misc.h"
#include "ri-util.h"
#include "properties.h"


#include "constants.h"
#include "rice.h"
#include "ricedg.h"

#include "complexity_ri.h"
#include "database.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "tiling.h"

#include "text.h"
#include "text-util.h"
#include "graph.h"
#include "pipsdbm.h"
#include "paf_ri.h"
#include "paf-util.h"
#include "resources.h"
#include "scheduling.h"


/* instantiation of the dependence graph */

typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;

/*====================================================================*/
/* This collection of functions implements Tarjan's algorithm to find 
 * the strongly connected components of a directed graph.
 *
 * this algorithm is presented in:
 *   Types de Donnees et Algorithmes
 *   Recherche, Tri, Algorithmes sur les Graphes
 *   Marie-Claude Gaudel, Michele Soria, Christine Froidevaux
 *   Collection Didactique
 *   INRIA
 *====================================================================*/


/* A set of macros to mark a vertex as 'not visited' or 'visited' and */
/* to check if a node has already been visited.                       */

#define MY_MIN(a,b) ((a)>(b)?(b):(a))

#define NEW_MARK 1
#define OLD_MARK 2
#define DFG_MARK_OLD(v) \
  (sccflags_mark(dfg_vertex_label_sccflags((dfg_vertex_label) \
					  vertex_vertex_label(v))) = OLD_MARK)
#define DFG_MARK_NEW(v) \
  (sccflags_mark(dfg_vertex_label_sccflags((dfg_vertex_label) \
					  vertex_vertex_label(v))) = NEW_MARK)
#define DFG_MARKED_NEW_P(v) \
  (sccflags_mark(dfg_vertex_label_sccflags((dfg_vertex_label) \
					  vertex_vertex_label(v))) == NEW_MARK)

#define DFG_VERTEX_ENCLOSING_SCC(v) \
   (sccflags_enclosing_scc(dfg_vertex_label_sccflags((dfg_vertex_label)\
                                 	  vertex_vertex_label(v))))


/* A set of variables shared by the functions of this package. The  */
/* stack contains the current SCC, i.e. the SCC currently being     */
/* built. Components is the result, i.e. a set of scc.               */

static int Count, StackPointer;
static vertex *Stack;
static sccs Components;

/*==================================================================*/
/* int dfg_is_in_stack(v) : this function checks if vertex v is in the 
 * stack.
 *
 * AC 93/10/18
 */

int dfg_is_in_stack(v)

 vertex  v;
{
 int     i;

 for (i = 0; i < StackPointer; i++)
     if (Stack[i] == v)  return(true);

 return(false);
}

/*==================================================================*/
/* dfg_low_link_compute() is the main function. Its behavior is 
 * explained in the book mentionned ealier.
 *
 * AC 93/10/18
 */

void dfg_low_link_compute(g, v)

 graph   g;
 vertex  v;
{
 dfg_vertex_label dvl = (dfg_vertex_label)vertex_vertex_label(v);
 sccflags fv = dfg_vertex_label_sccflags(dvl);

 DFG_MARK_OLD(v);

 sccflags_lowlink(fv) = Count;
 sccflags_dfnumber(fv) = Count;

 Count ++;

 Stack[StackPointer++] = v;

 MAPL(ps, {
      vertex s = successor_vertex(SUCCESSOR(CAR(ps)));
      if (s != (vertex)NIL)
	 {
          sccflags fs = dfg_vertex_label_sccflags((dfg_vertex_label)\
                                                vertex_vertex_label(s));
	  if (DFG_MARKED_NEW_P(s))
	     {
	      dfg_low_link_compute(g, s);
	      sccflags_lowlink(fv) = MY_MIN(sccflags_lowlink(fv),\
	         			 sccflags_lowlink(fs));
	     } 
	  else
	     {
	      if ((sccflags_dfnumber(fs) < sccflags_dfnumber(fv)) &&\
	        	dfg_is_in_stack(s)) 
	         sccflags_lowlink(fv) = MY_MIN(sccflags_dfnumber(fs),\
					    sccflags_lowlink(fv));
	     }
	 }
    }, vertex_successors(v));

 if (sccflags_lowlink(fv) == sccflags_dfnumber(fv)) 
    {
     scc      ns = make_scc(NIL, 0);
     vertex   p;
     sccflags fp;
     cons     *pv = NIL;

     do
	{
         p = Stack[--StackPointer];
	 fp = dfg_vertex_label_sccflags((dfg_vertex_label)\
					  vertex_vertex_label(p));
	 sccflags_enclosing_scc(fp) = ns;
	 pv = gen_nconc(pv, CONS(VERTEX, p, NIL));
	} while (v != p);

     scc_vertices(ns) = pv;
     sccs_sccs(Components) = gen_nconc(sccs_sccs(Components),\
					       CONS(SCC, ns, NIL));
    }
}

/*==================================================================*/ 
/* dfg_find_sccs is the interface function to compute the SCCs of a
 * graph. It marks all nodes as 'not visited' and then apply the
 * main function dfg_low_link_compute() on all vertices.
 * 
 * AC 93/10/19
 */

sccs dfg_find_sccs(g)

 graph  g;
{
 cons   *vertices = graph_vertices(g), *pv;

 Count = 1;
 StackPointer = 0;
 Stack = (vertex *)malloc(sizeof(vertex) * gen_length(vertices));
 Components = make_sccs(NIL);

 for (pv = vertices; pv != NIL; pv = CDR(pv)) 
    {
     vertex v = VERTEX(CAR(pv));
     dfg_vertex_label lv = (dfg_vertex_label) vertex_vertex_label(v);
     dfg_vertex_label_sccflags(lv) = make_sccflags(scc_undefined,1,0,0);
    }

 MAPL(pv, {
    	   vertex v = VERTEX(CAR(pv));
           if (DFG_MARKED_NEW_P(v))  dfg_low_link_compute(g, v);
          }, vertices);

 free(Stack);

 return(Components);
}

/*============================================================================*/
/* void fprint_sccs(FILE *fp, sccs obj): prints in the file "fp" the
 * list of strongly connected components "obj".
 *
 * AC 93/10/20
 */

void fprint_sccs(fp, obj)

 FILE  *fp;
 sccs  obj;
{
 list  nodes_l, su_l, df_l, scc_l;
 int   source_stmt, sink_stmt;
 int   iter = 0;

 fprintf(fp,"\n Graphe des Composantes Connexes :\n");
 fprintf(fp,"==================================\n");

 for(scc_l = sccs_sccs(obj); scc_l != NIL; scc_l = CDR(scc_l))
   {
    scc scc_an = SCC(CAR(scc_l));
    iter++;
    fprintf(fp,"\ncomposante %d :\n",iter);
    for(nodes_l=scc_vertices(scc_an);nodes_l !=NIL;nodes_l=CDR(nodes_l))
      {
       vertex crt_v = VERTEX(CAR(nodes_l));
       source_stmt = vertex_int_stmt(crt_v);
       fprintf(fp,"\tins_%d:\n", source_stmt);
       su_l = vertex_successors(crt_v);
       for( ; su_l != NIL; su_l = CDR(su_l))
         {
          successor su = SUCCESSOR(CAR(su_l));
          sink_stmt = vertex_int_stmt(successor_vertex(su));
          df_l= dfg_arc_label_dataflows((dfg_arc_label)successor_arc_label(su));
          for ( ; df_l != NIL; df_l = CDR(df_l))
             fprint_dataflow(fp, sink_stmt, DATAFLOW(CAR(df_l)));
         }
      }
   }
}

/*==================================================================*/
/* The following functions are used to build the reverse graph of a
 * dataflow graph
 *==================================================================*/

/*==================================================================*/
/* vertex  dfg_in_vertex_list((list) l, (vertex) v):
 * Input  : A list l of vertices.
 *          A vertex v of a dataflow graph.
 * Returns vertex_undefined if v is not in list l.
 * Returns v' that has the same statement_ordering than v.
 *
 * AC 93/10/19
 */

vertex dfg_in_vertex_list(l, v)

 list       l;
 vertex     v;
{
 vertex     ver;
 int        p, i;

 i = dfg_vertex_label_statement((dfg_vertex_label)\
                                        vertex_vertex_label(v));
 for ( ; !ENDP(l); POP(l)) 
    {
     ver = VERTEX(CAR(l));
     p = dfg_vertex_label_statement((dfg_vertex_label)\
                                          vertex_vertex_label(ver));
     if (p == i) return(ver);
    }

 return (vertex_undefined);
}


/*=======================================================================*/
/* graph dfg_reverse_graph((graph) g):
 * This function is used to reverse Pips's graph in order to have
 * all possible sources directly (Feautrier's dependance graph).
 *
 * AC 93/10/19
 */


graph dfg_reverse_graph(g)

 graph       g;
{
 graph       rev_graph = graph_undefined;
 list        vlist = NIL, li = NIL;
 successor   succ, succ2;
 vertex      v1, v2, v3, v4, v5;

 MAPL(ver_ptr,{
      v1 = VERTEX(CAR(ver_ptr));
      v5 = dfg_in_vertex_list(vlist, v1);
      if (v5 == vertex_undefined) 
         {
          v2 = make_vertex(copy_dfg_vertex_label((dfg_vertex_label)\
			     vertex_vertex_label(v1)), NIL);
          ADD_ELEMENT_TO_LIST(vlist, VERTEX, v2);
         }
      else v2 = v5;

      MAPL(succ_ptr, {
	   li = NIL;
           succ = SUCCESSOR(CAR(succ_ptr));
           v3  = successor_vertex(succ);
           v5  = dfg_in_vertex_list(vlist, v3);
           succ2 = make_successor((dfg_arc_label)\
		                     successor_arc_label(succ), v2);
           if (v5 == vertex_undefined) 
	      {
               ADD_ELEMENT_TO_LIST(li, SUCCESSOR, succ2);
               v4 = make_vertex(copy_dfg_vertex_label((dfg_vertex_label)\
                                  vertex_vertex_label(v3)), li);
               ADD_ELEMENT_TO_LIST(vlist, VERTEX, v4);
              }
           else
               ADD_ELEMENT_TO_LIST(vertex_successors(v5),\
                                                SUCCESSOR, succ2);
          }, vertex_successors(v1));
     }, graph_vertices(g));
 rev_graph = make_graph(vlist);

 return(rev_graph);
}

/***********************************************************************/ 
