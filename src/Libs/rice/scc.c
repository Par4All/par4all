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

#include "local.h"

/*

 this collection of functions implements Tarjan's algorithm to find the
 strongly connected components of a directed graph.

 this algorithm is presented in:
 Types de Donnees et Algorithmes
 Recherche, Tri, Algorithmes sur les Graphes
 Marie-Claude Gaudel, Michele Soria, Christine Froidevaux
 Collection Didactique
 INRIA

 the version implemented here has been modified because of Kennedy's
 algorithm requirements: SCCs are searched for on a sub-graph of graph g
 defined by:

 a set of nodes 'region'
 all arcs of the initial graph whose level is greater than 'level'

 */

/* 
 a set of macros to mark a vertex as 'not visited' or 'visited' and to
 check if a node has already been visited
 */
/* #define MIN(a,b) ((a)>(b)?(b):(a)) */

#define NEW_MARK 1
#define OLD_MARK 2
#define MARK_OLD(v) \
  (sccflags_mark(dg_vertex_label_sccflags((dg_vertex_label) \
					  vertex_vertex_label(v))) = OLD_MARK)
#define MARK_NEW(v) \
  (sccflags_mark(dg_vertex_label_sccflags((dg_vertex_label) \
					  vertex_vertex_label(v))) = NEW_MARK)
#define MARKED_NEW_P(v) \
  (sccflags_mark(dg_vertex_label_sccflags((dg_vertex_label) \
					  vertex_vertex_label(v))) == NEW_MARK)

/* 
 a set of variables shared by the functions of this package. the stack
 contains the current SCC, i.e. the SCC currently being built. Components
 is the result, i.e. a set of scc
 */
static int Count, StackPointer;
static vertex *Stack;

/*
 list of drivers functions used bt the SCC module
 */

static bool (*ignore_this_vertex_drv)(set, vertex) = 0;
static bool (*ignore_this_successor_drv)(vertex, set, successor, int) = 0;

/*
 drivers initialisation
 */

void set_sccs_drivers(ignore_this_vertex_fun, ignore_this_successor_fun)
bool (*ignore_this_vertex_fun)(set, vertex); bool (*ignore_this_successor_fun)(vertex, set, successor, int);
{
  pips_assert( "ignore_this_vertex_drv is set", ignore_this_vertex_drv == 0 );
  pips_assert( "ignore_this_successor_drv is set", ignore_this_successor_drv == 0 );

  ignore_this_vertex_drv = ignore_this_vertex_fun;
  ignore_this_successor_drv = ignore_this_successor_fun;
}

void reset_sccs_drivers() {
  pips_assert( "ignore_this_vertex_drv is not set", ignore_this_vertex_drv != 0 );
  pips_assert( "ignore_this_successor_drv is not set", ignore_this_successor_drv != 0 );
  ignore_this_vertex_drv = 0;
  ignore_this_successor_drv = 0;
}

/*
 LowlinkCompute is the main function. Its behavior is explained in the
 book mentionned ealier.

 g is a graph

 region and level define a sub-graph of g

 v is the current vertex
 */
void LowlinkCompute(graph g, set region, vertex v, int level, sccs Components) {
  dg_vertex_label dvl = (dg_vertex_label)vertex_vertex_label(v);
  sccflags fv = dg_vertex_label_sccflags(dvl);
  statement sv = ordering_to_statement(dg_vertex_label_statement(dvl));

  pips_debug(7, "vertex is %zd (%zd %zd %zd)\n", statement_number(sv),
      sccflags_mark(fv), sccflags_lowlink(fv), sccflags_dfnumber(fv));

  MARK_OLD(v);

  sccflags_lowlink(fv) = Count;
  sccflags_dfnumber(fv) = Count;

  Count++;

  Stack[StackPointer++] = v;
  FOREACH(SUCCESSOR, su, vertex_successors(v)) {

    if(!ignore_this_successor_drv(v, region, su, level)) {
      vertex s = successor_vertex(su);

      if(!ignore_this_vertex_drv(region, s)) {
        dg_vertex_label dsl = (dg_vertex_label)vertex_vertex_label(s);
        sccflags fs = dg_vertex_label_sccflags(dsl);
        statement ss = ordering_to_statement(dg_vertex_label_statement(dsl));

        pips_debug(7, "successor before is %zd (%zd %zd %zd)\n",
            statement_number(ss), sccflags_mark(fs),
            sccflags_lowlink(fs), sccflags_dfnumber(fs));

        if(MARKED_NEW_P(s)) {
          LowlinkCompute(g, region, s, level, Components);
          pips_debug(7, "successor after is %zd (%zd %zd %zd)\n",
              statement_number(ss), sccflags_mark(fs),
              sccflags_lowlink(fs), sccflags_dfnumber(fs));
          sccflags_lowlink(fv) = MIN(sccflags_lowlink(fv),
              sccflags_lowlink(fs));
        } else {
          if((sccflags_dfnumber(fs) < sccflags_dfnumber(fv)) && IsInStack(s)) {
            sccflags_lowlink(fv) = MIN(sccflags_dfnumber(fs),
                sccflags_lowlink(fv));
          }
        }
      }
    }
  }

  if(sccflags_lowlink(fv) == sccflags_dfnumber(fv)) {
    scc ns = make_scc(NIL, 0);
    vertex p;
    sccflags fp;
    cons *pv = NIL;

    do {
      p = Stack[--StackPointer];
      fp = dg_vertex_label_sccflags((dg_vertex_label)
          vertex_vertex_label(p));
      sccflags_enclosing_scc(fp) = ns;
      pv = gen_nconc(pv, CONS(VERTEX, p, NIL));
    } while(v != p);

    scc_vertices(ns) = pv;
    sccs_sccs(Components)
        = gen_nconc(sccs_sccs(Components), CONS(SCC, ns, NIL));
  }
}

/* this function checks if vertex v is in the stack */
int IsInStack(vertex v) {
  int i;

  for (i = 0; i < StackPointer; i++)
    if(Stack[i] == v)
      return (true);

  return (false);
}

/* 
 FindSccs is the interface function to compute the SCCs of a graph. It
 marks all nodes as 'not visited' and then apply the main function
 LowlinkCompute on all vertices.

 A vertex is processed only if it belongs to region. Later, successors
 will be processed if they can be reached through arcs whose level is
 greater or equal to level.

 g is a graph

 region and level define a sub-graph of g
 */
sccs FindSccs(graph g, set region, int level) {
  list vertices = graph_vertices(g);
  sccs Components;

  Count = 1;
  StackPointer = 0;
  Stack = (vertex *)malloc(sizeof(vertex) * gen_length(vertices));
  Components = make_sccs(NIL);
  FOREACH(VERTEX, v, vertices) {
    if(!ignore_this_vertex_drv(region, v)) {
      dg_vertex_label lv = (dg_vertex_label)vertex_vertex_label(v);
      sccflags fv = dg_vertex_label_sccflags(lv);
      if(fv == sccflags_undefined) {
        pips_debug (7, "fv has not been set so far");
        fv = make_sccflags(scc_undefined, 0, 0, 0);
        dg_vertex_label_sccflags(lv) = fv;
      }
      sccflags_mark(fv) = NEW_MARK;
    }
  }
  /* Bug FOREACH */FOREACH(VERTEX, v1, vertices) {
    if(!ignore_this_vertex_drv(region, v1))
      if(MARKED_NEW_P(v1)) {
        LowlinkCompute(g, region, v1, level, Components);
      }
  }

  free(Stack);

  ifdebug(3) {
    pips_debug(3, "Strongly connected components:\n");
    PrintSccs(Components);
    pips_debug(3, "End\n");
  }

  return (Components);
}

void ComputeInDegree(graph g, set region, int l) {
  FOREACH(VERTEX, v, graph_vertices(g))
  {
    if(!ignore_this_vertex_drv(region, v)) {
      scc sv = VERTEX_ENCLOSING_SCC(v);
      FOREACH(SUCCESSOR, su, vertex_successors(v))
      {
        if(!ignore_this_successor_drv(v, region, su, l)) {
          vertex s = successor_vertex(su);
          scc ss = VERTEX_ENCLOSING_SCC(s);
          if(sv != ss)
            scc_indegree(ss) += 1;
        }
      }
    }
  }
}

list TopSortSccs(graph __attribute__ ((unused)) g, set region, int l,sccs Components) {
  list lsccs = NIL, elsccs = NIL, no_ins = NIL;
  FOREACH(SCC, s, sccs_sccs(Components)) {
    if(scc_indegree(s) == 0)
      no_ins = CONS(SCC, s, no_ins);
  }

  while(no_ins != NIL) {
    list pcs;
    scc cs;

    pcs = no_ins;
    no_ins = CDR(no_ins);
    INSERT_AT_END(lsccs, elsccs, pcs);

    pips_debug(3, "updating in degrees ...\n");
    cs = SCC(CAR(pcs));
    FOREACH(VERTEX, v, scc_vertices(cs))
    {
      scc sv = VERTEX_ENCLOSING_SCC(v);
      FOREACH(SUCCESSOR, su, vertex_successors(v))
      {
        if(!ignore_this_successor_drv(v, region, su, l)) {
          vertex s = successor_vertex(su);
          scc ss = VERTEX_ENCLOSING_SCC(s);
          if(sv != ss && (scc_indegree(ss) -= 1) == 0
              && !ignore_this_vertex_drv(region, s)) {
            no_ins = CONS(SCC, ss, no_ins);
          }
        }
      }
    }
  }

  ifdebug(3) {
    fprintf(stderr, "[TopSortSccs] topological order:\n");
    FOREACH(SCC, s, lsccs) {
      fprintf(stderr, "( ");
      FOREACH(VERTEX, v, scc_vertices(s)) {
        statement st = vertex_to_statement(v);

        fprintf(stderr, "%td ", statement_number(st));
      }
      fprintf(stderr, ")   -->   ");
    }
    fprintf(stderr, "\n");
  }

  return (lsccs);
}

/*
 Don't forget to set the drivers before the call of this function and to
 reset after!
 */
list FindAndTopSortSccs(graph g, set region, int l) {
  list lsccs;
  sccs Components;

  ifdebug(8) {
    pips_debug(8, "Dependence graph:\n");
    prettyprint_dependence_graph(stderr, statement_undefined, g);
  }

  pips_debug(3, "computing sccs ...\n");

  Components = FindSccs(g, region, l);

  pips_debug(3, "computing in degrees ...\n");

  ComputeInDegree(g, region, l);

  pips_debug(3, "topological sort ...\n");

  lsccs = TopSortSccs(g, region, l,Components);

//  free_sccs(Components);
  gen_free_list(sccs_sccs(Components));
  free(Components);


  return (lsccs);
}

void PrintScc(scc s) {
  fprintf(stderr, "Scc's statements : ");
  FOREACH(VERTEX, v, scc_vertices(s)) {
    statement st = vertex_to_statement(v);

    fprintf(stderr, "%02td ", statement_number(st));
  }
  fprintf(stderr, " -  in degree : %td\n", scc_indegree(s));
}

void PrintSccs(sccs ss) {
  fprintf(stderr, "Strongly connected components:\n");
  if(!ENDP(sccs_sccs(ss))) {
    MAPL(ps, {PrintScc(SCC(CAR(ps)));}, sccs_sccs(ss));
  } else {
    fprintf(stderr, "Empty list of scc\n");
  }
}
