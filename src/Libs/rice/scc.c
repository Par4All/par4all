
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
static sccs Components;



/*
LowlinkCompute is the main function. its behavior is explained in the
book mentionned ealier.

g is a graph

region and level define a sub-graph of g

v is the current vertex 
*/
void LowlinkCompute(g, region, v, level)
graph g;
set region;
vertex v;
int level;
{
    dg_vertex_label dvl = (dg_vertex_label) vertex_vertex_label(v);
    sccflags fv = dg_vertex_label_sccflags(dvl);
    statement sv = ordering_to_statement(dg_vertex_label_statement(dvl));

    debug(7, "llc", "vertex is %d (%d %d %d)\n", statement_number(sv), 
	  sccflags_mark(fv), sccflags_lowlink(fv), sccflags_dfnumber(fv));

    MARK_OLD(v);

    sccflags_lowlink(fv) = Count;
    sccflags_dfnumber(fv) = Count;

    Count ++;

    Stack[StackPointer++] = v;

    MAPL(ps, {
	successor su = SUCCESSOR(CAR(ps));

	if (! ignore_this_successor(region, su, level)) {
	    vertex s = successor_vertex(su);

	    if (! ignore_this_vertex(region, s)) {
		dg_vertex_label dsl = (dg_vertex_label) vertex_vertex_label(s);
		sccflags fs = dg_vertex_label_sccflags(dsl);
		statement ss = 
		    ordering_to_statement(dg_vertex_label_statement(dsl));

		debug(7, "llc", "successor before is %d (%d %d %d)\n", 
		      statement_number(ss), sccflags_mark(fs),
		      sccflags_lowlink(fs), sccflags_dfnumber(fs));

		if (MARKED_NEW_P(s)) {
		    LowlinkCompute(g, region, s, level);
		    debug(7, "llc", "successor after is %d (%d %d %d)\n", 
			  statement_number(ss), sccflags_mark(fs),
			  sccflags_lowlink(fs), sccflags_dfnumber(fs));
		    sccflags_lowlink(fv) = MIN(sccflags_lowlink(fv),
					       sccflags_lowlink(fs));
		} 
		else {
		    if ((sccflags_dfnumber(fs) < sccflags_dfnumber(fv)) &&
			IsInStack(s)) {
			sccflags_lowlink(fv) = MIN(sccflags_dfnumber(fs),
						   sccflags_lowlink(fv));
		    }
		}
	    }
	}
    }, vertex_successors(v));

    if (sccflags_lowlink(fv) == sccflags_dfnumber(fv)) {
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
	} while (v != p);

	scc_vertices(ns) = pv;
	sccs_sccs(Components) = gen_nconc(sccs_sccs(Components), 
					  CONS(SCC, ns, NIL));
    }
}



/* this function checks if vertex v is in the stack */
int IsInStack(v)
vertex v;
{
    int i;

    for (i = 0; i < StackPointer; i++)
	if (Stack[i] == v)
	    return(TRUE);

    return(FALSE);
}



/* 
this function checks if a vertex v should be ignored, i.e. does not
belong to region
*/
bool ignore_this_vertex(region, v)
set region;
vertex v;
{
    dg_vertex_label dvl = (dg_vertex_label) vertex_vertex_label(v);
    statement st = ordering_to_statement(dg_vertex_label_statement(dvl));

    return(! set_belong_p(region, (char *) st));
}



/*
this function checks if a successoressor su of a vertex should be ignored,
i.e.  if it is linked through an arc whose level is less than 'level' or
if it does not belong to region
*/
bool ignore_this_successor(region, su, level)
set region;
successor su;
int level;
{
    if (ignore_this_vertex(region, successor_vertex(su)))
	return(TRUE);

    return(ignore_this_level((dg_arc_label) successor_arc_label(su), level));
}



/* 
this function checks if a successor su of a vertex is accessible
through an arc whose level is less than 'level'

dal is the arc label

level is the minimum level
*/
bool ignore_this_level(dal, level)
dg_arc_label dal;
int level;
{
    bool true_dep = get_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY" ) ;

    MAPL(pc, {
	conflict c = CONFLICT(CAR(pc));

	if( conflict_cone(c) != cone_undefined ) {
	    MAPL(pi, {
		if( INT(CAR(pi)) >= level ) {
		    if( true_dep ) {
			action s = effect_action( conflict_source( c )) ;
			action k = effect_action( conflict_sink( c )) ;

			return( action_write_p( s ) && action_read_p( k )) ;
		    } 
		    else {
			return(FALSE);
		    }
		}, cone_levels(conflict_cone(c)));
	     }
	}
    }, dg_arc_label_conflicts(dal));
    
    return(TRUE);
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
sccs FindSccs(g, region, level)
graph g;
set region;
int level;
{
    cons *vertices = graph_vertices(g);
    cons *pv;

    Count = 1;
    StackPointer = 0;
    Stack = (vertex *) malloc(sizeof(vertex) * gen_length(vertices));
    Components = make_sccs(NIL);
	
    for (pv = vertices; pv != NIL; pv = CDR(pv)) {
	vertex v = VERTEX(CAR(pv));
	if (! ignore_this_vertex(region, v)) {
	    dg_vertex_label lv = (dg_vertex_label) vertex_vertex_label(v);
	    sccflags fv = dg_vertex_label_sccflags(lv);
	    
	    sccflags_mark(fv) = NEW_MARK;
	}
    }

    MAPL(pv, {
	vertex v = VERTEX(CAR(pv));
	if (! ignore_this_vertex(region, v))
	    if (MARKED_NEW_P(v)) {
		LowlinkCompute(g, region, v, level);
	    }
    }, vertices);

    free(Stack);

    ifdebug(3) {
	debug(3, "FindSccs", "Strongly connected components:\n");
	PrintSccs(Components);    
	debug(3, "FindSccs", "End\n");
    }

    return(Components);
}



void ComputeInDegree(g, region, l)
graph g;
set region;
int l;
{
    MAPL(pv, {
	vertex v = VERTEX(CAR(pv));

	if (! ignore_this_vertex(region, v)) {
	    scc sv = VERTEX_ENCLOSING_SCC(v);
	    MAPL(ps, {
		successor su = SUCCESSOR(CAR(ps));
		if (! ignore_this_successor(region, su, l)) {
		    vertex s = successor_vertex(su);
		    scc ss = VERTEX_ENCLOSING_SCC(s);
		    if (sv != ss)
			scc_indegree(ss) += 1;
		}
	    }, vertex_successors(v));
	}
    }, graph_vertices(g));
}



cons *TopSortSccs(g, region, l)
graph g;
set region;
int l;
{
    cons *lsccs = NIL, *elsccs = NIL, *no_ins = NIL;

    MAPL(ps, {
	scc s = SCC(CAR(ps));
	if (scc_indegree(s) == 0)
	    no_ins = CONS(SCC, s, no_ins);
    }, sccs_sccs(Components));

    while (no_ins != NIL) {
	cons *pcs;
	scc cs;

	pcs = no_ins; no_ins = CDR(no_ins);
	INSERT_AT_END(lsccs, elsccs, pcs);

	debug(3, "TopSortSccs", "updating in degrees ...\n");
	cs = SCC(CAR(pcs));
	MAPL(pv, {
	    vertex v = VERTEX(CAR(pv));
	    scc sv = VERTEX_ENCLOSING_SCC(v);
	    MAPL(ps, {
		successor su = SUCCESSOR(CAR(ps));
		if (! ignore_this_successor(region, su, l)) {
		    vertex s = successor_vertex(su);
		    scc ss = VERTEX_ENCLOSING_SCC(s);
		    if (! ignore_this_vertex(region, s)) {
			if (sv != ss) {
			    if ((scc_indegree(ss) -= 1) == 0) {
				no_ins = CONS(SCC, ss, no_ins);
			    }
			}
		    }
		}
	    }, vertex_successors(v));
	}, scc_vertices(cs));
    }

    if (get_debug_level() >= 3) {
	fprintf(stderr, "[TopSortSccs] topological order:\n");
	MAPL(ps, {
	    scc s = SCC(CAR(ps));
	    fprintf(stderr, "( ");
	    MAPL(pv, {
		vertex v = VERTEX(CAR(pv));
		statement st = vertex_to_statement(v);

		fprintf(stderr, "%d ", statement_number(st));
	    }, scc_vertices(s));
	    fprintf(stderr, ")   -->   ");
	}, lsccs);
	fprintf(stderr, "\n");
    }

    return(lsccs);
}



cons *FindAndTopSortSccs(g, region, l)
graph g;
set region;
int l;
{
    cons *lsccs;

    ifdebug(8) {
	debug(8, "FindAndTopSortSccs", "Dependence graph:\n");
	prettyprint_dependence_graph(stderr, statement_undefined, g);
    }

    debug(3, "FindAndTopSortSccs", "computing sccs ...\n");
    Components = FindSccs(g, region, l);

    debug(3, "FindAndTopSortSccs", "computing in degrees ...\n");
    ComputeInDegree(g, region, l);

    debug(3, "FindAndTopSortSccs", "topological sort ...\n");
    lsccs = TopSortSccs(g, region, l);

    return(lsccs);
}


void PrintScc(s)
scc s;
{
    fprintf(stderr, "Scc's statements : ");
    MAPL(pv, {
	vertex v = VERTEX(CAR(pv));
	statement st = vertex_to_statement(v);

	fprintf(stderr, "%02d ", statement_number(st));
    }, scc_vertices(s));
    fprintf(stderr, " -  in degree : %d\n", scc_indegree(s));
}

void PrintSccs(ss)
sccs ss;
{
    fprintf(stderr, "Strongly connected components:\n");
    if(!ENDP(sccs_sccs(ss))) {
	MAPL(ps, {PrintScc(SCC(CAR(ps)));}, sccs_sccs(ss));
    }
    else {
	fprintf(stderr, "Empty list of scc\n");
    }
}
