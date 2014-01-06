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
/* Name     :   adg_graph.c
 * Package  :   array_dfg
 * Author   :   Arnauld LESERVOT
 * Date     :   93/06/27
 * Modified :
 * Documents:   Platonoff's thesis and Leservot's thesis
 *              "Dataflow Analysis of Array and Scalar References" P. FEAUTRIER
 * Comments :
 */

#define GRAPH_IS_DFG
#include "local.h"

/* External variables */
extern	int		Gcount_re;
hash_table 		Gvertex_number_to_statement;
extern  hash_table	Gstco_map;


/*=======================================================================*/
/* 			GRAPH FUNCTIONS					 */
/*=======================================================================*/


/*=======================================================================*/
/* graph   adg_pure_dfg( in_gr )			AL 18/02/94	
 * 
 * Returns a pure Data Flow Graph : without Entry or Exit Node.
 */
graph	adg_pure_dfg( in_gr )
graph	in_gr;
{
  list	  ver_ptr = NULL, verlist = NULL;
  vertex  v_entry = NULL, v_exit = NULL;

  debug(9, "adg_pure_dfg", "begin\n");
	
  /* First get entry and exit nodes from in_gr */
  for(ver_ptr = graph_vertices(in_gr); !ENDP(ver_ptr); POP(ver_ptr)) {
    vertex	ve = NULL;
    int		or;
    
    ve = VERTEX(CAR(ver_ptr));
    or = dfg_vertex_label_statement(vertex_vertex_label(ve));
    if (or == ENTRY_ORDER) { v_entry = ve; continue; }
    if (or == EXIT_ORDER)  { v_exit = ve; continue; }
  }
	
	
  /* Then duplicate nodes without exit and entry nodes */
  for(ver_ptr = graph_vertices(in_gr); !ENDP(ver_ptr); POP(ver_ptr)) {
    vertex  	ver, ver2 = NULL, ver5 = NULL;
    int		or = (int) NULL;
    list	succ_ptr = NIL;
    
    ver  = VERTEX(CAR( ver_ptr ));
    if ((ver == v_entry) || (ver == v_exit)) continue;
    
    or   = dfg_vertex_label_statement(vertex_vertex_label(ver));
    ver2 = adg_same_dfg_vertex_number( verlist, or );
    if ( ver2 == vertex_undefined ) {
	ver2 = make_vertex(vertex_vertex_label(ver),NIL);
      ADD_ELEMENT_TO_LIST( verlist, VERTEX, ver2 );
    }
	  
    for(succ_ptr = vertex_successors(ver); !ENDP(succ_ptr); POP(succ_ptr)){
      successor	succ, succ2 = NULL;
      vertex    ver3;
      int	ord = (int) NULL;
	    
      succ  = SUCCESSOR(CAR( succ_ptr ));
      ver3  = successor_vertex( succ );
      /* If this succ is an extremity, continue */
      if ((ver3 == v_entry) || (ver3 == v_exit)) continue;
      
      ord   = dfg_vertex_label_statement(vertex_vertex_label(ver3));
      ver5  = adg_same_dfg_vertex_number( verlist, ord );
      if ( ver5 == vertex_undefined ) {
	  ver5 = make_vertex(vertex_vertex_label(ver3),NIL);
	ADD_ELEMENT_TO_LIST( verlist, VERTEX, ver5 );
      }

      succ2 = make_successor(successor_arc_label(succ),ver5 );
      ADD_ELEMENT_TO_LIST(vertex_successors( ver2 ), SUCCESSOR, succ2 );
    }
  }
		
  debug(9, "adg_pure_dfg", "end\n");
  return make_graph(verlist);
}

/*=======================================================================*/
/* graph   adg_pure_dfg2( in_gr )			AL 18/02/94	
 * 
 * Returns a pure Data Flow Graph : without Entry or Exit Node.
 */
graph	adg_pure_dfg2( in_gr )
graph	in_gr;
{
  list	  vl = NULL;
  vertex  v_entry = NULL, v_exit = NULL;

  debug(9, "adg_pure_dfg2", "begin\n");
  in_gr = copy_graph( in_gr );
  /* First remove entry and exit nodes from in_gr */
  for(vl = graph_vertices(in_gr); !ENDP(vl); POP(vl)) {
    vertex	ve = NULL;
    int		or = (int) NULL;
    
    ve = VERTEX(CAR(vl));
    or = dfg_vertex_label_statement(vertex_vertex_label(ve));
    if (or == ENTRY_ORDER) v_entry = ve;
    if (or == EXIT_ORDER)  v_exit = ve;
  }
  gen_remove( &graph_vertices( in_gr ), v_entry );
  gen_remove( &graph_vertices( in_gr ), v_exit );

  /* Then remove successors that include exit node */
  for(vl = graph_vertices(in_gr); !ENDP(vl); POP(vl)) {
    vertex	ve = NULL;
    list	su_l = NULL;
    list	li = NIL;
    
    ve = VERTEX(CAR(vl));
    for(su_l = vertex_successors(ve); !ENDP(su_l); POP(su_l) ) {
      successor	succ;
      
      succ = SUCCESSOR(CAR( su_l ));
      if(successor_vertex(succ) != v_exit) continue;
      ADD_ELEMENT_TO_LIST(li, SUCCESSOR, succ);
    }
    for(; !ENDP(li); POP(li)) 
      {gen_remove( &vertex_successors(ve), SUCCESSOR(CAR(li)) );}
  }
  
  debug(9, "adg_pure_dfg2", "end\n");
  return in_gr;
}


/*=======================================================================*/
/* void adg_print_graph()
 * Prints a dg graph with vertex labels as they are, 
 * Do not try to translate vertex label to a statement number.
 */
void adg_print_graph(fd, mod_stat, mod_graph)
FILE *fd;
statement mod_stat;
graph     mod_graph;
{
  cons *pv1 = NULL, *ps = NULL, *pc = NULL;
  Ptsg gs = NULL;

  fprintf(fd, "\n ************************ Dependence graph ************************\n");

  for (pv1 = graph_vertices(mod_graph); !ENDP(pv1); pv1 = CDR(pv1)) {
    vertex v1 = VERTEX(CAR(pv1));
    int dvl1 = dg_vertex_label_statement((dg_vertex_label) vertex_vertex_label(v1) );

    for (ps = vertex_successors(v1); !ENDP(ps); ps = CDR(ps)) {
      successor su = SUCCESSOR(CAR(ps));
      vertex v2 = successor_vertex(su);
      int dvl2 = dg_vertex_label_statement(
				(dg_vertex_label) vertex_vertex_label(v2) );
      dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
      fprintf(fd, "\t%02d --> %02d with conflicts\n", dvl1, dvl2);
      
      for (pc = dg_arc_label_conflicts(dal); !ENDP(pc); pc = CDR(pc)) {
	conflict c = CONFLICT(CAR(pc));

	fprintf(fd, "\t\tfrom ");
	print_words(fd, words_effect(conflict_source(c)));

	fprintf(fd, " to ");
	print_words(fd, words_effect(conflict_sink(c)));

	if(conflict_cone(c) != cone_undefined){
	  fprintf(fd, " at levels ");
	  MAPL(pl, {
	    fprintf(fd, " %d", INT(CAR(pl)));
	  }, cone_levels(conflict_cone(c)));
	  
	  fprintf(fd, "\n");
	  gs = (Ptsg)cone_generating_system(conflict_cone(c));
	  if (!SG_UNDEFINED_P(gs)) sg_fprint(fd,gs,entity_local_name);
       
	}
	fprintf(fd, "\n");
      }
    }
  }
  fprintf(fd, "\n******************** End of Dependence graph ********************\n");
}

/*=======================================================================*/
/* int adg_number_to_ordering( (int) in_nb )			AL 21/10/93
 * Input  : Number of a vertex.
 * Output : The ordering of Statement associated to this vertex.
 * PRIVATE to this pass !!
 */
int adg_number_to_ordering( in_nb )
int in_nb;
{ return (int) hash_get( Gvertex_number_to_statement, (char *) in_nb ); }



/*=======================================================================*/
/* vertex adg_number_to_vertex( (graph) in_dfg, (int) in_nb )	AL 21/10/93
 * Input  : A graph and a number of a vertex.
 * Output : A vertex which statement number is equal to in_nb.
 * PRIVATE to this pass !!
 */
vertex adg_number_to_vertex( in_dfg, in_nb )
graph in_dfg;
int   in_nb;
{
  vertex ret_ver = vertex_undefined;
  list	 gl = graph_vertices( in_dfg );
  
  debug(9, "adg_number_to_vertex", "begin\n");
  for(; !ENDP(gl); POP(gl)) {
    vertex ver = VERTEX(CAR( gl ));
    dfg_vertex_label dvl = (dfg_vertex_label) vertex_vertex_label( ver );
    if ((dvl != dfg_vertex_label_undefined) && 
	(dfg_vertex_label_statement(dvl) == in_nb)) 
      ret_ver = ver;
  }
  debug(9, "adg_number_to_vertex", "end\n");
  return ret_ver;
}

/*=======================================================================*/
/* statement adg_vertex_to_statement( (vertex) in_ver )		AL 21/10/93
 * Input  : A vertex in_ver which has been reordered by 
 *		adg_reorder_statement_number.
 * Output : A statement corresponding to the vertex number.
 * WARNING ! Uses global variable Gvertex_number_to_statement.
 * PRIVATE to this pass !!
 */
statement adg_vertex_to_statement( in_ver )
vertex in_ver;
{ return ordering_to_statement( adg_vertex_to_ordering( in_ver ) ); }

/*=======================================================================*/
/* adg_vertex_to_ordering( (vertex) in_ver )			AL 21/10/93
 * Input  : A vertex in_ver 
 * Output : ordering of statement associated to in_ver
 * PRIVATE to this pass !!
 */
int adg_vertex_to_ordering( in_ver )
vertex in_ver;
{
  int  count = (int) NULL, order = -1;

  if (in_ver != vertex_undefined ) {
    count = dfg_vertex_label_statement((dfg_vertex_label) vertex_vertex_label(in_ver));
    debug(9,"adg_vertex_to_ordering","Number of ver : %d\n",count);
    order = (int) hash_get( Gvertex_number_to_statement, (char *) count );
    debug(9, "adg_vertex_to_ordering", " -> ordering %d\n", order);
  }
  return order;
}


/*============================================================================*/
bool integer_in_list_p(i, l)
int i;
list l;
{
 bool is_in_list = false;

 for( ; (l != NIL) && (! is_in_list); l = CDR(l))
    if(i == INT(CAR(l)))
       is_in_list = true;

 return(is_in_list);
}

/*=======================================================================*/
/* void adg_reorder_statement_number( (graph) in_dfg )		AL 21/10/93
 * Input  : A dfg graph with different vertices pointing on the same
 * 		statement. This is possible due to duplication of
 *		vextices whose statement are controled by a disjunction
 *		of predicates (See Doc about IF statement).
 * Output : Nothing. Each vertex has a new and distinct number for
 * 		the statement tag of it dfg_vertex_label tag. 
 * PRIVATE to this pass !!
 *
 * AP, oct 6th 1995: I change the way to choose the new numbers used to
 * reorder
 */
void adg_reorder_statement_number( in_dfg )
graph in_dfg;
{
  list	verl, l_num;
  int	order, num;
  
  l_num = NIL;
  Gvertex_number_to_statement = hash_table_make( hash_int, 0 );

  debug(7, "adg_reorder_statement_number", "begin\n");

  verl = graph_vertices( in_dfg );
  for(; !ENDP(verl); POP(verl)) {
    vertex v = VERTEX(CAR(verl));
    dfg_vertex_label vl = (dfg_vertex_label) vertex_vertex_label(v);
    if (vl !=  dfg_vertex_label_undefined ) {
      order = dfg_vertex_label_statement( vl );
      num =
	BASE_NODE_NUMBER + 10*statement_number(ordering_to_statement(order));
      while(integer_in_list_p(num, l_num))
	num++;
      l_num = CONS(INT, num, l_num);
      dfg_vertex_label_statement( vl ) = num;
      hash_put( Gvertex_number_to_statement, (char *) num, (char *) order );
      debug(9, "adg_reorder_statement_number",
	    "  Vertex-Statement : %d Ordering of Statement  : %d\n",
	    num, order );
    }
  }
    
  debug(7, "adg_reorder_statement_number", "end\n");
}

/*=======================================================================*/
/* vertex adg_same_dfg_vertex_number( (list) in_l, (int) in_i ) AL 27/10/93
 * Input  : A list of vertices and a number in_i.
 * Output : A vertex v in in_l which statement associated to it  
 * 		has a number equal to in_i.
 * PRIVATE to this pass !!
 */
vertex adg_same_dfg_vertex_number( in_l, in_i )
list in_l;
int  in_i;
{
  vertex          ver = NULL;

  debug(9, "adg_same_dfg_vertex_number", "doing\n");
  
  for(;!ENDP(in_l); POP(in_l)) {
    int prov_i;
    ver = VERTEX(CAR( in_l ));
    prov_i = dfg_vertex_label_statement((dfg_vertex_label) vertex_vertex_label(ver));
    if ( prov_i == in_i ) return( ver );
  }

  debug(9, "adg_same_dfg_vertex_number", "end\n");
  return vertex_undefined;
}

/*=======================================================================*/
/* void adg_update_dfg((quast) in_sou,(reference) in_ref,	AL 19/10/93
 *			(vertex) in_dest, (Psysteme) in_ps, (list) *in_lp )
 * Input  : A quast, a reference, destination vertex,
 *		a predicate representing present 
 *		control condition, and a list of vertices representing a DFG.
 * Output : Nothing. Just modify the graph *in_lp. See Doc.
 * PRIVATE to this pass !!
 */
void adg_update_dfg( in_sou, in_ref, in_dest, in_pa, in_context, in_test, in_gr, in_lp )
quast	  in_sou;
reference in_ref;
vertex	  in_dest;
Ppath	  in_pa;
Psysteme  in_context;
Psysteme  in_test;
graph	  in_gr;
list*	  in_lp;
{
  quast_value 	qv = NULL;

  debug(7, "adg_update_dfg", "begin\n");
  if (in_sou == quast_undefined) return;


  /* Presently, newparms should be empty */
  /* 	if (quast_newparms(in_sou) != NIL)  */
  /* fprintf(stderr, "\n Newparms list should be presently empty !\n"); */


  qv = quast_quast_value( in_sou );
  if (qv == quast_value_undefined) return;
  if (quast_value_quast_leaf_p( qv )) {
    Psysteme	dest_system = NULL; 
    dataflow 	df = NULL;
    quast_leaf	ql = NULL;
    vertex	pred_v = NULL;
    bool	get_it = false;
    list	qls = NIL;
    leaf_label	qll = NULL;
    int		sou_nb, dest_vls = (int) NULL;
    list	dfl = NIL;
    Pdisjunct	dj = NULL;

    pips_assert("adg_update_dfg",(in_dfg_vertex_list((list)*in_lp,in_dest)!= vertex_undefined));
    ql     = quast_value_quast_leaf( qv );
    qls    = quast_leaf_solution( ql );
    qll    = quast_leaf_leaf_label( ql );
    sou_nb = leaf_label_statement(qll);


    /* Build the new dataflows list of data flow */
    dj       = pa_path_to_disjunct( in_pa );
    dest_vls = dfg_vertex_label_statement(vertex_vertex_label(in_dest));
    if ((dest_vls != ENTRY_ORDER) && (dest_vls != EXIT_ORDER)) {
      dest_system = predicate_system
	(dfg_vertex_label_exec_domain(vertex_vertex_label(in_dest)));
    }

    for(; dj != NULL; dj = dj->succ) {
      Psysteme	prov_ps = SC_UNDEFINED;

      if (!SC_UNDEFINED_P(dj->psys) && sc_empty_p( dj->psys )) continue;

      if (dj->psys != SC_UNDEFINED) {
	prov_ps = sc_elim_redund(sc_append(sc_dup(dj->psys), in_context));
	if (prov_ps == SC_UNDEFINED) continue;
	prov_ps = adg_suppress_2nd_in_1st_ps(prov_ps, dest_system );
	if (prov_ps != NULL) {prov_ps->base = NULL; sc_creer_base( prov_ps ); }
      }
      df = make_dataflow( in_ref, qls, make_predicate( prov_ps ), communication_undefined );
      if (get_debug_level()>6) {
	adg_fprint_dataflow(stderr, 
			    dfg_vertex_label_statement(vertex_vertex_label(in_dest)),
			    df);
      }
      ADD_ELEMENT_TO_LIST( dfl, DATAFLOW, df );
    }


    /* Update graph */
    pred_v = adg_same_dfg_vertex_number( (list) *in_lp, sou_nb );
    if (pred_v == vertex_undefined) {
      successor	su = make_successor( make_dfg_arc_label( dfl ), in_dest);
      pred_v       = make_vertex(make_dfg_vertex_label(sou_nb,
						 predicate_undefined,
						 sccflags_undefined),
				 CONS(SUCCESSOR, su, NIL));
      ADD_ELEMENT_TO_LIST( *in_lp, VERTEX, pred_v );
    }
    else {
      list	succ_l = vertex_successors( pred_v );
      /* Do pred_v already has in_dest as a successor ? */
      for(; !ENDP(succ_l) && !(get_it); POP(succ_l)) {
	successor  succ   = SUCCESSOR(CAR( succ_l ));
	vertex	   vv     = successor_vertex( succ );

	if (vv != in_dest) continue;
	get_it = true;
	gen_nconc( dfg_arc_label_dataflows(successor_arc_label(succ) ), dfl );
      }
      /* If not, add to it the correct successor */
      if (!(get_it)) {
	successor su =  make_successor( make_dfg_arc_label(dfl), in_dest);
	ADD_ELEMENT_TO_LIST(  vertex_successors(pred_v), SUCCESSOR, su );
      }
    }
  }

  if (quast_value_conditional_p( qv )) {
    conditional	cond = quast_value_conditional( qv );
    quast	qt = NULL, qf = NULL;
    Psysteme	ps = NULL; 

    ps = predicate_system( conditional_predicate( cond ) );
    adg_sc_update_base( &ps );
    qt = conditional_true_quast( cond );
    adg_update_dfg( qt, in_ref, in_dest,
		   pa_intersect_system( in_pa, ps ), 
		   in_context, in_test, in_gr, in_lp  );
    qf = conditional_false_quast( cond );
    if ((qf != quast_undefined) && 
	(quast_quast_value( qf ) != quast_value_undefined)) {
      adg_update_dfg( qf, in_ref, in_dest,
		     pa_intersect_complement( in_pa, ps ), 
		     in_context, in_test, in_gr, in_lp );
    }
  }
  debug(7, "adg_update_dfg", "end\n");
}

/*=======================================================================*/
/* list adg_get_exec_domain( static_control stco )		AL 21/07/93
 * Input : A static_control stco.
 * Output: A list of predicates corresponding to the conditions
 * 	   on enclosing loops combined with enclosing tests.
 * PRIVATE to this pass !!
 */
list adg_get_exec_domain( stco )
static_control stco;
{
  predicate 	loop_pred = NULL;
  list 		test_pred_l = NULL, prl = NIL;

  debug(9, "adg_get_exec_domain", "begin\n");

  /* Get existing predicate for the exec_domain */
  test_pred_l = adg_get_disjunctions( static_control_tests(stco) );
  loop_pred   = adg_get_predicate_of_loops(static_control_loops(stco));
  if( loop_pred != predicate_undefined ) {
    list prov_list = NIL;
    ADD_ELEMENT_TO_LIST( prov_list, PREDICATE, loop_pred);
    prl = adg_make_disjunctions(test_pred_l, prov_list);
  }
  else prl = test_pred_l;
  if( prl == NIL ) ADD_ELEMENT_TO_LIST( prl, PREDICATE, make_predicate(SC_RN));

  if(get_debug_level() >= 9) adg_fprint_predicate_list(stderr, prl);
  debug(9, "adg_get_exec_domain", "end\n");
  return( prl );
}

/*=======================================================================*/
/* list adg_same_order_in_dvl( list l, vertex ver )		AL 21/07/93
 * Input   : A list of vertices, each vertex has an unique ordering
 *		according to a statement.
 *	     A vertex ver of a Data Flow Graph. Its number is not equal
 *		to an ordering statement.
 * Output  : A list or vertices with same statement ordering as ver.
 * WARNING : Implicitly uses Gvertex_number_to_statement !
 * PRIVATE to this pass !!
 */
list adg_same_order_in_dvl( l, ver )
list 	l;
vertex 	ver;
{
  list ret_list = NIL;
  int  in;

  in = adg_vertex_to_ordering( ver );
  debug(9, "adg_same_order_in_dvl", "Ordering of ver : %d\n", in);
  for(; !ENDP(l); POP(l)) {
    vertex v   = VERTEX(CAR( l ));
    int    in2 = dfg_vertex_label_statement(vertex_vertex_label(v));
    if( in2 != in ) continue;
    ADD_ELEMENT_TO_LIST( ret_list, VERTEX, v);
    debug(9, "adg_same_order_in_dvl", "-> %d\n", in2);
    break;
  }
  return( ret_list );
}

/*=======================================================================*/
/* graph adg_dup_disjunctive_nodes( graph g, statement_mapping stco_map)
 * Input : A dependence graph or a partial one.			AL 20/07/93
 * 	   A static_control mapping on each statement.
 * Output: A graph whose nodes controlled by a disjunctive predicate
 * 		are duplicated. 
 *		Labels associated to vertex are dfg_vertex_label
 * 		whose predicate are set to one of the disjunctive
 *		part of the englobing tests.
 *		Successors arc-labels are initial dg_arc_label
 * PRIVATE to this pass !!
 */
graph adg_dup_disjunctive_nodes( g, stco_map )
graph 			g;
statement_mapping 	stco_map;
{
  list 	l = NULL, ver_list = NIL;
  
  debug(7, "adg_dup_disjunctive_nodes", "begin \n");
  l = graph_vertices( g );
  for(; !ENDP( l ); POP(l)) {
    vertex	     ver = NULL;
    list             new_ver_l = NULL, succ_list = NULL;
    list             new_succ_list = NIL;
    int		     in;
    dfg_vertex_label dvl = NULL;

    ver = VERTEX(CAR( l ));
    in  = statement_ordering(vertex_to_statement( ver ));
    succ_list = vertex_successors( ver );

    /* Do the associated vertices successors exist ? */
    for(; !ENDP(succ_list); POP( succ_list )) {
      successor 	succ = SUCCESSOR(CAR( succ_list ));
      vertex    	v = successor_vertex( succ );
      list 		prl = NIL;
      static_control 	stco = NULL;
      statement	        st = NULL;
      int		in2 = (int) NULL;


      if( adg_list_same_order_in_dg( ver_list, v ) != NIL ) continue;

      st   = vertex_to_statement( v );
      in2  = statement_ordering( st );
      stco = (static_control) GET_STATEMENT_MAPPING( stco_map, st );
      prl  = adg_get_disjunctions(static_control_tests(stco));
      
      if (prl == NIL) {
	dvl = make_dfg_vertex_label(in2,  predicate_undefined,  sccflags_undefined );
	ADD_ELEMENT_TO_LIST( ver_list, VERTEX, make_vertex( dvl, NIL ) );
	continue;
      }
      for(; !ENDP(prl); POP(prl)) {
	predicate  pred = PREDICATE(CAR( prl ));
	if (pred != predicate_undefined) {
	  Psysteme ps = predicate_system( pred );
	  ps->base = NULL; sc_creer_base( ps );
	}
	dvl = make_dfg_vertex_label(in2, pred, sccflags_undefined);
	ADD_ELEMENT_TO_LIST( ver_list, VERTEX, make_vertex( dvl, NIL));
      }
                        
    }


    /* Does vertex associated to ver exist in the new ver_list ?*/
    if( adg_list_same_order_in_dg( ver_list, ver ) == NIL ) {
      list 		prl = NIL;
      static_control 	stco = NULL;
      
      stco = (static_control) GET_STATEMENT_MAPPING(stco_map, 
						    vertex_to_statement( ver ));
      prl  = adg_get_disjunctions(static_control_tests(stco));
      if (prl == NIL) {
	dvl = make_dfg_vertex_label(in, predicate_undefined,sccflags_undefined );
	ADD_ELEMENT_TO_LIST( ver_list, VERTEX, make_vertex( dvl, NIL ) );
      }
      else for(; !ENDP(prl); POP(prl)) {
	predicate  pred = PREDICATE(CAR( prl ));
	if (pred != predicate_undefined) {
	  Psysteme ps = predicate_system( pred );
	  ps->base    = NULL; sc_creer_base( ps );
	}
	dvl = make_dfg_vertex_label(in, pred,  sccflags_undefined);
	ADD_ELEMENT_TO_LIST( ver_list, VERTEX,  make_vertex( dvl, NIL));
      }
    }

    /* Build the new successors list */
    succ_list = vertex_successors( ver );
    for(; !ENDP(succ_list); POP(succ_list)) {
      successor	   succ = SUCCESSOR(CAR( succ_list ));
      vertex       v    = successor_vertex( succ );
      dfg_arc_label dal  = successor_arc_label( succ );
      list         sl   = adg_list_same_order_in_dg( ver_list, v );
      
      for(; !ENDP(sl); POP(sl)) 
	{ ADD_ELEMENT_TO_LIST( new_succ_list, SUCCESSOR,
	  make_successor( copy_dfg_arc_label( dal ), VERTEX(CAR(sl)) ) ); }
    }
    
    /*Associate a duplicated successor list to each new vertex */
    new_ver_l = adg_list_same_order_in_dg( ver_list, ver );
    for(; !ENDP(new_ver_l); POP(new_ver_l)) 
      { vertex_successors( VERTEX(CAR( new_ver_l )) ) = new_succ_list;}
  }

  debug(7, "adg_dup_disjunctive_nodes", "end \n");	
  return( make_graph( ver_list ) );
}


/*=======================================================================*/
/* list adg_write_reference_list( (vertex) ver, (effect) reff )	AL 08/07/93
 * Returns a list of vertices whose statement write 
 * the read effect reff.
 * Vertex ver is a DG node.
 */
list adg_write_reference_list( ver, reff )
vertex ver;
effect reff;
{
  list 		ls = NULL, ret_list = NIL;
  reference	r1 = NULL, rsink = NULL;

  debug( 9, "adg_write_reference_list", "begin\n" );
  r1 = effect_any_reference( reff );
  ls = vertex_successors( ver );

  for(; !ENDP( ls ); POP( ls ) ) {
    successor	 succ     = SUCCESSOR(CAR( ls ));
    vertex	 succ_ver = successor_vertex( succ );
    dg_arc_label dgal     = (dg_arc_label) successor_arc_label( succ );
    list         conflist = dg_arc_label_conflicts( dgal );
    
    for(; !ENDP( conflist ); POP( conflist )) {
      rsink = effect_any_reference(conflict_sink(  CONFLICT(CAR( conflist )) ));
      if (!(reference_equal_p( r1, rsink )) ) continue;
      ADD_ELEMENT_TO_LIST( ret_list, VERTEX, succ_ver );
    }
  }
	
  debug( 9, "adg_write_reference_list", "end\n" );
  return( ret_list );
}

/*=======================================================================*/
/* bool in_effect_list_p( (list) l, (effect) eff ) 		AL 08/07/93
 * Returns True if the effect list l has an element whose reference 
 * is the same as the reference of eff.
 * returns False in other cases.
 */
bool in_effect_list_p( l, eff )
list l;
effect eff;
{
  bool 		ret_bo = false;
  reference 	r1 = NULL;

  debug(9, "in_effect_list_p", "begin\n");
  r1 = effect_any_reference( eff );

  for(; !ENDP(l) && !ret_bo ; POP(l) ) {
    reference r2 = effect_any_reference(EFFECT(CAR( l )));
    if ( reference_equal_p( r1, r2 ) ) ret_bo = true;
  }

  debug(9, "in_effect_list_p", "end\n");
  return( ret_bo );
}
		
/*=======================================================================*/
/* list read_reference_list( ver, ent_l1, ent_l2 )		AL 18/02/94
 * Returns a list of read effect for vertex ver. 
 * Vertex ver should have dg_arc_label arc_labels.
 * This function should be very simple knowing effects of each 
 * statement : we should just scan the read effects of statement vertex.
 */
list	read_reference_list( ver, ent_l1, ent_l2 )
vertex  ver;
list	ent_l1, ent_l2;
{
  list      rl  = NIL, effs = NIL, aux_l;
  statement sta = statement_undefined;
  effect    eff = effect_undefined;    /* Current effect */
  entity    ent = entity_undefined;
	
  sta  = adg_vertex_to_statement( ver );
  debug( 7, "read_reference_list", "statement number : %02d\n", statement_number(sta) );
  effs = load_proper_rw_effects_list( sta );

  /* Put in rl effects readen by ver and remove from it 
   * effect whose variable are in ent_l1 or in ent_l2.
   */
  for(aux_l = effs; !ENDP(aux_l); POP(aux_l)) {
    eff = EFFECT(CAR(aux_l));
    ent = reference_variable( effect_any_reference( eff ) );
    if ( !action_read_p(effect_action( eff ))) continue;
    if ( is_entity_in_list_p( ent, ent_l1 ))   continue;
    if ( is_entity_in_list_p( ent, ent_l2 ))   continue;
    if ( in_effect_list_p( rl, eff )  )        continue;
    ADD_ELEMENT_TO_LIST( rl, EFFECT, eff );
  }
    
  debug( 7, "read_reference_list", "end\n" );
  return( rl );
}


/*=======================================================================*/
/* graph adg_only_call_WR_dependence( (graph) g )		AL 06/07/93
 * Returns a graph from a dependence graph with only Write-Read
 * conflicts and assignements statements.
 */
graph 	adg_only_call_WR_dependence( g )
graph	g;
{
  list	l, new_vertices = NIL;

  debug(7, "adg_only_call_WR_dependence", "begin \n");
  
  l = graph_vertices( g );
  for(; !ENDP(l); POP(l)) {
    list 		succ_list = NULL;
    list 		new_succ_list = NIL;
    vertex		ver, new_ver = NULL;
    dfg_vertex_label	dgvl = NULL;
    
    ver = VERTEX(CAR( l ));
    /* We only keep call statement */
    if (!assignment_statement_p(ordering_to_statement
				(dfg_vertex_label_statement(
	vertex_vertex_label( ver )))) )  continue;

    succ_list = vertex_successors( ver );
    if (adg_same_order_in_dg(new_vertices,ver)==vertex_undefined){
      dgvl    = vertex_vertex_label( ver );
      new_ver = make_vertex( dgvl,(list) NIL );
    }
    for(;!ENDP( succ_list ); POP( succ_list )) {
      successor	        s = NULL, new_succ = NULL;
      list 		conflist = NIL;
      list		new_conflist = NIL;
      dg_arc_label	dgal = NULL, vl = NULL;
      vertex		succ_ver = NULL, new_succver = NULL;
      
      s        = SUCCESSOR(CAR( succ_list ));
      succ_ver = successor_vertex( s );
      conflist = dg_arc_label_conflicts((dg_arc_label) successor_arc_label( s ) );
      
      for(;!ENDP( conflist ); POP( conflist )) {
	conflict 	conf = NULL;
	effect		eff1 = NULL;
	effect		eff2 = NULL;
	
	conf = CONFLICT(CAR( conflist ));
	eff1 = (effect) conflict_source( conf );
	eff2 = (effect) conflict_sink( conf );
	if ( action_write_p(effect_action(eff1)) && action_read_p(effect_action(eff2)) )
	  ADD_ELEMENT_TO_LIST( new_conflist,CONFLICT, conflict_dup( conf ) );
      }

      if (new_conflist != NIL) {
	new_succver = adg_same_order_in_dg( new_vertices, succ_ver );
	if (new_succver == vertex_undefined) {
	    vl          = (dg_arc_label) vertex_vertex_label(succ_ver);
	    new_succver = make_vertex((dfg_arc_label) vl, (list) NIL);
	}
	dgal     = make_dg_arc_label( new_conflist );
	new_succ = make_successor( (dfg_arc_label) dgal, new_succver );
	ADD_ELEMENT_TO_LIST( new_succ_list, SUCCESSOR, new_succ );
      }
    }
    if (new_succ_list != NIL ) vertex_successors( new_ver ) = (list) new_succ_list;
    
    ADD_ELEMENT_TO_LIST( new_vertices, VERTEX, new_ver );
  }

  debug(7, "adg_only_call_WR_dependence", "end \n" );
  return( make_graph( new_vertices ) );
}
						

/* effect effect_dup(effect eff)
 * input    : an effect.
 * output   : a copy of the input effect (no sharing).
 * modifies : 
 */
  /* FI: the meaning of effect_dup() is unclear; should any syntactically
     correct effect by duplicable? Or this function be restricted
     to semantically correct effects? I do not understand why
     SC_UNDEFINED was used instead of transformer_undefined in
     contexts for scalar variables (8 August 1992) */

static effect effect_dup(eff)
effect eff;
{
  effect ne;
  Psysteme sc = effect_system(eff);
  
  if(sc == SC_UNDEFINED) {
    ne = make_simple_effect(effect_reference(eff),
			    effect_action(eff),
			    effect_approximation(eff));
  }
  else {
    pips_assert("effect_dup", ! SC_UNDEFINED_P(sc));
    ne = make_convex_effect(effect_reference(eff),
			    effect_action(eff),
			    effect_approximation(eff),
			    sc);
  }
  return ne;
}


/*=======================================================================*/
/* conflict conflict_dup( (conflict) conf ) 			AL 06/07/93
 * Duplicates a conflict
 * copy_conflict should now be used.
 */
conflict conflict_dup( conf )
conflict conf;
{
  list		levlist = NIL;
  cone	 	co = NULL, new_cone = cone_undefined;
  conflict	new_conf = NULL;
  int		in = (int) NULL;
  Ptsg		cgs = NULL, new_gs = NULL ;

  debug(9, "conflict_dup", "begin\n");

  co      = conflict_cone( conf );
  if (co != cone_undefined) {
    MAPL( int_ptr, {
      in = INT(CAR( int_ptr ));
      ADD_ELEMENT_TO_LIST( levlist, INT, in );
    }, cone_levels( co ) );
    cgs = (Ptsg) cone_generating_system( co );
    if (cgs != (Ptsg) NIL) new_gs = sg_dup( cgs );
    new_cone = make_cone( levlist, new_gs );
  }
  new_conf = make_conflict(effect_dup(conflict_source( conf )),
			   effect_dup(conflict_sink( conf )),
			   new_cone );
  debug(9, "conflict_dup", "end\n");
  return( new_conf );
}


/*=======================================================================*/
/* dg_arc_label dg_arc_label_dup((dg_arc_label) dg_al) 		AL 05/07/93
 * Duplicates the dg_arc_label.
 */
dg_arc_label dg_arc_label_dup( dg_al )
dg_arc_label dg_al;
{
  list		conflist = NIL;
  dg_arc_label	ret_dg_al = NULL;
  
  debug(9, "dg_arc_label_dup", "begin\n");
  MAPL( conf_ptr,{
    conflict  new_conf = conflict_dup( CONFLICT(CAR( conf_ptr )) );
    ADD_ELEMENT_TO_LIST( conflist, CONFLICT, new_conf );
  }, dg_arc_label_conflicts( dg_al ) );
  
  ret_dg_al = make_dg_arc_label( conflist );
  debug(9, "dg_arc_label_dup", "end\n");
  return( ret_dg_al );
}

/*=======================================================================*/
/* dg_vertex_label dg_vertex_label_dup( 			AL 05/07/93
 * Duplicates the dg_vertex_label.
 */
dg_vertex_label dg_vertex_label_dup( dg_vl )
dg_vertex_label dg_vl;
{
  debug(9, "dg_vertex_label_dup", "doing for statement n. : %d\n",
	(int) dg_vertex_label_statement( dg_vl ) );
  return( make_dg_vertex_label(dg_vertex_label_statement(dg_vl), statement_undefined, -1, sccflags_undefined));
}

/*=======================================================================*/
/* vertex dg_vertex_dup( (vertex) ver )				AL 05/07/93
 * duplicates a vertex of the Dependence Graph
 * Be VERY CAREFULL : this code could loop for ever if one
 * successor of 'ver' points on it.
 * PRIVATE use !
 */
vertex	dg_vertex_dup( ver )
vertex  ver;
{
  list		  succ_list = NIL;
  dg_vertex_label vl = NULL, ve = NULL;
  successor	  succ = NULL;
  dg_arc_label	  al = NULL;
  vertex	  ret_ver = NULL;
	
  debug(9, "dg_vertex_dup", "statement n. : %02d\n", 
	(int) statement_number(vertex_to_statement( ver )) );
  vl = copy_dg_vertex_label((dg_vertex_label) vertex_vertex_label(ver) );
  MAPL( succ_ptr, {
    succ     = SUCCESSOR(CAR( succ_ptr ));
    al       = dg_arc_label_dup((dg_arc_label) successor_arc_label( succ ) );
    ve       = dg_vertex_dup( successor_vertex( succ ) );
    ADD_ELEMENT_TO_LIST( succ_list, SUCCESSOR, make_successor( al, ve ) );
  }, vertex_successors( ver ) );
	
  ret_ver = make_vertex( vl, succ_list );
  debug(9, "dg_vertex_dup", "end\n");
  return( ret_ver );
}

	
/*=======================================================================*/
/* list adg_list_same_order_in_dg( (list) in_l, (vertex) in_v ) AL 27/10/93
 * Input  : A list of vertices of a dg and a vertex of this type.
 * Output : All the vertices which associated ordering has same
 *		ordering as the one linked to in_v.
 * PRIVATE use !
 */
list adg_list_same_order_in_dg( in_l, in_v )
list in_l;
vertex in_v;
{
  vertex 	ver = NULL;
  int	 	in;
  statement	s = NULL;
  list		ret_l = NIL;

  debug(9, "adg_list_same_order_in_dg", "begin\n");
  in = statement_ordering( vertex_to_statement( in_v ) );
  for(;!ENDP(in_l); POP(in_l)) {
    ver = VERTEX(CAR( in_l ));
    s   = vertex_to_statement( ver );
    if ( statement_ordering(s) == in ) ADD_ELEMENT_TO_LIST( ret_l, VERTEX, ver );
  }
  debug(9, "adg_list_same_order_in_dg", "end \n");
  
  return ret_l;
}

/*=======================================================================*/
/* vertex  adg_same_order_in_dg( (list) l, (vertex) v )		AL 30/06/93
 * Input  : A list l of vertices.
 *	    A vertex v of a dependence graph.
 * Returns vertex_undefined if v is not in list l.
 * Returns v' that has the same statement_ordering than v.
 * COMMON use possible.
 */
vertex adg_same_order_in_dg( l, v )
list l;
vertex v;
{
  vertex 	ver = NULL;
  int	 	in;
  statement	s = NULL;

  debug(9, "adg_same_order_in_dg", "doing \n");

  in = statement_ordering( vertex_to_statement( v ) );
  for(;!ENDP(l); POP(l)) {
    ver = VERTEX(CAR( l ));
    s   = vertex_to_statement( ver );
    if ( statement_ordering(s) == in ) return( ver );
  }

  return vertex_undefined;
}


/*=======================================================================*/
/* graph adg_reverse_graph( (graph) g ) 			AL 29/06/93
 * This function is used to reverse Pips's graph in order to have 
 * all possible sources directly (Feautrier's dependance graph).
 */ 
graph	adg_reverse_graph( g )
graph	g;
{
  list		verlist = NIL;
  successor	succ = NULL;

  debug(7, "adg_reverse_graph", "begin \n");
  MAPL( ver_ptr, {
    vertex		ver = NULL;
    vertex		ver2 = NULL;
    vertex 		ver5 = NULL;

    ver  = VERTEX(CAR( ver_ptr ));
    ver5 = adg_same_order_in_dg( verlist, ver );
    if ( ver5 == vertex_undefined ) {
      ver2 = make_vertex( (dg_vertex_label) vertex_vertex_label( ver ),(list) NIL );
      ADD_ELEMENT_TO_LIST( verlist, VERTEX, ver2 );
    }
    else {ver2 = ver5;}
    
    MAPL( succ_ptr, {
      list 	 li = NIL;
      successor	 succ2 = NULL;
      vertex	 ver3 = NULL;
      vertex     ver4 = NULL;
      
      succ  = SUCCESSOR(CAR( succ_ptr ));
      ver3  = successor_vertex( succ );
      ver5  = adg_same_order_in_dg( verlist, ver3 );
      succ2 = make_successor(dg_arc_label_dup((dg_arc_label)successor_arc_label(succ)),ver2 );
      if ( ver5 == vertex_undefined ) {
	ADD_ELEMENT_TO_LIST( li, SUCCESSOR, succ2 );
	ver4 = make_vertex( (dg_vertex_label) vertex_vertex_label(ver3),(list) li );
	ADD_ELEMENT_TO_LIST( verlist, VERTEX, ver4 );
      }
      else { ADD_ELEMENT_TO_LIST( vertex_successors( ver5 ),  SUCCESSOR, succ2 );}
    }, vertex_successors( ver ) );

  }, graph_vertices( g ) );
		
  debug(7, "adg_reverse_graph", "end  \n");
  return( make_graph( verlist ) );
}
/*=======================================================================*/
