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
/* Name     :	array_dfg.c
 * Package  :	array_dfg
 * Author   : 	Arnauld LESERVOT
 * Date     :	93/06/27
 * Modified :
 * Documents:	Platonoff's thesis and Leservot's thesis
 *		"Dataflow Analysis of Array and Scalar References" P. FEAUTRIER
 * Comments :
 */

#define GRAPH_IS_DFG
#include "local.h"

/* Local defines */
#define NEXT(cp) (((cp) == NIL) ? NIL : (cp)->cdr)

/* Global variables */
extern			hash_table	Gvertex_number_to_statement;
int			Gcount_re;
int			Gcount_ie;
statement_mapping	Gstco_map;
list			Gstructural_parameters;

int my_pip_count;
int my_fai_count;

static hash_table		Gforward_substitute_table;

bool my_sc_faisabilite( in_ps )
Psysteme in_ps;
{
  my_fai_count++;
  return(sc_rational_feasibility_ofl_ctrl(in_ps, NO_OFL_CTRL,true));
}

/*=======================================================================*/
/* graph adg_dataflowgraph( 	(statement) 	    module_statement,
 *				(statement_mapping) static_control_map,
 * 				(graph)	   	    reversed_dg )
 *								AL 93/07/01 
 * To compute the Array Data Flow Graph, we need :
 * code, static_control, dependance_graph.
 */
graph                   adg_dataflowgraph( mod_stat, stco_map, dup_dg )
statement               mod_stat;
statement_mapping       stco_map;
graph                   dup_dg;
{
  graph   ret_graph = graph_undefined; /* Return dfg graph */
  list	  ret_verl = NIL;		/* Return list of vertices */
  list    dest_ver_l = NIL;
  vertex  entry_v = NULL;
  quast	  entry_q = NULL;

  debug(1, "adg_dataflowgraph", "begin \n");

  /* Initialization to have an entry node */
  /* We first put entry node in the return list */
  entry_v = make_vertex(make_dfg_vertex_label(ENTRY_ORDER, 
					     predicate_undefined,
					     sccflags_undefined), NIL);
  ADD_ELEMENT_TO_LIST( ret_verl, VERTEX, entry_v );
  entry_q = make_quast( make_quast_value(is_quast_value_quast_leaf,
					 make_quast_leaf( NIL, make_leaf_label(ENTRY_ORDER, 0) )),
		       NIL );

  hash_put( Gvertex_number_to_statement, (char *) ENTRY_ORDER, (char *)
	   ENTRY_ORDER );


  /* We run over each vertex of the input graph */
  for(dest_ver_l = graph_vertices( dup_dg );!ENDP(dest_ver_l);POP(dest_ver_l)) {
    vertex           ret_dest_ver = NULL, dest_ver = NULL;
    list             dest_succ = NIL, dest_loops = NIL; 
    list	     dest_readl = NIL, dest_psl = NIL, dest_lcl = NIL;
    list	     dest_args = NIL, prov_l = NIL;
    predicate        dest_pred = NULL;
    Psysteme	     dest_test_context = NULL, dest_loop_context = NULL;
    Psysteme	     dest_context = NULL;
    static_control   dest_stco;
    int		     dest_nb, dest_order;
    dfg_vertex_label ret_dest_dvl = NULL;
    statement	     dest_st = NULL;
    predicate	     prov_pr = NULL;


    /* Get destination vertex and information linked to it */
    dest_ver   = VERTEX(CAR( dest_ver_l ));
    dest_st    = adg_vertex_to_statement(dest_ver);
    dest_succ  = vertex_successors( dest_ver );
    dest_nb    = dfg_vertex_label_statement
	((dfg_vertex_label) vertex_vertex_label( dest_ver ));
    dest_order = adg_number_to_ordering( dest_nb );
    dest_stco  = (static_control) GET_STATEMENT_MAPPING(stco_map, dest_st);
    dest_loops = static_control_loops(  dest_stco );
    dest_psl   = static_control_params( dest_stco );
    dest_lcl   = adg_get_loop_indices(  dest_loops );

    dest_pred = dfg_vertex_label_exec_domain(vertex_vertex_label(dest_ver));
    if (!predicate_undefined_p(dest_pred)) 
	dest_test_context = predicate_system( dest_pred );
    else dest_test_context = SC_RN;

    prov_pr = adg_get_predicate_of_loops(dest_loops);
    if (prov_pr != predicate_undefined) 
	dest_loop_context = predicate_system( prov_pr );
    else dest_loop_context = SC_RN;
    dest_context = sc_append(sc_dup( dest_test_context ),dest_loop_context);
    if (dest_context != SC_UNDEFINED) adg_sc_update_base(&dest_context);
    if ((dest_context != NULL)&&(!my_sc_faisabilite(dest_context))) continue;

    ret_dest_ver = adg_same_dfg_vertex_number( ret_verl, dest_nb );
    ret_dest_dvl = make_dfg_vertex_label( dest_nb, 
					 make_predicate( sc_dup(dest_context) ),
					 sccflags_undefined );
    if (ret_dest_ver == vertex_undefined) {
      ret_dest_ver = make_vertex( ret_dest_dvl, NIL );
      ADD_ELEMENT_TO_LIST( ret_verl, VERTEX, ret_dest_ver );
    }
    else 
	vertex_vertex_label(ret_dest_ver) = ret_dest_dvl;



    /* We search for all the read effects of vertex dest_ver 
     * and try to find the source for each read effect in dest_ver.
     */
    dest_readl = read_reference_list( dest_ver, dest_lcl, dest_psl );
    if (get_debug_level() > 3) { /* For debug purpose */
      fprintf(stderr, "\n========================================\n");
      fprintf(stderr, "Destination Statement (ordering %d) :\n",dest_order);
      print_statement( adg_number_to_statement( dest_nb ) );
      fprintf(stderr, "Read Effects :\n");
      print_effects( dest_readl );
    }
    for(; !ENDP( dest_readl ); POP( dest_readl )) {
      effect 	dest_read_eff = NULL;
      list   	sou_l = NULL, sou_lll = NIL;
      list	cand_l = NIL;
      int	max_depth = 0;
      int	prov_i;
      quast 	source = quast_undefined;

      /* Get the current read_effect of destination */
      dest_read_eff = EFFECT(CAR( dest_readl ));
      dest_args     = reference_indices(effect_any_reference(dest_read_eff));
      if (get_debug_level() > 3) { /* For debug purpose */
	fprintf(stderr, "\n\n-->  Source of Effect ?  ");
	print_effects( CONS(EFFECT, dest_read_eff, NIL) );
      }


      /* Search for successors (in fact predecessors : input 
       * graph is reversed compared to dg graph !) that write 
       * the dest_read_eff and put their vertices in sou_l.
       * Then, order them by decreasing order of stat. number.
       */
      sou_l = adg_write_reference_list( dest_ver, dest_read_eff );
      if (sou_l == NIL) {
	/* No sources : Comes from Entry point */
	adg_fill_with_quast( &source, entry_q );

	/* Debugging */
	debug(9,"adg_dataflowgraph","No candidates => Entry Flow\n");
	if (get_debug_level() > 2) {
	  fprintf(stderr,  "\n ------  Final Source  ------\n");
	  imprime_special_quast( stderr, source );
	}

	adg_update_dfg( source,
		       effect_any_reference( dest_read_eff ),
		       ret_dest_ver,
		       pa_full(),
		       NULL,
		       NULL,
		       dup_dg,
		       &ret_verl );

	continue;
      }
      sou_l = adg_decreasing_stat_order_sort( sou_l );


      /* Build the source leaf label list sou_lll.
       * This list of leaf labels links a vertex number and a depth.
       */
      for(; !ENDP(sou_l); POP(sou_l)) {
	vertex      v   = VERTEX(CAR( sou_l ));
	int         dep = stco_common_loops_of_statements(stco_map,
					      adg_vertex_to_statement( v ), dest_st);
	leaf_label  lel = make_leaf_label( dfg_vertex_label_statement(
                                    (dfg_vertex_label) vertex_vertex_label(v)), dep );
	ADD_ELEMENT_TO_LIST(sou_lll, LEAF_LABEL, lel);
	debug(9,"adg_dataflowgraph", "\nPossible source : stat %d at depth %d\n", 
	      statement_number( adg_vertex_to_statement(v) ), dep);
      }


      /* Explode candidates for each depth and then,
       * build the candidate list cand_l by decreasing order.
       */ 
      max_depth = 0;
      for(prov_l = sou_lll; !ENDP(prov_l); POP(prov_l)) {
	int dep2 = leaf_label_depth(LEAF_LABEL(CAR( prov_l )));
	if( dep2 > max_depth ) max_depth = dep2;
      }
      for(prov_i = max_depth; prov_i >= 0; prov_i-- ) {
	prov_l = sou_lll;
	for(; !ENDP(prov_l); POP(prov_l) ) {
	  leaf_label prov_lel = LEAF_LABEL(CAR(prov_l));
	  int dd = leaf_label_depth( prov_lel );
	  int nb = leaf_label_statement( prov_lel );
	  if( prov_i <= dd ) 
	    ADD_ELEMENT_TO_LIST(cand_l, LEAF_LABEL,make_leaf_label( nb, prov_i ));
	}
      }    
      max_depth = 0;		/* we will reuse it after */


      /* We run over all possible candidates 
       * and compute to see how it could contribute to the source
       */
      for(; !ENDP( cand_l ); POP(cand_l) ) {
	vertex		sou_v;
	int		sou_order, sou_d;
	leaf_label  	sou_lel; 
	predicate	sou_pred = NULL;
	list		sou_lcl = NULL, sou_args = NIL;
	Psysteme	sou_ps = SC_RN;
	statement	sou_s;
	static_control	sou_stco;
	list		sou_psl;
	list		sou_loops;
	list		ent_l = NULL, renamed_l = NULL, merged_l = NULL;
	list		prov_l1 = NIL, prov_l2 = NIL;
	Psysteme	prov_ps = SC_RN;
	Psysteme	loc_context = SC_RN;
	Pvecteur	prov_pv = NULL;
	quast		prov_q = NULL, sou_q = NULL;
	Pposs_source 	poss = NULL; 
	quast		*local_source = NULL;
	Ppath		local_path;


	/* Get possible source vertex and informations linked to it */
	sou_lel   = LEAF_LABEL(CAR( cand_l ));
	sou_v     = adg_number_to_vertex( dup_dg, leaf_label_statement(sou_lel) );
	sou_s     = adg_vertex_to_statement( sou_v );
	sou_d     = leaf_label_depth( sou_lel );
	sou_order = statement_ordering( sou_s );
	sou_stco  = (static_control) GET_STATEMENT_MAPPING( stco_map, sou_s );
	sou_psl   = static_control_params( sou_stco );
	sou_loops = static_control_loops(sou_stco);
	max_depth = adg_number_of_same_loops(sou_loops, dest_loops );
	sou_lcl   = adg_get_loop_indices( sou_loops );


	/* If this candidate is not possible, see the next.
	 * Two cases : candidate and destination are in the
	 * same deepest loop and dest is before candidate ;
	 * or candidate is not valid with the present source.
	 */
	if ((sou_d == max_depth) && adg_is_textualy_after_p(sou_s, dest_st)) continue;
	poss       = adg_path_possible_source(&source, sou_v, sou_d, pa_full(), TAKE_LAST);
	local_path = (Ppath) poss->pat;
	/* Not a possible source => get the next candidate */
	if (pa_empty_p( local_path )) continue;

	if PA_UNDEFINED_P(local_path) prov_ps = SC_UNDEFINED;
	else prov_ps = local_path->psys;
	local_source = (quast*) (poss->qua);
	loc_context  = sc_append(sc_dup(prov_ps), dest_context);
	prov_ps      = SC_UNDEFINED; /* will be reuse ? */

	/* For debug purpose */
	if (get_debug_level() > 3) { 
	  fprintf(stderr, "\nPossible Source Statement (ordering %d) ",sou_order);
	  fprintf(stderr, "at depth %d :\n", sou_d);
	  print_statement( sou_s );
	}

				
	/* Get the f(u) = g(b) psystem 
	 * We first duplicate arguments expressions,
	 * then we rename entities that are at 
	 * a deeper depth than sou_d and forward
	 * subsitute those new entities in the 
	 * expressions 
	 */
	sou_args = reference_indices(syntax_reference(expression_syntax(
                                      EXPRESSION(CAR(call_arguments(instruction_call(
					    statement_instruction(sou_s) ))))  )));
	if(gen_length(sou_args) != gen_length(dest_args)) {
	  pips_internal_error("No coherence between the source array and destination array !");
	}

			
	/* Rename entities at rank > sou_d
	 * and update Gforward_substitute_table 
	 */
	for(prov_i=0; prov_i < sou_d; prov_i++) POP(sou_lcl);
	Gforward_substitute_table = hash_table_make(hash_pointer, 2+gen_length(sou_lcl));
	renamed_l = adg_rename_entities(sou_lcl, Gforward_substitute_table);

		
	/* Make corresponding indices equal in source and dest
	 * F(u) = g(b) and put it in sou_ps.
	 */
	prov_l1 = dest_args;
	for(prov_l2=sou_args;!ENDP(prov_l2);POP(prov_l2)){
	  expression sou_e = NULL, dest_e = NULL;
	  Pvecteur   pvec  = NULL;
	  Psysteme   pss   = NULL;

	  dest_e = EXPRESSION(CAR(prov_l1));
	  POP( prov_l1 );
	  sou_e = copy_expression( EXPRESSION(CAR(prov_l2)));
	  forward_substitute_in_exp(&sou_e, Gforward_substitute_table);

	  pvec = vect_substract(EXPRESSION_PVECTEUR(sou_e),
				EXPRESSION_PVECTEUR(dest_e));
	  if (pvec != NULL) {
	    pss    = sc_make( contrainte_make(pvec), CONTRAINTE_UNDEFINED );
	    sou_ps = sc_append(sou_ps, sc_dup(pss));
	  }
	}


	/* Build the sequencing predicate */
	if ( (sou_d >= 0) && (sou_d < max_depth) ) {
	  entity  indice1 = ENTITY( gen_nth(sou_d,dest_lcl) );
	  entity  indice2 = ENTITY( gen_nth(sou_d,dest_lcl) );
	  
	  if (renamed_l != NIL) indice1 = ENTITY(CAR(renamed_l));
	  /* compute indice1 + indice2 + 1 */
	  prov_pv = vect_add( vect_new(TCST, VALUE_ONE),
		  vect_substract(vect_new((Variable) indice1, VALUE_ONE),
				 vect_new((Variable) indice2, VALUE_ONE)) );
	  sou_ps = sc_append(sou_ps,
			     sc_make(CONTRAINTE_UNDEFINED, contrainte_make( prov_pv )));
	}
			
	/* append at the end p.s. of source to those of dest.
	 * Concatenate the three lists to build Psys.
	 * according to the order :
	 * source-variables,sink-variables,struc.params
	 */
	merged_l = adg_merge_entities_lists(dest_psl,sou_psl);
	ent_l    = gen_append( renamed_l, gen_append( dest_lcl, merged_l ) );
			


	/* Build source Psysteme (IF and DO contraints).
	 * Build the context and rename variables .
	 */
	/* Get predicate that comes from an IF statement */
	sou_pred = dfg_vertex_label_exec_domain(
						vertex_vertex_label( sou_v ));
	if (sou_pred != predicate_undefined) prov_ps = adg_sc_dup(predicate_system(sou_pred));
	/* Get predicate that comes from enclosing DO */
	prov_pr = adg_get_predicate_of_loops( sou_loops );
	if (prov_pr != predicate_undefined) {
	  prov_ps = sc_append( prov_ps, predicate_system( prov_pr ) );
	}
	/* Rename entities in the source context system */
	HASH_MAP( k, v, {
	  char* vval = (char *) reference_variable(
			      syntax_reference(expression_syntax((expression) v) ));
	  sc_variable_rename(prov_ps, (Variable) k, (Variable) vval);
	}, Gforward_substitute_table );
	hash_table_free(Gforward_substitute_table);


	/* Append sous_ps (F(u) = g(b) and seq. predicate)
	 * with prov_ps (IF and DO constraints).
	 */
	sou_ps = adg_suppress_2nd_in_1st_ps( sc_append(sou_ps, prov_ps), loc_context);
	if ((sou_ps != NULL) && !my_sc_faisabilite( sou_ps )) continue;

	/* Compute the new candidate source.
	 * We try to call PIP only if necesary.
	 */
	if (get_debug_level() > 4) {
	  fprintf(stderr, "\nSource Psysteme :\n");
	  fprint_psysteme(stderr, sou_ps);
	  if (sou_ps != SC_UNDEFINED) pu_vect_fprint(stderr, sou_ps->base);

	  fprintf(stderr, "\nContext Psysteme :\n");
	  fprint_psysteme(stderr, loc_context);
	  if (loc_context != SC_RN)  pu_vect_fprint(stderr,loc_context->base);
	}
	/* If there is no condition on source...*/
	if (sou_ps == SC_UNDEFINED) {
	  sou_q = make_quast( make_quast_value( is_quast_value_quast_leaf,
					      quast_leaf_undefined), NIL );
	}
	else if (gen_length(renamed_l) == 0) {
	  prov_ps = sc_append( sc_dup(sou_ps), loc_context );
	  if( (prov_ps == NULL) || my_sc_faisabilite(prov_ps)) {
	    prov_q = make_quast( make_quast_value( is_quast_value_quast_leaf,
						  quast_leaf_undefined ), NIL );
	    sou_q  = make_quast( make_quast_value(is_quast_value_conditional,
						 make_conditional( make_predicate(sou_ps),
								  prov_q, quast_undefined)
						 ), NIL );
	  }
	  else sou_q = quast_undefined;
	}
	else  {
	  Pvecteur pv_unknowns;

	  /* Order the psysteme according to ent_l */
	  sort_psysteme( sou_ps, adg_list_to_vect(ent_l, true) );
	  pv_unknowns = list_to_base(renamed_l);
	  sou_q = pip_integer_max(sou_ps, loc_context, pv_unknowns);
	  my_pip_count++;
	  if (get_debug_level() > 4) imprime_special_quast( stderr, sou_q );
	  sou_q = adg_compact_quast( sou_q );
	}
	adg_enrichir( sou_q, sou_lel );
	if (get_debug_level() > 4) {
	  fprintf(stderr, "\nPresent source quast :\n");
	  imprime_special_quast( stderr, sou_q );
	}
			

	/* Find the new source and simplify it */
	adg_path_max_source(local_source, &sou_q, local_path, dest_psl, TAKE_LAST );

	if (get_debug_level() > 4) {
	  fprintf(stderr, "\n Updated Source :\n");
	  imprime_special_quast( stderr, source );
	}
      }


      /* Fill "quast_undefined" part of the source 
       * with ENTRY node.
       */
      adg_fill_with_quast( &source, entry_q );


      /* Build the new Data Flow Graph with the new source*/
      if (get_debug_level() > 2) {
	fprintf(stderr, "\n ------  Final Source  ------\n");
	imprime_special_quast( stderr, source );
      }

      adg_update_dfg( source, 
		     effect_any_reference( dest_read_eff ),
		     ret_dest_ver,
		     pa_full(),
		     dest_context,
		     dest_test_context,
		     dup_dg,
		     &ret_verl );
    }
  }

  ret_graph = make_graph( ret_verl );
  debug(1, "adg_dataflowgraph", "end \n");
  return( ret_graph );

}

/*=======================================================================*/
/* void array_dfg( (char*) module_name ) 			AL 93/06/29
 * 
 * It computes the array data flow graph 
 * using Feautrier's algorithm. This kind of graph detects the real 
 * dependances between arrays. It could be computed on a static control
 * program. The original code is prepared by the static_controlize
 * package. See its comments for more details.
 */
boolean	array_dfg( mod_name )
char* 	mod_name;
{
  extern int	       	Gcount_re;
  extern int		Gcount_ie;
  graph			dg = NULL, rev_dg = NULL, wr_dg = NULL; 
  graph			dup_dg = NULL, ret_dfg = NULL;
  entity	       	ent = NULL;
  statement 		mod_stat = NULL;
  static_control       	stco = NULL;
  string       		ss = NULL; /* summary or not ? */
  bool      		SUMMARY = false;
  
  /* Initialize debugging functions */
  debug_on("ARRAY_DFG_DEBUG_LEVEL");
  if (get_debug_level() > 0) 
    user_log("\n\n *** COMPUTE ARRAY DATA FLOW GRAPH for %s\n",mod_name);


  my_pip_count = 0;
  my_fai_count = 0;

  /* Initialization of the pass */
  Gcount_re = 0;
  Gcount_ie = 0;
  ent       = local_name_to_top_level_entity( mod_name );
  set_current_module_entity(ent); /* set current_module_entity to ent ... */
  
  mod_stat  = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);
  Gstco_map = (statement_mapping) db_get_memory_resource(DBR_STATIC_CONTROL,
							 mod_name, true);
  
  /* If the input program is not a static_control one, return */
  stco	 = (static_control) GET_STATEMENT_MAPPING(Gstco_map, mod_stat);
  if ( !static_control_yes( stco )) {
    user_error( "array_dfg",
	       "\n CAN'T APPLY FEAUTRIER'S ALGORITHM :\n This is not a static control program !\n" );
  }
  Gstructural_parameters = static_control_params( stco );
  set_proper_rw_effects((statement_effects) 
		       db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, true));

  /* What will we compute ? */
  SUMMARY = ((ss = getenv("SUMMARY")) != NULL)? atoi(ss) : false;
  
  
  /* We need the dependance graph for a first source approximation.
   * The graph is first reversed to have the possible source statement.
   * Then we take only the WR dependances.
   * At the end : duplicate nodes "a la Redon" for IF statement.
   */
  dg	  = (graph) db_get_memory_resource( DBR_DG, mod_name, true );
  rev_dg  = adg_reverse_graph( dg );
  wr_dg   = adg_only_call_WR_dependence( rev_dg );
  dup_dg  = adg_dup_disjunctive_nodes( wr_dg, Gstco_map );
  
  /* We reorder the statement number linked to each vertex
   * in order to distinguich duplicated vertices 
   */
  adg_reorder_statement_number( dup_dg );
  
  /* We compute the core of the pass */
  if (!SUMMARY) 
  { ret_dfg = adg_dataflowgraph( mod_stat, Gstco_map, dup_dg );}
  else ret_dfg = adg_dataflowgraph_with_extremities(mod_stat, Gstco_map, dup_dg);
  
  
  /* End of the program */
  if (get_debug_level() > 0) fprint_dfg(stderr, ret_dfg);
  if (get_debug_level() > 8) fprint_dfg(stderr, adg_pure_dfg(ret_dfg));
  
  DB_PUT_MEMORY_RESOURCE( DBR_ADFG, strdup(mod_name), ret_dfg);
  
  if (get_debug_level() > 0)  {
    printf("\n PIP CALLS : %d\n", my_pip_count);
    printf("\n FAI CALLS : %d\n", my_fai_count);
  }

  if (get_debug_level() > 0) user_log("\n\n *** ARRAY_DFG done\n");
  debug_off();

  reset_proper_rw_effects();
  reset_current_module_entity();
  reset_current_module_statement();

  return(true);
}

/*=======================================================================*/
