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
/* Name     :	adg_summary.c
 * Package  :	array_dfg
 * Author   : 	Arnauld LESERVOT
 * Date     :	94/02/10
 * Modified :
 * Documents:	"Le Calcul de l'Array Data Flow Graph dans PIPS
 *		Partie II : Implantation dans PIPS" A. LESERVOT
 *		"Dataflow Analysis of Array and Scalar References" P. FEAUTRIER
 * Comments :
 */

#define GRAPH_IS_DG
#include "local.h"

/* Local defines */
#define NEXT(cp) (((cp) == NIL) ? NIL : (cp)->cdr)

/* Global variables */
extern 	hash_table 		Gvertex_number_to_statement;
extern	int			Gcount_re;
extern	statement_mapping	Gstco_map;
extern  list			Gstructural_parameters;

extern int my_pip_count;

/*=======================================================================*/
/* Pentity_vertices pev_new()                                      AL 15/02/94
 * Allocates a Pentity_vertices.
 */
Pentity_vertices pev_new()
{
  Pentity_vertices ret_pev = NULL;
  
  ret_pev = (Pentity_vertices) malloc( sizeof( Sentity_vertices ) );
  if (ret_pev == NULL) {
    (void) fprintf(stderr,"pev_new: Out of memory space\n");
    exit(-1);
  }
  ret_pev->ent = (void *) NULL;
  ret_pev->lis = (void *) NULL;
  return ret_pev;
}

/*=======================================================================*/
/* list adg_get_read_entity_vertices( (graph) in_dg )		AL 94/02/10
 *
 * Returns a list of Pentity_vertices, that represents all
 * variables read in the input graph and vertices that read each 
 * of these variables. We do not keep entities that are in in_l,
 * which represent structural parameters.
 * PRIVATE use.
 */
list adg_get_read_entity_vertices( in_dg, in_l )
graph in_dg;
list  in_l;
{
  list   vl = NULL, ret_list = NIL;
  
  /* Initialization */
  debug(9, "adg_get_read_entity_vertices", "begin \n");
  
  /* Run over all vertices to take their read variables */
  for( vl = graph_vertices( in_dg ); !ENDP(vl); POP(vl) ) {
    vertex		ver = NULL;	/* Current vertex */
    statement	        sta = NULL;	/* Statement of vertex */
    list 		effs = NULL;    /* effects linked to ver */
    list 		ent_l = NIL;	/* variable read by ver */

    /* Take current vertex and effects links to it */
    ver = VERTEX(CAR( vl ));
    sta = adg_vertex_to_statement( ver );
    effs = load_proper_rw_effects_list( sta );


    /* Put in ent_l variables readen by ver */
    for(; !ENDP(effs); POP(effs)) {
      effect 	eff = NULL;	/* Current effect */
      entity	ent = NULL;	/* variable readden by effect eff */
      
      eff = EFFECT(CAR( effs ));
      ent = reference_variable( effect_any_reference( eff ) );
      if (!action_read_p(effect_action( eff ))) continue;
      if (is_entity_in_list_p( ent, ent_l )) continue;
      if (is_entity_in_list_p( ent, in_l )) continue;
      ADD_ELEMENT_TO_LIST( ent_l, ENTITY, ent );
    }

    /* Update our list */
    /* We scan over the readen variables in ent_l... */
    for(; !ENDP(ent_l); POP(ent_l)) {
      entity			ent = NULL;
      list			prov_l = NULL;
      Pentity_vertices	        pev = NULL;
      boolean			found = false;

      ent = ENTITY(CAR( ent_l ));
      
      /* ... to see if it is already in our ret_list */
      for(prov_l = ret_list; !ENDP(prov_l); POP(prov_l)) {
	Pentity_vertices	pev2 = NULL;
	
	pev2 = (Pentity_vertices) CHUNK(CAR( prov_l ));
	if (pev2->ent != ent) continue;
	
	/* We find it : we update pev2 */
	ADD_ELEMENT_TO_LIST(pev2->lis, VERTEX, ver);
	found = true;
	break;
      }
      if (found) continue;

      /* entity ent is not in our ret_list : we add it */
      pev = pev_new();
      pev->ent = ENTITY(CAR( ent_l ));
      ADD_ELEMENT_TO_LIST(pev->lis, VERTEX, ver);
      ADD_ELEMENT_TO_LIST(ret_list, CHUNK, (chunk *) pev);
    }
  }

  debug(9, "adg_get_read_entity_vertices", "end \n");
  return ret_list;
}
		

/*=======================================================================*/
/* list adg_get_write_entity_vertices( (graph) in_dg )		AL 94/02/10
 *
 * Returns a list of Pentity_vertices, that represents all
 * variables written in the input graph and vertices that write each 
 * of these variables.
 * PRIVATE use.
 */
list adg_get_write_entity_vertices( in_dg )
graph in_dg;
{
  list 			vl = NULL, ret_list = NIL;

  /* Initialization */
  debug(9, "adg_get_write_entity_vertices", "begin \n");

  /* Run over all vertices to take their write variable */
  for( vl = graph_vertices( in_dg ); !ENDP(vl); POP(vl) ) {
    vertex	        ver = NULL;	/* Current vertex */
    statement	        sta = NULL;	/* statement of ver */
    list 	        effs = NULL;	/* effects linked of sta */
    entity 		w_ent = NULL;	/* variable written by sta */
    list		prov_l = NULL;
    boolean		found = false;
    Pentity_vertices	pev = NULL;

    /* Take current vertex and effects links to it */
    ver = VERTEX(CAR( vl ));
    sta = adg_vertex_to_statement( ver );
    if (!assignment_statement_p( sta )) continue;
    effs = load_proper_rw_effects_list( sta );

    /* Put in ent_l variables readen by ver */
    for(; !ENDP(effs); POP(effs)) {
      effect 	eff = NULL;	/* Current effect */
      entity	ent = NULL;	/* variable readden by effect eff */

      eff = EFFECT(CAR( effs ));
      ent = reference_variable( effect_any_reference( eff ) );
      if (!action_write_p(effect_action( eff ))) continue;
      w_ent = ent;
      break;
    }

    /* Update our list */
    /* Is w_ent already in our ret_list ?*/
    for(prov_l = ret_list; !ENDP(prov_l); POP(prov_l)) {
      Pentity_vertices	pev2 = NULL;
      
      pev2 = (Pentity_vertices) CHUNK(CAR( prov_l ));
      if (pev2->ent != w_ent) continue;

      /* We find it : we update pev2 */
      ADD_ELEMENT_TO_LIST(pev2->lis, VERTEX, ver);
      found = true;
      break;
    }
    if (found) continue;
    
    /* entity w_ent is not in our ret_list : we add it */
    pev = pev_new();
    pev->ent = w_ent;
    ADD_ELEMENT_TO_LIST(pev->lis, VERTEX, ver);
    ADD_ELEMENT_TO_LIST(ret_list, CHUNK, (chunk *) pev);
  }

  debug(9, "adg_get_write_entity_vertices", "end \n");
  return ret_list;
}



/*=======================================================================*/
/* graph adg_dataflowgraph_with_extremities( 	
 *				(statement) 	    module_statement,
 *				(statement_mapping) static_control_map,
 * 				(graph)	   	    reversed_dg )
 *								AL 94/02/10 
 */
graph                   adg_dataflowgraph_with_extremities( 
					mod_stat, stco_map, dup_dg )
statement               mod_stat;
statement_mapping       stco_map;
graph                   dup_dg;
{
  graph   	ret_graph = graph_undefined;
  list		ret_verl = NIL;
  vertex        ret_dest_ver = NULL;
  list          dest_psl = NULL;
  predicate	prov_pr = NULL;
  list		write_list = NULL, read_list = NULL;
  quast		entry_q = NULL, exit_q = NULL;
  vertex	entry_v = NULL, exit_v = NULL;


  debug(1, "adg_dataflowgraph_with_extremities", "begin \n");

  /* We first put entry node in the return list */
  entry_v = make_vertex( make_dfg_vertex_label( ENTRY_ORDER,
					       predicate_undefined, sccflags_undefined ), NIL );
  ADD_ELEMENT_TO_LIST( ret_verl, VERTEX, entry_v );
  entry_q = make_quast( make_quast_value(is_quast_value_quast_leaf,
					 make_quast_leaf( NIL, make_leaf_label(ENTRY_ORDER, 0) )),
		       NIL );
  hash_put( Gvertex_number_to_statement, (char *) ENTRY_ORDER, (char *) ENTRY_ORDER );


  /* Make exit_v : the exit node */
  exit_v = make_vertex( make_dfg_vertex_label( EXIT_ORDER, 
					      predicate_undefined, sccflags_undefined ), NIL);
  ADD_ELEMENT_TO_LIST( ret_verl, VERTEX, exit_v );
  exit_q = make_quast( make_quast_value(is_quast_value_quast_leaf,
					make_quast_leaf( NIL, make_leaf_label(EXIT_ORDER, 0) )),
		      NIL );
  hash_put( Gvertex_number_to_statement, (char *) EXIT_ORDER, (char *) EXIT_ORDER );




  /******************************************************************* 
   * WRITE EFFECTS
   */
  if (get_debug_level() > 3) { 			/* For debug purpose */
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Destination Statement: EXIT NODE \n");
  }


  /* Compute source quast for the exit node */
  ret_dest_ver = exit_v;
  write_list = adg_get_write_entity_vertices( dup_dg );
  for(; !ENDP( write_list ); POP( write_list )) {
    Pentity_vertices	pev = NULL;
    list		sou_l = NULL;
    entity		dest_ent = NULL;
    Psysteme		dest_context = SC_RN;
    list		dims = NIL;	/* Dim des tableaux */
    list		dest_indic = NIL; /* Indices of Exit */
    reference		dest_ref = NULL;
    int			ie = (int) NULL, dims_length = (int) NULL;
    quast		source = quast_undefined;
		

    /* Get write entity and vertices that write it */
    pev = (Pentity_vertices) CHUNK(CAR( write_list ));
    dest_ent = pev->ent;
    sou_l = pev->lis;

    /* Build reference associated to destination entity */
    dims = variable_dimensions(type_variable(entity_type(dest_ent)));
    dims_length = gen_length( dims );
    for(ie = 1; ie <= dims_length; ie++) {
      entity	ind = NULL;
      ind = adg_get_integer_entity( ie );
      ADD_ELEMENT_TO_LIST( dest_indic, EXPRESSION, entity_to_expression(ind) );
    }
    dest_ref = make_reference( dest_ent, dest_indic ); 


    /* No sources : ENTRY writes it */
    if (sou_l == NIL) {
      /* No sources : Comes from Entry point */
      adg_fill_with_quast( &source, entry_q );
      
      /* Debugging */
      debug(9,"adg_dataflowgraph",
	    "No candidates => Entry Flow\n");
      if (get_debug_level() > 2) {
	fprintf(stderr,
		"\n ------  Final Source  ------\n");
	imprime_special_quast( stderr, source );
      }
      
      adg_update_dfg( source,
		     dest_ref,
		     ret_dest_ver,
		     pa_full(),
		     dest_context,
		     SC_RN,
		     dup_dg,
		     &ret_verl );
      
      continue;
    }
    sou_l = adg_decreasing_stat_order_sort( sou_l );
    

    /* Get context of destination entity */
    for( ie = 1; !ENDP(dims); POP(dims), ie++) {
      Pvecteur	pv = NULL;
      dimension	dim = DIMENSION(CAR( dims ));
      entity	ind = NULL;
      Psysteme	pss = NULL;

      ind = adg_get_integer_entity( ie );
      pv = vect_substract(EXPRESSION_PVECTEUR(dimension_lower( dim )),
			  vect_new((Variable) ind , VALUE_ONE) );
      pss = sc_make(CONTRAINTE_UNDEFINED,contrainte_make(pv));
      dest_context = sc_append(dest_context, sc_dup(pss));
      pv = vect_substract( vect_new((Variable) ind , VALUE_ONE),
			  EXPRESSION_PVECTEUR(dimension_upper( dim )) );
      pss = sc_make(CONTRAINTE_UNDEFINED,contrainte_make(pv));
      dest_context = sc_append(dest_context, sc_dup(pss));
    }


    /* We run over all possible candidates 
     * and compute to see how it could contribute to the source
     */
    for(; !ENDP( sou_l ); POP(sou_l) ) {
      vertex		sou_v = NULL;
      int		sou_order = (int) NULL, sou_d = (int) NULL;
      leaf_label  	sou_lel = NULL; 
      predicate	        sou_pred = NULL;
      list		sou_lcl = NULL, sou_args = NIL;
      statement	        sou_s = NULL;
      static_control	sou_stco = NULL;
      list		sou_psl = NULL;
      list		sou_loops = NULL;
      Psysteme	        sou_ps = SC_RN;
      Psysteme	        prov_ps = SC_RN;
      Psysteme	        loc_context = SC_RN;
      quast		sou_q = NULL;
      Pposs_source 	poss = NULL; 
      quast		*local_source = NULL;
      Ppath		local_path = NULL;
      int		max_depth = (int) NULL;
      int		enclosing_nb = (int) NULL;



      /* Get possible source vertex and informations 
       * linked to it 
       */
      sou_v = VERTEX(CAR( sou_l ));
      sou_d = 0;
      sou_lel = make_leaf_label( dfg_vertex_label_statement(
                                (dfg_vertex_label) vertex_vertex_label(sou_v)), 0);

      sou_s = adg_vertex_to_statement( sou_v );
      sou_order = statement_ordering( sou_s );
      sou_stco =  (static_control) GET_STATEMENT_MAPPING( stco_map,sou_s );
      sou_psl = static_control_params( sou_stco );
      sou_loops = static_control_loops(sou_stco);
      max_depth = 0;
      sou_lcl = adg_get_loop_indices( sou_loops );
      enclosing_nb = gen_length( sou_lcl );

      
      /* If this candidate is not possible, see the next.
       * if candidate is not valid with the present source.
       */
      poss = adg_path_possible_source(&source, sou_v, sou_d, pa_full(), TAKE_LAST);
      local_path = (Ppath) poss->pat;
      /* Not a possible source => get the next candidate */
      if (is_pa_empty_p( local_path )) continue;
      
      if (local_path == PA_UNDEFINED) prov_ps =  SC_UNDEFINED;
      else prov_ps = local_path->psys;
      local_source = (quast*) (poss->qua);

      loc_context = sc_append(sc_dup(prov_ps), dest_context);
      prov_ps = SC_RN;

      /* For debug purpose */
      if (get_debug_level() > 2) { 
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
      sou_args = reference_indices(syntax_reference(
					expression_syntax(EXPRESSION(
					CAR(call_arguments(
					instruction_call(
					statement_instruction(sou_s) )))
					))));


      
      /* Make corresponding indices equal in source and dest
       * F(u) = g(b) and put it in sou_ps.
       */
      for(ie = 1; !ENDP(sou_args); POP(sou_args), ie++){
	expression sou_e = NULL;
	Pvecteur   pvec = NULL, exit_pv = NULL;
	Psysteme   pss = NULL;
	
	exit_pv = vect_new((Variable) adg_get_integer_entity(ie),VALUE_ONE);
	sou_e = copy_expression(EXPRESSION(CAR(sou_args)));
	pvec = vect_substract( exit_pv, EXPRESSION_PVECTEUR( sou_e ));
	if (pvec != NULL) {
	  pss = sc_make( contrainte_make(pvec),CONTRAINTE_UNDEFINED );
	  sou_ps = sc_append(sou_ps, sc_dup(pss));
	}
      }
			
      
      /* Build source Psysteme (IF and DO contraints).
       * Build the context and rename variables .
       */
      /* Get predicate that comes from an IF statement */
      sou_pred = dfg_vertex_label_exec_domain( (dfg_vertex_label)vertex_vertex_label( sou_v ));
      if (sou_pred != predicate_undefined) 
	prov_ps = adg_sc_dup(predicate_system(sou_pred));
      
      /* Get predicate that comes from enclosing DO */
      prov_pr = adg_get_predicate_of_loops( sou_loops );
      if (prov_pr != predicate_undefined) 
	prov_ps = sc_append( prov_ps, predicate_system( prov_pr ) );


      /* Append sous_ps (F(u) = g(b) and seq. predicate)
       * with prov_ps (IF and DO constraints).
       */
      sou_ps = adg_suppress_2nd_in_1st_ps(  sc_append(sou_ps, prov_ps), loc_context);
      if ((sou_ps != NULL) && !my_sc_faisabilite( sou_ps )) continue;




      /* Compute the new candidate source.
       * We try to call PIP only if necesary.
       */
      if (get_debug_level() > 4) {
	fprintf(stderr, "\nSource Psysteme :\n");
	fprint_psysteme(stderr, sou_ps);
	if (sou_ps != SC_UNDEFINED) 
	  pu_vect_fprint(stderr, sou_ps->base);

	fprintf(stderr, "\nContext Psysteme :\n");
	fprint_psysteme(stderr, loc_context);
	if (loc_context != SC_RN) pu_vect_fprint(stderr,loc_context->base);
      }
      /* If there is no condition on source...*/
      if (sou_ps == SC_UNDEFINED) {
	sou_q = make_quast( make_quast_value(is_quast_value_quast_leaf,
					    quast_leaf_undefined), NIL );
      }
      else if (enclosing_nb == 0) {
	quast	prov_q = NULL;
	
	prov_ps = sc_append(sc_dup(sou_ps),loc_context);
	if( (prov_ps == NULL) || my_sc_faisabilite(prov_ps)) {
	  prov_q = make_quast( make_quast_value(
                                                is_quast_value_quast_leaf,
                                                quast_leaf_undefined ), NIL );
	  sou_q = make_quast( make_quast_value(
				   is_quast_value_conditional,
				   make_conditional( make_predicate(sou_ps),
						    prov_q, quast_undefined) ), 
			     NIL );
	}
	else sou_q = quast_undefined;
      }
      else  {
	/* Order the psysteme according to ent_l */
	Pvecteur prov_pv = NULL;

	prov_pv = adg_list_to_vect(sou_lcl, false);
	sou_q = pip_integer_max( sou_ps , loc_context , prov_pv);
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
      adg_path_max_source( local_source, &sou_q, local_path, dest_psl, TAKE_LAST );
      
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
		   dest_ref,
		   ret_dest_ver,
		   pa_full(),
		   dest_context,
		   SC_RN,
		   dup_dg,
		   &ret_verl );
  }

  if (get_debug_level() > 0) fprint_dfg(stderr, make_graph(ret_verl));


  /******************************************************************* 
   * READ EFFECTS
   *
   * Here, the order is reversed: we compute for each reference the 
   * first operation that reads it. Destination is always the entry point,
   * and the source an operation of the program.
   */
  if (get_debug_level() > 3) { 			/* For debug purpose */
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Destination Statement: ENTRY NODE \n");
  }
	

  /* Compute source quast for the entry node */
  ret_dest_ver = entry_v;
  read_list = adg_get_read_entity_vertices( dup_dg, Gstructural_parameters );
  for(; !ENDP( read_list ); POP( read_list )) {
    Pentity_vertices	pev = NULL;
    list		sou_l = NULL;
    entity		dest_ent = NULL;
    Psysteme		dest_context = SC_RN;
    list		dims = NIL;	  /* Dim des tableaux */
    list		dest_indic = NIL; /* Indices of Exit */
    reference		dest_ref = NULL;
    int			ie = (int) NULL, dims_length = (int) NULL;
    quast		source = quast_undefined;
		

    /* Get read entity and vertices that read it */
    pev = (Pentity_vertices) CHUNK(CAR( read_list ));
    dest_ent = pev->ent;
    sou_l = pev->lis;

    /* Build reference associated to destination entity */
    dims = variable_dimensions(type_variable(entity_type(dest_ent)));
    dims_length = gen_length( dims );
    for(ie = 1; ie <= dims_length; ie++) {
      entity	ind = NULL;
      ind = adg_get_integer_entity( ie );
      ADD_ELEMENT_TO_LIST( dest_indic, EXPRESSION, entity_to_expression(ind) );
    }
    dest_ref = make_reference( dest_ent, dest_indic ); 
    

    /* No sources : EXIT reads it */
    if (sou_l == NIL) {
      /* No sources : Comes from exit point */
      adg_fill_with_quast( &source, exit_q );

      /* Debugging */
      debug(9,"adg_dataflowgraph", "No candidates => Exit Flow\n");
      if (get_debug_level() > 2) {
	fprintf(stderr,"\n ------  Final Source  ------\n");
	imprime_special_quast( stderr, source );
      }

      adg_update_dfg( source,
		     dest_ref,
		     ret_dest_ver,
		     pa_full(),
		     dest_context,
		     SC_RN,
		     dup_dg,
		     &ret_verl );
      
      continue;
    }
    sou_l = adg_increasing_stat_order_sort( sou_l );
    
    
    /* Get context of destination entity */
    for( ie = 1; !ENDP(dims); POP(dims), ie++) {
      Pvecteur	pv = NULL;
      dimension	dim = DIMENSION(CAR( dims ));
      entity	ind = NULL;
      Psysteme	pss = NULL;

      ind = adg_get_integer_entity( ie );
      pv = vect_substract( EXPRESSION_PVECTEUR(dimension_lower( dim )),
			  vect_new((Variable) ind , VALUE_ONE) );
      pss = sc_make(CONTRAINTE_UNDEFINED,contrainte_make(pv));
      dest_context = sc_append(dest_context, sc_dup(pss));
      pv = vect_substract( vect_new((Variable) ind , VALUE_ONE),
			  EXPRESSION_PVECTEUR(dimension_upper( dim )) );
      pss = sc_make(CONTRAINTE_UNDEFINED,contrainte_make(pv));
      dest_context = sc_append(dest_context, sc_dup(pss));
    }


    /* We run over all possible candidates 
     * and compute to see how it could contribute to the source
     */
    for(; !ENDP( sou_l ); POP(sou_l) ) {
      vertex		sou_v = NULL;
      int		sou_order = (int) NULL, sou_d = (int) NULL;
      leaf_label  	sou_lel = NULL; 
      predicate	        sou_pred = NULL;
      list		sou_lcl = NULL;
      statement	        sou_s = NULL;
      static_control	sou_stco = NULL;
      list		sou_psl = NULL, sou_loops = NULL;
      list		sou_read = NIL, sou_effs = NIL;
      Psysteme	        sou_ps = SC_RN, sou_context = SC_RN;
      Psysteme	        prov_ps = SC_RN;
      Psysteme	        loc_context = SC_RN;
      quast		sou_q = NULL;
      Pposs_source 	poss = NULL; 
      quast		*local_source = NULL;
      Ppath		local_path = NULL;
      int		max_depth = (int) NULL;
      int		enclosing_nb = (int) NULL;



      /* Get possible source vertex and informations 
       * linked to it 
       */
      sou_v = VERTEX(CAR( sou_l ));
      sou_d = 0;
      sou_lel = make_leaf_label( dfg_vertex_label_statement(
                                (dfg_vertex_label) vertex_vertex_label(sou_v)), 0);
			
      sou_s = adg_vertex_to_statement( sou_v );
      sou_order = statement_ordering( sou_s );
      sou_stco =  (static_control) GET_STATEMENT_MAPPING( stco_map, sou_s );
      sou_psl = static_control_params( sou_stco );
      sou_loops = static_control_loops(sou_stco);
      max_depth = 0;
      sou_lcl = adg_get_loop_indices( sou_loops );
      enclosing_nb = gen_length( sou_lcl );


      /* If this candidate is not possible, see the next.
       * if candidate is not valid with the present source.
       */
      poss = adg_path_possible_source(&source, sou_v, sou_d, pa_full(), TAKE_FIRST);
      local_path = (Ppath) poss->pat;
      /* Not a possible source => get the next candidate */
      if (is_pa_empty_p( local_path )) continue;
      
      local_source = (quast*) &source;
      loc_context = sc_dup( dest_context );

      
      /* Build source Psysteme (IF and DO contraints).
       */
      /* Get predicate that comes from an IF statement */
      sou_pred = dfg_vertex_label_exec_domain( (dfg_vertex_label)vertex_vertex_label( sou_v ));
      if (sou_pred != predicate_undefined) {
	sou_context = adg_sc_dup(predicate_system(sou_pred));
      }

      /* Get predicate that comes from enclosing DO */
      prov_pr = adg_get_predicate_of_loops( sou_loops );
      if (prov_pr != predicate_undefined) 
	sou_context = sc_append( prov_ps,predicate_system( prov_pr ) );
      
			

				
      /* Get all different effects that reads dest_ent */
      sou_effs = load_proper_rw_effects_list( sou_s );

      /* Put in ent_l variables readen by ver */
      for(; !ENDP(sou_effs); POP(sou_effs)) {
	effect 	eff = NULL;	/* Current effect */
	entity	ent = NULL;	/* variable readden by effect eff */
	
	eff = EFFECT(CAR( sou_effs ));
	if (!action_read_p(effect_action( eff ))) continue;
	ent = reference_variable( effect_any_reference( eff ) );
	if ( ent != dest_ent)  continue;
	if (is_entity_in_list_p( ent, sou_lcl )) continue;
	ADD_ELEMENT_TO_LIST( sou_read, EFFECT, eff );
      }

      
      for(; !ENDP(sou_read); POP(sou_read)) {
	effect	sou_eff = NULL;
	list	sou_args = NULL;
	

	sou_eff = EFFECT(CAR( sou_read ));
	/* For debug purpose */
	if (get_debug_level() > 2) { 
	  fprintf(stderr, "\nPossible Source Statement (ordering %d) ",sou_order);
	  fprintf(stderr, "at depth %d :\n", sou_d);
	  print_statement( sou_s );
	  fprintf(stderr,"for effect :\n");
	  print_words(stderr, words_effect(sou_eff));
	}
	
	/* Get the f(u) = g(b) psystem 
	 * We first duplicate arguments expressions,
	 * then we rename entities that are at 
	 * a deeper depth than sou_d and forward
	 * subsitute those new entities in the 
	 * expressions 
	 */
	sou_args = reference_indices(effect_any_reference(sou_eff));
	
	/* Make corresponding indices equal in source and dest
	 * F(u) = g(b) and put it in sou_ps.
	 */
	for(ie = 1; !ENDP(sou_args); POP(sou_args), ie++){
	  expression sou_e = NULL;
	  Pvecteur   pvec = NULL, exit_v = NULL;
	  Psysteme   pss = NULL;
	  
	  exit_v = vect_new((Variable) adg_get_integer_entity(ie),VALUE_ONE);
	  sou_e = copy_expression(EXPRESSION(CAR(sou_args)));
	  pvec = vect_substract( exit_v,EXPRESSION_PVECTEUR( sou_e ));
	  if (pvec != NULL) {
	    pss = sc_make( contrainte_make(pvec), CONTRAINTE_UNDEFINED );
	    sou_ps = sc_append(sou_ps, sc_dup(pss));
	  }
	}


			 


	/* Append sous_ps (F(u) = g(b) and seq. predicate)
	 * with prov_ps (IF and DO constraints).
	 */
	sou_ps = adg_suppress_2nd_in_1st_ps( sc_append(sou_ps, sou_context), loc_context);
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
	else if (enclosing_nb == 0) {
	  quast	prov_q = NULL;

	  prov_ps = sc_append(sc_dup(sou_ps),loc_context);
	  if( (prov_ps == NULL) || my_sc_faisabilite(prov_ps)) {
	    prov_q = make_quast( make_quast_value( is_quast_value_quast_leaf,
						  quast_leaf_undefined ), NIL );
	    sou_q = make_quast( make_quast_value( is_quast_value_conditional,
						 make_conditional(
						         make_predicate(sou_ps),
							 prov_q, quast_undefined)
						 ), NIL );
	  }
	  else sou_q = quast_undefined;
	}
	else  {
	  /* Order the psysteme according to ent_l */
	  Pvecteur prov_pv = NULL;

	  prov_pv = adg_list_to_vect(sou_lcl, false);
	  sou_q = pip_integer_max( sou_ps ,  loc_context , prov_pv);
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
	adg_path_max_source(local_source, &sou_q, local_path, dest_psl, TAKE_FIRST );
	
	if (get_debug_level() > 4) {
	  fprintf(stderr, "\n Updated Source :\n");
	  imprime_special_quast( stderr, source );
	}
      }
    }

    /* Fill "quast_undefined" part of the source
     * with EXIT node.
     */
    adg_fill_with_quast( &source, exit_q );


    /* Build the new Data Flow Graph with the new source*/
    if (get_debug_level() > 2) {
      fprintf(stderr, "\n ------  Final Source  ------\n");
      imprime_special_quast( stderr, source );
    }

    adg_update_dfg( source, 
		   dest_ref,
		   ret_dest_ver,
		   pa_full(),
		   dest_context,
		   SC_RN,
		   dup_dg,
		   &ret_verl );
  }


  if (get_debug_level() > 0) fprint_dfg(stderr, make_graph(ret_verl));


  ret_graph = make_graph( ret_verl );
  debug(1, "adg_dataflowgraph", "end \n");
  return( ret_graph );

}
/*=======================================================================*/
