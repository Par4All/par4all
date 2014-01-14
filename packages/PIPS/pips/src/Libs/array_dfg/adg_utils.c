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
/* Name     :   adg_utils.c
 * Package  :   array_dfg
 * Author   :   Arnauld LESERVOT
 * Date     :   93/06/27
 * Modified :
 * Documents:   Platonoff's thesis and Leservot's thesis
 *              "Dataflow Analysis of Array and Scalar References" P. FEAUTRIER
 * Comments :
 */

#define GRAPH_IS_DG
#include "local.h"

/* Global variables */
extern	int			Gcount_re;
extern  statement_mapping	Gstco_map;
extern	boolean			PATH_METHOD;

/*=======================================================================*/
/*			USEFULL FUNCTIONS				 */
/*=======================================================================*/

/*=======================================================================*/
/* void	adg_fill_with_quast( in_pq, in_q )			AL 17/02/94
 */
void adg_fill_with_quast( in_pq, in_q )
quast*	in_pq;
quast	in_q;
{
  debug(8, "adg_fill_with_quast", "begin\n");	
  if (get_debug_level() > 7) {
    fprintf(stderr, "\n Input quast :\n");
    imprime_special_quast( stderr, *in_pq );
    fprintf(stderr, "\n To fill with :\n");
    imprime_special_quast( stderr, in_q );
  }
  if (*in_pq == quast_undefined) *in_pq = in_q;
  else { 
    quast_value	qqv = quast_quast_value( *in_pq );
    if (quast_value_conditional_p( qqv )) {
      conditional	cond = quast_value_conditional( qqv );
      adg_fill_with_quast( &(conditional_true_quast( cond )), in_q );
      adg_fill_with_quast( &(conditional_false_quast( cond )),in_q );
    }
  }
  if (get_debug_level() > 7) {
    fprintf(stderr, "\n Output quast :\n");
    imprime_special_quast( stderr, *in_pq );
  }
  debug(8, "adg_fill_with_quast", "end\n");	
}


/*=======================================================================*/
/* AL 94/02/14
 */
entity adg_get_integer_entity( in_i )
int	in_i;
{
  extern  int Gcount_ie;
  entity  new_ent = NULL, mod_ent = NULL;
  char    *name = NULL, *name2 = NULL, *num = NULL;
  
  
  debug(9, "adg_get_integer_entity", "begin \n");
  num     = i2a(in_i);
  mod_ent = get_current_module_entity();
  name    = strdup(concatenate("I", (char*) num, (char *) NULL));
  free(num);
  
  /* If a Renamed Entity already exists, we use it ;
   * else, we make a new one and increment Gcount_re.
   */
  if ( in_i <= Gcount_ie ) {
    new_ent = FindOrCreateEntity(
		strdup(concatenate(ADFG_MODULE_NAME, entity_local_name(mod_ent), (char*) NULL)),
		(char*) name );
  }
  else {
    Gcount_ie++;
    name2 = strdup(concatenate( ADFG_MODULE_NAME, entity_local_name(mod_ent),
			       MODULE_SEP_STRING, name, NULL ));
    new_ent = make_entity(name2,
			  make_type(is_type_variable,
			 make_variable(make_basic(is_basic_int,UUINT(4)),NIL)),
			  make_storage(is_storage_ram, ram_undefined),
			  make_value(is_value_unknown, UU));
  }

  if (get_debug_level() > 7) fprintf(stderr, "New %s\n", entity_local_name( new_ent ));
        
  debug(9, "adg_get_integer_entity", "end \n");
  return( new_ent );
}


/*=======================================================================*/
/* quast adg_compact_quast( in_q )				AL 1/12/93
 * Compact a quast with a lot of undefined leaves.
 * Could be costly extended to more general cases.
 * Usefull to compact a quast provided by PIP.
 */
quast adg_compact_quast( in_q )
quast in_q;
{
  quast		new_true = NULL, new_false = NULL, ret_q = quast_undefined;
  quast_value	qv = NULL;
  conditional	cond = NULL;
  Psysteme	ps1 = NULL;

  debug( 9, "adg_compact_quast", "begin\n");
  if ( in_q == quast_undefined ) return quast_undefined;
  
  qv = quast_quast_value( in_q );
  if ( qv == quast_value_undefined ) return quast_undefined;
  if ( quast_value_quast_leaf_p( qv ) ) return in_q;
  
  cond      = quast_value_conditional( qv );
  ps1       = predicate_system( conditional_predicate( cond ) );
  new_true  = adg_compact_quast( conditional_true_quast( cond ) ); 
  new_false = adg_compact_quast( conditional_false_quast( cond ) ); 
  
  if (adg_quast_equal_p( new_true, new_false )) 
    {free_quast( new_false );  return new_true;}
  else if ((new_true != quast_undefined) &&
	   (quast_quast_value(new_true) != quast_value_undefined) &&
	   (quast_value_conditional_p(quast_quast_value(new_true)))) {
    conditional co = NULL;
    quast	cfq = NULL, cft = NULL;
    
    co = quast_value_conditional(quast_quast_value(new_true));
    cfq = conditional_false_quast( co );
    cft = conditional_true_quast( co );
    if( adg_quast_equal_p( cfq, new_false ) ) {
      Psysteme	ps = NULL;
      
      ps1->base = NULL; sc_creer_base(ps1);
      ps = sc_append(sc_dup(ps1), predicate_system(conditional_predicate( co )));
      ret_q = make_quast( make_quast_value(is_quast_value_conditional, 
					  make_conditional(make_predicate(ps), 
							   cft, new_false) ), 
			 quast_newparms( in_q ) );
    }
    else ret_q = make_quast( make_quast_value(is_quast_value_conditional, 
					     make_conditional(make_predicate(ps1), 
							      new_true, new_false) ), 
			    quast_newparms( in_q ) );
  }
  else {
    ret_q = make_quast(make_quast_value(is_quast_value_conditional, 
					make_conditional(make_predicate(ps1), 
							 new_true, new_false) ), 
		       quast_newparms( in_q ) );  
  }

  debug( 9, "adg_compact_quast", "end\n");
  return ret_q;
}


/*=======================================================================*/
/* bool adg_quast_equal_p() 					AL 1/12/93
 * Returns true if the 2 input quasts are equal to quast_undefined.
 * Could be extented.
 */
bool adg_quast_equal_p( in_q1, in_q2 )
quast in_q1, in_q2;
{ return (( in_q1 == quast_undefined ) && (in_q2 == quast_undefined)); }


/*=======================================================================*/
/* bool adg_quast_value_equal_p()				AL 1/12/93
 * NOT used, NOT tested.
 */
bool adg_quast_value_equal_p( in_qv1, in_qv2 )
quast_value in_qv1, in_qv2;
{
  conditional 	cond1 = NULL, cond2 = NULL;
  Psysteme	ps1 = NULL, ps2 = NULL;

  if ((in_qv1 == quast_value_undefined)&&(in_qv2 == quast_value_undefined)) return true;
  if ((in_qv1 == quast_value_undefined)||(in_qv2 == quast_value_undefined)) return false;
  
  if ( quast_value_quast_leaf_p(in_qv1) && quast_value_quast_leaf_p(in_qv2)) 
    return adg_quast_leaf_equal_p(quast_value_quast_leaf(in_qv1),
				  quast_value_quast_leaf(in_qv2));
  if (quast_value_quast_leaf_p(in_qv1)||quast_value_quast_leaf_p(in_qv2)) return false;
  
  ps1 = predicate_system(conditional_predicate( quast_value_conditional( in_qv1 ) ));
  ps2 = predicate_system(conditional_predicate( quast_value_conditional( in_qv2 ) ));
  if ((adg_suppress_2nd_in_1st_ps(ps1, ps2) != SC_UNDEFINED) || 
      (adg_suppress_2nd_in_1st_ps(ps2, ps1) != SC_UNDEFINED) ) return false;
  return adg_quast_equal_p(conditional_true_quast(cond1),conditional_false_quast(cond2));
}

/*=======================================================================*/
/* bool adg_quast_leaf_equal_p()				AL 1/12/93
 * NOT used, NOT tested.
 */
bool adg_quast_leaf_equal_p( in_ql1, in_ql2 )
quast_leaf in_ql1, in_ql2;
{
  leaf_label	ll1 = NULL, ll2 = NULL;
  
  if ((in_ql1 == quast_leaf_undefined)&&(in_ql2 == quast_leaf_undefined)) return true;
  if ((in_ql1 == quast_leaf_undefined)||(in_ql2 == quast_leaf_undefined)) return false;
  
  ll1 = quast_leaf_leaf_label( in_ql1 );
  ll2 = quast_leaf_leaf_label( in_ql2 );
  if (ll1 == leaf_label_undefined) {
    if (ll2 == leaf_label_undefined) return true;
    else return false;
  }
  else {
    if (ll2 == leaf_label_undefined) return false;
    else return( ((int) leaf_label_statement( ll1 ) ==
		  (int)   leaf_label_statement( ll2 )) &&
		((int) leaf_label_depth( ll1 ) ==
		 (int) leaf_label_depth( ll2 )) );
  }
}


/*=======================================================================*/
/* bool adg_quast_leaves_equal_p( (quast) in_q1, (quast) in_q2 ) 
 * Returns True if the Two input quast have same leaf-labels.
 */
bool adg_quast_leaf_label_equal_p( in_q1, in_q2 )
quast in_q1, in_q2;
{
  quast_value 	qv1 = NULL, qv2 = NULL;
  quast_leaf	ql1 = NULL, ql2 = NULL;
  leaf_label	ll1 = NULL, ll2 = NULL;
	
  debug(9, "adg_quast_leaves_equal_p", "doing\n");
  if ((in_q1 == quast_undefined)||(in_q2 == quast_undefined)) return false;
  
  qv1 = quast_quast_value( in_q1 );
  qv2 = quast_quast_value( in_q2 );
  if ((qv1 == quast_value_undefined)||(qv2 == quast_value_undefined) ||
      quast_value_conditional_p( qv1 )||quast_value_conditional_p( qv2 )) return false;
  
  ql1 = quast_value_quast_leaf( qv1 );
  ql2 = quast_value_quast_leaf( qv2);
  if ((ql1 == quast_leaf_undefined)||(ql2 == quast_leaf_undefined)) return false;
  
  ll1 = quast_leaf_leaf_label( ql1 );
  ll2 = quast_leaf_leaf_label( ql2 );
  if (ll1 == leaf_label_undefined) {
    if (ll2 == leaf_label_undefined) return true;
    else return false;
  }
  else {
    if (ll2 == leaf_label_undefined) return false;
    else return( ((int) leaf_label_statement( ll1 ) == 
		  (int) 	leaf_label_statement( ll2 )) && 
		((int) leaf_label_depth( ll1 ) == 
		 (int) leaf_label_depth( ll2 )) );
  }
}


/*=======================================================================*/
bool adg_quast_leaf_solution_equal_p( in_q1, in_q2 )
quast in_q1, in_q2;
{
  quast_value 	qv1, qv2;
  quast_leaf	ql1 = NULL, ql2 = NULL;
  list		ll1 = NULL, ll2 = NULL;

  debug(9, "adg_quast_leaf_solution_equal_p", "doing\n");
  if ((in_q1 == quast_undefined) && (in_q2 == quast_undefined)) return true;
  if ((in_q1 == quast_undefined)||(in_q2 == quast_undefined))   return false;

  qv1 = quast_quast_value( in_q1 ); 
  qv2 = quast_quast_value( in_q2 );
  if ((qv1 == quast_value_undefined) && (qv2 == quast_value_undefined)) return true;
  if ((qv1 == quast_value_undefined)||(qv2 == quast_value_undefined) ||
    quast_value_conditional_p( qv1 )||quast_value_conditional_p( qv2 )) return false;

  ql1 = quast_value_quast_leaf( qv1 );
  ql2 = quast_value_quast_leaf( qv2);
  if ((ql1 == quast_leaf_undefined) && (ql2 == quast_leaf_undefined)) return true;
  if ((ql1 == quast_leaf_undefined)||(ql2 == quast_leaf_undefined))   return false;

  ll1 = quast_leaf_solution(ql1);
  ll2 = quast_leaf_solution(ql2);
  if (gen_length(ll1) != gen_length( ll2 )) return false;
  for(; !ENDP(ll1); POP(ll1) , POP(ll2)) {
    Pvecteur pv1 = EXPRESSION_PVECTEUR( EXPRESSION(CAR(ll1)) );
    Pvecteur pv2 = EXPRESSION_PVECTEUR( EXPRESSION(CAR(ll2)) );
    if (!vect_equal(pv1, pv2)) return false;
  }
  return true;
}


/*=======================================================================*/
/* Psysteme adg_sc_dup( (Psysteme) in_ps )			AL 09/11/93
 * Input  : A Psysteme in_ps.
 * Output : A duplicated Psysteme. Smooth version of sc_dup :
 * 		We avoid any verificatio on the contraintes.
 * PRIVATE use !
 */
Psysteme adg_sc_dup( in_ps ) 
Psysteme in_ps;
{
  Psysteme cp = SC_UNDEFINED;
  Pcontrainte eq = NULL, eq_cp = NULL;
  
  debug(9, "adg_sc_dup", "begin\n");
  if (!SC_UNDEFINED_P(in_ps)) {
    cp = sc_new();
    
    for (eq = in_ps->egalites; eq != NULL; eq = eq->succ) {
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = vect_dup(contrainte_vecteur(eq));
      sc_add_egalite(cp, eq_cp);
    }

    for(eq=in_ps->inegalites;eq!=NULL;eq=eq->succ) {
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = vect_dup(contrainte_vecteur(eq));
      sc_add_inegalite(cp, eq_cp);
    }

    if(in_ps->dimension==0) {
      cp->dimension = 0;
      cp->base = VECTEUR_UNDEFINED;
    }
    else {
      cp->dimension = in_ps->dimension;
      cp->base = base_reversal(vect_dup(in_ps->base));
    }
  }
  debug(9, "adg_sc_dup", "end\n");
  return(cp);
}

/*=======================================================================*/
/* bool adg_is_textualy_after_p( (statement) in_s1, (statement) in_s2 )
 * Input  : Two statements in_s1 and in_s2.			AL 08/11/93
 * Output : True if in_s1 is before in_s2 in the text program.
 * WARNING: This function compares the statement number of the two 
 *		statements => these numbers should already be ordered.
 */
bool adg_is_textualy_after_p( in_s1, in_s2 )
statement in_s1, in_s2;
{ return (statement_number(in_s1) >= statement_number(in_s2));}


/*=======================================================================*/
/* void adg_sc_update_base( (Psysteme*) in_pps )		AL 05/11/93
 * Input  : A pointer on Psysteme in_pps.
 * Done   : Update base of in_ps.
 * PRIVATE use !
 */
void adg_sc_update_base( in_pps )
Psysteme* in_pps;
{
  Psysteme	in_ps = (Psysteme) *in_pps;

  if ((in_ps != SC_UNDEFINED) && (in_ps != SC_RN)) {
    if ((in_ps->nb_eq == 0) && (in_ps->nb_ineq == 0) ) *in_pps = SC_RN; 
    else { in_ps->base = (Pbase) NULL; sc_creer_base( in_ps ); }
  }
}

/*=======================================================================*/
/* Psysteme adg_suppress_2nd_in_1st_ps( (Psysteme) in_ps1, (Psysteme) in_ps2 )
 *                                                              AL 03/11/93
 * Input  : 2 Psystemes.
 * Output : Psysteme : Scan in_ps1 and remove from it Pcontraintes
 *              in in_ps2. No sharing, No remove input object.
 * PUBLIC use possible.
 */
Psysteme adg_suppress_2nd_in_1st_ps( in_ps1, in_ps2 )
Psysteme in_ps1, in_ps2;
{
  Psysteme        ret_ps = SC_RN;
  Pcontrainte     eq1 = NULL, eq2 = NULL, ineq1 = NULL, ineq2 = NULL;
  
  debug(9, "adg_suppress_2nd_in_1st_ps", "begin\n");
  if ( in_ps1 == SC_RN ) RETURN(9, "adg_suppress_2nd_in_1st_ps", ret_ps);
  if ( in_ps2 == SC_RN ) RETURN(9, "adg_suppress_2nd_in_1st_ps", in_ps1);
  for (eq1 = in_ps1->egalites; eq1 != NULL; eq1 = eq1->succ) {
    bool ok = true;
    for (eq2 = in_ps2->egalites; eq2 != NULL; eq2 = eq2->succ)
      { if (vect_equal(eq1->vecteur, eq2->vecteur)) {ok=false; break; }}
    if (ok) ret_ps = sc_append( ret_ps, 
		       sc_make(contrainte_make(eq1->vecteur),CONTRAINTE_UNDEFINED));
  }
  for (ineq1 = in_ps1->inegalites; ineq1 != NULL; ineq1 = ineq1->succ) {
    bool ok = true;
    for (ineq2 = in_ps2->inegalites; ineq2 != NULL; ineq2 = ineq2->succ) 
      { if (vect_equal(ineq1->vecteur, ineq2->vecteur)) {ok=false; break;} }
    if (ok) ret_ps = sc_append( ret_ps, 
		      sc_make( CONTRAINTE_UNDEFINED, contrainte_make(ineq1->vecteur)));
  }

  debug(9, "adg_suppress_2nd_in_1st_ps", "end\n");
  return ret_ps;
}

/*=======================================================================*/
/* int adg_number_of_same_loops( (list) in_l1, (list) in_l2 )	AL 28/10/93
 * Input  : Two lists of loops in_l1 and in_l2.
 * Output : Number of loops in the two lists that have the same 
 *		statement ordering.
 * PUBLIC use possible.
 */
int adg_number_of_same_loops( in_l1, in_l2 )
list in_l1, in_l2;
{
  int count = 0;
  
  debug(9, "adg_number_of_same_loops", "begin\n");
  for(; !ENDP(in_l1); POP(in_l1)) {
    list ll     = in_l2;
    int  order1 = statement_ordering( loop_body( LOOP(CAR(in_l1)) ) );
    for(; !ENDP(ll); POP(ll)) {
      if (order1 == statement_ordering(loop_body(LOOP(CAR(ll))))) count++;
    }
  }
  debug(9,"adg_number_of_same_loops","number of same loop = %d\n", count);
  return count;
}

/*=======================================================================*/
/* statement adg_number_to_statement( (int) in_nb )		AL 25/10/93
 * Input  : Number of a vertex.
 * Output : A statement associated to this vertex.
 * PRIVATE use !
 */
statement adg_number_to_statement( in_nb )
int in_nb;
{ return ordering_to_statement( adg_number_to_ordering( in_nb ) ); }


/*=======================================================================*/
/* void adg_enrichir( (quast) in_qu, (leaf_label) in_ll )	AL 21/10/93
 * Input  : A quast and a leaf label.
 * Output : Nothing. Just put each leaf_label of quast in_qu equal to in_ll
 *		and update the solution.
 * WARNING ! Using global variable : Gstco_map.
 * PRIVATE use !
 */
void adg_enrichir( in_qu, in_ll )
quast 		in_qu;
leaf_label 	in_ll;
{
  quast_value	qv = NULL;
  
  debug(9, "adg_enrichir", "begin \n");
  if( in_qu == quast_undefined ) {}
  else if((qv = quast_quast_value(in_qu))!= quast_value_undefined) {
    if (quast_value_conditional_p(qv)) {
      conditional cond = quast_value_conditional(qv);
      if(cond != conditional_undefined) {
	adg_enrichir(conditional_true_quast(cond), in_ll);
	adg_enrichir(conditional_false_quast(cond), in_ll);
      }
    }
    if(quast_value_quast_leaf_p(qv)) {
      int		nb, dep, count = 0;
      statement	        stat;
      static_control	sc;
      quast_leaf	ql = NULL;
      list		qs = NIL, prov_l = NIL, ind = NIL;
      
      /* We get the indices of statement linked to in_ll */
      nb   = leaf_label_statement( in_ll );
      dep  = leaf_label_depth( in_ll );
      stat = adg_number_to_statement( nb );
      sc   = (static_control) GET_STATEMENT_MAPPING( Gstco_map, stat );
      ind  = adg_get_loop_indices(static_control_loops(sc));
      
      /* Get the dep first indices and put them in prov_l */
      for(; !ENDP(ind) && (count < dep); POP(ind), count++) {
	expression exp = entity_to_expression( ENTITY(CAR( ind )) );
	ADD_ELEMENT_TO_LIST( prov_l, EXPRESSION, exp );
      }
      
      /* Add to prov_l the solutions of quast */
      ql = quast_value_quast_leaf( qv );
      if (ql != quast_leaf_undefined) qs = quast_leaf_solution( ql );
      for(; !ENDP(qs) ; POP(qs)) {
	expression exp = EXPRESSION(CAR( qs ));
	ADD_ELEMENT_TO_LIST( prov_l, EXPRESSION, exp );
      }
      quast_value_quast_leaf( qv ) = make_quast_leaf( prov_l, in_ll );
    }
  }
  debug(9, "adg_enrichir", "end \n");
}

/*=======================================================================*/
/* predicate predicate_dup( (predicate) in_pred )		AL 18/10/93
 * Input  : A predicate in_pred.
 * Output : A duplicated predicate ret_pred.
 * PUBLIC use possible.
 */
predicate predicate_dup( in_pred )
predicate in_pred;
{
  if ( in_pred != predicate_undefined ) 
    return make_predicate( sc_dup( predicate_system(in_pred) ) );
  else return predicate_undefined;
}

/*=======================================================================*/
/* dfg_vertex_label_dup( (dfg_vertex_label) in_dvl )		AL 18/10/93
 * Input  : A dfg_vertex_label in_dvl
 * Output : A duplicated dfg_vertex_label.
 * WARNING: tag sccflags is always put to sccflags_undefined !!
 * 		Should not be used anymore : copy_dfg_vertex_label exists !
 * PRIVATE use .
 */
dfg_vertex_label dfg_vertex_label_dup( in_dvl )
dfg_vertex_label in_dvl;
{
  dfg_vertex_label ret_dvl = dfg_vertex_label_undefined;
	
  debug(9, "dfg_vertex_label_dup", "begin \n");
  if( in_dvl != dfg_vertex_label_undefined ) {
    ret_dvl = make_dfg_vertex_label( 
			    dfg_vertex_label_statement( in_dvl ),
			    predicate_dup( dfg_vertex_label_exec_domain(in_dvl) ),
			    sccflags_undefined );
  }
  debug(9, "dfg_vertex_label_dup", "end \n");
  return ret_dvl;
}

/*=======================================================================*/
/* bool adg_simple_ineg_p( (Psysteme) in_ps )			AL 22/10/93
 * Input  : A psysteme in_ps
 * Output : true if in_ps has only one inegality in it.
 *		FALSE otherwhise.
 * PUBLIC use.
 */
bool adg_simple_ineg_p( in_ps )
Psysteme in_ps;
{ return (in_ps != SC_RN)?((in_ps->nb_eq == 0) && (in_ps->nb_ineq == 1)):false; }


/*=======================================================================*/
/* quast adg_max_of_leaves( tsou, tsou2, in_i, in_pa, take_last)
 * Compute max of two quasts. in_i is the order to ta  ...
 * PRIVATE use.
 */
quast adg_max_of_leaves( tsou, tsou2, in_i, in_pa, take_last)
quast 	*tsou, tsou2;
int	in_i;
Ppath 	in_pa;
bool take_last;
{
  quast_value	in_qv = NULL, in_qv2 = NULL;
  int 		dep = 0, dep2 = 0;
  int		nb = 0, nb2 = 0, max_depth = 0;
  leaf_label	ll = NULL, ll2 = NULL;
  quast		ret_q = quast_undefined;
  quast_leaf 	ql = NULL, ql2 = NULL;
  statement	st = NULL, st2 = NULL;
  Psysteme	delt_sc = NULL, delt_sc1 = NULL, delt_sc2 = NULL;
  Ppath		new_pa1 = NULL, new_pa2 = NULL, new_pa = NULL;
  Pvecteur	pvec = NULL, pv1 = NULL, pv2 = NULL, diff = NULL;
  bool 	cut_space=false, after2=false, after1=false, tt=false;

  debug(9, "adg_max_of_leaves", "begin\n");

  in_qv  = quast_quast_value( *tsou );
  in_qv2 = quast_quast_value( tsou2 );
  ql     = quast_value_quast_leaf( in_qv );
  ql2    = quast_value_quast_leaf( in_qv2 );
  ll     = quast_leaf_leaf_label(ql);
  ll2    = quast_leaf_leaf_label(ql2);

  if (ll == leaf_label_undefined) 
    {   *tsou = copy_quast(tsou2); RETURN(9, "adg_max_of_leaves", *tsou); }
  if (ll2 == leaf_label_undefined) RETURN(9, "adg_max_of_leaves", copy_quast(*tsou));
  
  dep 	= leaf_label_depth(ll);
  dep2 	= leaf_label_depth(ll2);
  nb 	= leaf_label_statement(ll);
  nb2	= leaf_label_statement(ll2);
  
  if (take_last) {
    if (dep > dep2) RETURN(9, "adg_max_of_leaves", copy_quast(*tsou));
    if (dep2 > dep) {
      quast_quast_value( *tsou ) = copy_quast_value(in_qv2);
      RETURN(9, "adg_max_of_leaves", *tsou);
    }
  }
  else {
    if (dep2 > dep) RETURN(9, "adg_max_of_leaves", copy_quast(*tsou));
    if (dep > dep2) {
      quast_quast_value( *tsou ) = copy_quast_value(in_qv2);
      RETURN(9, "adg_max_of_leaves", *tsou);
    }
  }
		
  /*The two leaves are at the same depth : we have three cases*/
  st  = adg_number_to_statement( nb );
  st2 = adg_number_to_statement( nb2 );
  
  /* If we are in the deepest position : take textual order */
  max_depth = stco_common_loops_of_statements(Gstco_map, st, st2);
  if ((dep == max_depth) || (in_i == max_depth)) {
    tt = (statement_number(st) >= statement_number(st2));
    if (take_last) {
      if (tt) RETURN(9, "adg_max_of_leaves", copy_quast(*tsou));
      *tsou = copy_quast(tsou2);
      RETURN(9, "adg_max_of_leaves", *tsou);
    }
    else {
      if (!tt) RETURN(9, "adg_max_of_leaves", copy_quast(*tsou));
      *tsou = copy_quast(tsou2);
      RETURN(9, "adg_max_of_leaves", *tsou);
    }
  }

  /* Take the in_i element of solutions */
  pv1 = EXPRESSION_PVECTEUR(EXPRESSION(gen_nth(in_i, quast_leaf_solution(ql))));
  pv2 = EXPRESSION_PVECTEUR(EXPRESSION(gen_nth(in_i, quast_leaf_solution( ql2 ))));
  in_i++;

  /*If *tsou and tsou2 distances to the source could be equal*/
  diff      = vect_substract( pv1, pv2 );
  pvec      = diff;
  delt_sc   = sc_make(contrainte_make(pvec), CONTRAINTE_UNDEFINED);
  new_pa    = pa_intersect_system( in_pa, delt_sc );
  cut_space = pa_faisabilite(pa_intersect_system(in_pa, delt_sc));

  /* See if tsou2 could be after *tsou */
  pvec     = vect_add( vect_new(TCST, VALUE_ONE), diff );
  delt_sc2 = sc_make(CONTRAINTE_UNDEFINED, contrainte_make(pvec));
  new_pa2  = pa_intersect_system( in_pa, delt_sc2 );
  after2   = pa_faisabilite( new_pa2 );

  /* See if *tsou could be after tsou2 */
  vect_chg_sgn( diff );
  pvec     = vect_add( vect_new(TCST, VALUE_ONE), diff );
  delt_sc1 = sc_make(CONTRAINTE_UNDEFINED, contrainte_make(pvec));
  new_pa1  = pa_intersect_system( in_pa, delt_sc1 );
  after1   = pa_faisabilite( new_pa1 );
  
  /* Only one of the two is after the other. */
  if (!cut_space) {
    if (take_last) {
      if(after1) ret_q = copy_quast( *tsou );
      else if(after2) ret_q = copy_quast( tsou2 );
      else pips_internal_error("Bad cutting space !");
    }
    else {
      if(after2) ret_q = copy_quast( *tsou );
      else if(after1) ret_q = copy_quast( tsou2 );
      else pips_internal_error("Bad cutting space !");
    }
  }
  /* Both quast could be a source: look at a deeper stage. */
  else if (after1 && after2) {
    quast	ts = NULL, ts2 = NULL;
    
    if (take_last) {
      ts = copy_quast( *tsou );
      ts2 = copy_quast( tsou2 );
    }
    else {
      ts = copy_quast( tsou2 );
      ts2 = copy_quast( *tsou );
    }
    ret_q  = make_quast(make_quast_value(is_quast_value_conditional, 
			   make_conditional(
				make_predicate(delt_sc), 
				adg_max_of_leaves( tsou, tsou2, in_i, new_pa, take_last),
				make_quast( make_quast_value(
				  is_quast_value_conditional,
				    make_conditional( 
					make_predicate( delt_sc2 ), ts2, ts )),
					   NIL )   )), 
			NIL  );
  }
  /* !after1 or !after2 */
  /* only after1 holds */
  else if (after1) ret_q  = take_last?copy_quast( *tsou ):copy_quast(tsou2);
  /* only after2 holds */
  else if (after2) ret_q = take_last?copy_quast( tsou2 ):copy_quast( *tsou );
  /* !after1 and !after2 */
  else 	ret_q = adg_max_of_leaves( tsou, tsou2, in_i, new_pa, take_last);

  debug(9, "adg_max_of_leaves", "end \n");
  return( ret_q );
}

/*=======================================================================*/
/* quast adg_path_max_source( quast *tsou, quast *tsou2, 
 *	     list psl, Ppath in_pa, bool take_last )	        AL 04/08/93
 */
quast adg_path_max_source( tsou, tsou2, in_pa, psl, take_last )
quast	    *tsou, *tsou2;
list	    psl;
Ppath	    in_pa;
boolean	    take_last;
{
  quast	 	ret_tsou = quast_undefined;
  quast_value	in_qv    = NULL, in_qv2 = NULL;
  
  debug(9, "adg_path_max_source", "begin \n");
  /* Trivial cases */
  if(*tsou2 == quast_undefined) RETURN(9,"adg_path_max_source",*tsou);
  in_qv2 = quast_quast_value( *tsou2 );
  if(in_qv2 == quast_value_undefined) RETURN(9,"adg_path_max_source",*tsou);


  if(*tsou == quast_undefined) 
    { *tsou = *tsou2;   RETURN(9,"adg_path_max_source",*tsou); }

  /* Case the quast of *tsou is a conditional */
  in_qv  = quast_quast_value( *tsou );
  if (quast_value_conditional_p(in_qv)) {
    conditional qvcond  = quast_value_conditional( in_qv );
    Psysteme    cond    = predicate_system(conditional_predicate(qvcond));
/*    quast       qt      = conditional_true_quast( qvcond );
    quast       qf      = conditional_false_quast( qvcond );*/
    Ppath       ct 	= pa_intersect_system( in_pa, cond );
    Ppath       cf 	= pa_intersect_complement( in_pa, cond );

    adg_path_max_source(&(conditional_true_quast(qvcond)),tsou2,ct,psl,take_last);
    adg_path_max_source(&(conditional_false_quast(qvcond)),tsou2,cf,psl,take_last);
    
    ret_tsou = *tsou;
  }

  /* Case the quast of *tsou is a leaf and *tsou2 a conditional*/
  else if (quast_value_conditional_p(in_qv2)) {
    Ppath	ct, cf;
    conditional qvcond  = quast_value_conditional( in_qv2 );
    predicate   pred    = conditional_predicate( qvcond );
    Psysteme    cond    = predicate_system( pred ); 
    quast       qt      = conditional_true_quast( qvcond );
    quast       qf      = conditional_false_quast( qvcond );
    quast	qll     = *tsou;
    
    cond->base = NULL; sc_creer_base( cond );
    ct 	       = pa_intersect_system( in_pa, cond );
    cf 	       = pa_intersect_complement( in_pa, cond );

    if (!pa_faisabilite( ct ))    adg_path_max_source(tsou,&qf,in_pa,psl,take_last);
    else if (!pa_faisabilite(cf)) adg_path_max_source(tsou,&qt,in_pa,psl,take_last);
    else {
      quast q3 = copy_quast( qll );
      quast q4 = copy_quast( qll );
      quast q1 = adg_path_max_source(&q3, &qt, ct, psl, take_last);
      quast q2 = adg_path_max_source(&q4, &qf, cf, psl, take_last);
      
      if (adg_quast_leaf_solution_equal_p(q1, q2)) *tsou = q1;
      else *tsou = make_quast( make_quast_value(is_quast_value_conditional,
					       make_conditional(pred,q1, q2)),psl);
    }
  
    ret_tsou = *tsou;
  }

  /* The two quasts are leaves */
  else if (quast_value_quast_leaf_p(in_qv2) && quast_value_quast_leaf_p(in_qv)) {
    int 	dep, dep2, nb, nb2, max_depth;
    quast	q2 = quast_undefined;
    statement	st, st2;
    bool 	tt;
    quast_leaf  ql  = quast_value_quast_leaf( in_qv );
    quast_leaf  ql2 = quast_value_quast_leaf( in_qv2 );
    leaf_label  ll  = quast_leaf_leaf_label(ql);
    leaf_label  ll2 = quast_leaf_leaf_label(ql2);
    
    if (ll == leaf_label_undefined) 
      { *tsou = copy_quast(*tsou2);   RETURN(9, "adg_path_max_source", *tsou2); }
    if (ll2 == leaf_label_undefined)  RETURN(9, "adg_path_max_source", *tsou);
    
    dep   = leaf_label_depth(ll);
    dep2  = leaf_label_depth(ll2);
    nb 	  = leaf_label_statement(ll);
    nb2	  = leaf_label_statement(ll2);
    
    if (take_last) {
      if (dep > dep2) RETURN(9, "adg_path_max_source", *tsou);
      if (dep2 > dep) {
	quast_quast_value( *tsou ) = copy_quast_value(in_qv2);
	RETURN(9, "adg_path_max_source", *tsou2);
      }
    }
    else {
      if (dep2 > dep ) RETURN(9, "adg_path_max_source", *tsou);
      if (dep  > dep2) {
	quast_quast_value( *tsou ) = copy_quast_value(in_qv2);
	RETURN(9, "adg_path_max_source", *tsou2);
      }
    }

    /* The two statement have the same depth dep */
    st  = adg_number_to_statement( nb );
    st2 = adg_number_to_statement( nb2 );

    max_depth = stco_common_loops_of_statements(Gstco_map, st, st2);
    if (dep == max_depth) {
      tt = (statement_number(st) >= statement_number(st2));
      if (take_last) {
	if (tt) RETURN(9, "adg_path_max_source", *tsou);
	*tsou = copy_quast(*tsou2);
	RETURN(9, "adg_path_max_source", *tsou2);
      }
      else {
	if (!tt) RETURN(9, "adg_path_max_source", *tsou);
	*tsou = copy_quast(*tsou2);
	RETURN(9, "adg_path_max_source", *tsou2);
      }
    }
    
    q2 = adg_max_of_leaves( tsou, *tsou2, dep, in_pa, take_last );
    quast_newparms( q2 ) = gen_concatenate(quast_newparms(*tsou),
					   quast_newparms(*tsou2) );
    quast_quast_value( *tsou ) = quast_quast_value( q2 );
    ret_tsou = *tsou;
  }
  else pips_internal_error("Anormal quast ");
  
  debug(9, "adg_path_max_source", "end \n");
  return( ret_tsou );
}


/*=======================================================================*/
/* Pvecteur adg_list_to_vect(list in_list, bool with_tcst)	AL 07/10/93
 * Input 	: A list of entities and a boolean.
 * Output	: A Pvecteur sorted by the in_list order.
 *		  TCST is added at the end if with_tcst is set to TRUE.
 * PUBLIC use.
 */
Pvecteur adg_list_to_vect( in_list, with_tcst )
list	in_list;
bool with_tcst;
{
  Pvecteur pv2 = NULL, *pvec = NULL, prov_vec = VECTEUR_NUL;

  debug(9, "adg_list_to_vect", "begin \n");

  /* Build a Pvecteur according to the reverse order of the in_list */
  pvec = &prov_vec;
  for(; !ENDP(in_list); POP(in_list)) 
    { vect_add_elem( pvec, (Variable) ENTITY(CAR( in_list )), (Value) 1 ); }

  /* Add the TCST var or not */
  if (with_tcst) vect_add_elem( pvec, TCST, VALUE_ONE);
  
  /* Reverse the vecteur to recover the order of the input list */
  pv2 = vect_reversal( *pvec );

  debug(9, "adg_list_to_vect", "end \n");
  return pv2;
}

/*=======================================================================*/
/* Psysteme adg_build_Psysteme( predicate in_pred, list in_list )
 * Input  : A predicate in_pred.				AL 29/07/93
 *	    A list of entities in_list.
 * Output : A Psysteme from in_pred ordered according to the list in_list.
 * PUBLIC use.
 */
Psysteme adg_build_Psysteme( in_pred, in_list )
predicate	in_pred;
list		in_list;
{
  Psysteme	ret_ps = NULL;
  Pvecteur	pv2 = NULL;

  debug(9, "adg_build_Psysteme", "begin \n");
  if ((in_list == NIL) && (in_pred != predicate_undefined)) {
    pips_internal_error("Error : there is no correspondance between input Psysteme and input entity list");
  }

  ret_ps = (Psysteme) predicate_system( in_pred );
  pv2    = adg_list_to_vect( in_list, true );
	
  /* Call a sort function */
  sort_psysteme( ret_ps, pv2 );

  debug(9, "adg_build_Psysteme", "end \n");
  return( ret_ps );
}

/*=======================================================================*/
/* Pposs_source adg_path_possible_source(  quast in_tsou, vertex in_ver, 
 *				int in_dep, Psysteme in_ps )
 * Input  : The present solution source in_tsou			AL 29/07/93
 * 		the vertex in_ver to examine and its depth in_dep.
 *		and the candidate context in_ps.
 *		If take_last is true, we select leaves according to 
 *		the sequential order (the last one wins).
 *		If it is false, the first one wins.
 * Output : Node of quast in_tsou under which we could find a source. 
 * PRIVATE use.
 */
Pposs_source 	adg_path_possible_source(in_tsou, in_ver, in_dep, in_pa, take_last )
quast	 	*in_tsou;
vertex 		in_ver;
int 		in_dep;
Ppath		in_pa;
bool 	take_last;
{
  bool		ret_bool = false;
  quast_value	qu_v = NULL;
  Pposs_source 	ret_psou;
  
  debug(8, "adg_path_possible_source", "begin \n");
  
  ret_psou      = (Pposs_source) malloc(sizeof(Sposs_source));
  ret_psou->pat = in_pa; 
  ret_psou->qua = in_tsou;

  if ( *in_tsou == quast_undefined ) RETURN(8,"adg_path_possible_source 1",ret_psou);
  qu_v = quast_quast_value( *in_tsou );
  if (qu_v == quast_value_undefined ) RETURN(8,"adg_path_possible_source 2",ret_psou);
  
  if (quast_value_conditional_p(qu_v)) {
    Ppath	  pat, paf;
    Pposs_source  pt, pf;
    conditional   qvc    = quast_value_conditional(qu_v);
    quast*        tq     = &(conditional_true_quast( qvc ));
    quast*        fq     = &(conditional_false_quast( qvc ));
    Psysteme      loc_ps = predicate_system( conditional_predicate( qvc ) );
    
    adg_sc_update_base( &loc_ps );
    pat    = pa_intersect_system( in_pa, loc_ps );
    paf    = pa_intersect_complement( in_pa, loc_ps );
    pt     = adg_path_possible_source( tq, in_ver, in_dep, pat, take_last );
    pf     = adg_path_possible_source( fq, in_ver, in_dep, paf, take_last );
    
    if      (pa_empty_p( pt->pat )) ret_psou = pf;
    else if (pa_empty_p( pf->pat )) ret_psou = pt;
  }
  else if (quast_value_quast_leaf_p(qu_v)) {
    leaf_label qu_ll    = quast_leaf_leaf_label(quast_value_quast_leaf(qu_v));
    int        qu_d     = leaf_label_depth( qu_ll ); 
    statement  sou_s    = adg_vertex_to_statement(in_ver); 
    int        sou_nb   = statement_number( sou_s ); 
    statement  qu_s     = adg_number_to_statement( leaf_label_statement(qu_ll) );
    int        qu_order = statement_number( qu_s ); 
    int        nesting  = stco_common_loops_of_statements( Gstco_map,qu_s, sou_s );
    bool       tt       = ( sou_nb <= qu_order );
    
    /* The last wins. General usage. */
    if (take_last) {
      if( qu_d != in_dep ) {
	if( (nesting > qu_d) || (nesting > in_dep) )  {
	  if(qu_d <= in_dep) ret_bool = true; }
	else ret_bool = tt;
      }
      else ret_bool = true;
    }
    /* The first wins. Used to compute summary read effects */
    else {
      if( qu_d != in_dep ) {
	if( (nesting > qu_d) || (nesting > in_dep) )  {
	  if(qu_d <= in_dep) ret_bool = false;
	  else ret_bool = true;
	}
	else ret_bool = !tt;
      }
      else ret_bool = true;
    }
    
    /* This vertex is not a possible source */
    if (!ret_bool)  ret_psou->pat =  pa_empty();
  }
  
  debug(8, "adg_path_possible_source", "end \n");
  return( ret_psou );
}


/*=======================================================================*/
/* list adg_increasing_stat_order_sort( list in_list )		AL 28/07/93
 * Input  : A list of vertices in_list.
 * Output : A list of vertices sorted by decreasing order
 * PRIVATE use.
 */
list adg_increasing_stat_order_sort( in_list )
list in_list;
{ return(gen_nreverse( adg_decreasing_stat_order_sort( in_list ))); }

/*=======================================================================*/
/* list adg_decreasing_stat_order_sort( list in_list )		AL 28/07/93
 * Input  : A list of vertices in_list.
 * Output : A list of vertices sorted by decreasing order
 *		of statement_ordering attached to each vertex ret_list.
 * Method : We scan in_list. For each vertex ver of in_list, we scan ret_list
 *		with l1. l2 points on predecessor of l1.
 *		According to the order of ver compared to the order
 *		of v1 and v2, we insert (or not) ver between l2 and l1.
 * PRIVATE use.
 */
list adg_decreasing_stat_order_sort( in_list )
list in_list;
{
  list 	ret_list = NIL, prov_l = NIL;
  
  debug(9,"adg_decreasing_stat_order_sort", "begin\n");
  for(; !ENDP(in_list); POP(in_list)) {
    bool    cont  = true;
    list    l2    = NIL; 
    vertex  ver   = VERTEX(CAR( in_list ));
    int     order = statement_number(adg_vertex_to_statement( ver ));
    list    l1    = ret_list;
    
    prov_l = NIL;
    ADD_ELEMENT_TO_LIST( prov_l, VERTEX, ver );
    
    while( cont ) {
	int 	order2 = 0;
      
      /* ret_list is presently empty : we initialize it
       * with the new vertex ver	*/
      if ((l1 == NIL) && (l2 == NIL )) {
	ret_list = prov_l;
	cont = false;
      }
      /* We are at the end of ret_list : we add the new
       * vertex at the end of our list */ 
      else if ((l1 == NIL) && (l2 != NIL)) {
	l2->cdr = prov_l;
	cont = false;
      }
      /* Core of the pass : compares new input vertex ver */
      else {
	order2 = statement_number(adg_vertex_to_statement(VERTEX(CAR(l1))));
	if ((order >= order2) && (l2 == NIL)) {
	  ret_list = prov_l;
	  ret_list->cdr = l1;
	  cont = false;
	}
	else if ((order >= order2) && (l2 != NIL)) {
	  l2->cdr = prov_l;
	  prov_l->cdr = l1;
	  cont = false;
	}
	else {
	  cont = true;
	  l2 = l1;
	  POP( l1 );
	}
      }
    }
  }
  debug(9,"adg_decreasing_stat_order_sort",  "returned list length : %d \n",
	gen_length( ret_list ));
  return( ret_list );
}
	
	 
/*=======================================================================*/
/* list adg_merge_entities_lists( list l1, list l2 )		AL 23/07/93
 * Input  : Two lists of entities.
 * Output : A list of entities, union of the l1 and l2.
 *		Append the new entities of l2 after list l1.
 * PUBLIC use.
 */
list adg_merge_entities_lists( l1, l2 )
list l1, l2;
{
  list 	ret_list = NIL;

  debug(9, "adg_merge_entities_list", "begin \n");
  for(; !ENDP(l1); POP(l1) ) 
    { ADD_ELEMENT_TO_LIST( ret_list, ENTITY,  ENTITY(CAR( l1 )) );}
  for(; !ENDP(l2); POP( l2 )) {
    entity ent = ENTITY(CAR(l2));
    if ( gen_find_eq(ent, ret_list) == chunk_undefined )
      ADD_ELEMENT_TO_LIST( ret_list, ENTITY, ent );
  }
  
  debug(9, "adg_merge_entities_list", "end \n");
  return( ret_list );
}
		
		

/*=======================================================================*/
/* list adg_rename_entities( list le, hash_table fst)	AL 22/07/93
 * Input  : A list of entities le and a hash table fst.
 *
 * Output : A list of entities with names RE1 RE2 RE3 ...
 * 	according to a global counter Gcount_re. If we already have
 * 	created enough amount of new entities, we reuse them; else,
 *	we create new ones. Gcount_re is the total number of such entities.
 *	The corresponding changes are kept in the hash_table
 *	Gforward_substitute_table ("fst") given in argument.
 * PRIVATE use.
 */
list adg_rename_entities(le, fst)
list le;
hash_table fst;
{ 
  list ret_l   = NIL;
  int  counter = 0;

  debug(9, "adg_rename_entities", "begin \n");
  for(;!ENDP(le); POP(le)) {
    extern  int Gcount_re;
    entity  in_ent = NULL, new_ent = NULL, mod_ent = NULL;
    char    *name  = NULL, *name2 = NULL, *num = NULL;
    
    counter++;
    num     = i2a(counter);
    mod_ent = get_current_module_entity(); 
    name    = strdup(concatenate("RE", num, (char *) NULL));
    free(num);

    /* If a Renamed Entity already exists, we use it ;
     * else, we make a new one and increment Gcount_re.
     */
    if ( counter <= Gcount_re ) {
      new_ent = FindOrCreateEntity( 
			 strdup(concatenate(ADFG_MODULE_NAME, 
					    entity_local_name(mod_ent),
					    (char*) NULL)),
			 name );
    }
    else {
      Gcount_re++;
      name2 = strdup(concatenate( ADFG_MODULE_NAME, 
				 entity_local_name(mod_ent), 
				 MODULE_SEP_STRING, name, NULL ));
      new_ent = make_entity(name2,
			    make_type(is_type_variable,
		      make_variable(make_basic(is_basic_int,UUINT(4)),NIL)),
			    make_storage(is_storage_ram, ram_undefined),
			    make_value(is_value_unknown, UU));
    }

    ADD_ELEMENT_TO_LIST(ret_l, ENTITY, new_ent);
    
    in_ent = ENTITY(CAR(le));
    hash_put(fst, (char*) in_ent, (char*) entity_to_expression(new_ent));
    if (get_debug_level() > 7) {
      fprintf(stderr, "Old %s -> New %s\n",
	      entity_local_name( in_ent ),
	      entity_local_name( new_ent ));
    }
  }
  debug(9, "adg_rename_entities", "end \n");
  return( ret_l );
}
 
/*=======================================================================*/
/* list adg_get_loop_indices( list ll )				AL 22/07/93
 * Input  : A list of loops ll.
 * Output : A list of entities representing indices of loops in ll.
 * PUBLIC use.
 */
list adg_get_loop_indices( ll )
list ll;
{
  list ret_list = NIL;
  for(; !ENDP(ll); POP(ll)) 
    { ADD_ELEMENT_TO_LIST( ret_list, ENTITY, loop_index(LOOP(CAR( ll ))) ); }
  return( ret_list );
}
/*=======================================================================*/
