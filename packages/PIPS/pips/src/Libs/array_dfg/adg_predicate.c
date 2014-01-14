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
 * Documents:   "Le Calcul de l'Array Data Flow Graph dans PIPS
 *              Partie II : Implantation dans PIPS" A. LESERVOT
 *              "Dataflow Analysis of Array and Scalar References" P. FEAUTRIER
 * Comments :
 */

#define GRAPH_IS_DG
#include "local.h"

/* External variables */
extern	int		Gcount_re;

/*=======================================================================*/
/*			PREDICATE FUNCTIONS				 */
/*=======================================================================*/

/*=======================================================================*/
/* predicate adg_get_predicate_of_loops( list loops )		AL 15/07/93
 * Input  : a list of loops.
 * Output : a predicate which represents a conjonction of all 
 *	    conditions on each loop's indexes.
 * COMMON use.
 */
predicate adg_get_predicate_of_loops( loops )
list loops;
{
  predicate	ret_pred = predicate_undefined;
  Psysteme	new_sc = NULL;
  Pvecteur	pv = NULL;

  debug(9, "adg_get_predicate_of_loops", "begin\n");
  new_sc = sc_new();
  for(; !ENDP( loops ); POP( loops )) {
    loop 		l = NULL;
    range		ran = NULL;
    expression	low = NULL, upp = NULL;
    Pvecteur	pvi = NULL;

    l = LOOP(CAR( loops ));
    pvi = vect_new( (Variable) loop_index( l ), VALUE_ONE);
    ran = loop_range( l );
    low = range_lower( ran );
    upp = range_upper( ran );
    pv = vect_substract( EXPRESSION_PVECTEUR(low), pvi );
    sc_add_inegalite( new_sc, contrainte_make(pv) );
    pv = vect_substract( pvi, EXPRESSION_PVECTEUR(upp) );
    sc_add_inegalite( new_sc, contrainte_make(pv) );
  }
    
  if ((new_sc->nb_ineq != 0) || (new_sc->nb_eq != 0)) {
    new_sc->base = NULL;
    sc_creer_base(new_sc);
    ret_pred = make_predicate(new_sc);
  }
    
  if ((get_debug_level() >= 9) && (ret_pred != predicate_undefined))
    adg_fprint_psysteme(stderr, (Psysteme) predicate_system(ret_pred) );
  debug(9, "adg_get_predicate_of_loops", "end\n");
  return( ret_pred );
}


/*=======================================================================*/
/* list adg_predicate_list_dup( (list) ps_l )			AL 13/07/93
 * Duplicates a list of Predicates.
 * Was used to simulate union. dj_ functions should be used now.
 */
list adg_predicate_list_dup( ps_l )
list ps_l;
{
  list ret_list = NIL;

  debug(9, "adg_predicate_list_dup", "begin \n");
  for(; !ENDP( ps_l ); POP( ps_l )) {
    Psysteme sc = NULL, new_sc = NULL;
    
    sc = (Psysteme) predicate_system( PREDICATE(CAR( ps_l )) );
    new_sc = sc_dup( sc );
    ADD_ELEMENT_TO_LIST(ret_list,PREDICATE,make_predicate(new_sc));
  }
	
  debug(9, "adg_predicate_list_dup", "end \n");
  return( ret_list );
}
		
/*=======================================================================*/
/* list adg_make_disjunctions( (list) ps_l1, (list) ps_l2 )	AL 13/07/93
 * Takes in two list of Predicate and returns a list of Predicates.
 * Was used to simulate union. dj_ functions should be used now.
 */
list adg_make_disjunctions( ps_l1, ps_l2 )
list ps_l1, ps_l2;
{
  list 	ps_list = NIL, psl1 = NULL, psl2 = NULL, l1 = NULL, l2 = NULL;

  debug(9, "adg_make_disjunctions", "begin \n");
  psl1 = adg_predicate_list_dup( ps_l1 );
  psl2 = adg_predicate_list_dup( ps_l2 );
  if (psl1 == NIL) RETURN(9, "adg_make_disjunctions", psl2);
  if (psl2 == NIL) RETURN(9, "adg_make_disjunctions", psl1);

  for( l1 = psl1; !ENDP(l1); POP( l1 )) {
    Psysteme p1 = NULL;
    
    p1 = (Psysteme) predicate_system(PREDICATE(CAR( l1 )));
    for (l2 = psl2; !ENDP(l2); POP( l2 )) {
      Psysteme 	p2 = NULL;
      predicate	pred = NULL;
      
      p2 = (Psysteme) predicate_system(PREDICATE(CAR( l2 )));
      pred = make_predicate( sc_append( sc_dup(p2), p1 ) );
      ADD_ELEMENT_TO_LIST( ps_list, PREDICATE, pred );
    }
  }

  debug(9, "adg_make_disjunctions", "end \n");
  return( ps_list );
}

/*=======================================================================*/
/* list adg_get_conjonctions( (expression) ndf_exp )		AL 13/07/93
 * Returns a list of Predicates made of conjonctions, 
 * from an ndf expression. 
 * Was used to simulate union. dj_ functions should be used now.
 */
list adg_get_conjonctions( ndf_exp )
expression ndf_exp;
{
  syntax  syn = NULL;
  list	ps_list = NIL, psl1 = NIL, psl2 = NIL;
  list 	args = NIL;
  call 	c = NULL;

  debug(9, "adg_get_conjonctions", "exp : %s\n",
	words_to_string(words_expression( ndf_exp )) );
  syn = expression_syntax( ndf_exp );
  if (!syntax_call_p( syn )) RETURN(9, "adg_get_conjonctions", ps_list );

  c = syntax_call( syn );
  args = call_arguments( c );
  if (ENTITY_OR_P(call_function( c ))) {
    psl1 = adg_get_conjonctions(EXPRESSION(CAR( args )));
    psl2 = adg_get_conjonctions(EXPRESSION(CAR(CDR( args ))));
    ps_list = gen_nconc( ps_list, psl1 );
    ps_list = gen_nconc( ps_list, psl2 );
  }
  else if (ENTITY_AND_P( call_function(c) )) {
    Psysteme	ps1, ps2;
    /* We are in a conjonction form case */
    psl1 = adg_get_conjonctions( EXPRESSION(CAR( args )) );
    ps1 = (Psysteme) predicate_system(PREDICATE(CAR( psl1 )));
    psl2 = adg_get_conjonctions( EXPRESSION(CAR(CDR( args ))) );
    ps2 = (Psysteme) predicate_system(PREDICATE(CAR( psl2 )));
    ps1 = sc_append( ps1, ps2 );
    ADD_ELEMENT_TO_LIST( ps_list, PREDICATE, make_predicate(ps1));
  }
  else if (ENTITY_GREATER_OR_EQUAL_P( call_function(c) ))  {
      normalized nexp;
      expression e;
      Psysteme	new_sc = NULL;
    
    new_sc = sc_new();
    e = EXPRESSION(CAR( args ));
    nexp = NORMALIZE_EXPRESSION(e);
    if(expression_to_int(EXPRESSION(CAR(CDR( args )))) != 0)
      RETURN(9, "adg_get_conjonctions", ps_list);
    
    if( normalized_linear_p(nexp) ) {
      Pvecteur new_vec = (Pvecteur) normalized_linear(nexp);
      vect_chg_sgn( new_vec );
      sc_add_inegalite(new_sc, contrainte_make(new_vec));
    }
    else {
      fprintf(stderr, "\nNon linear expression :");
      fprintf(stderr, " %s\n", words_to_string(words_expression(ndf_exp)));
    }
    sc_creer_base( new_sc );
    ADD_ELEMENT_TO_LIST(ps_list,PREDICATE,make_predicate(new_sc));
  }
  else
    pips_internal_error("Expression : %s is not in a normal disjunctive form !", 
	       words_to_string(words_expression( ndf_exp )) );
  
  debug(9, "adg_get_conjonctions", "end \n");
  return( ps_list );
}

/*=======================================================================*/
/* list adg_get_disjunctions( (list) exp_l )			AL 13/07/93
 * exp_l is a list of normal disjunctive form expressions.
 * It returns a list of Predicates. These Predicates, put together
 * in a disjunctive form have the same bool value as a system
 * of incoming expressions.
 */
list adg_get_disjunctions( exp_l )
list exp_l;
{
  list	ps_list = NIL;
  
  debug( 7, "adg_get_disjunctions", "begin \n" );
  for(; !ENDP( exp_l ); POP( exp_l )) {
    expression	exp = NULL;
    list 	ps_l = NIL;
    
    exp = EXPRESSION(CAR( exp_l ));
    ps_l = adg_get_conjonctions( exp );
    ps_list = adg_make_disjunctions( ps_list, ps_l );
  }

  if (get_debug_level()>8) adg_fprint_predicate_list( stderr, ps_list );
  debug( 7, "adg_get_disjunctions", "end \n" );
  return( ps_list );
}
	

/*=========================================================================*/
/* this is Alexis's function adg_expressions_to_predicate (see mapping) */
/* predicate adg_expressions_to_predicate(list exp_l): returns the predicate
 * that has the inequalities given as expressions in "exp_l". For example:
 * if A is an expresion of "exp_l" then we'll have the inequality A <= 0 in the
 * predicate.
 *
 * If an expression is not linear, we warn the user.
 *
 * Note: if "exp_l" is empty then we return an undefined predicate.
 */
predicate my_adg_expressions_to_predicate(exp_l)
list exp_l;
{
  predicate new_pred = NULL;
  Psysteme new_sc = NULL;

  debug(9, "my_adg_expressions_to_predicate", "begin \n");
  if(exp_l == NIL) RETURN(9, "my_adg_expressions_to_predicate",predicate_undefined );

  new_sc = sc_new();
  for(; !ENDP(exp_l) ; POP( exp_l) ) {
    expression exp = EXPRESSION(CAR( exp_l ));
    normalized nexp = NULL;
    
    unnormalize_expression( exp );
    nexp = NORMALIZE_EXPRESSION(exp);
    if( normalized_linear_p(nexp) ) {
      Pvecteur new_vec = (Pvecteur) normalized_linear(nexp);
      sc_add_inegalite(new_sc, contrainte_make(new_vec));
    }
    else {
      fprintf(stderr, "\nNon linear expression :");
      fprintf(stderr, " %s\n", words_to_string(words_expression(exp)));
    }
  }

  sc_creer_base(new_sc);
  new_pred = make_predicate(new_sc);
  
  if ((get_debug_level() >= 9) && (new_pred != predicate_undefined))
    adg_fprint_psysteme(stderr, predicate_system(new_pred) );
  debug(9, "my_adg_expressions_to_predicate", "end \n");
  return(new_pred);
}
/*=======================================================================*/







