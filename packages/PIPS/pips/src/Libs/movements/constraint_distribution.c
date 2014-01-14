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
/*
 * PACKAGE MOVEMENTS
 *
 * Corinne Ancourt  - juin 1990
 */

#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"



/* Distribution of the constraints of the system ps in three systems.
 * System ps contains only contraints having to be used to generate 
 * bound expressions for the index of rank "rank".
 * sc_neg contains the constraints corresponding to lower bounds for 
 * the variable of rank "rank".
 * sc_pos contains the constraints corresponding to upper bounds for 
 * the variable of rank "rank".
 * sc_test contains the constraints having to be added in the guard 
 *
*/

void 
bound_distribution(pps,index_base,sc_info,nb_loop,sc_neg,sc_pos,sc_test)
Psysteme *pps;
Pbase index_base;
int sc_info[][4];
int nb_loop;
Psysteme *sc_neg,*sc_pos,sc_test;
{
    Pcontrainte ineq;
    Variable var;
    int higher_rank,rank;

    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"bound_distribution","begin\n");


    for (rank = 1; rank<=nb_loop; rank++) {
	var = variable_of_rank(index_base,rank);

	sc_neg[rank] = sc_init_with_sc(pps[rank]);
	sc_pos[rank] = sc_init_with_sc(pps[rank]);
	
	for (ineq = pps[rank]->inegalites; 
	     !CONTRAINTE_UNDEFINED_P(ineq); 
	     ineq=ineq->succ) {
	    if ((higher_rank = search_higher_rank(ineq->vecteur,
						  index_base))>0) {
		if (sc_info[rank][1]) {
		    /* This condition is true if the variable must be kept 
		       like loop index. Then all the constraints are kept in
		       the system (and not in the system of guards)  */
		    if (higher_rank<= rank) {
			/* This condition is true when the constraint constrains 
			   directly the variable */

			if (vect_coeff(var,ineq->vecteur) < 0) 
			    insert_ineq_begin_sc(sc_neg[rank],ineq);
			else 
			    insert_ineq_begin_sc(sc_pos[rank],ineq);
		    }
		    else {		
			/* This condition is true when the variable of rank 
			   "higher_rank" could be eliminated from the 
			   two constraints ineq and ineq->succ in order 
			   to obtain a new constraint on the variable "var" */
			int left_rank,right_rank;
			Value left_coeff,right_coeff;
			Variable left_var, right_var;
		
			constraint_integer_combination
			    (index_base,ineq,ineq->succ,higher_rank,
			     &right_var, &right_rank, &right_coeff,
			     &left_var, &left_rank, &left_coeff);
	
			/* the two constraints ineq and ineq->succ are added to 
			   the end of the system */
			if (((right_rank>left_rank) && (right_coeff >0)) 
			    || ((right_rank < left_rank) && (left_coeff>0))) 
			    insert_2ineq_end_sc(sc_pos[rank],ineq);
			else
			    insert_2ineq_end_sc(sc_neg[rank],ineq);
			ineq = ineq->succ;
		    }
		}
		else {
		    /* If some constraints are in the system constraining a 
		       variable which must not be kept as loop index, these 
		       constraints are put in guards */
		    insert_2ineq_end_sc(sc_test,ineq);
		    ineq = ineq->succ;
		}
	    }  
	}
	ifdebug(9) {
	    (void) fprintf(stderr,"systeme negatif \n");
	    sc_fprint(stderr,sc_neg[rank],(get_variable_name_t)entity_local_name);
	    (void) fprintf(stderr,"systeme positif \n");
	    sc_fprint(stderr,sc_pos[rank],(get_variable_name_t)entity_local_name);
	}
    }
    debug(8,"bound_distribution","end\n");
    debug_off();
}





/* Distribution of the constraints of the system sc into several systems.
 * System sc contains all the  contraints having to be used to generate 
 * bound expressions for all loop indices.
 *
 * A new system is defined for each system variable.
 * Each new system defines the set of constraints needed for the generation 
 * of loop bound expressions for one variable. 
 *
 * The constraint constraining directly a variable of rank "rank_hr"
 * is added to the system corresponding to the variable "var_hr"
 * except if this variable must not be kept (var_hr is not a loop index).
 * In this last case, the constraint must be combined with another constraint 
 * in order to eliminate the variable "var_hr" and to deduce a constraint 
 * constrainnig another variable of rank "rank".
 * In that case, the two constraints are added to the  system corresponding 
 * to the variable "var = variable_of_rank(rank)"
 *
*/

void constraint_distribution(
  Psysteme sc,
  Psysteme *bound_systems,
  Pbase index_base,
  int sc_info[][4])
{
    Pcontrainte pc1,pc2;
    int rank,rank_hr,rank_pc1,rank_pc2;
    Value coeff1,coeff2;
    int sign1,sign2,i;
    Variable var_hr;
    Psysteme sc2 = sc_init_with_sc(sc);

    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"constraint_distribution","begin\n");


    for (pc1=sc->inegalites; !CONTRAINTE_UNDEFINED_P(pc1); pc1 = pc1->succ) {
	if ((rank_hr = search_higher_rank(pc1->vecteur,index_base))>0) {
	    if (sc_info[rank_hr][1]) {
		/* This condition is true if the variable must be kept like 
		   loop index. All the constraints constraining directly the 
		   variable of rank "rank_hr" are kept in the system 
		   bound_systems[rank_hr] */

		insert_ineq_end_sc(bound_systems[rank_hr],pc1);
		insert_ineq_end_sc(sc2,pc1);
	    }
	    else {
		var_hr = variable_of_rank(index_base,rank_hr);
		rank_pc1 = rank_of_variable
		    (index_base,
		     search_var_of_higher_rank(pc1->vecteur,
					       index_base,
					       var_hr));
		
		coeff1 = vect_coeff(var_hr,pc1->vecteur);
		sign1 = value_sign(coeff1);
		for (pc2 = pc1; !CONTRAINTE_UNDEFINED_P(pc2); 
		     pc2= pc2->succ) {
		    coeff2 = vect_coeff(var_hr,pc2->vecteur);
		    sign2 = value_sign(coeff2);	
		    if (value_notzero_p(coeff2) && sign1 == -sign2 
			&& !bound_redund_with_sc_p(sc2,pc1,pc2,var_hr)) {
			/* this condition is true if the combination of the 
			   two constraints pc1 and pc2 is not redundant for the  
			   system. Then the two constraints are added to the 
			   system of the variable of higher rank */

			Variable var2 = search_var_of_higher_rank(pc2->vecteur,
								  index_base,
								  var_hr);
			rank_pc2 = rank_of_variable(index_base,var2);

			if (rank_pc1 !=rank_pc2)
			    rank=(rank_pc1 >rank_pc2) ? rank_pc1:rank_pc2;
			else rank = rank_hr;
			insert_ineq_end_sc(bound_systems[rank],pc1);
			insert_ineq_end_sc(bound_systems[rank],pc2);
			insert_ineq_end_sc(sc2,pc1);
			insert_ineq_end_sc(sc2,pc2);
		    }	
		}
	    }
	}
    }
    ifdebug(8) {
	for (i=1;i<=vect_size(index_base);i++) {
	    (void) fprintf(stderr,"Le systeme sur la var. %d est:\n",i);
	    sc_fprint(stderr, bound_systems[i], (get_variable_name_t) entity_local_name);
	}
    }

    debug(8,"constraint_distribution","end\n");
    debug_off();
}

void egalite_distribution(
  Psysteme sc,
  Psysteme *bound_systems,
  Pbase index_base)
{
    Pcontrainte pc1;
    int rank;

    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"egalite_distribution","begin\n");

    for (pc1=sc->egalites; !CONTRAINTE_UNDEFINED_P(pc1); pc1 = pc1->succ) {
	if ((rank = search_higher_rank(pc1->vecteur,index_base))>0) {
	    sc_add_eg(bound_systems[rank],pc1);
	}
    }
    debug(8,"egalite_distribution","end\n");
    debug_off();
}

