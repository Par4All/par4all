/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

extern Variable variable_of_rank();
extern Variable search_var_of_higher_rank();

/* This  function returns true:
 *  if all positive OR all negative coefficients of the variable 
 *  var in the system are equal to 1.
 * That's mean that the FM projection can be used without problem on 
 * integer domain
*/
bool var_with_unity_coeff_p(sc, var)
Psysteme sc;
Variable var;
{
    register Pcontrainte pc;

    if (!var_in_sc_p(sc,var))
	return false;

    for (pc = sc->inegalites; !CONTRAINTE_UNDEFINED_P(pc); pc = pc->succ) 
    {
	Value coeff = vect_coeff(var,pc->vecteur);
	if (value_gt(coeff, VALUE_ONE) ||
	    value_lt(coeff, VALUE_MONE))
	    return false;
    }
    for (pc = sc->egalites; !CONTRAINTE_UNDEFINED_P(pc); pc = pc->succ) 
    {
	Value coeff = vect_coeff(var,pc->vecteur);
	if (value_gt(coeff, VALUE_ONE) ||
	    value_lt(coeff, VALUE_MONE))
	    return false;
    }

    return true;
}

/* This function gives information about the variables and the constraints of
 * the system. These informations are stored in the array sc_info.
 *
 * the first information: if the variable must be kept as loop index
 * then sc_info[rank of the variable][1] is greater than 1.
 *
 * the second information:  sc_info[rank of the variable][2] is the number
 * of constraints constraining the variable as upper bound.
 *
 * ths third information: sc_info[rank of the variable][3] is the number
 * of constraints constraining the variable as lower bound.
 *
*/

void sc_integer_projection_information(
  Psysteme sc,
  Pbase index_base,
  int sc_info[][4],
  int dim_h,
  int n)
{

    Pcontrainte ineq,pc;
    Variable var_hr,hvr1,hvr2,right_var,left_var;
    int rank_hr,right_rank,left_rank;
    int sign1,sign2;
    Value coeff1,coeff2,right_coeff,left_coeff;
    bool find_one = false;
    register int i;
    register int j;

    /* Initialisation of the array sc_info */
    // ??? should it be from 0 to n-1
    for (i=1; i<=n; i++)
      for (j=2; j<=3; j++)
        sc_info[i][j]=0;

    for (i=1; i<=dim_h; i++)
      sc_info[i][1]=1;

    for (i=dim_h+1; i<=n; i++)
      sc_info[i][1]=0;

    /* Computation of variables that must be kept as loop indices. */

    for (ineq = sc->inegalites;
	 !CONTRAINTE_UNDEFINED_P(ineq); ineq=ineq->succ) {

	find_one=false;
	if ((rank_hr= search_higher_rank(ineq->vecteur,index_base)) >dim_h) {

	    /* if the variable rank is greater than n, the constraints may be 
	       eliminated if they do not appear in constraints of higher 
	       rank and  having to be keept  */

	    var_hr=variable_of_rank(index_base,rank_hr);
	    coeff1 = vect_coeff(var_hr,ineq->vecteur);
	    sign1 = value_sign(coeff1);

	    for (pc = ineq;
		 !CONTRAINTE_UNDEFINED_P(pc) && !find_one;
		 pc = pc->succ) {

		coeff2 = vect_coeff(var_hr,pc->vecteur);
		sign2 = value_sign(coeff2);	
		if (value_notzero_p(coeff2) && sign1 == -sign2) {
		    hvr1 =search_var_of_higher_rank(ineq->vecteur,
						    index_base,var_hr);
		    hvr2 =search_var_of_higher_rank(pc->vecteur,
						    index_base,var_hr);
		    if ((hvr1 ==hvr2) && 
			(rank_of_variable(index_base,hvr1) >dim_h)) {
			sc_info[rank_of_variable(index_base,hvr1)][1] ++;
			find_one = true;
		    }
		}
	    }
	}
    }


    /* Computation of the number of constraints contraining a variable
       either as lower bound or upper bound */

    for (ineq = sc->inegalites;
	 !CONTRAINTE_UNDEFINED_P(ineq); ineq=ineq->succ) {
	if ((rank_hr= search_higher_rank(ineq->vecteur,index_base))>0) {
	    var_hr=variable_of_rank(index_base,rank_hr);
	    coeff1 = vect_coeff(var_hr,ineq->vecteur);	
	
	    if (rank_hr >dim_h) {
		if (sc_info[rank_hr][1]) {

		    /* If the variable is a loop index then the constraint 
		       ineq gives an upper or a lower bound directly */

		    if (value_pos_p(coeff1)) sc_info[rank_hr][2] ++;
		    else sc_info[rank_hr][3] ++;
		}
		else {

		    /* If the variable is not a loop index then the constraint 
		       ineq combined with another constraint pc gives an upper
		       or  a lower bound for another variable  */

		    for (pc = ineq;
			 !CONTRAINTE_UNDEFINED_P(pc);
			 pc = pc->succ) {
	
			coeff2 = vect_coeff(var_hr,pc->vecteur);
			sign2 = value_sign(coeff2);
			sign1 = value_sign(coeff1);	
		    
			if (value_notzero_p(coeff2) && sign1 == -sign2) {
			    constraint_integer_combination
				(index_base, ineq, pc, rank_hr,
				 &right_var, &right_rank, &right_coeff,
				 &left_var, &left_rank, &left_coeff);
			    if (right_rank>left_rank) { 
				if  (value_pos_p(right_coeff))
				    sc_info[right_rank][2]++;
				else
				    sc_info[right_rank][3]++;
			    }
			    else   if (right_rank<left_rank){
				if (value_pos_p(left_coeff))
				    sc_info[left_rank][2]++;   
				else
				    sc_info[left_rank][3]++;
			    }
			}   
		    }
		}
	    }
	    else 
		/* If the variable is a loop index then the constraint 
		   ineq gives an upper or a lower bound directly */

		if (value_neg_p(vect_coeff(var_hr,ineq->vecteur)))
		    sc_info[rank_hr][3] ++;
		else 
		    sc_info[rank_hr][2] ++;

	}
    }
}


/* This function computes the coefficients of the constraint resulting 
 * from the elimination of the variable of rank "rank" from the 2 
 * inequalities ineq1 and ineq2.
 * Assuming that:
 * ineq1 is the constraint      a0 X2 + + E0 + b0 <= 0
 * ineq2 is the constraint      - a1 X2 + E1 + b1 <=0
 *                where E1 contains X1, E0 contains X0 and a0,a1 >0
 * then the result of the fonction will be:
 *   right_var = X0, right_rank=rank(X0), right_ceoff=ceofficient of X0 in E0
 *   left_var = X1, left_rank=rank(X10, left_coeff=coefficient of X1 in E1
*/

void 
constraint_integer_combination(
  Pbase index_base,
  Pcontrainte ineq1,
  Pcontrainte ineq2,
  int rank,
  Variable *right_var, /* RIGHT */
  int *right_rank,
  Value *right_coeff,
  Variable *left_var,  /* LEFT */
  int *left_rank,
  Value *left_coeff)
{


    Pcontrainte right_ineg= CONTRAINTE_UNDEFINED; 
    Pcontrainte left_ineg = CONTRAINTE_UNDEFINED;
    Variable el_var= variable_of_rank(index_base,rank);

    if (value_pos_p(vect_coeff(el_var,ineq1->vecteur))) {
	right_ineg = ineq1;
	left_ineg = ineq2;			
    }
    else { 
	if (value_pos_p(vect_coeff(el_var,ineq2->vecteur))) {
	    right_ineg = ineq2;
	    left_ineg = ineq1;
	} 
    }
    *right_var = search_var_of_higher_rank(right_ineg->vecteur,
					   index_base,
					   el_var);
    *left_var = search_var_of_higher_rank(left_ineg->vecteur,
					  index_base,
					  el_var);
    *right_rank = rank_of_variable(index_base,*right_var); 
    *left_rank = rank_of_variable(index_base,*left_var); 
    *right_coeff = vect_coeff(*right_var,right_ineg->vecteur);
    *left_coeff = vect_coeff(*left_var,left_ineg->vecteur);

}
