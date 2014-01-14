/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* This function returns true if the inequation ineq is redundant for the
 * system ps and false otherwise.
 * sc and ineq are not modified by the function.
 *
 * Inequality ineq may not be redundant wrt ps for rational numbers and
 * nevertheless true is returned if it is redundant wrt integer points.
 *
 * A bug found here: if we have input sc==NULL with ineq=NULL, then we'll have to test the satisfiability
 * of a sc that has a base null and an ineq (core dump): because in sc_add_inegalite, we test only if the 
 * pointers (sc,ineg) are null.
 * The encountered case is that the pointer not null but his all elements are null
 * or except only one element. This means the constraint is not valid.
 *
 * DN 2/1/2003
 * The same with eq_redund_with_sc_p 
 * Modifs:
 * - change _dup to _copy
 * - correct the bug
 */
 
bool ineq_redund_with_sc_p(sc, ineq)
Psysteme sc;
Pcontrainte ineq;
{
  Psysteme ps;
  Pcontrainte ineg;
  bool result = false;  
 
  if (CONTRAINTE_NULLE_P(ineq)) {
    /*nothing to test: 0==0 is intrinsically redundant */
    return true;
  }

  ps = sc_copy(sc);
  ineg = contrainte_copy(ineq);
  contrainte_reverse(ineg);
  sc_add_inegalite(ps,ineg);

  base_rm(sc_base(ps));
  sc_base(ps) = BASE_NULLE;
  sc_creer_base(ps);

  /* test de sc_faisabilite avec la nouvelle inegalite      */
  if (!sc_rational_feasibility_ofl_ctrl(ps,OFL_CTRL,true))
    result = true;
  sc_rm(ps);
  return(result);
}


/* bool eq_redund_with_sc_p(sc, eq)
 * Psysteme sc;
 * Pcontrainte eq;
 *
 *     IN: sc, eq
 *    OUT: returned boolean
 *
 * true if eq is redundant with sc
 * (c) FC 16/05/94
 */
bool eq_redund_with_sc_p(sc, eq)
Psysteme sc;
Pcontrainte eq;
{
    if (!ineq_redund_with_sc_p(sc, eq)) /* eq considered as an inequality */
	return(false);
    else
    {
	Pcontrainte
	    c = contrainte_copy(eq);
	bool
	    res = ineq_redund_with_sc_p(sc, (contrainte_chg_sgn(c), c));
	contrainte_free(c);
	return(res);
    }
}


/* Psysteme extract_nredund_subsystem(s1, s2)
 * Psysteme s1, s2;
 *
 *       IN: s1, s2
 *      OUT: returned Psysteme
 *
 * returns the constraints of s1 that are not redundant with s2
 *
 * (c) FC 16/05/94
 */
Psysteme extract_nredund_subsystem(s1, s2)
Psysteme s1, s2;
{
    Psysteme
	new = SC_UNDEFINED;
    Pcontrainte
	c = CONTRAINTE_UNDEFINED,
	eq = CONTRAINTE_UNDEFINED,
	in = CONTRAINTE_UNDEFINED,
	cnew = CONTRAINTE_UNDEFINED; /* temporary */
	
    /* inequalities 
     */
    for(c=sc_inegalites(s1);
	c!=CONTRAINTE_UNDEFINED;
	c=c->succ)
	/* could be inlined to avoid costly sc_copy inside ineq_redund_with_sc_p
	 */
	if (!ineq_redund_with_sc_p(s2, c)) 
	    cnew = contrainte_copy(c),
	    cnew->succ = in,
	    in = cnew;
    
    /* equalities
     */
    for(c=sc_egalites(s1);
	c!=CONTRAINTE_UNDEFINED;
	c=c->succ)
	if (!eq_redund_with_sc_p(s2, c))
	    cnew = contrainte_copy(c),
	    cnew->succ = eq,
	    eq = cnew;

    new = sc_make(eq, in);
    return(sc_nredund(&new), new);
}


/* Psysteme build_sc_nredund_1pass_ofl_ctrl(Psysteme ps, int ofl_ctrl)
 * input    : a system in which redundancies must be eliminated, and an
 *            integer indicating how overflow errors must be handled.
 * output   : Computes a new system sc from the system ps, where each 
 *            constraint of the system ps is added to the new system sc, 
 *            if the constraint is not redundant with the system sc 
 *            previously computed. 
 * modifies : 
 * comment  : 
 *            The set of equalities is copied as such and ignored by 
 *            redundancy checks.
 *
 *            if ofl_ctrl == 0 overflow errors are trapped in the 
 *               routine which checks if the system is feasible.
 *            if ofl_ctrl == 2 overflow errors are trapped in the 
 *              sc_rational_feasibility_ofl_ctrl  routine that 
 *              keeps the constraint in the system whatever its redundancy 
 *              characteristic
 *            if ofl_ctrl == 1 overflow errors are forwarded to the
 *               calling routine.
 *
 *            the redundancy elimination assumes integer points. The
 *            rational set defined by sc may be enlarged.
 *
 */
void build_sc_nredund_1pass_ofl_ctrl(psc, ofl_ctrl)
Psysteme volatile *psc;
int ofl_ctrl;
{

    Psysteme sc;
    Psysteme ps = *psc;
    Pcontrainte ineq, ineg;
    int init_exception_thrown = linear_number_of_exception_thrown;

    if (SC_UNDEFINED_P(ps) || sc_rn_p(ps) || sc_empty_p(ps)) 
	return;

    sc = sc_init_with_sc(ps);
    if (!sc_rational_feasibility_ofl_ctrl(ps,OFL_CTRL,true)) {
      sc=sc_empty(base_dup(ps->base));
      sc_rm(ps);
      *psc =sc;
      return;
    }

    sc->egalites = contraintes_copy(ps->egalites);
    sc->nb_eq = ps->nb_eq;
    for (ineq = ps->inegalites;
	 !CONTRAINTE_UNDEFINED_P(ineq) &&
	   /* if more than 6 exceptions are thrown from within the loop,
	      the loop is stopped. */
	   linear_number_of_exception_thrown-init_exception_thrown<7;
	 ineq=ineq->succ) 
    {
	ineg = contrainte_copy(ineq);
	contrainte_reverse(ineg); 
	
	sc_add_inegalite(sc,ineg);

	if (sc_rational_feasibility_ofl_ctrl(sc,ofl_ctrl,true))
	    contrainte_reverse(ineg);		
	else {
	    sc->inegalites = sc->inegalites->succ;
	    ineg->succ = NULL;
	    contrainte_rm(ineg);
	    sc->nb_ineq--;
	}
    }

    if (linear_number_of_exception_thrown-init_exception_thrown>=7)
      fprintf(stderr, "[build_sc_nredund_1pass_ofl_ctrl] "
	      "too many exceptions in redundancy elimination... function stopped.\n");

    sc_rm(ps);
    *psc = sc;
} 

void sc_safe_build_sc_nredund_1pass(ps)
Psysteme volatile *ps;
{   

  if (!sc_rn_p(*ps) && !sc_empty_p(*ps))
    {
      Pbase b = base_copy(sc_base(*ps));
      /*  *ps = sc_sort_constraints_simplest_first(*ps, b); see version 1.16*/
      build_sc_nredund_1pass(ps);
      if (*ps == SC_EMPTY)
	*ps = sc_empty(b);
      else {
	base_rm(sc_base(*ps)); 
	(*ps)->base = base_copy(b);
      }
    }
}


/* Computation of a new system sc from the system ps, where each 
 * constraint of the system ps is added to the new system sc, 
 * if the constraint is not redundant with the system sc previously 
 * computed.
 * 
 * The set of equalities is copied as such and ignored by redundancy checks.
 */
void build_sc_nredund_1pass(ps)
Psysteme volatile *ps;
{
    build_sc_nredund_1pass_ofl_ctrl(ps,OFL_CTRL);
} 

void build_sc_nredund_2pass_ofl_ctrl(psc,ofl_ctrl)
Psysteme volatile *psc;
int ofl_ctrl;
{
  Psysteme ps = *psc;
  static int francois_check = 0;
  Pvecteur ip;
  if(francois_check && !SC_UNDEFINED_P(ps))
    ip = vect_make_dense(ps->base, 1LL, 0LL, 0LL, 100LL, 0LL);

  if (SC_UNDEFINED_P(ps) || sc_rn_p(ps) || sc_empty_p(ps))
    return;

  *psc = sc_normalize(ps);
  if(francois_check)
    assert(sc_belongs_p(*psc, ip));
ifscdebug(5) {   
		  fprintf(stderr, "after normalize: \n");  
		  sc_default_dump(*psc);
		}
  assert(!SC_UNDEFINED_P(*psc));
  build_sc_nredund_1pass_ofl_ctrl(psc, ofl_ctrl);
  if(francois_check)
    assert(sc_belongs_p(*psc, ip));
ifscdebug(5) {   
		  fprintf(stderr, "after first nredund: \n");  
		  sc_default_dump(*psc);
		}
  build_sc_nredund_1pass_ofl_ctrl(psc, ofl_ctrl);
  if(francois_check)
    assert(sc_belongs_p(*psc, ip));
}


void sc_safe_build_sc_nredund_2pass(ps)
Psysteme volatile *ps;
{   

  if (!sc_rn_p(*ps) && !sc_empty_p(*ps))
    {
      Pbase b = base_copy(sc_base(*ps));
      build_sc_nredund_2pass(ps);
      if (*ps == SC_EMPTY)
	*ps = sc_empty(b);
      else {
	base_rm(sc_base(*ps)); 
	(*ps)->base = base_copy(b);
      }
    }
}

/* void  build_sc_nredund_2pass
 * Psysteme *psc;
 *
 */

void build_sc_nredund_2pass(Psysteme volatile *psc)
{
    if (SC_UNDEFINED_P(*psc)) 
	return;
    else
	build_sc_nredund_2pass_ofl_ctrl(psc, OFL_CTRL);
}


/* This function returns true if the constraint ineq can be eliminated 
 * from the system sc and false oterwise. 
 * It assumes that two constraints at least must be kept for constraining
 * the variable "var_hr" in the system.
 * the array "tab_info" contains the useful informations allowing to know
 * the number of constraints constraining  each variable as upper or 
 * lower bounds.
*/

static bool sc_elim_triang_integer_redund_constraint_p
    (pc2,index_base,ineq,var_hr,tab_info,rank_max)
Pcontrainte pc2;
Pbase index_base;
Pcontrainte ineq;
Variable var_hr;
int tab_info[][4];
int *rank_max;
{
    int rank_hr = rank_of_variable(index_base,var_hr);
    Value coeff = vect_coeff(var_hr,ineq->vecteur);
    int sign = value_sign(coeff);
    bool result=false;
    bool trouve=false;
    *rank_max=rank_hr;

    if (tab_info[rank_hr][1]) {

	/* This condition is true if the variable is a loop index. 
	   As the constraint constrains directly the variable, 
	   this constraint must be kept if there is not enough 
	   remainding constraints  
	   */

	if (((sign >0) && (tab_info[rank_hr][2]>1))
	    || ((sign <0) && (tab_info[rank_hr][3]>1)))
	    result = true;
    }
    else {
	register Pcontrainte pc;

	for (pc = pc2;
	     !CONTRAINTE_UNDEFINED_P(pc) && !trouve;
	     pc = pc->succ) {

	    Value coeff2 = vect_coeff(var_hr,pc->vecteur);
	    int sign2 = value_sign(coeff2);
	    int right_rank, left_rank;
	    Value right_coeff, left_coeff;
	    Variable right_var,left_var;
				    
	    if (value_notzero_p(coeff2) && sign == -sign2) {
		constraint_integer_combination(index_base,ineq,pc,rank_hr,
			       &right_var,&right_rank,&right_coeff,
			       &left_var,&left_rank,&left_coeff);
		*rank_max = MAX(right_rank,left_rank);
		if (((right_rank>left_rank) 
		     && (((value_pos_p(right_coeff)) && 
			  (tab_info[right_rank][2] <=1))
			 || (value_neg_p(right_coeff) && 
			     (tab_info[right_rank][3] <=1)))) 
		    || ((right_rank<left_rank) 
			&& ((value_pos_p(left_coeff) && 
			     (tab_info[left_rank][2]<=1)) 
			    || (value_neg_p(left_coeff) && 
				(tab_info[left_rank][3] <=1)))))
		    trouve = true;
	    }   
	}
	if (!trouve) result = true;
    } 
    return result;

}


/* Computation of a new system sc from the system ps, where each
 * constraint of the system ps is added to the new system sc,
 * if the constraint is not redundant with the system sc previously
 * computed.
 *
 * The difference with the function build_sc_nredund is that at least
 * 2 constraints are kept for each variable: one for the upper bound
 * and the other for the lower bound.
 *
 * The rational set defined by ps may be enlarged by this procedure
 * because an integer constraint negation is used.
 *
 * Modifs:
 *  - change _dup to _copy.
 *  - the parameters of contrainte_dup, _copy, _reverse are not changed like base_dup, so it's ok.
 */

Psysteme build_integer_sc_nredund(
  volatile Psysteme ps,
  Pbase index_base,
  int tab_info[][4],
  int loop_level,
  int dim_h __attribute__ ((unused)),
  int n __attribute__ ((unused)))
{

  volatile Psysteme sc = sc_new();
  Pcontrainte eq;
  // Automatic variables read in a CATCH block need to be declared volatile as
  // specified by the documentation
  volatile Pcontrainte ineq, pred;
  int rank_hr,rank_max = 0;
  Variable var_hr;
  Value coeff;
  volatile int sign;

  if (SC_UNDEFINED_P(ps) || SC_EMPTY_P(ps) || sc_empty_p(ps) )
    return ps;
  sc->base = base_copy(ps->base);
  sc->dimension = ps->dimension;

  for (eq = ps->egalites;
       !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ) {
    Pcontrainte pc=contrainte_copy(eq);
    sc_add_eg(sc,pc);
  }

  if (!CONTRAINTE_UNDEFINED_P(ps->inegalites))  {

    sc->inegalites = contrainte_copy(ps->inegalites);
    sc->nb_ineq +=1;
    for (pred = ps->inegalites,ineq = (ps->inegalites)->succ;
         !CONTRAINTE_UNDEFINED_P(ineq); ineq=ineq->succ) {

	    Pcontrainte volatile ineg = contrainte_copy(ineq);
	    sc_add_inegalite(sc,ineg);

	    // search the characteristics of the variable of higher rank in
      // the constraint ineq
	    if (( rank_hr= search_higher_rank(ineq->vecteur,index_base)) >0) {
        var_hr=variable_of_rank(index_base,rank_hr);
        coeff=vect_coeff(var_hr,ineq->vecteur);
        sign = value_sign(coeff);

        if (sc_elim_triang_integer_redund_constraint_p
            (ps->inegalites,index_base,ineq, var_hr,tab_info, &rank_max)
            && (rank_max >= loop_level)) {

          /* this condition is true if the constraint can be
             eliminated from the system, that means if this is
             not the last constraint on the variable or if all
             the constraints on this variable can be
             eliminated (the rank of variable is greater the
             number of loops) */

          contrainte_reverse(ineg);
          CATCH(overflow_error) {
            pred = pred->succ;
            contrainte_reverse(ineg);
          }
          TRY {
            // test de sc_faisabilite avec la nouvelle inegalite
            if (!sc_rational_feasibility_ofl_ctrl(sc,OFL_CTRL,true)) {

              // si le systeme est non faisable ==>
              // inegalite redondante ==> elimination de cette inegalite
              sc->inegalites = sc->inegalites->succ;
              ineg->succ = NULL;
              contrainte_rm(ineg);
              sc->nb_ineq -=1;
              pred->succ = ineq->succ;

              // mise a jour du nombre de contraintes restantes
              // contraingnant la variable  de rang rank_hr
              if (sign >0) tab_info[rank_hr][2] --;
              else if (sign <0) tab_info[rank_hr][3]--;
            }
		    else {
          pred = pred->succ;
          contrainte_reverse(ineg);
        }
            UNCATCH(overflow_error);
          }
        }
	    }
    }
  }
  return sc;
}

/* This  function returns true if the constraint C (resulting of the
 *  combination of the two constraints ineq1 and ineq2) is redundant 
 * for the system sc, and false otherwise.
 *
 * Assume that ineq1 = coeff1 (positive) * var + E1 <=0
 *             ineq2 = coeff2 (negative) * var +E2 <=0
 *             C = coeff1 * E2 - coeff2 * E1 - coeff1*coeff2-coeff1 <=0
 *
*/

bool bound_redund_with_sc_p(sc,ineq1,ineq2,var)
Psysteme sc;
Pcontrainte ineq1,ineq2;
Variable var;
{

  volatile Pcontrainte posit, negat;
  Pcontrainte ineg = CONTRAINTE_UNDEFINED;
  bool result = false;

  if (!CONTRAINTE_UNDEFINED_P(ineq1) && !CONTRAINTE_UNDEFINED_P(ineq2)) {

    if (value_pos_p(vect_coeff(var,ineq1->vecteur))) {
	    posit = contrainte_copy(ineq1);
	    negat = contrainte_copy(ineq2);
    }
    else  {
	    posit = contrainte_copy(ineq2);
	    negat = contrainte_copy(ineq1);
    }

    CATCH(overflow_error)
	    result = false;
    TRY {
	    ineg = sc_integer_inequalities_combination_ofl_ctrl
        (sc, posit, negat, var, &result, FWD_OFL_CTRL);
	    contrainte_rm(ineg);
	    UNCATCH(overflow_error);
    }

    contrainte_rm(posit);
    contrainte_rm(negat);
  }
  return result;
}
