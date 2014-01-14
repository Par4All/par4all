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

 /* package sc
  */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sc-private.h"


/*   COMPARISON of CONSTRAINTS
 *
 *
 *
 *
 *
 */

static Pbase 
  rbase_for_compare  = BASE_NULLE, 
  others_for_compare = BASE_NULLE;
static bool
  inner_for_compare,
  complex_for_compare;

static void set_info_for_compare(base, sort_base, inner_first, complex_first)
Pbase base, sort_base;
bool inner_first, complex_first;
{
    Pbase btmp=BASE_NULLE;
    assert(BASE_NULLE_P(rbase_for_compare) &&
	   BASE_NULLE_P(others_for_compare));

    /* the base is reversed! inner indexes first!!
     */
    rbase_for_compare  = base_normalize(base_reversal(sort_base));
    btmp = base_difference(base, sort_base);
    others_for_compare =  base_normalize(btmp);
    inner_for_compare = inner_first;
    complex_for_compare = complex_first;
}

static void reset_info_for_compare()
{
    base_rm(rbase_for_compare),  rbase_for_compare=BASE_NULLE;
    base_rm(others_for_compare), others_for_compare=BASE_NULLE;
}

#define ADD_COST (1)
#define MUL_COST (1)
#define AFF_COST (1)
static int cost_of_constant_operations(v) 
Pvecteur v;
{
    int cost = AFF_COST;
    Pbase b;
    Value val;
    
    /*   constant
     */
    if (value_notzero_p(vect_coeff(TCST, v)))
	cost += ADD_COST;

    /*   other variables
     */
    for (b=others_for_compare; b!=(Pvecteur)NULL; b=b->succ)
    {
	val = vect_coeff(var_of(b), v);
	val = value_abs(val);

	if (value_notzero_p(val))
	    cost+=value_one_p(val)? ADD_COST: (MUL_COST+ADD_COST);
    }

    return(cost);
}

/* for qsort, returns "is simpler than"
 *
 *    - : v1 < v2
 *    0 : v1==v2
 *    + : v1 > v2
 *
 * with the following criterion 
 *
 *  1/ ranks
 *  2/ coef of comparable ranks, +-1 or simpler...
 *  3/ 
 *
 * rational: 
 *  - loop sizes are assumed to be infinite
 *  - invariant code motion
 *  - induction variables recognized
 */
#define DB_RESULT(e)							 \
{									 \
      int result = (e);							 \
      fprintf(stderr, "[compare_the_constraints]\n");			 \
      vect_debug(v1); vect_debug(v2);					 \
      fprintf(stderr, "%s\n", result==0 ? "=" : (result>0 ? ">" : "<")); \
      return(result);							 \
}

#define RESULT(e) { return (e); }

#define RETURN_HARDER(b) RESULT(complex_for_compare ? (b) : -(b))
#define RETURN_ORDER(b) RESULT(inner_for_compare ? (b) : -(b))
#define same_sign_p(v,w) ((value_neg_p(v) && value_neg_p(w)) || \
			  (value_pos_p(v) && value_pos_p(w)))

/* returns -1: c1<c2, 0: c1==c2, +1: c1>c2
 */
static int compare_the_constraints(pc1, pc2)
Pcontrainte *pc1, *pc2;
{
    Pvecteur
	v1 = (*pc1)->vecteur,
	v2 = (*pc2)->vecteur;
    int null_1, null_2, i, irank=0, cost_1, cost_2;
    Value val_1=VALUE_ZERO, val_2=VALUE_ZERO, 
          val=VALUE_ZERO, val_p=VALUE_ZERO;
    Pbase b, high=NULL;

    /*  for each inner first indexes,
     *  the first constraint with a null coeff while the other one is non 
     *  null is the simplest.
     */
    for (i=1, b=rbase_for_compare; !BASE_NULLE_P(b); i++, b=b->succ)
    {
	val_1 = vect_coeff(var_of(b), v1), null_1 = value_zero_p(val_1),
	val_2 = vect_coeff(var_of(b), v2), null_2 = value_zero_p(val_2);

	if (irank==0 && !same_sign_p(val_1,val_2))
	    RETURN_ORDER(value_neg_p(val_1) && value_neg_p(val_2)? 
			 value_compare(val_2,val_1): 
			 value_compare(val_1,val_2));

	if (null_1 ^ null_2) {
	    if (irank==0)
	    { RETURN_ORDER(value_compare(null_1,null_2));}
	    else
	    { RETURN_HARDER(value_compare(null_1,null_2));} 
	}
	if (irank==0 && (!null_1||!null_2)) 
	    val=val_1, val_p=val_2, irank=i, high=b;
    }

    if (value_ne(val_p,val))
	RETURN_HARDER(value_neg_p(val_1) && value_neg_p(val_2)? 
		      value_compare(val_2,val_1): 
		      value_compare(val_1,val_2));
    
    /*   constant operations
     */
    cost_1 = cost_of_constant_operations(v1),
    cost_2 = cost_of_constant_operations(v2);

    if (cost_1!=cost_2) RETURN_HARDER(cost_2-cost_1);

    /*   compare the coefficients for the base
     */
    for (b=high==NULL ? NULL : high->succ; !BASE_NULLE_P(b); b=b->succ)
    {
	val_1 = vect_coeff(var_of(b), v1),
	val_2 = vect_coeff(var_of(b), v2);
	
	if (value_ne(val_1,val_2))
	    RETURN_HARDER(value_neg_p(val_1) && value_neg_p(val_2)? 
			  value_compare(val_1,val_2): 
			  value_compare(val_2,val_1));
    }

    /*   do it for the for the parameters
     */
    for (b=others_for_compare; !BASE_NULLE_P(b); b=b->succ)
    {
	val_1 = vect_coeff(var_of(b), v1),
	val_2 = vect_coeff(var_of(b), v2);
	
	if (value_ne(val_1,val_2))
	    RETURN_HARDER(value_compare(val_2,val_1));
    }
    
    /*   at last the constant
     */
    val_1 = vect_coeff(TCST, v1),
    val_2 = vect_coeff(TCST, v2);

    RETURN_HARDER(value_pos_p(val)? 
		  value_compare(val_2,val_1): 
		  value_compare(val_1,val_2));
}

static int compare_the_constraints_debug(pc1, pc2)
Pcontrainte *pc1, *pc2;
{
    int b1, b2;
    b1 = compare_the_constraints(pc1, pc2),
    b2 = compare_the_constraints(pc2, pc1);
    assert((b1+b2)==0);
    return b1;
}

/* returns the highest rank pvector of v in b, of rank *prank
 */
Pvecteur highest_rank_pvector(v, b, prank)
Pvecteur v;
Pbase b;
int *prank;
{
    Pbase pb;
    Pvecteur pv, result=(Pvecteur) NULL;
    Variable var;
    int rank;

    for (*prank=-1, rank=1, pb=b;
	 !BASE_NULLE_P(pb);
	 pb=pb->succ, rank++)
    {
	var = var_of(pb);
	
	for (pv=v; pv!=NULL; pv=pv->succ)
	    if (var_of(pv)==var) 
	    {
		result=pv;
		*prank=rank;
		continue;
	    }
    }

    return(result);
}

/*  sorts the constraints according to the compare function,
 *  and set the number of constraints for each index of the sort base
 */

Pcontrainte constraints_sort_info(c, sort_base, compare, info)
Pcontrainte c;
Pbase sort_base;
int (*compare)();
two_int_infop info;
{
    Pcontrainte pc, *tc;
    Pvecteur phrank;
    int	i, rank,
	nb_of_sort_vars = vect_size(sort_base),
	nb_of_constraints = nb_elems_list(c);

    if (nb_of_constraints<=1) return(c);

    tc   = (Pcontrainte*) malloc(sizeof(Pcontrainte)*nb_of_constraints);

    for (i=0; i<=nb_of_sort_vars; i++)
	info[i][0]=0, info[i][1]=0;

    /*   the constraints are put in the table
     *   and info is set.
     */
    for (i=0, pc=c; pc!=NULL; i++, pc=pc->succ)
    {
	tc[i] = pc;
	if (!BASE_NULLE_P(sort_base))
	{
	    phrank = highest_rank_pvector(pc->vecteur, sort_base, &rank);
	    info[rank==-1?0:rank][rank==-1?0:value_pos_p(val_of(phrank))]++;
	}
    }
    
   qsort(tc, nb_of_constraints, sizeof(Pcontrainte), compare);

    /*  the list of constraints is generated again
     */
    for (i=0; i<nb_of_constraints-1; i++)
    {
	tc[i]->succ = tc[i+1];
    }
    tc[nb_of_constraints-1]->succ=NULL;
    c = tc[0];

    /*   clean!
     */
    free(tc);
    return(c);
}

Pcontrainte constraints_sort_with_compare(c, sort_base, compare)
Pcontrainte c;
Pbase sort_base;
int (*compare)();
{
    int n = vect_size(sort_base)+1;
    two_int_infop info;

    info = (two_int_infop) malloc(sizeof(int)*2*n);

    c = constraints_sort_info(c, sort_base, compare, info);

    free(info);
    return(c);
}

Pcontrainte contrainte_sort(c, base, sort_base, inner_first, complex_first)
Pcontrainte c;
Pbase base, sort_base;
bool inner_first, complex_first;
{
    set_info_for_compare(base, sort_base, inner_first, complex_first);
    c = constraints_sort_with_compare(c, sort_base, compare_the_constraints);
    reset_info_for_compare();

    return(c);
}


Psysteme sc_sort_constraints(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    ps->inegalites = 
	contrainte_sort(ps->inegalites, ps->base, base_index, true, true);

    return(ps);
}

/* sort  contrainte c, base b, 
 * relatively to sort_base, as defined by the switches.
 *
 * inner_first: innermost first
 * complex_first: the more complex the likely to be put earlier
 */

/* Psysteme sc_triang_elim_redond(Psysteme ps, Pbase base_index):
 * elimination des contraintes lineaires redondantes dans le systeme ps 
 * par test de faisabilite de contrainte inversee; cette fonction est
 * utilisee pour calculer des bornes de boucles, c'est pourquoi il peut
 * etre necessaire de garder des contraintes redondantes afin d'avoit
 * toujours au moins une borne inferieure et une borne superieure
 * pour chaque indice.
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme	    : Le systeme initial est modifie. Il est egal a NULL si 
 *		      le systeme initial est non faisable.
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme ps    : systeme lineaire 
 *
 *  Attention: pour chaque indice dans base_index, il doit rester au moins deux
 *             contraintes correspondantes (une positive et une negative).
 *             C'est la seule difference avec la fonction sc_elim_redond().
 *
 *             contrainte_reverse() is used. Rational points may be added
 *             by this procedure.
 *
 * Yi-Qing YANG
 *
 * Modifications:
 *  - add a normalization step for inequalities; if they are not normalized,
 *    i.e. if the GCD of the variable coefficients is not 1, the constant
 *    term of the inverted constraint should be carefully updated using
 *    the GCD?!? (Francois Irigoin, 30 October 1991)
 *  - the variables are sorted in order to get a deterministic result
 *    (FC 26 Sept 94)
 *  - the feasible overflow function is called, and only if the constraint
 *    is not the last one (performance bug of the previous version)
 *    (FC 27 Sept 94)
 *  - a warning is displayed if many inequalities are to be dealt with,
 *    instead of returning the system as is.
 *    (FC 28 Sept 94)
 *  - sc_normalize inserted, in place of many loop that were 
 *    doing nearly the same. (FC 29/09/94)
 */
/* extern char *entity_local_name(); */

Psysteme sc_triang_elim_redund(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    Pcontrainte ineq, ineq1;
    int level, n = vect_size(base_index)+1;
    two_int_infop info;

    ps = sc_normalize(ps);
     
    if (ps==NULL)
	return(NULL);

    if (ps->nb_ineq > NB_INEQ_MAX1) 
	fprintf(stderr,
		"[sc_triang_elim_redund] warning, %d inequalities\n",
		ps->nb_ineq);

    if (!sc_integer_feasibility_ofl_ctrl(ps, OFL_CTRL,true))
    {
	sc_rm(ps), ps=NULL;
	return(NULL);
    }

    info = malloc(sizeof(int)*2*n);

    set_info_for_compare(ps->base, base_index, true, true);
    ps->inegalites = constraints_sort_info(ps->inegalites, 
					   base_index,
					   compare_the_constraints, 
					   info);
    reset_info_for_compare();

    for (ineq = ps->inegalites; ineq != NULL; ineq = ineq1)
    {
	ineq1 = ineq->succ;
	level = level_contrainte(ineq, base_index);

	/* only the variables that have more than one 
	 * constraints on a given size and that deal with 
	 * the variables of base_index are tested.
	 *
	 * an old comment suggested that keeping contraints on variables
	 * out of base_index would help find redundancy on the base_index
	 * contraints, but this should not be true anymore, since the
	 * variables are sorted... just help to deal with larger systems...
	 *
	 * FC 28/09/94
	 */
	if (level!=0 && info[abs(level)][level<0?0:1]>1)
	{
	    /* inversion du sens de l'inegalite par multiplication
	     * par -1 du coefficient de chaque variable
	     */
	    contrainte_reverse(ineq);

	    /* test de sc_faisabilite avec la nouvelle inegalite 
	     */
	    if (sc_integer_feasibility_ofl_ctrl(ps, OFL_CTRL, true))
		/* restore the initial constraint */
		contrainte_reverse(ineq);
	    else
	    {
		eq_set_vect_nul(ineq),		
		info[abs(level)][level<0?0:1]--;		
	    }
	}
    }
    sc_elim_empty_constraints(ps,0);
    ps = sc_kill_db_eg(ps);

    free(info);
    return(ps);
}

/* void move_n_first_constraints(source, target, n)
 * Pcontrainte *source, *target;
 * int n;
 *
 * moves the n first constraints from source to target, in order.
 */
void move_n_first_constraints(source, target, n)
Pcontrainte *source, *target;
int n;
{
    Pcontrainte tmp, nth;

    if (n==0) return; /* nothing to be done */

    /*  nth points to the nth constraint.
     */
    for (nth=*source; n>1; n--, nth=nth->succ);

    tmp = *target, *target = *source, *source = nth->succ, nth->succ = tmp;
    
}

/* void sc_triang_elim_redund_n_first(s, n)
 * Psysteme s;
 * int n;
 *
 * tries a triangular redundancy elimination on the n first constraints,
 * which *must all* deal with the same side of the same index.
 * if n is 0, nothing is done, but nothing is reported.
 *
 * contrainte_reverse() is used and rational points may be added
 */
void sc_triang_elim_redund_n_first(s, n)
Psysteme s;
int n;
{
    int tested, removed;
    Pcontrainte ineq;

    if (n<=1) return; /* nothing to be done */

    for (ineq=sc_inegalites(s), tested=0, removed=0;
	 removed<n-1 && tested<n;
	 tested++, ineq=ineq->succ)
    {
	contrainte_reverse(ineq);

	if (sc_integer_feasibility_ofl_ctrl(s, OFL_CTRL, true))
	    contrainte_reverse(ineq); /* restore */
	else
	    eq_set_vect_nul(ineq), removed++; /* remove */
    }
}

Psysteme sc_build_triang_elim_redund(s, indexes)
Psysteme s;
Pbase indexes; /* outer to inner */
{
    Pcontrainte
	old ;
    int level, side, n_other_constraints, n = vect_size(indexes)+1;
    two_int_infop info;

    s = sc_normalize(s);
    if (s==NULL || sc_nbre_inegalites(s)==0) return(s);

    info = (two_int_infop) malloc(sizeof(int)*2*n);

    /* sort outer first and complex first
     */
    set_info_for_compare(s->base, indexes, false, true);
    s->inegalites = constraints_sort_info(s->inegalites, 
					  indexes,
					  compare_the_constraints_debug,
					  info);
    reset_info_for_compare();

    /* remove the redundancy on others
     * then triangular clean of what remains.
     */

    n_other_constraints = info[0][0]+info[0][1];
    old = sc_inegalites(s), sc_inegalites(s) = NULL, sc_nbre_inegalites(s) = 0;
    
    move_n_first_constraints(&old, &s->inegalites, n_other_constraints);
    sc_nbre_inegalites(s) = n_other_constraints;
    s = sc_elim_redund(s);

    /* what if s is empty or null ???
     */
    
    /*  build the non redundant triangular system for each level and side.
     */
    for (level=1; level<n; level++)
    {
	for (side=0; side<=1; side++)
	    if (info[level][side]) 
	    {
		move_n_first_constraints(&old, &sc_inegalites(s), 
					 info[level][side]);
		sc_nbre_inegalites(s)+=info[level][side];
		
		sc_triang_elim_redund_n_first(s, info[level][side]);
	    }
    }

    assert(old==NULL); 

    /*  clean!
     */
    sc_elim_empty_constraints(s, 0);
    s = sc_kill_db_eg(s);
    free(info);

    return(s);
}

Psysteme sc_sort_constraints_simplest_first(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    ps->inegalites = 
	contrainte_sort(ps->inegalites, ps->base, base_index, false, false);

    return(ps);
}


/*   That is all
 */
