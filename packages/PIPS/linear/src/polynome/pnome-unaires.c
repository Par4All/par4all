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

/********************************************************** pnome-unaires.c
 *
 * UNARY OPERATIONS ON POLYNOMIALS
 *
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <boolean.h>
#include "arithmetique.h"
#include "vecteur.h"
#include "polynome.h"


/* void polynome_negate(Ppolynome *ppp);
 *  changes sign of polynomial *ppp.
 *  !usage: polynome_negate(&pp);
 */
void polynome_negate(ppp)
Ppolynome *ppp;
{
    Ppolynome curpp;

    if ( !POLYNOME_UNDEFINED_P(*ppp) && !POLYNOME_NUL_P(*ppp) )
	for(curpp = *ppp; curpp != POLYNOME_NUL; curpp = polynome_succ(curpp))
	    monome_coeff(polynome_monome(curpp)) = - monome_coeff(polynome_monome(curpp));
}

/* Ppolynome polynome_opposed(Ppolynome pp);
 *  changes sign of polynomial pp.
 *  !usage: pp = polynome_negate(pp);
 */
Ppolynome polynome_opposed(pp)
Ppolynome pp;
{
    Ppolynome curpp;

    if ( !POLYNOME_UNDEFINED_P(pp) && !POLYNOME_NUL_P(pp) )
	for(curpp = pp; curpp != POLYNOME_NUL; curpp = polynome_succ(curpp))
	    monome_coeff(polynome_monome(curpp)) = - monome_coeff(polynome_monome(curpp));

    return pp;
}


/* Ppolynome polynome_sum_of_power(Ppolynome ppsup, int p)
 *  calculates the sum of i^p for i=1 to (ppsup),
 *  returns the polynomial sigma{i=1, ppsup} (i^p).
 *  It does the job well until p=13; after, it goes wrong
 *  (the Bernouilli numbers are computed until p=12)
 */

Ppolynome polynome_sum_of_power(ppsup, p)
Ppolynome ppsup;
int p;
{
    Ppolynome ppresult, ppacc;
    int i;

    if (POLYNOME_UNDEFINED_P(ppsup))
	return (POLYNOME_UNDEFINED);
    if (p < 0)
	polynome_error("polynome_sum_of_power", "negative power: %d\n", p);
    else if (p == 0)
	ppresult = polynome_dup(ppsup);
    else {
	if ( polynome_constant_p(ppsup) ) {    /* if the upper bound is constant ... */
	    double factor, result = 0;
	    double cste = (double)polynome_TCST(ppsup);

	    if (cste<1) {
	      /* FI: That means, no iteration is executed whatsoever,
		 isn't it?

		 Also, polynome_error() does stop the execution and we
		 are in trouble for Linear/C3 Library. We should init some exit
		 function towards pips_internal_error().
	      */
	      /*
		polynome_error("polynome_sum_of_power",
			       "compute a sum from 1 to %f!\n", (float) cste);
	      */
		ppresult = POLYNOME_NUL;
	    }
	    /*else if (cste==1)
		ppresult = POLYNOME_NUL;*/
	    else {
		result = intpower(cste, p) * ((double) (cste / (p+1)) + 0.5);
		factor = ((double) p/2);

		for (i=1; 0<p-2*i+1; i++) {
		    result += (intpower(cste, p-2*i+1)
			       * ((double) (Bernouilli(i) * factor)));
		    factor *= - ((double) (p-2*i+1)*(p-2*i)) / ((double) (2*i+1)*(2*i+2));
		}
		ppresult = make_polynome((float) result, TCST, VALUE_ONE);
	    }
	}
	else {    /* if the upper bound is a non-constant polynomial ... */
	    float factor;
	      /*  (ppsup^(p+1)) / (p+1)  */
	    ppresult = polynome_power_n(ppsup, p+1);
	    polynome_scalar_mult(&ppresult, (float) 1/(p+1));
	      /*  1/2 * ppsup^p  */
	    ppacc = polynome_power_n(ppsup, p);
	    polynome_scalar_mult(&ppacc, (float) 1/2);
	    polynome_add(&ppresult, ppacc);
	    polynome_rm(&ppacc);

	    factor = ((float) p / 2);
	    /* computes factors p(p-1).../(2i!) incrementally */

	    for (i=1; 0 < p-2*i+1; i++) {
		/* the current term of the remaining of the sum is:     */
		/* Ti = (1/(2i)!)*(Bi*p*(p-1)* . *(p-2*i+2)*ppsup^(p-2*i+1)) */

		ppacc = polynome_power_n(ppsup, p-2*i+1);
		polynome_scalar_mult(&ppacc, (float) Bernouilli(i) * factor);

		polynome_add(&ppresult, ppacc);
		polynome_rm(&ppacc);

		factor *= -((float)(p-2*i+1)*(p-2*i))/((float)(2*i+1)*(2*i+2));
	    }
       }
    }
    return(ppresult);
}

/* Ppolynome polynome_sigma(Ppolynome pp, Variable var, Ppolynome ppinf, ppsup)
 *  returns the sum of pp when its variable var is moving from ppinf to ppsup.
 *  Neither ppinf nor ppsup must contain variable var.
 */
Ppolynome polynome_sigma(pp, var, ppinf, ppsup)
Ppolynome pp;
Variable var;
Ppolynome ppinf, ppsup;
{
    Ppolynome ppacc, pptemp, ppfact, ppresult = POLYNOME_NUL;
    int i;

    if (POLYNOME_UNDEFINED_P(pp) || POLYNOME_UNDEFINED_P(ppinf)
	|| POLYNOME_UNDEFINED_P(ppsup)) 
	return (POLYNOME_UNDEFINED);

    for(i = 0; i <= polynome_degree(pp, var); i++) {
	/* compute:
	 *     sum(ppinf,ppsup) ppfact * x ^ i 
	 * as:
	 *     ppfact * ( sum(1, ppsup) x ^ i -
	 *                sum(1, ppinf) x ^ i +
	 *                ppfin ^ i )
	 * where:
	 * ppfact is the term associated to x ^ i in pp
	 *
	 * Note that this decomposition is correct wrt standard
	 * mathematical notations if and only if:
	 *     ppsup >= ppinf >= 1
	 * although the correct answer can be obtained when
	 *     ppsup >= ppinf
	 *
	 * Thus:
	 *      sum(1, ppsup) x ^ i
	 * is extended for ppsup < 1 and defined as:
	 *      - (sum(ppsup, -1) x ^ i)
	 */
	ppfact = polynome_factorize(pp, var, i);

	if (!POLYNOME_NUL_P(ppfact)) {
	    ppacc  = polynome_sum_of_power(ppsup, i);
	    /* LZ: if ppinf == 1: no need to compute next sigma (pptemp), 
	     * nor ppinf^i (FI: apparently not implemented) */
	    pptemp = polynome_sum_of_power(ppinf, i);

	    polynome_negate(&pptemp);
	    polynome_add(&ppacc, pptemp);
	    /* ppacc = sigma{1,ppsup} - sigma{1,ppinf} */
	    polynome_rm(&pptemp);

	    /* ppacc == (sigma{k=1,ppsup} k^i) - (sigma{k=1,ppinf} k^i)  */

	    pptemp = polynome_power_n(ppinf, i);
	    polynome_add(&ppacc, pptemp);
	    polynome_rm(&pptemp);

	    pptemp = polynome_mult(ppfact, ppacc);

	    polynome_add(&ppresult, pptemp);

	    polynome_rm(&pptemp);
	    polynome_rm(&ppacc);
	}
    }
    return(ppresult);
}


/* Ppolynome polynome_sort((Ppolynome *) ppp, bool (*is_inferior_var)())
 *  Sorts the polynomial *ppp: monomials are sorted by the private routine
 *  "is_inferior_monome" based on the user one "is_inferior_var".
 *  !usage: polynome_sort(&pp, is_inferior_var);
 */
Ppolynome polynome_sort(ppp, is_inferior_var)
Ppolynome *ppp;
int (*is_inferior_var)(Pvecteur *, Pvecteur *);
{
    Ppolynome ppcur;
    Ppolynome ppsearchmin;
    Pmonome pmtemp;

    if ((!POLYNOME_NUL_P(*ppp)) && (!POLYNOME_UNDEFINED_P(*ppp))) {
	for (ppcur = *ppp; ppcur != POLYNOME_NUL; ppcur = polynome_succ(ppcur)) {
	    Pmonome pm = polynome_monome(ppcur);
	    Pvecteur pv = monome_term(pm);
	    pv = vect_sort(pv, vect_compare);
	    /* pv = vect_tri(pv, is_inferior_var); */
	}
	for (ppcur = *ppp; polynome_succ(ppcur) != POLYNOME_NUL; ppcur = polynome_succ(ppcur)) {
	    for(ppsearchmin  = polynome_succ(ppcur);
		ppsearchmin != POLYNOME_NUL;
		ppsearchmin  = polynome_succ(ppsearchmin)) {

		if (!is_inferior_monome(polynome_monome(ppsearchmin),
				        polynome_monome(ppcur), is_inferior_var)) {
		    pmtemp = polynome_monome(ppsearchmin);
		    polynome_monome(ppsearchmin) = polynome_monome(ppcur);
		    polynome_monome(ppcur) = pmtemp;
		}
	    }
	}
    }
    return (*ppp);
}

/* void polynome_chg_var(Ppolynome *ppp, Variable v_old, Variable v_new)
 * replace the variable v_old by v_new 
 */
void polynome_chg_var(ppp,v_old,v_new)
Ppolynome *ppp;
Variable v_old,v_new;
{
    Ppolynome ppcur;

    for (ppcur = *ppp; ppcur != POLYNOME_NUL; ppcur = polynome_succ(ppcur)) {
	Pmonome pmcur = polynome_monome(ppcur);
	/* Should it be comparated against MONOME_NUL (that is different
	   of 0) instead? */
	if ( pmcur != NULL ) {
	    Pvecteur pvcur = monome_term(pmcur);

	    vect_chg_var(&pvcur,v_old,v_new);
	}
    }
}
