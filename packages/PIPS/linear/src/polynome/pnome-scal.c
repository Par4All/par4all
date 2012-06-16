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

/****************************************************************** pnome-scal.c
 *
 * SCALAR OPERATIONS ON POLYNOMIALS
 *
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <boolean.h>
#include <math.h>
#include "arithmetique.h"
#include "vecteur.h"
#include "polynome.h"


/* void polynome_scalar_mult(Ppolynome* ppp, float factor)
 *  (*ppp) = factor * (*ppp)
 *  !usage: polynome_scalar_mult(&pp, factor);
 */
void polynome_scalar_mult(ppp, factor)
Ppolynome *ppp;
float factor;
{
    Ppolynome curpp;

    if ((!POLYNOME_NUL_P(*ppp)) && (!POLYNOME_UNDEFINED_P(*ppp))) {
	if (factor == 0)
	    polynome_rm(ppp);   /* returns *ppp pointing to POLYNOME_NUL */
	else
	    for(curpp = *ppp; curpp != POLYNOME_NUL; curpp = polynome_succ(curpp))
		monome_coeff(polynome_monome(curpp)) *= factor;
    }
}

/* Ppolynome polynome_scalar_multiply(Ppolynome pp, float factor)
 *  pp = factor * (pp)
 *  !usage: pp = polynome_scalar_mult(pp, factor);
 */
Ppolynome polynome_scalar_multiply(pp, factor)
Ppolynome pp;
float factor;
{
    Ppolynome curpp;

    if ((!POLYNOME_NUL_P(pp)) && (!POLYNOME_UNDEFINED_P(pp))) {
	if (factor == 0)
	    pp = polynome_free(pp);
	else
	    for(curpp = pp; curpp != POLYNOME_NUL; curpp = polynome_succ(curpp))
		monome_coeff(polynome_monome(curpp)) *= factor;
    }
    return pp;
}


/* void polynome_scalar_add(Ppolynome* ppp, float term)
 *  (*ppp) = (*ppp) + term
 *  !usage: polynome_scalar_add(&pp, term);
 */
void polynome_scalar_add(ppp, term)
Ppolynome *ppp;
float term;
{
    if ((term != 0) && (!POLYNOME_UNDEFINED_P(*ppp))) {
	Pmonome pmtoadd = make_monome((float) term, TCST, VALUE_ONE);
	polynome_monome_add(ppp, pmtoadd);
	monome_rm(&pmtoadd);
    }
}

/* Ppolynome polynome_scalar_addition(Ppolynome pp, float term)
 *  pp = pp + term
 *  !usage: pp = polynome_scalar_add(pp, term);
 */
Ppolynome polynome_scalar_addition(pp, term)
Ppolynome pp;
float term;
{
    if ((term != 0) && (!POLYNOME_UNDEFINED_P(pp))) {
	Pmonome pmtoadd = make_monome((float) term, TCST, VALUE_ONE);
	pp = polynome_monome_addition(pp, pmtoadd);
	monome_rm(&pmtoadd);
    }
    return pp;
}


/* Ppolynome polynome_power_n(Ppolynome pp, int n)
 *  returns pp ^ n  (n>=0)
 *
 * Modification:
 *  - treat n < 0 if pp is a monomial.
 *    LZ 6 Nov. 92
 */
Ppolynome polynome_power_n(pp, n)
Ppolynome pp;
int n;
{
    if (POLYNOME_UNDEFINED_P(pp)) 
	return (POLYNOME_UNDEFINED);
    else if (POLYNOME_NUL_P(pp)) {
	if(n>0)
	    return POLYNOME_NUL;
	else if (n == 0)
	    return make_polynome(1.0, TCST, VALUE_ONE);
	else
	    return POLYNOME_UNDEFINED;
    }
    else if (n < 0) {
	if ( is_polynome_a_monome(pp) ) {
	    int i,m=-n;
	    Ppolynome pptemp, ppresult = polynome_dup(pp);

	    for(i = 1; i < m; i++)	{
		pptemp = polynome_mult(ppresult, pp);
		polynome_rm(&ppresult);
		ppresult = pptemp;
	    }
	    return(polynome_div(make_polynome(1.0, TCST, VALUE_ONE), 
				ppresult));
	}
	else
	    polynome_error("polynome_power_n",
			   "negative power n=%d"
			   " and polynome is not a monome\n",n);
    }
    else if (n == 0) 
	return(make_polynome(1.0, TCST, VALUE_ONE));
    else if (n == 1) 
	return(polynome_dup(pp));
    else if (n > 1) {
	int i;
	Ppolynome pptemp, ppresult = polynome_dup(pp);

	for(i = 1; i < n; i++)	{
	    pptemp = polynome_mult(ppresult, pp);
	    polynome_rm(&ppresult);
	    ppresult = pptemp;
	}
	return(ppresult);
    }
    /* FI: a unique return would be welcome! No enough time for cleaning */
    polynome_error("polynome_power_n", "Cannot happen!\n");
    return POLYNOME_UNDEFINED;
}

/* computes the n-root of polynomial if possible, that is if all 
 * exponents are multiple of n
 * return POLYNOME_UNDEFINED if not possible symbolically
 */
Ppolynome polynome_nth_root(Ppolynome p, int n) {
    Ppolynome pp = polynome_dup(p);
    for(p=pp;!POLYNOME_NUL_P(p);p=polynome_succ(p)) {
        Pmonome m = polynome_monome(p);
        Pvecteur v ;
        monome_coeff(m)=powf(monome_coeff(m),1.f/n);
        for(v = monome_term(m); !VECTEUR_NUL_P(v); v=vecteur_succ(v)) {
            if(vecteur_val(v)%n == 0) {
                vecteur_val(v)/=n; 
            }
            else if(vecteur_var(v)!=(Variable)TCST){
                polynome_rm(&pp);
                return POLYNOME_UNDEFINED;
            }
        }
    }
    return pp;
}


/* Ppolynome number_replaces_var(Ppolynome pp, Variable var, float num)
 *  returns a copy of polynomial pp where variable var is replaced by
 *  a floating-point number: num
 */
Ppolynome number_replaces_var(pp, var, num)
Ppolynome pp;
Variable var;
float num;
{
    Pmonome pmnum;
    Ppolynome ppnum, ppnew;

    if (POLYNOME_UNDEFINED_P(pp)) 
	return (POLYNOME_UNDEFINED);
    else {
	pmnum = make_monome(num, TCST, VALUE_ONE);
	ppnum = monome_to_new_polynome(pmnum);
	ppnew = polynome_var_subst(pp, var, ppnum);
	polynome_rm(&ppnum);		/* removes also the monomial pmnum */

	return(ppnew);
    }
}


/* Ppolynome polynome_incr(Ppolynome pp)
 *  returns pp + 1.
 *  pp is NOT duplicated.
 */
Ppolynome polynome_incr(pp)
Ppolynome pp;
{
    if (POLYNOME_UNDEFINED_P(pp)) 
	return (POLYNOME_UNDEFINED);
    else {
	Pmonome one = make_monome(1.0, TCST, VALUE_ONE);

	polynome_monome_add(&pp, one);
	monome_rm(&one);

	return(pp);
    }
}


/* Ppolynome polynome_decr(Ppolynome pp)
 *  returns pp - 1.
 *  pp is NOT duplicated.
 */
Ppolynome polynome_decr(pp)
Ppolynome pp;
{
    if (POLYNOME_UNDEFINED_P(pp)) 
	return (POLYNOME_UNDEFINED);
    else {
	Pmonome minus_one = make_monome(-1.0, TCST, VALUE_ONE);

	polynome_monome_add(&pp, minus_one);
	monome_rm(&minus_one);

	return(pp);
    }
}
