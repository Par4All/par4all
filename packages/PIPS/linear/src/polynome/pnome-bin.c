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

/******************************************************************* pnome-bin.c
 *
 * BINARY OPERATIONS ON POLYNOMIALS
 *
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "polynome.h"

/* void polynome_monome_add(Ppolynome* ppp, Pmonome pm)
 *  PRIVATE
 *  Add monomial pm to polynomial *ppp, in place.
 *  There is no new polynomial malloc.
 *  Monomial pm doesn't become part of the polynomial:
 *  it is duplicated if needed.
 *  !usage: polynome_monome_add(&pp, pm);
 */
void polynome_monome_add(ppp, pm)
Ppolynome *ppp;
Pmonome pm;
{
    Ppolynome prevpp = POLYNOME_NUL;
    float coeff;

    if (POLYNOME_NUL_P(*ppp)) {
	*ppp = monome_to_new_polynome(monome_dup(pm));
    }
    else if (POLYNOME_UNDEFINED_P(*ppp))
	;
    else if (MONOME_UNDEFINED_P(pm)) {
	polynome_rm(ppp);
	*ppp = POLYNOME_UNDEFINED;
    }
    else if (!MONOME_NUL_P(pm)) {
	Ppolynome curpp;
	for(curpp = *ppp; curpp != POLYNOME_NUL; prevpp = curpp,curpp = polynome_succ(curpp)) {
	    if (monome_colin(polynome_monome(curpp), pm)) {
		coeff = monome_coeff(polynome_monome(curpp)) + monome_coeff(pm);
		if ((coeff < PNOME_MACH_EPS) && (coeff > -PNOME_MACH_EPS)) {
		    /* This monomial is null now. We free it */
		    if (curpp == *ppp) 
			*ppp = polynome_succ(*ppp);
		    else 
			polynome_succ(prevpp) = polynome_succ(curpp);
		    polynome_succ(curpp) = POLYNOME_NUL;
		    polynome_rm(&curpp);
		    curpp = ( prevpp==POLYNOME_NUL ? *ppp : prevpp );
		    if ( curpp == POLYNOME_NUL ) /* no element in polynome */
			*ppp = POLYNOME_NUL;
		}
		else            /* Save new value of monomial coefficient. */
		    monome_coeff(polynome_monome(curpp)) = coeff;
		break;
	    }
	}

	if ( curpp == POLYNOME_NUL && !POLYNOME_NUL_P(*ppp) ) { 
	    /* Add a copy of the monomial at the end */
	    good_polynome_assert("polynome_monome_add about prevpp before",prevpp);
	    if ( polynome_succ(prevpp) == POLYNOME_NUL ) {
		if ( MONOME_NUL_P(pm) || MONOME_UNDEFINED_P(pm) )
		    printf("monome is poor\n");
		else 
		    polynome_succ(prevpp) = monome_to_new_polynome(monome_dup(pm));
	    }
	    good_polynome_assert("polynome_monome_add about prevpp at end",prevpp);
	}
    }
    good_polynome_assert("polynome_monome_add about *ppp at end",*ppp);
}

/* Ppolynome polynome_monome_addition(Ppolynome pp, Pmonome pm)
 *  PRIVATE
 *  Add monomial pm to polynomial pp, in place.
 *  There is no new polynomial malloc.
 *  Monomial pm doesn't become part of the polynomial:
 *  it is duplicated if needed.
 *  !usage: pp = polynome_monome_add(pp, pm);
 */
Ppolynome polynome_monome_addition(pp, pm)
Ppolynome pp;
Pmonome pm;
{
    Ppolynome prevpp = POLYNOME_UNDEFINED;

    if (POLYNOME_NUL_P(pp)) {
	pp = monome_to_new_polynome(monome_dup(pm));
    }
    else if (POLYNOME_UNDEFINED_P(pp))
	;
    else if (MONOME_UNDEFINED_P(pm)) {
	pp = polynome_free(pp);
	pp = POLYNOME_UNDEFINED;
    }
    else if (!MONOME_NUL_P(pm)) {
	Ppolynome curpp;
	for(curpp = pp; curpp != POLYNOME_NUL; prevpp = curpp,curpp = polynome_succ(curpp)) {
	    if (monome_colin(polynome_monome(curpp), pm)) {
		float coeff = monome_coeff(polynome_monome(curpp)) + monome_coeff(pm);

		if ((coeff < PNOME_MACH_EPS) && (coeff > -PNOME_MACH_EPS)) {
		    /* This monomial is null now. We free it */
		    if (curpp == pp) 
			pp = polynome_succ(pp);
		    else 
			polynome_succ(prevpp) = polynome_succ(curpp);
		    polynome_succ(curpp) = POLYNOME_NUL;
		    polynome_rm(&curpp);
		    curpp = ( prevpp==POLYNOME_NUL ? pp : prevpp );
		    if ( curpp == POLYNOME_NUL ) /* no element in polynome */
			pp = POLYNOME_NUL;
		}
		else            /* Save new value of monomial coefficient. */
		    monome_coeff(polynome_monome(curpp)) = coeff;
		break;
	    }
	}

	if ( curpp == POLYNOME_NUL && !POLYNOME_NUL_P(pp) ) { 
	    /* Add a copy of the monomial at the end */
	    good_polynome_assert("polynome_monome_add about prevpp before",prevpp);
	    if ( polynome_succ(prevpp) == POLYNOME_NUL ) {
		if ( MONOME_NUL_P(pm) || MONOME_UNDEFINED_P(pm) )
		    printf("monome is poor\n");
		else 
		    polynome_succ(prevpp) = monome_to_new_polynome(monome_dup(pm));
	    }
	    good_polynome_assert("polynome_monome_add about prevpp at end",prevpp);
	}
    }
    good_polynome_assert("polynome_monome_add about pp at end",pp);
    return pp;
}

/* void polynome_add(Ppolynome* ppp, Ppolynome pp2)
 *  (*ppp) = (*ppp) + pp2.
 *  !usage: polynome_add(&pp, pp2);
 */
void polynome_add(ppp, pp2)
Ppolynome *ppp,pp2;
{
    if (POLYNOME_NUL_P(*ppp)) {
	if ( !POLYNOME_NUL_P(pp2))
	    *ppp = polynome_dup(pp2);
	else
	    *ppp = POLYNOME_NUL;
    }
    else if (POLYNOME_UNDEFINED_P(pp2)) {
	polynome_rm(ppp);
	*ppp = POLYNOME_UNDEFINED;
    }
    else if (!POLYNOME_UNDEFINED_P(*ppp)) {
	for (;pp2 != POLYNOME_NUL; pp2 = polynome_succ(pp2)) {
	    polynome_monome_add(ppp, polynome_monome(pp2));
	}
    }
}

/* Ppolynome polynome_addition(Ppolynome pp, Ppolynome pp2)
 *  pp = pp + pp2.
 *  !usage: pp = polynome_add(pp, pp2);
 */
Ppolynome polynome_addition(pp, pp2)
Ppolynome pp,pp2;
{
    Ppolynome newpp = POLYNOME_UNDEFINED;

    if (POLYNOME_NUL_P(pp)) {
	if ( !POLYNOME_NUL_P(pp2))
	    newpp = polynome_dup(pp2);
	else
	    newpp = POLYNOME_NUL;
    }
    else if (POLYNOME_UNDEFINED_P(pp2)) {
	pp = polynome_free(pp);
	newpp = POLYNOME_UNDEFINED;
    }
    else if (!POLYNOME_UNDEFINED_P(pp)) {
	newpp = pp;
	for (;pp2 != POLYNOME_NUL; pp2 = polynome_succ(pp2)) {
	    newpp = polynome_monome_addition(newpp, polynome_monome(pp2));
	}
    }
    return newpp;
}


/* void monome_monome_mult(Pmonome *pm, Pmonome pm2)
 *  PRIVATE
 *  (*pm) = (*pm) * pm2.
 *  !usage: monome_monome_mult(&pm, pm2);
 */
static void monome_monome_mult(ppm, pm2)
Pmonome *ppm, pm2;
{
    Pvecteur produit;

    if (MONOME_UNDEFINED_P(*ppm))
	;
    else if (MONOME_UNDEFINED_P(pm2)) {
	monome_rm(ppm);
	*ppm = MONOME_UNDEFINED;
    }
    else if (!MONOME_NUL_P(*ppm)) {
	if (MONOME_NUL_P(pm2))
	    monome_rm(ppm);	/* returns ppm pointing to MONOME_NUL */
	else {
	    if (MONOME_CONSTANT_P(pm2))
		produit = vect_dup(monome_term(*ppm));
	    else if (MONOME_CONSTANT_P(*ppm))
		produit = vect_dup(monome_term(pm2));		
	    else {
		if (var_of(monome_term(*ppm))==var_of(monome_term(pm2)) &&
		    value_zero_p(value_plus(val_of(monome_term(*ppm)),
					    val_of(monome_term(pm2))))) {
		    /* M^-1 *M should be 1 .   26/10/92    LZ 
		     and the vecteur must be same. 09/11/92 */
		    produit = vect_new(TCST, VALUE_ONE);
		}
		else
		    produit = vect_add(monome_term(*ppm), monome_term(pm2));
	    }
	    monome_coeff(*ppm) *= monome_coeff(pm2);
	    vect_rm(monome_term(*ppm)); 
	    monome_term(*ppm) = produit;
	}
    }
}

/* Ppolynome polynome_monome_mult(Ppolynome pp, Pmonome pm)
 *  PRIVATE
 *  returns pp * pm.
 */
Ppolynome polynome_monome_mult(pp, pm)
Ppolynome pp;
Pmonome pm;
{
    Ppolynome curpp, ppdup;
    if (POLYNOME_UNDEFINED_P(pp) || MONOME_UNDEFINED_P(pm))
	return (POLYNOME_UNDEFINED);
    else if (POLYNOME_NUL_P(pp) || MONOME_NUL_P(pm))
	return (POLYNOME_NUL);
    else {
	ppdup = polynome_dup(pp);
	for (curpp = ppdup ; curpp != POLYNOME_NUL; curpp = polynome_succ(curpp))
	    monome_monome_mult(&(polynome_monome(curpp)), pm);
	return (ppdup);
    }
}


/* Ppolynome polynome_mult(Ppolynome pp1, Ppolynome pp2)
 *  returns pp1 * pp2.
 */
Ppolynome polynome_mult(pp1, pp2)
Ppolynome pp1, pp2;
{
    Ppolynome mp2 = POLYNOME_UNDEFINED;

    if (POLYNOME_UNDEFINED_P(pp1) || POLYNOME_UNDEFINED_P(pp2))
	return (POLYNOME_UNDEFINED);
    if (POLYNOME_NUL_P(pp1) || POLYNOME_NUL_P(pp2))
	return (POLYNOME_NUL);
    else {
	Ppolynome pppartiel, ppresult = POLYNOME_NUL;

	for (mp2 = pp2 ; mp2 != POLYNOME_NUL; mp2 = polynome_succ(mp2)) {
	    pppartiel = polynome_monome_mult(pp1, polynome_monome(mp2));
	    polynome_add(&ppresult, pppartiel);
	    polynome_rm(&pppartiel);
	}
	return (ppresult);
    }
}

/*  Pmonome monome_monome_div(Pmonome pm1, Pmonome pm2)
 *  PRIVATE
 *  (pm1) = (pm1) / pm2.
 *  !usage: monome_monome_div(pm, pm2);
 *  Lei Zhou , 09/07/91
 */
Pmonome monome_monome_div(pm1, pm2)
Pmonome pm1, pm2;
{
    if (MONOME_UNDEFINED_P(pm1))
	return (MONOME_UNDEFINED);
    else if (MONOME_UNDEFINED_P(pm2)) {
	monome_rm(&pm1);
	return (MONOME_UNDEFINED);
    }
    else if (!MONOME_NUL_P(pm1)) {
	if (MONOME_NUL_P(pm2)) {
	    monome_rm(&pm1);	/* returns ppm pointing to MONOME_NUL */
	    return (MONOME_UNDEFINED);
	}
	else {
	    Pmonome pmr = new_monome();

	    if (MONOME_CONSTANT_P(pm2))
		monome_term(pmr) = vect_dup(monome_term(pm1));
	    else if (MONOME_CONSTANT_P(pm1)) {
		Pvecteur pv = vect_dup(monome_term(pm2));
		vect_chg_sgn(pv);
		monome_term(pmr) = pv;
		}
	    else {
		monome_term(pmr) = vect_substract(monome_term(pm1), monome_term(pm2));
		if ( monome_term(pmr) == VECTEUR_NUL )
		    monome_term(pmr) = vect_new(TCST, VALUE_ONE);
	    }
	    monome_coeff(pmr) = monome_coeff(pm1)/monome_coeff(pm2);

	    return (pmr);
	}
    }
    polynome_error("monome_monome_div","Unreachable...\n");
    return MONOME_UNDEFINED;
}

/* Ppolynome polynome_monome_div(Ppolynome pp, Pmonome pm)
 *  PRIVATE
 *  returns p = pp / pm.
 */
Ppolynome polynome_monome_div(pp, pm)
Ppolynome pp;
Pmonome pm;
{
    if (POLYNOME_UNDEFINED_P(pp) || MONOME_UNDEFINED_P(pm))
	return (POLYNOME_UNDEFINED);
    else if (POLYNOME_NUL_P(pp) || MONOME_NUL_P(pm))
	return (POLYNOME_NUL);
    else {
	Ppolynome ppresult = POLYNOME_NUL;
	Ppolynome curpp, ppdup;
	ppdup = polynome_dup(pp);

	for (curpp = ppdup ; curpp != POLYNOME_NUL; curpp = polynome_succ(curpp)) {
	    Pmonome pmtmp = monome_monome_div(polynome_monome(curpp), pm);
	    polynome_monome_add(&ppresult,pmtmp);
	}
	return (ppresult);
    }
}


/* Ppolynome polynome_div(Ppolynome pp1, Ppolynome pp2)
 *  returns p = pp1 / pp2.
 */
Ppolynome polynome_div(pp1, pp2)
Ppolynome pp1, pp2;
{
    if (POLYNOME_UNDEFINED_P(pp1) || POLYNOME_UNDEFINED_P(pp2)
	                          || POLYNOME_NUL_P(pp2)   )
	return (POLYNOME_UNDEFINED);
    if (POLYNOME_NUL_P(pp1))
	return (POLYNOME_NUL);
    else {
	Ppolynome ppresult;

	if (is_single_monome(pp2)) {
	    ppresult = polynome_monome_div(pp1, polynome_monome(pp2));
	}
	else {
	    fprintf(stdout,"The divider has at least two elements!\n");
	    exit(3);
	}
	return (ppresult);
    }
}
/*============================================================================*/
/* Ppolynome vecteur_to_polynome(Pvecteur pv): translates a Pvecteur into a
 * Ppolynome.
 */
Ppolynome vecteur_to_polynome(Pvecteur pv)
{
  Ppolynome pp;

  if(VECTEUR_NUL_P(pv))
    pp = POLYNOME_NUL;
  else {
    Pvecteur vec;

    pp = NULL;
    for(vec = pv; vec != NULL; vec = vec->succ) {
      Variable var = vecteur_var(vec);
      float val = VALUE_TO_FLOAT(vecteur_val(vec));
      Ppolynome newpp;

      newpp = make_polynome(val, var, VALUE_ONE);
      polynome_succ(newpp) = pp;
      pp = newpp;
    }
  }

  return(pp);
}
