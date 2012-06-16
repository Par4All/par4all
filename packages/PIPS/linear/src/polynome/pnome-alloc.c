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

/***************************************************************** pnome-alloc.c
 *
 * CREATING, DUPLICATING AND FREEING A POLYNOME
 *
 */

/*LINTLIBRARY*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "polynome.h"

/* allocation of an unitialized monome (to avoid various direct
 * unchecked call to malloc)
 */
Pmonome new_monome()
{
    Pmonome pm = (Pmonome) malloc(sizeof(Smonome));
    if (pm == NULL) {
	(void) fprintf(stderr,"new_monome: Out of memory space\n");
	/* (void) fprintf(stderr, "%10.3f MB", 
		       (sbrk(0) - etext)/(double)(1 << 20)); */
	abort();
	/*exit(-1);*/
    }
    return pm;
}

/* allocation of an unitialized polynome (to avoid various direct
 * unchecked call to malloc)
 */
Ppolynome new_polynome()
{
    Ppolynome pp = (Ppolynome) malloc(sizeof(Spolynome));
    if (pp == NULL) {
	(void) fprintf(stderr,"new_polynome: Out of memory space\n");
	/* (void) fprintf(stderr, "%10.3f MB", 
	   (sbrk(0) - etext)/(double)(1 << 20)); */
	abort();
	/*exit(-1);*/
    }
    return pp;
}

/* Pmonome make_monome(float coeff, Variable var, Value exp)
 *  PRIVATE
 *  allocates space for, and creates, the monome "coeff*var^exp" 
 */
Pmonome make_monome(coeff, var, exp)
float coeff;
Variable var;
Value exp;
{
    if (coeff == 0)
	return (MONOME_NUL);
    else {
	Pmonome pm = new_monome();
	monome_coeff(pm) = coeff;
	if (value_zero_p(exp))
	    monome_term(pm) = vect_new(TCST, VALUE_ONE);
	else
	    monome_term(pm) = vect_new(var, exp);

	return(pm);
    }
}

/* Ppolynome make_polynome(float coeff, Variable var, Value exp)
 *  PRIVATE
 *  allocates space for, and creates, the polynome "coeff*var^exp"
 */
Ppolynome make_polynome(coeff, var, exp)
float coeff;
Variable var;
Value exp;
{
    Pmonome m = make_monome(coeff, var, exp);
    Ppolynome p = monome_to_new_polynome(m);
    return p;
}


/* Ppolynome monome_to_new_polynome(Pmonome pm)
 *  PRIVATE
 *  allocates space for, and creates the polynomial containing
 *  the monomial pointed by pm, which is NOT duplicated
 *  but attached to the polynomial.
 */
Ppolynome monome_to_new_polynome(pm)
Pmonome pm;
{
    if (MONOME_NUL_P(pm)) 
	return (POLYNOME_NUL);
    else if (MONOME_UNDEFINED_P(pm)) 
	return (POLYNOME_UNDEFINED);
    else {
	Ppolynome pp = new_polynome();
	polynome_monome(pp) = pm;
	polynome_succ(pp) = POLYNOME_NUL;
	return (pp);
    }
}

/* Pmonome monome_dup(Pmonome pm)
 *  PRIVATE
 *  creates and returns a copy of pm
 */
Pmonome monome_dup(pm)
Pmonome pm;
{
    if (MONOME_NUL_P(pm)) 
	return (MONOME_NUL);
    else if (MONOME_UNDEFINED_P(pm)) 
	return (MONOME_UNDEFINED);
    else {
	Pmonome pmd = new_monome();
	monome_coeff(pmd) = monome_coeff(pm);
	monome_term(pmd) = vect_dup(monome_term(pm));
	return(pmd);
    }
}


/* void monome_rm(Pmonome* ppm)
 *  PRIVATE
 *  frees space occupied by monomial *ppm
 *  returns *ppm pointing to MONOME_NUL
 *  !usage: monome_rm(&pm);
 */
void monome_rm(ppm)
Pmonome *ppm;
{
    if ((!MONOME_NUL_P(*ppm)) && (!MONOME_UNDEFINED_P(*ppm))) {
	vect_rm((Pvecteur) monome_term(*ppm));
	free((char *) *ppm);
    }
    *ppm = MONOME_NUL;
}


/* void polynome_rm(Ppolynome* ppp)
 *  frees space occupied by polynomial *ppp
 *  returns *ppp pointing to POLYNOME_NUL
 *  !usage: polynome_rm(&pp);
 */
void polynome_rm(ppp)
Ppolynome *ppp;
{
    Ppolynome pp1 = *ppp, pp2;

    if (!POLYNOME_UNDEFINED_P(*ppp)) {
	while (pp1 != POLYNOME_NUL) {
	    pp2 = polynome_succ(pp1);
	    monome_rm(&polynome_monome(pp1));
	    free((char *) pp1);               /* correct? */
	    pp1 = pp2;
	}
	*ppp = POLYNOME_NUL;
    }
}

/* Ppolynome polynome_free(Ppolynome pp)
 *  frees space occupied by polynomial pp
 *  returns pp == POLYNOME_NUL
 *  !usage: polynome_rm(pp);
 */
Ppolynome polynome_free(pp)
Ppolynome pp;
{
    Ppolynome pp1 = pp, pp2;

    if (!POLYNOME_UNDEFINED_P(pp)) {
	while (pp1 != POLYNOME_NUL) {
	    pp2 = polynome_succ(pp1);
	    monome_rm(&polynome_monome(pp1));
	    free((char *) pp1);               /* correct? */
	    pp1 = pp2;
	}
    }
    return POLYNOME_NUL;
}


/* Ppolynome polynome_dup(Ppolynome pp)
 *  creates and returns a copy of pp
 */
Ppolynome polynome_dup(pp)
Ppolynome pp;
{
    Ppolynome ppdup, curpp;

    if (POLYNOME_NUL_P(pp)) 
	return (POLYNOME_NUL);
    else if (POLYNOME_UNDEFINED_P(pp)) 
	return (POLYNOME_UNDEFINED);
    else {
	ppdup = monome_to_new_polynome(monome_dup(polynome_monome(pp)));
	curpp = ppdup;
	while ((pp = polynome_succ(pp)) != POLYNOME_NUL) {
	    polynome_succ(curpp) =
		monome_to_new_polynome(monome_dup(polynome_monome(pp)));
	    curpp = polynome_succ(curpp);
	}
	return (ppdup);
    }
}
