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

/************************************************************* pnome-reduc.c
 *
 * REDUCTIONS ON POLYNOMIALS
 *
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
#include "polynome.h"

/* Ppolynome polynome_var_subst(Ppolynome pp, Variable var, Ppolynome ppsubst)
 *  creates and returns a Ppolynome copied from pp in which every occurrence of
 *  variable var is substituted by polynomial ppsubst, which must not contain var.
 */
Ppolynome polynome_var_subst(pp, var, ppsubst)
Ppolynome pp;
Variable var;
Ppolynome ppsubst;
{
    Ppolynome ppsubst_n, ppmult;
    Ppolynome newpp = POLYNOME_NUL;
    Pmonome curpm, newpm;
    int varpower;
    
    if (POLYNOME_UNDEFINED_P(pp))
	return (POLYNOME_UNDEFINED);
    else {
	for( ; pp != POLYNOME_NUL; pp = polynome_succ(pp)) {
	    curpm = polynome_monome(pp);
	    if ((varpower = (int) vect_coeff(var, monome_term(curpm))) == 0)
		polynome_monome_add(&newpp, curpm);
	    else {
	    	/* the monomial curpm contains the variable var.  */
		/* We duplicate it, remove variable var from it,  */
		/* we multiply it with ppsubst^n (where n was the */
		/* power of var), and we add the result to newpp. */
		
		if (POLYNOME_UNDEFINED_P(ppsubst))
		    return (POLYNOME_UNDEFINED);
		else {
		    newpm = monome_del_var(curpm, var);
		    ppsubst_n = polynome_power_n(ppsubst, varpower);
		    ppmult = polynome_monome_mult(ppsubst_n, newpm);
		    polynome_add(&newpp, ppmult);

		    polynome_rm(&ppmult);
		    polynome_rm(&ppsubst_n);
		    monome_rm(&newpm);
		}
	    }
	}
	return(newpp);
    }
}

/* int polynome_degree(Ppolynome pp, Variable var)
 *  returns the degree of polynomial pp viewed as a polynomial
 *  of one variable, var.
 *  If pp is POLYNOME_UNDEFINED: abort. [???]
 */
int polynome_degree(pp, var)
Ppolynome pp;
Variable var;
{
    int power, deg = 0;

    /* polynome_degree: polynome is undefined */
    assert(!POLYNOME_UNDEFINED_P(pp));
    for( ; pp != POLYNOME_NUL; pp = polynome_succ(pp)) {
	power = (int) vect_coeff(var, monome_term(polynome_monome(pp)));
	if (deg < power) deg = power;
    }
    return(deg);
}

/* int polynome_max_degree(Ppolynome pp)
 *  returns the degree of polynomial pp 
 *  Let's hope there aren't too many negative powers...
 *  If pp is POLYNOME_UNDEFINED: abort. [???]
 */
int polynome_max_degree(Ppolynome pp)
{
    int power, deg = 0;
    Ppolynome m = POLYNOME_UNDEFINED;

    /* polynome_degree: polynome is undefined */
    assert(!POLYNOME_UNDEFINED_P(pp));
    for(m = pp ; m != POLYNOME_NUL; m = polynome_succ(m)) {
	power = (int) vect_sum(monome_term(polynome_monome(m)));
	if (deg < power) deg = power;
    }
    return deg;
}


/* Ppolynome polynome_factorize(Ppolynome pp, Variable var, int n)
 *  returns the (polynomial) coefficient of var^n in polynomial pp
 */
Ppolynome polynome_factorize(pp, var, n)
Ppolynome pp;
Variable var;
int n;
{
    Ppolynome ppfact = POLYNOME_NUL;
    Pmonome pm;

    if (POLYNOME_UNDEFINED_P(pp))
	return (POLYNOME_UNDEFINED);
    else {
	for( ; pp != POLYNOME_NUL; pp = polynome_succ(pp))
	    if (n == (int) vect_coeff(var, monome_term(polynome_monome(pp)))) {
		pm = monome_del_var(polynome_monome(pp), var);
		polynome_monome_add(&ppfact, pm);
	    }

	return(ppfact);
    }
}

/* float polynome_TCST(Ppolynome pp)
 *  returns the constant term of polynomial pp.
 *  If pp is POLYNOME_UNDEFINED: abort. [???]
 */
float polynome_TCST(pp)
Ppolynome pp;
{
    Pmonome pm;
    Pvecteur pvTCST = vect_new((Variable) TCST, VALUE_ONE);

    /* polynome_TCST: polynome is undefined */
    assert(!POLYNOME_UNDEFINED_P(pp));
    for( ; pp != POLYNOME_NUL; pp = polynome_succ(pp) ) {
	pm = polynome_monome(pp);
	if (vect_equal(pvTCST, monome_term(pm)))
	    return(monome_coeff(pm));
    }

    vect_rm(pvTCST);
    return ((float) 0);

}

/* bool polynome_constant_p(Ppolynome pp)
 *  return true if pp is a constant polynomial
 *  (including null polynomial)
 *  If pp is POLYNOME_UNDEFINED: abort. [???]
 */
bool polynome_constant_p(pp)
Ppolynome pp;
{
    /* polynome_constant_p: polynome is undefined */
    assert(!POLYNOME_UNDEFINED_P(pp));

    if (POLYNOME_NUL_P(pp)) 
	return(true);
    else {
	Pvecteur pvTCST = vect_new((Variable) TCST, VALUE_ONE);
	bool b = (vect_equal(pvTCST, monome_term(polynome_monome(pp)))
		     && (polynome_succ(pp) == POLYNOME_NUL));

	vect_rm(pvTCST);
	return(b);
    }
}


/* Pbase polynome_used_var(Ppolynome pp, bool *is_inferior_var())
 *  PRIVATE
 *  Returns, in a Pbase, a list of the variables used in pp,
 *  sorted according to the function is_inferior_var()
 */
Pbase polynome_used_var(pp, is_inferior_var)
Ppolynome pp;
int (*is_inferior_var)(Pvecteur *, Pvecteur *);
{
    Pbase b = BASE_NULLE;
    Pbase b2 = BASE_NULLE;
    Ppolynome pm = POLYNOME_UNDEFINED;

    if (!POLYNOME_UNDEFINED_P(pp)) {
	for (pm = pp; !POLYNOME_NUL_P(pm); pm = polynome_succ(pm)) {
	    b2 = base_union(b, (Pbase) monome_term(polynome_monome(pm)));
	    b = b2;
	}
    
	/* FI: I do not understand what has been done here! (20/09/95) */
	/* b2 = (Pbase) vect_tri((Pvecteur) b, is_inferior_var); */
	/* FI: vect_compare() seems only good when Value=char * */
	/* b2 = (Pbase) vect_sort((Pvecteur) b, vect_compare); */
	b2 = (Pbase) vect_sort((Pvecteur) b, is_inferior_var);
	vect_rm((Pvecteur) b);
    }
    else {
	polynome_error("polynome_used_var", 
		       "POLYNOME_UNDEFINED out of domain\n");
	b2 = BASE_UNDEFINED;
    }
    return b2;
}


/* bool polynome_contains_var(Ppolynome pp, Variable var)
 *  PRIVATE
 *  returns true if variable var is in polynomial pp.
 */
bool polynome_contains_var(pp, var)
Ppolynome pp;
Variable var;
{
    if ( POLYNOME_UNDEFINED_P(pp) )
	return (false);
    else {
	for ( ; pp != POLYNOME_NUL; pp = polynome_succ(pp))
	    if (vect_coeff(var, monome_term(polynome_monome(pp))) != 0)
		return (true);
	return(false);
    }
}


/* bool polynome_equal(Ppolynome pp1, Ppolynome pp2)
 *  return (pp1 == pp2)
 *  >>>TO BE CONTINUED<<<
 */
bool polynome_equal(pp1, pp2)
Ppolynome pp1,pp2;
{
    Ppolynome ppcopy1 = polynome_dup(pp1);
    Ppolynome ppcopy2 = polynome_dup(pp2);

    polynome_sort(&ppcopy1, default_is_inferior_pvarval);
    polynome_sort(&ppcopy2, default_is_inferior_pvarval);
    

    /* TO BE CONTINUED */
    polynome_error ("polynome_equal", "To be implemented!\n");
    return false;
}
