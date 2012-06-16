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

/* Package sc
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <string.h>
#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Transform each equality in two inequalities
 */

void sc_transform_eg_in_ineg(sc)
Psysteme sc;
{

    Pcontrainte eg,pc1,pc2;

    for (eg = sc->egalites; 
	 !CONTRAINTE_UNDEFINED_P(eg); 
	 eg=eg->succ) 
    {
	pc1 = contrainte_dup(eg);
	pc2 = contrainte_dup(eg);
	vect_chg_sgn(pc2->vecteur);
	sc_add_ineg(sc,pc1);
	sc_add_ineg(sc,pc2);
    }
    
    sc->egalites = contraintes_free(sc->egalites);
    sc->nb_eq = 0;

    sc = sc_elim_db_constraints(sc); 
}


/* Transform the two constraints A.x <= b and -A.x <= -b of system sc into 
 * an equality  A.x == b.
 */
void sc_transform_ineg_in_eg(sc)
Psysteme sc;
{
    Pcontrainte pc1, pc2, pc1_succ, c_tmp;
    bool found;

    for (pc1 = sc->inegalites, pc1_succ = (pc1==NULL ? NULL : pc1->succ); 
	 !CONTRAINTE_UNDEFINED_P(pc1);
	 pc1 = pc1_succ, pc1_succ = (pc1==NULL ? NULL : pc1->succ))
    {
	for (pc2 = pc1->succ, found=false; 
	     !CONTRAINTE_UNDEFINED_P(pc2) && !found; 
	     pc2 = pc2->succ)
	{
	    Pvecteur pv = vect_add(pc1->vecteur, pc2->vecteur); 
	    
	    if (VECTEUR_NUL_P(pv)) /* True if the constraints are opposed. */
	    {
		c_tmp = contrainte_dup(pc1);

		sc_add_eg(sc, c_tmp);
		eq_set_vect_nul(pc2);
		eq_set_vect_nul(pc1);
		found = true;
	    }
	    
	    vect_rm(pv);
	}
    } 

    sc = sc_elim_double_constraints(sc);
    /* ??? humm, the result is *not* returned */
}

/* sc_find_equalities(Psysteme * ps)
 *
 * what: updates *ps to an equivalent system with more/better equalities.
 * in/out: *ps
 * how:   (1) Maslov PLDI'92;   (2) (a <= x <= a) => (x == a)
 *
 * (1) Inspired by Vadim Maslov, PLDI'92 "delinearization". It is based on
 * the following property (a 1 page theorem and proof in his paper):
 * 
 * Prop: forall a,n,m in Z, a>0, |m|<a, (an+m=0  <=>  n=0, m=0)
 * Proof: [<=] trivial; [=>] a|n|=|m|<a => |n|<1 => n=0 => m=0;
 *
 * From this property, typical of dependence tests on linearized accesses,
 * two simple equations are derived from a bigger one and inequalities.
 * Our purpose is to identify the simpler equalities and to substitute
 * them in the system, not to perform an actual dependence test.
 *
 * The issue is to identify candidates n and m in equalities, so as to test
 * the bounds on m and match the property conditions. We will not assume
 * cartesian constraints on the variables, but rather compute it by projection.
 * We will use the gcd condition before projecting to find m bounds. 
 * By doing so, we may lose some instances if a variable is in fact a constant.
 *
 * FC, Apr 09 1997.
 */
static int abscmp(Pvecteur * pv1, Pvecteur * pv2)
{
    Value v1 = value_abs(val_of(*pv1)), v2 = value_abs(val_of(*pv2));
    return value_compare(v1, v2);
}

void sc_find_equalities(Psysteme * ps)
{
    Variable vtmp = (Variable) "local temporary variable";
    Pcontrainte eq;
 
    sc_transform_ineg_in_eg(*ps);

    /* FOR EACH EQUALITY
     */
    for (eq = sc_egalites(*ps); eq; eq = eq->succ)
    {
	Pvecteur m, an;
	Value c, a;
	
	an = vect_dup(contrainte_vecteur(eq));
	m = vect_new(vtmp, VALUE_MONE);

	c = vect_coeff(TCST, an);
	vect_erase_var(&an, TCST);

	/* the vector is sorted by increassing absolute coeffs.
	 * the the m/an separation is performed on a per-abs(coeff) basis.
	 */
	vect_sort_in_place(&an, abscmp);
	
	/* BUILD POSSIBLE AN AND M (AN + M + C == 0)
	 */
	while (!VECTEUR_NUL_P(an))
	{
	    Value refc, coeff, vmin, vmax, d, r, dp, rp, delta, ma;
	    Psysteme ns;

	    /* accumulate next absolute coeff in m 
	     */
	    refc = value_abs(val_of(an));

	    do
	    {
		Pvecteur v = m; /* insert head(an) in front of m */
		m = an;
		an = an->succ;
		m->succ = v;
		coeff = VECTEUR_NUL_P(an)? VALUE_ZERO: value_abs(val_of(an));
	    } 
	    while (value_eq(coeff, refc));

	    /* WITH AN, COMPUTES A AND CHECK A "GCD" CONDITION ???
	     */
	    if (VECTEUR_NUL_P(an))
		continue; /* break... */

	    a = vect_pgcd_all(an); 
	    if (value_ge(refc, a))
		continue; /* to WHILE(an) */

	    /* COMPUTE M BOUNDS, IF ANY
	     */
	    ns = sc_dup(*ps);
	    sc_add_egalite(ns, contrainte_make(vect_dup(m)));
	    base_rm(sc_base(ns));
	    sc_creer_base(ns);

	    if (!sc_minmax_of_variable(ns, vtmp, &vmin, &vmax)) /* kills ns */
	    {
		/* the system is not feasible. */
		vect_rm(m); 
		vect_rm(an);
		sc_rm(*ps);
		*ps = sc_empty(BASE_NULLE);
		return;
	    }

	    if (value_min_p(vmin) || value_max_p(vmax))
	    {
		/* well, if m is not bounded, the larger m won't be either,
		 * thus we can shorten the an loop...
		 */
		vect_rm(an), an = VECTEUR_NUL;
		vect_rm(m), m = VECTEUR_NUL;
		continue; /* break... */
	    }

	    /* now, we must compute the shifts...
	     *
	     * an + m + c == 0,   vmin <= m <= vmax ;
	     * c = a d + r,  0 <= r < a, 
	     * a (n+d) + (m+r) == 0,   vmin+r <= m+r <= vmax+r,
	     * (vmax+r) = a d' + r', 0 <= r' < a,
	     * vmin+r-ad' <= (m+r-ad') <= r' < a
	     * 
	     * question: -a < vmin+r-ad ?
	     * if YES: m+r-ad'==0 and n+d+d'==0
	     * assert: vmin+r-ad<=0 (or not feasible!)
	     */

	    /* c = a d + r 
	     */
	    d = value_pdiv(c, a); 
	    r = value_pmod(c, a); 

	    value_addto(vmax, r);
	    value_addto(vmin, r);
	    
	    /* vmax' = a dp + rp
	     */
	    dp = value_pdiv(vmax, a); 
	    rp = value_pmod(vmax, a);

	    /* delta = -ad'
	     */
	    delta = value_minus(rp, vmax); 
	    vmax = rp;
	    value_addto(vmin, delta);
	    ma = value_uminus(a);
	    
	    if (value_lt(ma, vmin)) /* CONDITION |m|<a */
	    {
		Value n_shift, m_shift;

		Pvecteur n = vect_div(an, a); /* an modified */
		n_shift = value_plus(d,dp);
		vect_add_elem(&n, TCST, n_shift);
		
		vect_erase_var(&m, vtmp);
		m_shift = value_plus(r, delta);
		vect_add_elem(&m, TCST, m_shift);

		/* INSERT m==0 [ahead] and n==0 [in place of the old one]
		 */
		sc_add_egalite(*ps, contrainte_make(m)); /* m pointed to */
		
		vect_rm(contrainte_vecteur(eq));
		contrainte_vecteur(eq) = vect_dup(n);

		/* KEEPS ON WITH N AND resetted M
		 */
		m = vect_new(vtmp, VALUE_MONE);
		an = n;
	    }
	}

	/* an == VECTEUR_NUL */
	vect_rm(m), m=VECTEUR_NUL;
    }
}

/* that is all
 */
