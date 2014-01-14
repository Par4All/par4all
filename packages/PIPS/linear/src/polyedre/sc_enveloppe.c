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

 /* package polyedre: enveloppe convexe de deux systemes lineaires
  *
  * Ce module est range dans le package polyedre bien qu'il soit utilisable
  * en n'utilisant que des systemes lineaires (package sc) parce qu'il
  * utilise lui-meme des routines sur les polyedres.
  *
  * Francois Irigoin, Janvier 1990
  * Corinne Ancourt, Fabien Coelho from time to time (1999/2000)
  */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"

#include "polyedre.h"

/* Psysteme sc_enveloppe_chernikova_ofl_ctrl(Psysteme s1, s2, int ofl_ctrl)
 * input    : 
 * output   : 
 * modifies : s1 and s2 are NOT modified.
 * comment  : s1 and s2 must have a basis.
 * 
 * s = enveloppe(s1, s2);
 * return s;
 *
 * calcul d'une representation par systeme
 * lineaire de l'enveloppe convexe des polyedres definis par les systemes
 * lineaires s1 et s2
 */
Psysteme sc_enveloppe_chernikova_ofl_ctrl(s1, s2, ofl_ctrl)
Psysteme s1;
Psysteme s2;
int ofl_ctrl;
{
    Psysteme s = SC_UNDEFINED;
    bool catch_performed = false;
    /* mem_spy_begin(); */

    assert(!SC_UNDEFINED_P(s1) && !SC_UNDEFINED_P(s2));

    switch (ofl_ctrl) 
    {
    case OFL_CTRL :
	ofl_ctrl = FWD_OFL_CTRL;
	catch_performed = true;
	CATCH(overflow_error|timeout_error) {
	  //CATCH(overflow_error) {
	    /* 
	     *   PLEASE do not remove this warning.
	     *
	     *   BC 24/07/95
	     */
	    fprintf(stderr, "[sc_enveloppe_chernikova_ofl_ctrl] "
		    "arithmetic error occured\n" );
	    s = sc_rn(base_dup(sc_base(s1)));
	    break;
	}		
    default: {
    
	    if (SC_RN_P(s2) || sc_rn_p(s2) || sc_dimension(s2)==0
		|| sc_empty_p(s1) || !sc_faisabilite_ofl(s1)) 
	    {
		Psysteme sc2 = sc_dup(s2);
		sc2 = sc_elim_redond(sc2);
		s = SC_UNDEFINED_P(sc2)? sc_empty(base_dup(sc_base(s2))): 
		    sc2;
	    }
	    else 
		if (SC_RN_P(s1) ||sc_rn_p(s1) || sc_dimension(s1)==0   
		    || sc_empty_p(s2) || !sc_faisabilite_ofl(s2)) 
		{
		    Psysteme sc1 = sc_dup(s1);
		    sc1 = sc_elim_redond(sc1);
		    s = SC_UNDEFINED_P(sc1)? 
			sc_empty(base_dup(sc_base(s1))): sc1;
		}
		else 
		{
		    /* calcul de l'enveloppe convexe */
		    s = sc_convex_hull(s1,s2);
		    /* printf("systeme final \n"); sc_dump(s);  */
		}
	    if (catch_performed)
		UNCATCH(overflow_error|timeout_error);
	}
    }
    /* mem_spy_end("sc_enveloppe_chernikova_ofl_ctrl"); */
    return s;
}

Psysteme sc_enveloppe_chernikova(Psysteme s1, Psysteme s2)
{
  return sc_enveloppe_chernikova_ofl_ctrl((s1), (s2), OFL_CTRL);
} 

/* call chernikova with compatible base.
 */
static Psysteme actual_convex_union(Psysteme s1, Psysteme s2)
{
  Psysteme s;

  /* same common base! */
  Pbase b1 = sc_base(s1), b2 = sc_base(s2), bu = base_union(b1, b2);
  int d1 = sc_dimension(s1), d2 = sc_dimension(s2), du = vect_size(bu);

  sc_base(s1) = bu;
  sc_dimension(s1) = du;
  sc_base(s2) = bu;
  sc_dimension(s2) = du;

  /* call chernikova directly.
     sc_common_projection_convex_hull improvements have already been included.
  */
  s = sc_enveloppe_chernikova(s1, s2);

  /* restaure initial base */
  sc_base(s1) = b1;
  sc_dimension(s1) = d1;
  sc_base(s2) = b2;
  sc_dimension(s2) = d2;

  base_rm(bu);

  return s;
}

/* implements FC basic idea of simple fast cases...
 * returns s1 v s2. 
 * s1 and s2 are not touched.
 * The bases should be minimum for best efficiency!
 * otherwise useless columns are allocated and computed.
 * a common base is rebuilt in actual_convex_union.
 * other fast cases may be added?
 */
Psysteme elementary_convex_union(Psysteme s1, Psysteme s2)
{
  bool
    b1 = sc_empty_p(s1),
    b2 = sc_empty_p(s2);
    
  if (b1 && b2)
    return sc_empty(base_union(sc_base(s1), sc_base(s2)));

  if (b1) {
    Psysteme s = sc_dup(s2);
    Pbase b = base_union(sc_base(s1), sc_base(s2));
    base_rm(sc_base(s));
    sc_base(s) = b;
    return s;
  }
  
  if (b2) {
    Psysteme s = sc_dup(s1);
    Pbase b = base_union(sc_base(s1), sc_base(s2));
    base_rm(sc_base(s));
    sc_base(s) = b;
    return s;
  }
  
  if (sc_rn_p(s1) || sc_rn_p(s2) || 
      !vect_common_variables_p(sc_base(s1), sc_base(s2)))
    return sc_rn(base_union(sc_base(s1), sc_base(s2)));

  /*
  if (sc_dimension(s1)==1 && sc_dimension(s2)==1 && 
      sc_base(s1)->var == sc_base(s2)->var)
  {
    // fast computation...
  }
  */

  return actual_convex_union(s1, s2);
}


/********************************************** SET BASED TRANSITIVE CLOSURE */

/* put base variables in set.
   returns whether something was put.
 */
static bool base_to_set(linear_hashtable_pt s, Pvecteur b)
{
  bool modified = false;

  for (; b; b=b->succ)
    if (b->var && !linear_hashtable_isin(s, b->var)) 
    {
      linear_hashtable_put(s, b->var, b->var);
      modified = true;
    }

  return modified;
}

/* returns whether c contains variables of vars. */
static bool contains_variables(Pvecteur v, linear_hashtable_pt vars)
{
  for (; v; v = v->succ)
    if (v->var && linear_hashtable_isin(vars, v->var))
      return true;
  return false;
}

/* one pass only of transitive closure.
 * returns whether vars was modified.
 * appends extracted constraints to ex.
 */
static bool 
transitive_closure_pass(Pcontrainte * pc, Pcontrainte * ex, 
			linear_hashtable_pt vars)
{
  Pcontrainte c, cp, cn;
  bool modified = false;

  for (c=*pc, 
	 cp = CONTRAINTE_UNDEFINED, 
	 cn = c? c->succ: CONTRAINTE_UNDEFINED; 
       c;
       cp = c==cn? cp: c,
	 c = cn, 
	 cn = c? c->succ: CONTRAINTE_UNDEFINED)
  {
    if (contains_variables(c->vecteur, vars)) 
    {
      modified |= base_to_set(vars, c->vecteur);
      c->succ = *ex, *ex = c;
      if (cp) cp->succ = cn; else *pc = cn;
      c = cn;
    }
  }

  return modified;
}

/* transtitive extraction of constraints.
 */
static Psysteme transitive_closure_system(Psysteme s, linear_hashtable_pt vars)
{
  Pcontrainte e = CONTRAINTE_UNDEFINED, i = CONTRAINTE_UNDEFINED;
  bool modified;

  do {
    modified = transitive_closure_pass(&s->egalites, &e, vars);
    modified |= transitive_closure_pass(&s->inegalites, &i, vars);
  }
  while (modified);

  sc_fix(s);
  return sc_make(e, i);
}

/* returns constraints from s which may depend on variables in b1 and b2.
 * these constraints are removed from s, hence s is modified.
 */
static Psysteme
transitive_closure_from_two_bases(Psysteme s, Pbase b1, Pbase b2)
{
  Psysteme st;
  linear_hashtable_pt vars = linear_hashtable_make();

  base_to_set(vars, b1);
  base_to_set(vars, b2);
  st = transitive_closure_system(s, vars);
  linear_hashtable_free(vars);

  return st;
}

/*********************************************** HOPEFULLY CUTE CONVEX UNION */

/* returns s1 v s2.
 * initial systems are not changed.
 * 
 * v convex union
 * u union
 * n intersection
 * T orthogonality
 * 
 * (1) CA: 
 * 
 * P1 v P2 = (P n X1) v (P n X2) = 
 * let P = P' n P'' 
 *   so that P' T X1 and P' T X2 and P' T P'' built by transitive closure,
 * then P1 v P2 = (P' n (P'' n X1)) v (P' n (P'' n X2)) =
 *                 P' n ((P'' n X1) v (P'' n X2))
 *
 * Proof by considering generating systems:
 * Def: A T B <=> var(A) n var(B) = 0
 * Prop: A n B if A T B
 *  Let A = (x,v,l), B = (y,w,m)
 *  A n B = { z | z = (x) \mu + (v) d + (l) e + (0) f
 *                    (0)     + (0)   + (0)     (I)
 *            and z = (0) \nu + (0) g + (0) h + (I) f'
 *                    (y)     + (w)   + (m)     (0)
 *            with \mu>0, \nu>0, \sum\mu=1, \sum\nu=1, d>0, g>0 }
 *  we can always find f and f' equals to the other part (by extension) so
 *  A n B = { z | z = (x 0) \mu + (v 0) d + (l 0) e
 *                    (0 y) \nu   (0 w) g   (0 m) h 
 *            with \mu \nu d g constraints... }
 *  It is a convex : ((xi)  , (v 0), (l 0))
 *                    (yj)ij, (0 w), (0 m)
 *  we just need to prove that Cmn == Cg defined as
 *    (x 0) \mu  ==  (xi)   \gamma with >0 and \sum = 1
 *    (0 y) \nu      (yj)ij
 *  . Cg in a convex.
 *  . Cg \in Cmn since ((xi)\(yj) ij) in Cmn 
 *    with \mu = \delta i, \nu = \delta j
 *  . Cmn \in Cg by chosing \gamma_ij = \mu_i\nu_j, which >0 and \sum = 1
 * Prop: A T B and A T C => A T (B v C) and A T (B n C)
 * Theo: A T B and A T C => (A n B) v (A n C) = A n (B v C)
 *   compute both generating systems with above props. they are equal.
 *
 * (2) FI:
 *
 * perform exact projections of common equalities.
 * no proof at the time. It looks ok anyway.
 * 
 * (3) FC:
 * 
 * base(P) n base(Q) = 0 => P u Q = Rn
 * and some other very basic simplifications...
 * which are not really interesting if no decomposition is performed?
 *
 * on overflows FI suggested.
 * 
 * 1/ we want to compute : (A n B1) u (A n B2)
 * 2/ we chose to approximate it as : (A n B1) v (A n B2)
 * 3/ we try Corinne's factorization with transitive closure
 *    A' n ((A'' n B1) v (A'' n B2))
 * 4/ on overflow, we can try A n (B1 v B2) // NOT IMPLEMENTED YET
 * 5/ if we have one more overflow, then A looks good as an approximation.
 */
Psysteme sc_cute_convex_hull(Psysteme is1, Psysteme is2)
{
  Psysteme s1, s2, sc, stc, su, scsaved;
  int current_overflow_count;

  s1 = sc_dup(is1);
  s2 = sc_dup(is2);

  /* CA: extract common disjoint part.
   */
  sc = extract_common_syst(s1, s2);
  scsaved = sc_dup(sc);
  stc = transitive_closure_from_two_bases(sc, s1->base, s2->base);

  /* FI: in sc_common_projection_convex_hull
     note that equalities are not that big a burden to chernikova?
   */
  sc_extract_exact_common_equalities(stc, sc, s1, s2);

  /* fast sc_append */
  s1 = sc_fusion(s1, sc_dup(stc));
  sc_fix(s1);

  s2 = sc_fusion(s2, stc);
  sc_fix(s2);

  stc = NULL;

  current_overflow_count = linear_number_of_exception_thrown;

  su = elementary_convex_union(s1, s2);
  
  /* usually we use V (convex hull) as a U (set union) approximation.
   * as we have : (A n B1) U (A n B2) \in A
   * the common part of both systems is an approximation of the union!
   * sc_rn is returned on overflows (and some other case).
   * I don't think that the result is improved apart 
   * when actual overflow occurs. FC/CA.
   */
  if (current_overflow_count!=linear_number_of_exception_thrown)
  {
    if (su) sc_rm(su), su = NULL;
    if (sc) sc_rm(sc), sc = NULL;
    sc = scsaved;
  }
  else
  {
    sc_rm(scsaved);
  }

  scsaved = NULL; /* dead. either rm or moved as sc. */
  sc_rm(s1); 
  sc_rm(s2);

  /* better compatibility with previous version, as the convex union
   * normalizes the system and removes redundancy, what is not done
   * if part of the system is separated. Other calls may be considered here?
   */
  sc_transform_ineg_in_eg(sc); /* ok, it will look better. */
  /* misleading... not really projected. { x==y, y==3 } => { x==3, y==3 } */
  sc_project_very_simple_equalities(sc);

  /* sc, su: fast union of disjoint. */
  sc = sc_fusion(sc, su);

  /* regenerate the expected base. */
  if (sc) 
  {
    sc_fix(sc);
    if (sc_base(sc)) base_rm(sc_base(sc));
  }
  else
  {
      sc = sc_rn(NULL);
  }
  sc_base(sc) = base_union(sc_base(is1), sc_base(is2));
  sc_dimension(sc) = vect_size(sc_base(sc));
  
  return sc;
}

/* take the rectangular bounding box of the systeme @p sc,
 * by projecting each constraint of the systeme against each of the basis
 * in @p pb
 *
 * SG: this is basically a renaming of sc_projection_on_variables ...
 */
Psysteme sc_rectangular_hull(Psysteme sc, Pbase pb) {
    volatile Psysteme rectangular = SC_UNDEFINED;
    rectangular = sc_projection_on_variables(sc,pb,pb);
    CATCH(overflow_error) {
        ;
        // it does not matter if we fail ...
    }
    TRY {
        sc_nredund(&rectangular);
        UNCATCH(overflow_error);
    }
    return rectangular;
}
