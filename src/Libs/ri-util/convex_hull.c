/*
 * $Id$
 *
 * Functions which should be in linear.
 * Here because we're using some newgen stuff.
 */

#include <stdio.h>

#include "linear.h"
#include "genC.h"   /* for set */

/*************************************** NEWGEN SET BASED TRANSITIVE CLOSURE */

/* put base variables in set.
   returns whether something was put.
 */
static boolean base_to_set(set s, Pvecteur b)
{
  boolean modified = FALSE;

  for (; b; b=b->succ)
    if (b->var && !set_belong_p(s, b->var)) 
    {
      (void) set_add_element(s, s, b->var);
      modified = TRUE;
    }

  return modified;
}

/* returns whether c contains variables of vars. */
static boolean contains_variables(Pvecteur v, set vars)
{
  for (; v; v = v->succ)
    if (v->var && set_belong_p(vars, v->var))
      return TRUE;
  return FALSE;
}

/* one pass only of transitive closure.
 * returns whether vars was modified.
 * appends extracted constraints to ex.
 */
static boolean 
transitive_closure_pass(Pcontrainte * pc, Pcontrainte * ex, set vars)
{
  Pcontrainte c, cp, cn;
  boolean modified = FALSE;

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
static Psysteme transitive_closure_system(Psysteme s, set vars)
{
  Pcontrainte e = CONTRAINTE_UNDEFINED, i = CONTRAINTE_UNDEFINED;
  boolean modified;

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
  set vars = set_make(set_pointer);

  base_to_set(vars, b1);
  base_to_set(vars, b2);
  st = transitive_closure_system(s, vars);
  set_free(vars);

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
 * Prop: A T B and A T C => A T (B v C)
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
 */
Psysteme sc_cute_convex_hull(Psysteme is1, Psysteme is2)
{
  Psysteme s1, s2, sc, stc, su;

  s1 = sc_dup(is1);
  s2 = sc_dup(is2);

  /* CA: extract common disjoint part.
   */
  sc = extract_common_syst(s1, s2);
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

  /* how useful it is:
  fprintf(stderr, "common eg=%d/%d, in=%d/%d\n",
	  sc_nbre_egalites(sc), 
	  sc_nbre_egalites(sc)+sc_nbre_egalites(s1)+sc_nbre_egalites(s2),
	  sc_nbre_inegalites(sc),
	 sc_nbre_inegalites(sc)+sc_nbre_inegalites(s1)+sc_nbre_inegalites(s2));
  */

  su = elementary_convex_union(s1, s2);
  // su = actual_convex_union(s1, s2);

  sc_rm(s1); 
  sc_rm(s2);

  /* sc, su union of disjoint */
  sc = sc_fusion(sc, su);

  sc_fix(sc);
  if (sc_base(sc)) base_rm(sc_base(sc));
  sc_base(sc) = base_union(sc_base(is1), sc_base(is2));
  sc_dimension(sc) = vect_size(sc_base(sc));
  
  return sc;
}

/********************************************************************** PIPS */

#include "ri.h"
#include "ri-util.h"
#include "misc.h"

Psysteme cute_convex_union(Psysteme s1, Psysteme s2)
{
  Psysteme s;

  ifdebug(9) {
    pips_debug(9, "IN\nS1 and S2:\n");
    sc_fprint(stderr, s1, entity_local_name);
    sc_fprint(stderr, s2, entity_local_name);
  }

  s = sc_cute_convex_hull(s1, s2);

  ifdebug(9) {
    pips_debug(9, "S =\n");
    sc_fprint(stderr, s, entity_local_name);
    pips_debug(9, "OUT\n");
  }

  return s;
}
