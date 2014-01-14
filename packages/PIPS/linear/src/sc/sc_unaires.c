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

 /* package sur les systemes de contraintes sc
  *
  * Yi-Qing YANG, 20/05/92
  */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_elim_var(Psysteme sc, Variable v)`
 * This function eliminate all the contraints containing the variable v
 */
Psysteme sc_elim_var(Psysteme sc, Variable v)
{
  Pcontrainte pr = NULL, eq;

  if (sc == NULL) return sc;
  else {
    eq = sc->egalites;
    while (eq != NULL) {
	    if (vect_coeff(v,eq->vecteur) != 0) {
        if (pr == NULL) {
          sc->egalites = eq = eq->succ;
        }
        else {
          pr->succ = eq = eq->succ;
        }
	    }
	    else {
        pr = eq;
        eq = eq->succ;
	    }
    }

    eq = sc->inegalites;
    pr = NULL;
    while (eq != NULL) {
	    if (vect_coeff(v,eq->vecteur) != 0) {
        if (pr == NULL) {
          sc->inegalites = eq = eq->succ;
        }
        else {
          pr->succ = eq = eq->succ;
        }
	    }
	    else {
        pr = eq;
        eq = eq->succ;
	    }
    }

    sc->nb_eq = nb_elems_list(sc->egalites);
    sc->nb_ineq = nb_elems_list(sc->inegalites);
    return sc;
  }
}

/* void sc_chg_var(Psysteme s, Variable v_old, Variable v_new)
 * this function replace the variable v_old in the system s by the
 * variable v_new.
 */
void sc_chg_var(Psysteme s, Variable v_old, Variable v_new)
{
  Pcontrainte pc;
  if (s != NULL){
    for (pc = s->egalites; pc != NULL; pc = pc->succ) {
	    vect_chg_var(&(pc->vecteur), v_old, v_new);
    }

    for (pc = s->inegalites; pc != NULL; pc = pc->succ) {
	    vect_chg_var(&(pc->vecteur), v_old, v_new);
    }
  }
}

/* the name is self explanatory, I guess. FC 24/11/94
 * the vectors of the system are sorted.
 * see man qsort about the compare functions.
 */
void sc_vect_sort(Psysteme s, int (*compare)(Pvecteur *, Pvecteur *))
{
  if (s==NULL || sc_empty_p(s) || sc_rn_p(s))
    return;

  contrainte_vect_sort(sc_egalites(s), compare);
  contrainte_vect_sort(sc_inegalites(s), compare);
}

/* SORT a Psysteme according to sort_base and compare (given to qsort).
 * Each constraint is first sorted according to the compare function.
 * Then list of constraints are sorted.
 *
 * The only expected property
 * is that two calls to this function with the same system (whatever its
 * order) and same sort_base that covers all variables and same compare
 * function should give the same result.
 *
 * The function is quite peculiar and the order is relevant for some
 * code generation issues...
 */
void sc_sort(
    Psysteme sc,
    Pbase sort_base,
    int (*compare)(Pvecteur*, Pvecteur*))
{
  sc_vect_sort(sc, compare);
  sc->inegalites =
    contrainte_sort(sc->inegalites, sc->base, sort_base, true, true);
  sc->egalites =
    contrainte_sort(sc->egalites, sc->base, sort_base, true, true);
}

static bool vect_printout_order_decided_p(Pvecteur v)
{
  bool decided_p = false;
  int positive_terms = 0;
  int negative_terms = 0;
  Pvecteur coord = VECTEUR_UNDEFINED;

  for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
    if(vecteur_var(coord)!= TCST)
      (value_pos_p(vecteur_val(coord))) ?
        positive_terms++ :  negative_terms++;
  }
  /* constant vectors are considered decided */
  decided_p = ((positive_terms!=negative_terms) ||
               (positive_terms==0 && negative_terms==0));

  return decided_p;
}

/* Try to guess the print out order for an equality already
   lexicographically sorted */
Pvecteur vect_printout_order(Pvecteur v,
                             int (*compare)(Pvecteur *, Pvecteur *))
{
  Pvecteur v_neg = VECTEUR_NUL;
  Pvecteur cc = VECTEUR_UNDEFINED;
  Pvecteur succ = VECTEUR_UNDEFINED;

  for(cc = v; !VECTEUR_NUL_P(cc); cc = succ) {
    succ = vecteur_succ(cc); /* before cc might be freed */
    if(vecteur_val(cc) < 0 || vecteur_var(cc)==TCST) {
      vect_add_elem(&v_neg, vecteur_var(cc), vecteur_val(cc));
      vect_erase_var(&v, vecteur_var(cc));
    }
  }

  vect_sort_in_place(&v, compare);
  vect_sort_in_place(&v_neg, compare);

  /* append v_neg to v */
  for(cc=v; !VECTEUR_NUL_P(cc); cc = vecteur_succ(cc)) {
    if(VECTEUR_NUL_P(vecteur_succ(cc))) {
      vecteur_succ(cc) = v_neg;
      break; /* do not follow v_neg for ever */
    }
  }

  return v;
}

/* Minimize first the lexico-graphic weight of each constraint according to
 * the comparison function "compare", and then sort the list of equalities
 * and inequalities by increasing lexico-graphic weight.
 *
 * Francois Irigoin
 */
void
sc_lexicographic_sort(
    Psysteme sc,
    int (*compare)(Pvecteur*, Pvecteur*))
{
  Pcontrainte c = CONTRAINTE_UNDEFINED;

  if (sc==NULL || sc_empty_p(sc) || sc_rn_p(sc))
    return;

  /* sort the system basis */
  vect_sort_in_place(&(sc->base), compare);

  /* sort each constraint */
  contrainte_vect_sort(sc_egalites(sc), compare);
  contrainte_vect_sort(sc_inegalites(sc), compare);

  /* minimize equations: when equations are printed out, terms are moved
     to eliminate negative coefficients and the inner lexicographic
     order is broken. Furthermore, an equation can be multiplied by -1
     and stay the same but be printed out differently. */
  for(c = sc_egalites(sc); !CONTRAINTE_UNDEFINED_P(c); c = contrainte_succ(c))
  {
    Pvecteur v = contrainte_vecteur(c);

    if(!VECTEUR_NUL_P(v) && !vect_printout_order_decided_p(v)) {
      int cmp = 0;
      Pvecteur v1 = vect_dup(v);
      Pvecteur v2 = vect_multiply(vect_dup(v), VALUE_MONE);

      v1 = vect_printout_order(v1, compare);
      v2 = vect_printout_order(v2, compare);
      cmp = vect_lexicographic_compare(v1, v2, compare);
      if(cmp>0) {
        contrainte_vecteur(c) =
          vect_multiply(contrainte_vecteur(c), VALUE_MONE);
        vect_rm(v1);
        vect_rm(v2);
      }
      else if(cmp<0) {
        vect_rm(v1);
        vect_rm(v2);
      }
      else {
        /* we are in trouble: v1 and v2 are different but not
	   comparable because the compare function is not strong
	   enough */
	/* This may occur if the same local names are used for two
	   different variables that can be live simultaneously. For
	   instance, "predict!predict_mb:pict_struct" and
	   "TOP-LEVEL:pict_struct" which are or seem to be both int
	   variables in mpeg2enc. We could take a default action here
	   or force the user to improve his/her comparison function
	   where more information is available. */
        assert(false);
      }
    }
  }

  /* sort equalities and inequalities */
  sc->egalites = equations_lexicographic_sort(sc->egalites, compare);
  sc->inegalites = inequalities_lexicographic_sort(sc->inegalites, compare);
}

/* remove constraints with large coefs,
 *   possibly to avoid overflows and to keep systems as simple as possible
 * @param sc system to consider, which may be modified
 * @param val maximum coef allowed, >=0, 0 meant to do nothing
 * @param equalities whether to process equalities
 * @param inequalities whether to process inequalities
 * @return whether the system was changed
 */
bool sc_remove_large_coef(Psysteme sc, Value val,
			     bool equalities, bool inequalities)
{
  bool changed = false;

  // nothing to do in some simple cases
  if (sc==NULL || sc_empty_p(sc) || sc_rn_p(sc))
    return changed;

  if (equalities && sc->egalites) {
    int nb = sc->nb_eq;
    sc->egalites = contrainte_remove_large_coef(sc->egalites, val);
    sc->nb_eq = nb_elems_list(sc->egalites);
    changed |= nb!=sc->nb_eq;
  }

  if (inequalities && sc->inegalites) {
    int nb = sc->nb_ineq;
    sc->inegalites = contrainte_remove_large_coef(sc->inegalites, val);
    sc->nb_ineq = nb_elems_list(sc->inegalites);
    changed |= nb!=sc->nb_ineq;
  }

  return changed;
}

/*   That is all
 */
