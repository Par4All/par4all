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

 /* package contrainte - operations sur les listes de contraintes
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
#include "contrainte.h"


/* bool contrainte_in_liste(Pcontrainte c, Pcontrainte lc): test de
 * l'appartenance d'une contrainte c a une liste de contrainte lc
 *
 * Les contrainte sont supposees normalisees (coefficients reduits par leur
 * PGCD, y compris les termes constants).
 *
 * On considere que la contrainte nulle, qui ne represente aucune contrainte
 * (elle est toujours verifiee) appartient a toutes les listes de contraintes.
 *
 */
bool contrainte_in_liste(Pcontrainte c, Pcontrainte lc)
{
  Pcontrainte c1;

  assert(!CONTRAINTE_UNDEFINED_P(c));

  if (CONTRAINTE_NULLE_P(c))
    return true;

  for (c1=lc; !CONTRAINTE_UNDEFINED_P(c1); c1=c1->succ) {
    if (vect_equal((c1->vecteur),(c->vecteur))) {
	    return true;
    }
  }
  return false;
}

/* Return the rank of constraint c in constraint list lc. 1 for the
 * first element, and so on, Fortran style. 0 when c is not in lc.
 *
 * The comparisons are based on the pointers, not on the values of the
 * constraints. It is mainly useful to detect cycles in constraint list.
 */
int constraint_rank(Pcontrainte c, Pcontrainte lc)
{
  Pcontrainte cc;
  int rank = 0;

  assert(!CONTRAINTE_UNDEFINED_P(c));

  if (CONTRAINTE_NULLE_P(c))
    ;
  else {
    for (cc=lc; !CONTRAINTE_UNDEFINED_P(cc); cc=cc->succ) {
      rank++;
      // This would be useful to detect duplicate constraints
      // if (vect_equal((c1->vecteur),(c->vecteur))) {

      // Physical check
      if(cc==c)
	break;
    }
  }

  return rank;
}

/* bool egalite_in_liste(Pcontrainte eg, Pcontrainte leg): test si une
 * egalite appartient a une liste d'egalites
 *
 * Une egalite peut avoir ete multipliee par moins 1 mais ses coefficients,
 * comme les coefficients des egalites de la liste, doivent avoir ete
 * reduits par leur PGCD auparavant
 *
 * Ancien nom: vect_in_liste1()
 */
bool egalite_in_liste(Pcontrainte v, Pcontrainte listev)
{
  Pcontrainte v1;

  if (v->vecteur == NULL) return(true);
  for (v1=listev;v1!=NULL;v1=v1->succ) {
    if (vect_equal((v1->vecteur),(v->vecteur)) ||
        vect_oppos((v1->vecteur),(v->vecteur))) {
	    return true;
    }
  }
  return false;
}

/* int nb_elems_list(Pcontrainte list): nombre de contraintes se trouvant
 * dans une liste de contraintes
 *
 * Ancien nom: nb_elems_eq()
 *
 * Cycles are not detected
 */
int nb_elems_list(Pcontrainte list)
{
  int i;
  Pcontrainte c;

  for(i=0, c=list;c!=NULL;i++,c=c->succ)
    ;

	return i;
}

/* Check if list l contains a cycle */
bool cyclic_constraint_list_p(Pcontrainte l)
{
  int i;
  Pcontrainte c;
  bool cyclic_p = false;

  for(i=0, c=l; c!=NULL; i++, c=c->succ) {
    int r = constraint_rank(c, l);
    if(r>0 && r<i) {
      cyclic_p = true;
      break;
    }
  }

  return cyclic_p;
}

/* Compute the number of elements in the list if it is less than n. n
   is assumed positive. A negative value is returned if the number of
   elements is strictly greater than n, for instance because the list
   is cyclic. */
int safe_nb_elems_list(Pcontrainte list, int n)
{
  int i;
  Pcontrainte c;

  assert(n>=0);

  for(i=0, c=list;c!=NULL && i<=n ;i++,c=c->succ)
    ;

  return i<=n? i : -1;
}

/* @return a constraint list without constraint with large coeffs
 * @param lc list of constraint, which is modified
 * @param val maximum value allowed for coefficients
 */
Pcontrainte contrainte_remove_large_coef(Pcontrainte lc, Value val)
{
  linear_assert("value must be positive", value_posz_p(val));

  Pcontrainte first = lc, previous = NULL;

  if (value_zero_p(val)) // nothing to do
    return lc;

  while (lc!=NULL)
  {
    if (vect_larger_coef_p(lc->vecteur, val))
    {
      // unlink and free
      Pcontrainte next = lc->succ;
      lc->succ = NULL;
      contrainte_free(lc);
      if (lc==first) first = next;
      if (previous) previous->succ = next;
      // "previous" constraint itself is unchanged
      lc = next;
    }
    else
    {
      previous = lc;
      lc = lc->succ;
    }
  }
  return first;
}
