/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

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


/* boolean contrainte_in_liste(Pcontrainte c, Pcontrainte lc): test de
 * l'appartenance d'une contrainte c a une liste de contrainte lc
 *
 * Les contrainte sont supposees normalisees (coefficients reduits par leur
 * PGCD, y compris les termes constants).
 *
 * On considere que la contrainte nulle, qui ne represente aucune contrainte
 * (elle est toujours verifiee) appartient a toutes les listes de contraintes.
 *
 */
boolean contrainte_in_liste(Pcontrainte c, Pcontrainte lc)
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

/* boolean egalite_in_liste(Pcontrainte eg, Pcontrainte leg): test si une
 * egalite appartient a une liste d'egalites
 *
 * Une egalite peut avoir ete multipliee par moins 1 mais ses coefficients,
 * comme les coefficients des egalites de la liste, doivent avoir ete
 * reduits par leur PGCD auparavant
 *
 * Ancien nom: vect_in_liste1()
 */
boolean egalite_in_liste(Pcontrainte v, Pcontrainte listev)
{
  Pcontrainte v1;

  if (v->vecteur == NULL) return(TRUE);
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
 */
int nb_elems_list(Pcontrainte list)
{
	int i;

	for(i=0;list!=NULL;i++,list=list->succ)
    ;

	return i;
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
