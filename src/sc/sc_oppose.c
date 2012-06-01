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

 /* package sc */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_oppose(Psysteme ps): calcul, pour un systeme de
 * contraintes sans egalites, du systeme de contraintes dont les
 * inegalites sont les negations des inegalites originelles;
 * attention, cela ne calcule pas le complementaire, qui n'est
 * d'aileurs pas un polyedre!
 *
 * Pour chaque inegalite de, AX <= B, on construit une inegalites "opposee"
 * AX > B approximee par -AX <= -B
 *
 * Un systeme non trivial risque fort d'etre transforme en systeme non
 * faisable. For instance, -1<=x<=1 is changed into 1<=-x<=-1.
 *
 * The function aborts if ps contains equalities.
 *
 * The constraint system ps is modified by side effect and
 * returned. No new system is allocated.
 */
Psysteme sc_oppose(ps)
Psysteme ps;
{
    Pcontrainte eq;

    if (ps->nb_eq != 0) {
	(void) fprintf(stderr,"sc_oppose: systeme contenant des egalites\n");
	abort();
    }

    for (eq = ps->inegalites; eq != (Pcontrainte )NULL; eq = eq->succ)
	(void) vect_multiply(eq->vecteur, VALUE_MONE);

    return(ps);
}
