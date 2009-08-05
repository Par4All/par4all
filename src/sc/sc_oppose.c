/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

 /* package sc */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_oppose(Psysteme ps):
 * calcul pour un systeme de contraintes sans egalites du systeme de
 * contraintes dont les inegalites sont les negations des inegalites
 * originelles; attention, cela ne calcule pas le polyedre complementaire!
 *
 * Un systeme non trivial risque fort d'etre transforme en systeme non
 * faisable.
 *
 * pour chaque inegalite donnee AX <= B, on construit une inegalites "opposee"
 * AX > B ie -AX <= -B
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
