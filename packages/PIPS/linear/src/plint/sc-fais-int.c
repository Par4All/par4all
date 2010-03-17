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

 /* package plint */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"
#include "matrix.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc(s)

/* boolean sys_int_fais(Psysteme sys1):
 * Test de faisabilite d'un systeme lineaire syst1 en nombres entiers par 
 * l'algorithme des congruences decroissantes (cf. livre ???, pp. ??? ).
 * Renvoie TRUE si le systeme est satisfiable (i.e. il definit un polyedre
 * convexe non vide), FALSE sinon.
 *
 * Ce test est exact, mais il est tres couteux en temps CPU.
 *
 * Le systeme de contrainte syst1 n'est pas modifie
 */
boolean sys_int_fais(sys1)
Psysteme sys1;
{
    Psysteme sys2 = NULL;
    Psommet fonct = fonct_min(sys1->dimension,sys1->base);
    Psolution sol1 = NULL;

    boolean is_faisable = FALSE;

    sys2=sc_dup(sys1);
    /*
     * Recherche d'une solution par l'algorithme des congruences
     * decroissantes a partir d'une fonction economique minimale (
     * recherche du minimum de la premiere variable du systeme)
     */
    if ((sys2 != NULL) && 
	( (sys2->egalites != NULL) || (sys2->inegalites != NULL))) 
	sys2 = plint(sys2,fonct,&sol1);
    else
	is_faisable = TRUE;

    if ((sys2 != NULL) && ((sys2->egalites != NULL)
			   || (sys2->inegalites != NULL)))
	is_faisable = TRUE;

    if (is_faisable)
	/* cas ou le systeme est faisable          */
	sc_rm(sys2);
    return (is_faisable);
}
