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

/* package plint */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrix.h"

#include "sommet.h"
#include "plint.h"

#define MALLOC(s,t,f) malloc(s)

/* Psysteme sys_int_redond(Psysteme sys):
 * elimination des contraintes lineaires redondantes d'un systeme lineaire 
 * en nombres entiers par tests de faisabilite exacts. Chaque inegalite est
 * inversee tour a tour, et la faisabilite de chacun des systemes ainsi
 * obtenus est teste par sys_int_fais(), l'algorithme des congruences
 * decroissantes.
 */
Psysteme sys_int_redond(sys)
Psysteme sys;
{

    Pcontrainte eq,eq1;

    sys = sc_normalize(sc_dup(sys));
    if (sys && (sys->nb_ineq <= NB_INEQ_MAX2) && sys_int_fais(sys) && 
	sys != NULL) {
	for (eq = sys->inegalites;
	     eq != NULL && sys->nb_ineq > 1;
	     eq = eq1)
	{
	    eq1 = eq->succ;
	    /* inversion du sens de l'inegalite par multiplication */
	    /* par -1 du coefficient de chaque variable            */
	    vect_chg_sgn(eq->vecteur);
	    vect_add_elem(&(eq->vecteur),TCST,VALUE_ONE);
	    /* test de faisabilite avec la nouvelle inegalite      */
	    if (sys_int_fais(sys) == false)
	    {
		/* si le systeme est non faisable 
		   ==> inegalite redondante
		   ==> elimination de cette inegalite         */
		eq_set_vect_nul (eq);
		sc_rm_empty_constraints(sys,0);
	    }
	    else
	    {
		vect_add_elem (&(eq->vecteur),TCST,VALUE_MONE);
		vect_chg_sgn(eq->vecteur);
	    }
	}
    }
    return (sys);
}
