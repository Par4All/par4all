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

/* package sc: elimination de redondance simple */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <string.h>
#include <stdio.h>
#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* void sc_rm_empty_constraints(Psysteme ps, bool process_equalities):
 * elimination des "fausses" contraintes du systeme ps, i.e. les contraintes ne
 * comportant plus de couple (variable,valeur), i.e. les contraintes qui
 * ont ete eliminees par la fonction 'eq_set_vect_nul', i.e. 0 = 0 ou
 * 0 <= 0
 * 
 * resultat retourne par la fonction: le systeme initial ps est modifie.
 * 
 * parametres de la fonction:
 *   !Psysteme ps: systeme lineaire 
 *   bool egalite: true s'il faut traiter la liste des egalites 
 *                    false s'il faut traiter la liste des inegalites
 *
 * Modifications:
 *  - the number of equalities was always decremented, regardless
 *    of the process_equalities parameter; Francois Irigoin, 30 October 1991
 */
void sc_rm_empty_constraints(ps, process_equalities)
Psysteme ps;
bool process_equalities;
{
    Pcontrainte pc, ppc;

    if (ps != SC_EMPTY)
	pc = (process_equalities) ? ps->egalites : ps->inegalites;
    else pc = NULL;
    ppc = NULL;

    while (pc != NULL) {
	if  (contrainte_vecteur(pc) == NULL) {
	    Pcontrainte p = pc;

	    if (ppc == NULL) {
		if (process_equalities)
		    ps->egalites = pc = pc->succ;
		else 
		    ps->inegalites = pc = pc->succ;
	    }
	    else {
		ppc->succ = pc = pc->succ;
	    }
	    contrainte_free(p);
	    if (process_equalities)
		ps->nb_eq--;
	    else
		ps->nb_ineq--;
	}
	else {
	    ppc = pc;
	    pc = pc->succ;
	}
    }
}

/* Psysteme sc_kill_db_eg(Psysteme ps):
 * elimination des egalites et des inegalites identiques ou inutiles dans
 * le systeme; plus precisemment:
 *
 * Pour les egalites, on elimine une equation si on a un systeme d'egalites 
 * de la forme :
 *
 *   a1/    Ax - b == 0,            ou  b1/        Ax - b == 0,              
 *          Ax - b == 0,                           b - Ax == 0,              
 *
 * ou   c1/    0 == 0
 *
 * Pour les inegalites, on elimine une inequation si on a un systeme 
 * d'inegalites de la forme :
 *  
 *   a2/    Ax - b <= c,             ou   b2/     0 <= const  (avec const >=0)
 *          Ax - b <= c             
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme   	    : Le systeme initial est modifie (si necessaire) et renvoye
 *       	      Si le systeme est non faisable (0 <= const <0 ou
 *                    0 = b), il est desalloue et NULL est
 *                    renvoye.
 *
 * Attention, on ne teste pas les proportionalites: 2*i=2 est different
 * de i = 1. Il faut reduire le systeme par gcd avant d'appeler cette
 * fonction sc_kill_db_eg()
 *
 * Notes:
 *  - le temps d'execution doit pouvoir etre divise par deux en prenant en
 * compte la symetrie des comparaisons et en modifiant l'initialisation
 * des boucles internes pour les "triangulariser superieurement".
 *  - la representation interne des vecteurs est utilisee pour les tests;
 * il faudrait tester la colinearite au vecteur de base representatif du
 * terme constant
 *
 * - so called triangular version, FC 28/09/94
 */

Psysteme sc_kill_db_eg(ps)
Psysteme ps;
{
    Pcontrainte
	eq1 = NULL,
	eq2 = NULL;

    if (ps == NULL) 
	return(NULL);

    for (eq1 = ps->egalites; 
	 eq1 != NULL; 
	 eq1 = eq1->succ) 
    {
	if ((vect_size(eq1->vecteur) == 1) && 
	    (eq1->vecteur->var == 0) && (eq1->vecteur->val != 0)) 
	{
	    /* b = 0 */
	    sc_rm(ps);
	    return(NULL);
	}

	for (eq2 = eq1->succ;
	     eq2 != NULL;
	     eq2 = eq2->succ)
	    if (egalite_equal(eq1, eq2))
		eq_set_vect_nul(eq2);
    }

    for (eq1 = ps->inegalites;
	 eq1 != NULL;
	 eq1 = eq1->succ)
    {
	if ((vect_size(eq1->vecteur) == 1) && (eq1->vecteur->var == 0))
	    if (eq1->vecteur->val <= 0)
		vect_rm(eq1->vecteur),
		eq1->vecteur = NULL;
	    else
	    {
		/* 0 <= b < 0 */
		sc_rm(ps);
		return(NULL);
	    }
	
	for (eq2 = eq1->succ;
	     eq2 != NULL;
	     eq2 = eq2->succ)
	    if (contrainte_equal(eq1,eq2))
		eq_set_vect_nul(eq2);
    }

    sc_rm_empty_constraints(ps, true);
    sc_rm_empty_constraints(ps, false);

    return (ps);
}

/* same as above, but returns an empty system if the system is not feasible*/
Psysteme sc_safe_kill_db_eg(ps)
Psysteme ps;
{
    Pcontrainte
	eq1 = NULL,
	eq2 = NULL;

    if (ps == NULL) 
	return(NULL);

    for (eq1 = ps->egalites; 
	 eq1 != NULL; 
	 eq1 = eq1->succ) 
    {
	if ((vect_size(eq1->vecteur) == 1) && 
	    (eq1->vecteur->var == 0) && (eq1->vecteur->val != 0)) 
	{
	    /* b = 0 */
	  Pbase base_tmp = ps->base;
	  ps->base = BASE_UNDEFINED;
	  sc_rm(ps);
	  ps =sc_empty(ps->base);
	  return(ps);
	}

	for (eq2 = eq1->succ;
	     eq2 != NULL;
	     eq2 = eq2->succ)
	    if (egalite_equal(eq1, eq2))
		eq_set_vect_nul(eq2);
    }

    for (eq1 = ps->inegalites;
	 eq1 != NULL;
	 eq1 = eq1->succ)
    {
	if ((vect_size(eq1->vecteur) == 1) && (eq1->vecteur->var == 0))
	    if (eq1->vecteur->val <= 0)
		vect_rm(eq1->vecteur),
		eq1->vecteur = NULL;
	    else
	    {
		/* 0 <= b < 0 */
	      Pbase base_tmp = ps->base;
	      ps->base = BASE_UNDEFINED;
	      sc_rm(ps);
	      ps =sc_empty(ps->base);
	      return(ps);
	    }

	for (eq2 = eq1->succ;
	     eq2 != NULL;
	     eq2 = eq2->succ)
	    if (contrainte_equal(eq1,eq2))
		eq_set_vect_nul(eq2);
    }

    sc_rm_empty_constraints(ps, true);
    sc_rm_empty_constraints(ps, false);

    return (ps);
}



/*   That is all
 */
