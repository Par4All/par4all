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

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "matrix.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc(s)
char * malloc();

/*
 * Cette fonction teste si l'un des termes constants des contraintes est negatif
 *
 *  resultat retourne par la fonction :
 *
 *  bool          : == true si l'un des termes constants est negatif
 *                  == false sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psommet  som  : systeme lineaire
 *
 */

bool const_negative(som)
Psommet som;
{
    Psommet ps;
    bool result = false;

    for (ps = som;
	 ps!= NULL && value_negz_p(vect_coeff(TCST,ps->vecteur));
	 ps= ps->succ);
    result = (ps == NULL) ? false : true;
    return (result);
}



/*
 * Si la partie constante d'une contrainte est negative, il faut que l'un des
 * coefficients des variables de la contrainte le soit aussi pour que le systeme 
 * reste borne.
 *
 *  resultat retourne par la fonction :
 *
 *  bool           : false si la contrainte montre que le systeme est non borne
 *		     true  sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psommet  eq    : contrainte du systeme  
 */

bool test_borne(eq)
Psommet eq;
{
    Pvecteur pv = NULL;
    bool result= false;

    if (eq) {
	pv = eq->vecteur;
	if (value_pos_p(vect_coeff(TCST,pv)))
	{
	    for (pv= eq->vecteur;pv!= NULL 
		 && ((pv->var ==NULL) || value_pos_p(pv->val))
		 ;pv= pv->succ);
	    result = (pv==NULL) ? false : true;
	}
	else result = true;
    }
    return (result);

}





/*
 * Cette fonction teste s'il existe une variable hors base de cout nul
 * dans le systeme
 *
 *  resultat retourne par la fonction :
 *
 *  bool           : true s'il existe une variable hors base de cout 
 *    		      nul
 *
 *  Les parametres de la fonction :
 *
 *  Psommet fonct  : fonction economique du  programme lineaire
 *  Pvecteur lvbase: liste des variables de base du systeme
 */

bool cout_nul(fonct,lvbase,nbvars,b)
Psommet fonct;
Pvecteur lvbase;
int nbvars;
Pbase b;
{
    Pvecteur liste1 = NULL;		/* liste des variables h.base de cout nul */
    Pvecteur pv=NULL;
    Pvecteur pv2=VECTEUR_NUL;
    register int i;
    bool result= false;

#ifdef TRACE
    printf(" ** Gomory - existe-t-il une var. h.base de cout  nul  \n");
#endif

    liste1 = vect_new(vecteur_var(b),VALUE_ONE);
    for (i = 1 ,pv2 = b->succ;
	 i< nbvars && !VECTEUR_NUL_P(pv2); 
	 i++, pv2=pv2->succ)
	vect_add_elem (&(liste1),vecteur_var(pv2),VALUE_ONE);
    if (fonct != NULL)
	for (pv = fonct->vecteur;pv != NULL;pv=pv->succ)
	    if (value_notzero_p(pv->val))
		vect_chg_coeff(&liste1,pv->var,0);
		
    for (pv = lvbase;pv != NULL;pv=pv->succ)
	if (value_notzero_p(pv->val))
	    vect_chg_coeff(&liste1,pv->var,0);
    result = (liste1 != NULL) ? true : false;

    vect_rm(liste1);
    return (result);

}

