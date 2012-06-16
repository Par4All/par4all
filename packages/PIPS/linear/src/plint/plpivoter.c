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

#define TRACE

/*
 * Remplacement de la variable VAR dans la contrainte EQ par sa valeur dans
 * la contrainte LIGNE.
 *
 *  resultat retourne par la fonction :
 *
 *  La contrainte EQ est modifiee.
 *
 *  Les parametres de la fonction :
 *
 *  Psommet eq     : contrainte du systeme
 *  Psommet ligne  : contrainte du systeme ( ligne pivot) 
 *  int     var    : variable pivot
 */


void pivoter_pas(eq,ligne,var)
Psommet eq;
Psommet ligne;
Variable var;
{


    Pvecteur pvec = NULL;
    Pvecteur ligne2;

    Value c1 = VALUE_ZERO;
    Value den;
    bool cst = false;
#ifdef TRACE
    printf(" --- pas - pivoter \n");
#endif
    if (ligne && eq) {
	Pvecteur pv3 = vect_dup(eq->vecteur);

	den = ligne->denominateur;
	cst = false;
	if ((eq != ligne) && value_notzero_p(c1 = vect_coeff(var,pv3))) {

	    ligne2 = vect_dup(ligne->vecteur);
	    value_product(eq->denominateur,den);
	    for (pvec =pv3;pvec!= NULL; pvec=pvec->succ) {
		Value tmp;
		if (pvec->var == NULL) cst = true;

		value_product(pvec->val,den);
		tmp = vect_coeff(pvec->var,ligne->vecteur);
		value_product(tmp,c1);
		value_substract(pvec->val,tmp);

		vect_chg_coeff(&ligne2,pvec->var,VALUE_ZERO);
	    }

	    for (pvec=ligne2;pvec!= NULL;pvec = pvec->succ)
		if (pvec->var != TCST)
		{
		    Value tmp = vect_coeff(pvec->var,ligne2);
		    value_product(tmp,c1);
		    vect_add_elem(&pv3,pvec->var,value_uminus(tmp));
		}
	    if (!cst)
	    {
		Value tmp = vect_coeff(TCST,ligne->vecteur);
		value_product(tmp,c1);
		vect_add_elem(&pv3,TCST,value_uminus(tmp));
	    }
	}
	eq->vecteur = pv3;
    }
}

/*
 * Operation "pivot" avec VAR comme variable pivot et LIGNE comme ligne
 * pivot.
 *
 *  resultat retourne par la fonction :
 *
 *  Le systeme initial est modifie.
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Psommet fonct  : fonction economique du  programme lineaire
 *  Psommet ligne  : ligne pivot
 *  int     var    : variable pivot   
 */
void pivoter(sys,ligne,var,fonct)
Psommet sys;
Psommet ligne;
Variable var;
Psommet fonct;

{

    Psommet sys1 = fonct;
    Psommet ps1 = NULL;
    int sgn_den = 1;
    Value den,tmp;
#ifdef TRACE
    printf(" *** on effectue le pivot \n");
#endif
    if (ligne) {

	den = vect_coeff(var,ligne->vecteur);
	if (value_neg_p(den))
	{
	    sgn_den = -1;
	    value_oppose(den);
	}
	if (fonct != NULL)
	    fonct->succ = sys;
	else sys1 = sys;

	/* mise a jour du denominateur   */
	tmp = ligne->denominateur;
	if (sgn_den==-1) value_oppose(tmp);
	(void) vect_multiply(ligne->vecteur,tmp);
	value_product(ligne->denominateur,den);
	den = ligne->denominateur;
	for (ps1 = sys1; ps1!= NULL; ps1=ps1->succ)
	    pivoter_pas(ps1,ligne,var);
	sommets_normalize(sys1);

    }
    if (fonct)
	fonct->succ = NULL;

}

