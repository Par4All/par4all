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

/* package vecteur
 * routines internes au package
 */

/*LINTLIBRARY*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

/* PRIVATE: introduit du sharing, ne garantit pas l'unicite de chaque
 * composante
 *
 * Pvecteur vect_chain(Pvecteur v_in, Variable var, Value coeff): 
 * ajout d'un vecteur colineaire au vecteur de base var et de coefficient
 * coeff au vecteur v_in; si la composante var de v_in etait nulle:
 *
 * si <v_in . evar> == 0 alors
 *	       ---->
 *    allocate v_out;
 *    ---->   --->         --> 
 *    v_out = v_in + coeff var;
 *	      --->
 *    destroy v_in;
 * sinon
 *    chaos!
 *
 * v_in ne doit plus etre utilise ulterieurement; l'utilisation
 * standard est donc:
 *  v = vect_chain(v,var,coeff);
 *
 * Pour preserver la coherence des vecteurs, l'utilisateur doit
 * appeler vect_add_elem(v,var,coeff) qui teste l'existence d'un coefficient
 * var nul dans v avant de creer un nouveau couple (var,coeff)
 *
 * On pourrait preferer le passage d'un Pvecteur * comme premier argument
 * et un return void.
 */
Pvecteur vect_chain(v_in,var,coeff)
Pvecteur v_in;
Variable var;
Value coeff;
{
    Pvecteur v_out;

    /* very expensive for EDF... FC/CA 06/07/2000. useless? */
    /* assert(vect_coeff(var, v_in)==0); */

    v_out = vect_new(var, coeff);
    v_out->succ = v_in;

    return (v_out);
}

/* PRIVATE: introduit du sharing; never used...
 *
 * Pvecteur vect_elem(Pvecteur vect, Variable var): retourne la valeur
 * du pointeur vers le couple correspondant a la variable var dans le
 * vecteur vect, s'il existe ou NULL sinon.
 */
Pvecteur vect_elem(vect,var)
Pvecteur vect;
Variable var;
{
    for( ;vect!=NULL;vect=vect->succ) {
	if (var_of(vect)==var) {
	    return(vect);
	}
    }
    return (NULL);
}

/* UNUSED - NOT TESTED
 *
 * Pvecteur vect_extract(Pvecteur pvec, Variable var):
 * fonction qui extrait le couple (variable,coefficient) du vecteur et renvoie
 * l'adresse d'un nouveau vecteur vers ce couple. On a donc un effet de
 * bord sur pvec (qui est impossible a assurer dans tous les cas) et un
 * retour de valeur
 *
 * --->    --->    --->   -->  -->
 * pvec := pvec - <pvec . var> var;
 *          ---->
 * allocate pvec1;
 *        ---->     --->   -->  -->
 * return(pvec1 := <pvec . var> var);
 * 
 * Notes:
 *  - ca ne peut pas marcher si var apparait dans le premier couple du vecteur
 *    car on n'a pas moyen de changer pvec, vu la passage par valeur; il
 *    faudrait donc soit changer le type de la procedure et passer un 
 *    Pvecteur *, soit autoriser les couples de valeur 0;
 *  - cette fonction n'est utile que si elle diminue le nombre d'allocations;
 *    elle commence malencontreusement par un vect_dup inconditionnel.
 */
Pvecteur vect_extract(pvec,var)
Pvecteur pvec;
Variable var;
{	
    Pvecteur pvec1 = vect_dup(pvec);
    Pvecteur var_val = pvec1;
    Pvecteur var_pred = pvec1;

    if (var_val != NULL)  {
	if  ( var_of(var_pred)== var) {
	    /* le couple interessant se trouve en tete */
	    pvec1 = var_pred->succ;
	    var_pred->succ = NULL;
	    return (var_pred);
	}
	else {
	    for(var_val=(var_val->succ);var_val!=NULL;var_pred = var_val,
		var_val=var_val->succ) {
		if (var_of(var_val)==var) {
		    var_pred->succ = var_val->succ;
		    var_val->succ = NULL;
		    return(var_val);
		}
	    }

	}
    }

    /* Sinon, c'est le vecteur 0 */
    return (NULL);
}

/* PRIVATE: marquage du couple var_val comme visite par remplacement de
 * var par -var dans le couple (OBSOLETE)
 *
 * Value vect_coeff_m(Variable var, Pvecteur vect)
 *
 * static Value vect_coeff_m(var,vect)
 * Variable var;
 * Pvecteur vect;
 * {
 *     for (; vect != NULL ; vect = vect->succ)
 * 	if (var_of(vect) == var) {
 * 	    marquer(vect);
 * 	    return(val_of(vect));
 * 	}
 *     return(0);
 * }
 */


/* PRIVATE
 * Pvecteur vect_tri_old(Pvecteur pvec): allocation d'un vecteur
 * prenant une valeur egale a celle de pvec mais dont les couples
 * (variable,valeur) sont tries dans "l'ordre croissant" des vecteurs de base 
 * (i.e. des variables)
 */
/*
static Pvecteur vect_tri_old(pvec)
Pvecteur pvec;
{
    Pvecteur pv1,pv2,pv3;

    Pvecteur debut = NULL;
    Pvecteur pred;

    if (pvec)
    {
	debut = vect_new(pvec->var,pvec->val);
	for (pv1 = pvec->succ;pv1!= NULL;pv1 = pv1->succ)
	{

	    pred = debut;
	    for (pv2 = debut;((pv2 != NULL) && (pv2->var < pv1->var));
		 pred = pv2,pv2=pv2->succ);

	    if ( pv2 == pred)
	    {
		pv3 = vect_new(pv1->var,pv1->val);
		pv3->succ = debut;
		debut = pv3;
	    }
	    else 
	    {
		pv3 = vect_new(pv1->var,pv1->val);
		pred->succ = pv3;
		pv3->succ = pv2;
	    }
	}
    }
    return (debut);
}
*/


/* Variable vect_first_var(Pvecteur pvec)
 *  retourne la premiere variable (au sens CAR) du vecteur pvec
 *  routine sale mais qui permet d'iterer sur les variables
 *  formant un vecteur ou une base.
 *  20/06/90 PB
 */
Variable vect_first_var(pvec)
Pvecteur pvec;
{
    return(pvec->var);
}

/* Pvecteur vect_reversal(Pvecteur vect_in); produces the reversal vector of 
 * the vect_in. vect_in is not killed.             
 * 12/09/91, YY 
 */
Pvecteur vect_reversal(vect_in)
Pvecteur vect_in;
{
    Pvecteur pv;
    Pvecteur vect_out = VECTEUR_NUL;
    
    for(pv=vect_in; !VECTEUR_NUL_P(pv); pv=pv->succ)
	vect_add_elem(&vect_out, vecteur_var(pv), vecteur_val(pv));
    return (vect_out);
}
 
