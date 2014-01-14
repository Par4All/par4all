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

#include "sommet.h"
#include "matrix.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(s,t,f) free((char *)(s))

/* void oter_lvbase(Psommet sys, Pvecteur lvbase):
 * Elimination d'une liste de variables contenues dans un vecteur
 * dans Gomory : elimination des variables de base du systeme  
 *
 *  resultat retourne par la fonction :
 *
 *  Le systeme initial est modifie.  
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire
 *  Pvecteur lvbase: liste des variables de base du systeme
 */
void oter_lvbase(sys,lvbase)
Psommet sys;
Pvecteur lvbase;
{
    Psommet ps1=NULL;
    Pvecteur pv;

#ifdef TRACE
    printf (" **** elimination d'une liste de variables \n");
#endif

    for (pv = lvbase; pv!=NULL; pv = pv->succ)
	for (ps1 = sys; ps1 != NULL; ps1 = ps1->succ)
	    vect_chg_coeff(&(ps1->vecteur),pv->var,0);
}

/* bool is_var_in_lvbase(Variable var, Pvecteur lvbase):
 * test de non appartenance d'une variable a un vecteur
 *
 * Ne pourais-tu pas obtenir le meme resultat avec un vect_coeff()?
 *
 *  resultat retourne par la fonction :
 *
 *  boolean	   : true si la variable n'appartient pas au vecteur
 *		     false sinon
 *
 *  Les parametres de la fonction :
 *
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     var    : variable du systeme   
 */
bool is_var_in_lvbase(var,lvbase)
Variable var;
Pvecteur lvbase;
{
    Pvecteur pv;
    for (pv = lvbase; 
	 pv!= NULL && ((pv->var!=var) || (pv->val == 0)); 
	 pv = pv->succ)	;
    if(pv == NULL)
	return (true);
    else 
	return (false);
}

/* void lvbase_add(Variable var, int no_ligne, Pvecteur * lvbase):
 * ajout d'un couple (variable de base, no ligne correspondante) dans la liste
 * des variables de base du systeme
 *
 *  Les parametres de la fonction :
 *
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  Variable     var    : variable du systeme   
 *  int    no_ligne: numero de la ligne 
 */
void lvbase_add(var,no_ligne,lvbase)
Variable var;
int no_ligne;
Pvecteur *lvbase;
{
    Pvecteur pv = NULL;
    Value vno_ligne = int_to_value(no_ligne);
    Pvecteur pl = vect_new(var,vno_ligne);
    bool trouve = false;

    for (pv = *lvbase; pv != NULL && (pv->val != vno_ligne);pv=pv->succ);
    if (pv !=NULL) trouve = true;
    for (pv = *lvbase;( pv != NULL) &&(trouve) ;pv=pv->succ) {
	if (value_ge(pv->val,vno_ligne))
	    value_increment(pv->val);
    }
    pl->succ = *lvbase;
    *lvbase = pl;
}

/* void lvbase_ote_no_ligne(int no_ligne, Pvecteur * lvbase):
 * Elimination de la variable de base correspondant au numero de ligne
 * NO_LIGNE de la liste des variables de base du systeme.
 *
 *  Les parametres de la fonction :
 *
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int    no_ligne: numero de la ligne 
 */
void lvbase_ote_no_ligne (no_ligne,lvbase)
int no_ligne;
Pvecteur *lvbase;
{
    Pvecteur pv;
    Pvecteur pred = *lvbase;
    Value vno_ligne = int_to_value(no_ligne);

#ifdef TRACE
    printf(" **** oter la variable d'indice %d \n",no_ligne);
#endif
    if (pred != NULL)
	if (value_eq(pred->val,vno_ligne))
	{
	    pv = (*lvbase)->succ;
	    (*lvbase)->succ = NULL;
	    vect_rm(*lvbase);
	    *lvbase = pv;
	}
	else {
	    for (pv = (*lvbase)->succ;
		 pv!= NULL && value_ne(pv->val,vno_ligne);
		 pv = pv->succ, pred = pred->succ);

	    if (value_eq(pv->val,vno_ligne)) {
		pred->succ = pv->succ;
		FREE(pv,VECTEUR,"ote_var_Liste");
	    }
	}
}

/* Variable find_vbase(Psommet eq, Pvecteur lvbase):
 * Recherche de la variable de base d'une contrainte
 *
 *  resultat retourne par la fonction :
 *
 *  int            : no de la variable de base
 *
 *  Les parametres de la fonction :
 *
 *  Psommet eq     : contrainte du systeme
 *  Pvecteur lvbase: liste des variables de base du systeme
 */
Variable find_vbase(eq,lvbase)
Psommet eq;
Pvecteur lvbase;
{
    Pvecteur pv=NULL;
    Variable var = NULL;
#ifdef TRACE
    printf(" *** recherche d'une variable de base \n");
#endif
    if (eq != NULL) {
	Value cst = vect_coeff(TCST,eq->vecteur);
	Pvecteur pv3;
	vect_chg_coeff(&(eq->vecteur),TCST,0);
	pv = vect_sort(eq->vecteur, vect_compare);
	for(pv3 =pv; pv3->succ!=NULL;pv3=pv3->succ);
	pv3->succ = vect_new(TCST,cst);
	for (;pv!= NULL 
	     && ((pv->val ==0) || is_var_in_lvbase(pv->var,lvbase));
	     pv=pv->succ) ;

	if (pv != NULL)
	    var = pv->var;
	vect_rm(pv);
    }
    return (var);
}

/* Variable coeff_no_ligne(lvbase, int no_ligne):
 * Recherche de la variable de base d'une contrainte.
 * La contrainte est caracterisee par son numero de ligne
 *
 *  resultat retourne par la fonction :
 *
 *  Variable 	   : numero de la variable
 *
 *  Les parametres de la fonction :
 *
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int    no_ligne: numero de la ligne 
 */
Variable coeff_no_ligne(lvbase,no_ligne)
Pvecteur lvbase;
int no_ligne;
{
    Pvecteur pv;
    Value vno_ligne = int_to_value(no_ligne);

    for (pv = lvbase; 
	 (pv!= NULL) && value_ne(pv->val,vno_ligne); 
	 pv= pv->succ);
    if (pv != NULL)
	return (pv->var);
    else 
	return((Variable) NULL);
}

