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

/* pour recuperer les declarations des fonctions de conversion de
 * sc en liste de sommets et reciproquement, bien que ca casse le
 * DAG des types de donnees
 */
#include "ray_dte.h"
#include "polyedre.h"
#include "matrix.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc(s)

/* Psommet ligne_pivot(Psommet sys, int nb_som, int * no_som):
 * recherche de la ligne pivot dans l'algorithme dual du simplexe
 *
 *  resultat retourne par la fonction : 
 *  Psommet        : prochaine contrainte "pivot"
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  int     nb_som : nombre de contraintes du systeme
 *  int     no_som : place de la contrainte dans le systeme (no de la ligne)
 */
Psommet ligne_pivot(sys,nb_som,no_som)
Psommet sys;
int nb_som;
int *no_som ;
{
    Psommet som1,som2;
    Psommet lpivot= NULL;
    double min;
    double coeff;
    int nb_som1 = nb_som;
    Value den;
    double dden;

#ifdef TRACE
    printf(" **** alg. dual - recherche de la ligne pivot \n");
#endif


    if (sys) den = sys->denominateur;
    /*  initialisation du minimum    */

    for (som2 = sys;som2 != NULL  
	 && value_posz_p(vect_coeff(TCST,som2->vecteur)) ;
	 som2 = som2->succ,nb_som1 --);
    den = som2->denominateur;
    dden = VALUE_TO_DOUBLE(den);

    min = -VALUE_TO_DOUBLE(vect_coeff(TCST,som2->vecteur))/dden;
    lpivot = som2;
    *no_som = nb_som1;
    nb_som1 --;
    for (som1 = som2->succ;som1 != NULL;som1 = som1->succ)
    {
	den = som1->denominateur;
	if ((coeff = -VALUE_TO_DOUBLE(vect_coeff(TCST,som1->vecteur))/dden) < 0
	    && coeff < min)
	{
	    min = coeff;
	    lpivot = som1;
	    *no_som = nb_som1;

	}  nb_som1--;
    }
    return(lpivot);
}

/* int var_pivotd(Psommet eq, Psommet fonct):
 * recherche de la nouvelle variable pivot entrant dans la base au cours de
 * l'execution de l'algorithme dual du simplexe
 *
 *  resultat retourne par la fonction :
 *  int            : variable "pivot" entrant dans la base 
 *
 *  Les parametres de la fonction :
 *
 *  Psommet eq     : contrainte du systeme
 *  Psommet fonct  : fonction economique du  programme lineaire
 *
 */
Variable var_pivotd(eq,fonct)
Psommet eq,fonct;
{
    Pvecteur pv= VECTEUR_NUL;
    double min=0, min2;
    Variable no_lpivot = NULL;
    Value cf2 = VALUE_ZERO;
    bool trouve = false;

#ifdef TRACE
    printf(" **** alg. dual - recherche de la variable pivot \n");
#endif

    if (eq && fonct)
    {
	/* initialisation du minimum   */
	Value cst = vect_coeff(TCST,eq->vecteur);
	Pvecteur pv3;
	vect_chg_coeff(&(eq->vecteur),TCST,0);
	pv = vect_sort(eq->vecteur, vect_compare);
	/* pv = vect_tri(eq->vecteur, is_inferior_variable); */
	for(pv3 =pv; pv3->succ!=NULL;pv3=pv3->succ);
	pv3->succ = vect_new(TCST,cst);
    
	for (; pv!= NULL && !trouve; pv = pv->succ) {
	    if ((var_of(pv) != NULL) && (value_neg_p(val_of(pv))) && 
		(value_posz_p(cf2 = vect_coeff(var_of(pv),fonct->vecteur))))
	    {
		double d1,d2 = 0;
		d1 = VALUE_TO_DOUBLE(pv->val) * 
		    VALUE_TO_DOUBLE(fonct->denominateur);
		d2 = VALUE_TO_DOUBLE(cf2) * 
		    VALUE_TO_DOUBLE(eq->denominateur) ;
		min = - d2/d1;
		trouve = true;
		no_lpivot = var_of(pv);
	    }
	}
	if (pv != VECTEUR_NUL) {
	    for (; pv!= NULL && trouve ;pv= pv->succ) {
		if ((var_of(pv) != NULL) && value_neg_p(val_of(pv)) &&
		    value_posz_p(cf2 = vect_coeff(var_of(pv),fonct->vecteur)))
		{
		    double d1,d2 = 0;
		    d1 = VALUE_TO_DOUBLE(cf2) *
			VALUE_TO_DOUBLE(eq->denominateur);
		    d2 = VALUE_TO_DOUBLE(pv->val) * 
			VALUE_TO_DOUBLE(fonct->denominateur);
		    min2 = -d1/d2;
		    if (min2 < min) {
			min = min2;
			no_lpivot = var_of(pv);
		    }
		}
	    }
	}
    }
    vect_rm(pv);

    return (no_lpivot);
}

/* bool dual_pivot_pas(Psommet * sys, Pvecteur * lvbase, int nb_som,
 *                        Psommet fonct, int * nbvars, Pbase * b):
 * execution d'un pas de l'algorithme dual du simplexe i.e.
 * - ajout des variables d'ecart
 * - recherche d'un pivot
 * - on effectue le pivot
 *
 *  resultat retourne par la fonction :
 *
 *  bool           :  == false si l'on n'a pas pu effectuer un pas de 
 * 			       l'algorithme dual 
 * 		      == true  si l'on a pu effectuer un pas de l'algorithme
 *
 *  le systeme est modifie. 
 *  Si le systeme retourne est NULL, c'est que le systeme est non borne. 
 *                   
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Psommet fonct  : fonction economique du  programme lineaire
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme.
 *  int     nbvars : nombre de variables du systeme.
 *  Pbase     * b : liste des variables du systeme
 */
bool dual_pivot_pas(sys,lvbase,nb_som,fonct,nbvars,b)
Psommet *sys;
Pvecteur *lvbase;
int nb_som;
Psommet fonct;
int *nbvars;
Pbase *b;
{

    Psommet ligne_entrant =NULL;
    Psommet *ps=NULL;
    int no_som = 0;
    bool result = true;

#ifdef TRACE
    printf(" *** alg. dual -  un pas de l'algorithme \n");
#endif
    ps = sys;

    *ps = var_ecart_sup(*ps,nb_som,lvbase,nbvars,b);

    /* recherche de la ligne pivot	*/
    ligne_entrant = ligne_pivot(*ps,nb_som,&no_som);
    if ((ligne_entrant != NULL))
    {
	Variable var_entrant = NULL;
	/* recherche de la variable pivot	*/
	var_entrant = var_pivotd(ligne_entrant,fonct);

	if (VARIABLE_DEFINED_P(var_entrant))
	{
	    /* operation PIVOT	*/
	    pivoter(*ps,ligne_entrant,var_entrant,fonct);
	    lvbase_ote_no_ligne(no_som,lvbase);
	    lvbase_add(var_entrant,no_som,lvbase);
	}
	else {
	    if (VARIABLE_UNDEFINED_P(var_entrant))
	    {
		sommets_rm(*sys);
		*sys = NULL;
		result = false;
#ifdef TRACE
		printf (" --- alg. dual - le probleme n'est pas borne \n");
#endif
	    }
	}
    }
    else 
    {
	result = false;
#ifdef TRACE
	printf (" --- alg. dual - plus de ligne pivot possible \n");
#endif
    }

    return (result);

}

/* bool dual_pivot(Psysteme sys, Psysteme * syst_result,
 *                    Psommet * fonct, int * nb_som, Pvecteur * lvbase):
 * Algorithme dual du simplexe 
 *  (c.f. Programmation mathematique - tome 1 . M.MINOUX (83) )  
 *
 *
 *  resultat retourne par la fonction :
 *
 *  bool           : == true  si le systeme a une solution
 *                   == false sinon
 *
 *  Le parametre SYST_RESULT est le systeme final  (dont on peut
 *  deduire la valeur des differentes variables du systeme)     .
 *  S'il est vide, c'est que le systeme initial est non faisable.    
 *
 *  Les parametres de la fonction :
 *
 *! Psysteme sys    	: systeme lineaire 
 *  Psysteme syst_result: systeme lineaire resultant de l'execution
 *! Psommet fonct  	: fonction economique du  programme lineaire
 *! Pvecteur lvbase	: liste des variables de base du systeme
 *! int     nb_som 	: nombre de contraintes du systeme
 *
 */
bool dual_pivot(sys,syst_result,fonct,nb_som,lvbase)
Psysteme sys;
Psysteme *syst_result;
Psommet *fonct;
int *nb_som;
Pvecteur *lvbase;
{
    Psommet som = (Psommet) NULL;
    int nbvars = 0;
    Pbase b = BASE_NULLE;
#ifdef TRACE
    printf(" ** algorithme dual du simplexe \n");
#endif
    if (sys != (Psysteme) NULL) {
	nbvars = sys->dimension;
	b = base_dup(sys->base);
	som = sys_som_conv(sys,nb_som);

	som = eq_in_ineq(&som,nb_som,lvbase);
	som = var_ecart_sup(som,*nb_som,lvbase,&nbvars,&b);

	while( (som != (Psommet) NULL) && 
	       (dual_pivot_pas(&som,lvbase,*nb_som,*fonct,&(sys->dimension),
			       &(sys->base))	== true) );

	*syst_result = som_sys_conv(som);
	if (*syst_result) {
	    (*syst_result)->dimension = sys->dimension;
	    (*syst_result)->base = base_dup(sys->base);
	}
    }
    if(som == (Psommet) NULL);
    else  sommets_rm(som);
  
    return((boolean) (som != (Psommet) NULL));
}

/* Psysteme dual(Psysteme ps, Psommet fonct):
 * Algorithme dual du simplexe avec initialisation des parametres.
 *
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme 		: systeme final dont on peut deduire la valeur
 * 			  des differentes variables du systeme.
 *
 *  NULL			: si le systeme initial est non faisable.    
 *
 *  Les parametres de la fonction :
 *
 *! Psysteme sys    	: systeme lineaire 
 *! Psommet fonct  	: fonction economique du  programme lineaire
 *
 */
Psysteme dual(sys,fonct)
Psysteme sys;
Psommet fonct;
{

    Psysteme syst_result = NULL;
    Pvecteur lvbase = NULL;
    int nb_som = 0;

    dual_pivot(sys,&syst_result,&fonct,&nb_som,&lvbase);
    vect_rm(lvbase);
    return(syst_result);
}
   
Psysteme dual_positive(sys,fonct)
Psysteme sys; /* Psommet ??? */
Psommet fonct;
{

    Psysteme syst_result = NULL;
    Pvecteur lvbase = NULL;
    int nb_som = 0;

    dual_pivot(sys,&syst_result,&fonct,&nb_som,&lvbase);
    if (!sol_positive(sys,lvbase,nb_som)) { 
	sc_rm(syst_result);
	syst_result=NULL;}

    vect_rm(lvbase);
    return(syst_result);
}
   
