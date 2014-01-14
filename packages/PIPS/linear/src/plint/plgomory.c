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

/* Psommet gomory_trait_eq(Psommet eq, Variable var):
 * Cette fonction utilise une contrainte du systeme en vue d'obtenir une coupe
 * de Gomory servant a rendre entiere une des variables du systeme
 *
 *  resultat retourne par la fonction :
 *
 *  Psommet        : inegalite correspondant a la coupe qu'il faut ajouter au 
 * 			systeme pour rendre la variable entiere
 *
 *  Les parametres de la fonction :
 *
 *  Psommet eq     : contrainte du systeme dont la variable de base est 
 *                   non entiere 
 *  Variable var   : variable de base non entiere
 */
Psommet gomory_trait_eq(eq,var)
Psommet eq;
Variable var;
{
    Psommet ps1 = NULL;
    Pvecteur pv1,pv2;
    Value in1;
    Value b0;
    Value lambda = VALUE_ONE;
    int sgn_op= 1;
    Value d = VALUE_ONE;

#ifdef TRACE
    printf(" ** Gomory - ajout d'une eq. verifiant les cond. de Gomory \n");
#endif

    if (eq != NULL && var != NULL)
    {
	/* calcul du determinant D	*/
	Value p1, p2,tmp;
	p1 = vect_coeff(var,eq->vecteur);
	value_absolute(p1);
	p2 =vect_pgcd_all(eq->vecteur);
	value_absolute(p2);

	d = value_div(p1,p2);

	/* calcul de la coupe de Gomory	*/
	ps1= (Psommet)MALLOC(sizeof(Ssommet),SOMMET,"gomory_trait_eq");
	ps1->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"gomory_trait_eq");
	pv1 = vect_dup(eq->vecteur);
	ps1->denominateur = VALUE_ONE;
	*(ps1->eq_sat) = -1;
	ps1->succ = NULL;
	vect_chg_coeff(&pv1,var,VALUE_ZERO);
	ps1->vecteur = pv1;
	tmp = vect_coeff(var,eq->vecteur);
	sgn_op = -value_sign(tmp);
	if (value_notzero_p(d)) {
	    for (pv2=pv1;pv2 != NULL; pv2= pv2->succ) {
		if (sgn_op==-1)
		    value_oppose(pv2->val);
		value_modulus(pv2->val,d);
		if (value_neg_p(pv2->val))
		    value_addto(pv2->val,d);
	    }
	}
	b0 = vect_coeff(TCST,pv1);
	value_absolute(b0);

	/* calcul de lambda = coefficient multiplicateur permettant
	   d'obtenir la coupe optimale		*/

	if (value_notzero_p(b0)) {
	    in1 = value_div(d,b0);
	    lambda = value_zero_p(value_mod(d,b0))? (in1-1) : in1;
	    if (value_zero_p(lambda)) 
		lambda = VALUE_ONE;
	}

	/* optimisation de la coupe de Gomory	*/
	if (value_gt(lambda,VALUE_ONE))
	    for (pv2 = pv1;pv2!= NULL; pv2=pv2->succ)
	    {
		value_product(pv2->val,lambda);
		value_modulus(pv2->val,d);
	    }

	vect_chg_sgn(pv1);

	ps1->vecteur = vect_clean(ps1->vecteur);
    }
    return (ps1);
}

/* Psommet gomory_eq(Psommet *sys, Pvecteur lvbase, int nb_som,
 *                   int * no_som, Variable * var):
 * Recherche d'une contrainte dont la variable de base est non entiere
 *
 *  resultat retourne par la fonction :
 *
 *  Psommet        :  contrainte dont la variable de base est non entiere
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 *  int     no_som : place de la contrainte dans le systeme (no de la ligne)
 *  Variable    var: variable non entiere  
 */
Psommet gomory_eq(sys,lvbase,nb_som,no_som,var)
Psommet *sys;
Pvecteur lvbase;
int nb_som;
int *no_som;
Variable *var;
{
    Psommet eq=NULL;
    Psommet result= NULL;
    int nb_som1 = nb_som;
    Value a;
    Value max = VALUE_ZERO;
    Variable v1 = NULL;
    bool borne = true;

#ifdef TRACE
    printf(" **  Gomory - recherche d'une variable non_entiere \n");
#endif
    *var = NULL;

    for (eq = *sys ;eq != NULL && borne; eq=eq->succ) {
	Value b0,den2;

	if ((borne = test_borne(eq))) {
	    if (((v1 = find_vbase(eq,lvbase))) !=  NULL) {
		den2 = vect_coeff(v1,eq->vecteur);
		b0 = value_uminus(vect_coeff(TCST,eq->vecteur));

		/* on a trouve une variable de base non entiere   */
		/* on choisit celle de plus grand denominateur     */

		if (value_notzero_p(value_mod(b0,den2))) { 
		    Value p1=eq->denominateur;
		    Value p2 = value_abs(b0);
		    Value p3 = pgcd(p2,p1);

		    a = value_div(p1,p3);
		    value_absolute(a);
		    if (value_gt(a,max)) {
			max = a;
			*var = v1;
			*no_som = nb_som1;
			result = eq;
		    }
		}
	    }
	    nb_som1 --;
	}
	else {
#ifdef TRACE	
	    printf (" --  le systeme est non borne !!! \n ");
#endif
	    sommets_rm(*sys);
	    *sys = NULL;
	}
    }
    if (value_notzero_p(max))
	return (result);
    else
	return (NULL);
}
