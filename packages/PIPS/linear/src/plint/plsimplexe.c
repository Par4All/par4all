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


#define TRACE
/* pour recuperer les declarations des fonctions de conversion de
 * sc en liste de sommets et reciproquement, bien que ca casse le
 * DAG des types de donnees
 */
#include "ray_dte.h"
#include "polyedre.h"
#include "matrix.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc(s)

/* Psommet lignes_entrant(Psommet sys, Variable var, int nb_som, int * no_som):
 * algorithme primal du simplexe - recherche de la nouvelle ligne pivot 
 *
 *  resultat retourne par la fonction :
 *
 *  Psommet        : contrainte correspondant a la nouvelle ligne pivot
 *  NULL              si l'on n'en a pas trouve
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  int     nb_som : nombre de contraintes du systeme
 *  int     no_som : place de la contrainte dans le systeme (no de la ligne)
 *  int     var    : variable pivot qui a ete choisie  
 */
Psommet lignes_entrant(sys,var,nb_som,no_som)
Psommet sys;
Variable var;
int nb_som;
int *no_som;
{

    Psommet ps = NULL;
    Psommet result = NULL;
    double ncoeff = 0;
    double min = 0;
    Value mod = 0;
    Value mod2 = 0;
    Value cst = 0;

#ifdef TRACE 
    printf (" **** alg. primal - recherche ligne pivot \n");
#endif
    for (ps = sys; 
	 ps != NULL && (value_negz_p(vect_coeff(var,ps->vecteur))
			|| value_pos_p(vect_coeff(TCST,ps->vecteur)));
	 ps = ps->succ,nb_som --);

    /* initialisation du minimum 	*/
    if (ps != NULL)
    {
	Value cv1 = 0;
	Value tmp1 = value_uminus(vect_coeff(TCST,ps->vecteur)),
	      tmp2 = vect_coeff(var,ps->vecteur),tmp;
	tmp = value_div(tmp1,tmp2);
	min = VALUE_TO_DOUBLE(tmp);
	mod = value_mod(tmp1,tmp2);
	*no_som = nb_som;
	result = ps;
	nb_som --;

	printf ("min = %f, mod = ",min);
	print_Value(mod); printf(", nb_som = %d\n",nb_som+1);

	for (ps = ps->succ; ps != NULL; ps = ps->succ) {
	    cv1 = vect_coeff(var,ps->vecteur);
	    cst = value_uminus(vect_coeff(TCST,ps->vecteur));
	    if (value_pos_p(cv1) && value_posz_p(cst))  { 
		Value tmp = value_div(cst,cv1);
		ncoeff = VALUE_TO_DOUBLE(tmp);
		mod2 = value_mod(cst,cv1) ;
		if ( ncoeff<min  || ( ncoeff ==min && 
				     value_lt(mod2,mod))) {
		    min = ncoeff;
		    mod = mod2;
		    result = ps;
		    *no_som = nb_som;
		}
	    }
	    nb_som --;
	}
    }
    return (result);
}

/* Variable var_pivots(Psommet fonct):
 * algorithme primal du simplexe - recherche de la variable pivot entrant dans
 * la base
 *
 *  resultat retourne par la fonction :
 *
 *  int            : variable pivot
 *
 *  Les parametres de la fonction :
 *
 *  Psommet fonct  : fonction economique du  programme lineaire
 */
Variable var_pivots(fonct)
Psommet fonct;
{
    Pvecteur pv1 = NULL;
    Pvecteur pv2 = NULL;
    Value coeff = 0;
    Value min = 0;
    Variable var = NULL;

#ifdef TRACE
    printf (" **** alg. primal - recherche de la variable pivot \n");
#endif

    if (fonct != NULL) {
	if (fonct->vecteur->succ != NULL) {
	    Value cst = vect_coeff(TCST,fonct->vecteur);
	    Pvecteur pv3;
	    vect_chg_coeff(&(fonct->vecteur),TCST,0);
	    pv1 = vect_sort(fonct->vecteur, vect_compare);
	    for(pv3 =fonct->vecteur; pv3->succ!=NULL;pv3=pv3->succ);
	    pv3->succ = vect_new(TCST,cst);
	}
	else pv1 = fonct->vecteur;
	for (;pv1 != NULL 
	     && ((pv1->var == NULL) || value_posz_p(pv1->val)) ; 
	     pv1=pv1->succ);

	/* initialisation du minimum	*/
	if ( pv1 != NULL) {
	    min = pv1->val;
	    var = pv1->var;
	}
	for (pv2 = pv1; pv2 != NULL; pv2 = pv2->succ) {
	    if ( (pv2->var != NULL) && value_neg_p((coeff = pv2->val))
		&& (value_lt(coeff,min))) {
		min = coeff;
		var = pv2->var;
	    }
	}

	vect_rm(pv1);
    }
 
#ifdef TRACE
    printf("variable pivot choisie: %s\n",var);
#endif
   
    return (var);
}

/* Psommet primal_pivot(Psommet sys, Pvecteur * lvbase, int nb_som,
 *                      Psommet fonct):
 * algorithme primal du simplexe
 *
 *  resultat retourne par la fonction :
 *
 *  Psommet        : systeme lineaire initial modifie.
 *  NULL		   : si le systeme de depart est non faisable.       
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Psommet fonct  : fonction economique du  programme lineaire
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 */
Psommet primal_pivot(sys,lvbase,nb_som,fonct)
Psommet sys;
Pvecteur *lvbase;
int nb_som;
Psommet fonct;
{
    Psommet lpivot =NULL;
    Psommet ps = sys;
    Pvecteur pv = VECTEUR_NUL;
    Variable var_entrant = NULL;
    int no_som = 0;
    bool non_borne = false;
    Psysteme ps2;
#ifdef TRACE
    printf (" *** algorithme primal du simplexe \n");
#endif

    /* recherche de la variable pivot 	*/
    while (ps != NULL && !non_borne && ((var_entrant = var_pivots(fonct)) != NULL)) {
	/* recherche de la ligne pivot	*/
	lpivot = lignes_entrant(ps,var_entrant,nb_som,&no_som);
	if (lpivot != NULL) {
	    /* operation PIVOT	*/
	    pivoter(ps,lpivot,var_entrant,fonct);
	    lvbase_ote_no_ligne(no_som,lvbase);
	    lvbase_add(var_entrant,no_som,lvbase);
	}
	else {				
	    /* cas ou la solution est non bornee: on renvoit le 
	       systeme car on considere qu'il admet une solution */
#ifdef TRACE
	    printf (" cas 4 ou le probleme n'est pas borne \n");
#endif
	    non_borne = true;
	}
	if (((ps2 = som_sys_conv(ps))) != NULL)
	    sc_fprint(stdout,ps2,*variable_default_name);
    }

    if (fonct)
	for (pv = fonct->vecteur;
	     ((pv!=NULL) && ((pv->var == NULL) || value_posz_p(pv->val)));
	     pv=pv->succ);
#ifdef TRACE
    if (pv == NULL)
	printf (" ---- alg. primal - plus de variable pivot possible \n");
    else
	printf (" ---- alg. primal - pas de solution \n");
#endif

    if (pv != NULL && !non_borne) { 
	sommets_rm(ps);
	ps = NULL;
    }

    sommets_normalize(ps);
    return (ps);
}

/* Psysteme primal(Psysteme ps, Psommet fonct):
 * Algorithme primal du simplexe avec fonction economique initialisee
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme        : systeme lineaire initial modifie.
 *
 *  Si le systeme final est vide, c'est que le systeme initial est non 
 *  faisable
 *
 *  Les parametres de la fonction :
 *
 *! Psysteme ps    : systeme lineaire 
 *! Psommet fonct   : fonction economique du  programme lineaire
 */
Psysteme primal(ps,fonct)
Psysteme ps;
Psommet fonct;
{
    Psysteme ps2 =NULL;
    Psommet ps1 = NULL;
    Pvecteur lvbase = NULL;
    Pvecteur lvsup = NULL;
    int nb_som = 0;
    int nbvars =0;
    Pbase b = BASE_NULLE;

#ifdef TRACE
    printf (" ** Algorithme du simplexe avec fonct. econom.\n");
#endif

    if (ps != NULL && fonct!= NULL)
    {

#ifdef TRACE
	sc_fprint (stdout,ps,*variable_default_name);
#endif
	nbvars = ps->dimension;
	b = base_dup(ps->base);
	ps1 = sys_som_conv(ps,&nb_som);

	if((ps1 = eq_in_ineq(&ps1,&nb_som,&lvbase)) != NULL) {
	    ps1 = add_var_sup(ps1,nb_som,&lvbase,&lvsup,&nbvars,
				&b,fonct);
	    ps1 = primal_pivot(ps1,&lvbase,nb_som,fonct);
	}
	else
	    printf (" le systeme est non faisable \n");
    }
    if ((ps2 = som_sys_conv(ps1)) != NULL) {
	ps2->dimension = ps->dimension;
	ps2->base = base_dup(ps->base);
	sommets_rm(ps1);
    }
    vect_rm(lvbase);
#ifdef TRACE
    sc_fprint(stdout,ps2,*variable_default_name);
#endif

    return (ps2);
}


bool primal_positive(ps,fonct)
Psysteme ps;
Psommet fonct;
{
    Psysteme ps2 =NULL;
    Psommet ps1 = NULL;
    Pvecteur lvbase = NULL;
    Pvecteur lvsup = NULL;
    int nb_som = 0;
    int nbvars =0;
    Pbase b = BASE_NULLE;
    bool result = false;

#ifdef TRACE
    printf (" ** Algorithme du simplexe avec fonct. econom.\n");
#endif

    if (ps != NULL && fonct!= NULL) {
#ifdef TRACE
	sc_fprint (stdout,ps,*variable_default_name);
#endif
	nbvars = ps->dimension;
	b = base_dup(ps->base);
	ps1 = sys_som_conv(ps,&nb_som);

	if((ps1 = eq_in_ineq(&ps1,&nb_som,&lvbase)) != NULL) {
	    ps1 = add_var_sup(ps1,nb_som,&lvbase,&lvsup,&nbvars,
			      &b,fonct);
	    ps1 = primal_pivot(ps1,&lvbase,nb_som,fonct);
	}
	else
	    printf (" le systeme est non faisable \n");
    }
    if ((ps2 = som_sys_conv(ps1)) != NULL) {
	ps2->dimension = nbvars;
	ps2->base = base_dup(b);
    }

    if (!sol_positive_simpl(ps1,lvbase,lvsup,nb_som)) { 
	sc_rm(ps2);
	ps2=NULL;}

#ifdef TRACE
    if (ps2) sc_fprint(stdout,ps2,*variable_default_name);
#endif
    if (ps2) result = true;
    sc_rm(ps2);
    sommets_rm(ps1);
    vect_rm(lvbase);

    return (result);
}
