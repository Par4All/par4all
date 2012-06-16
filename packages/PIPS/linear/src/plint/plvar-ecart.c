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

/* Psommet eq_in_ineq(Psommet * sys, int * nb_som, Pvecteur * lvbase):
 * Transformation des egalites du systeme en inegalites.
 * N egalites du systeme seront transformees en N+1 inegalites.
 *
 *  resultat retourne par la fonction :
 *
 *  Psommet        : systeme lineaire initial modifie.
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 */
Psommet eq_in_ineq(sys,nb_som,lvbase)
Psommet *sys;
int *nb_som;
Pvecteur *lvbase;
{
    Psommet eq=NULL;
    Psommet sys2;
    Psommet som2;
    Pvecteur pv3=NULL;
    Pvecteur pvsom=NULL;
    Pvecteur pvsom2 = NULL;
    bool egalite = false;
    Value den = VALUE_ONE;

#ifdef TRACE

    Psysteme ps1=NULL;
    Pvecteur pvec;

    printf(" ** transformation des equations en inequations \n");
#endif


    sys2 = *sys;

    for (eq = sys2;eq!= NULL;eq=eq->succ) {
	if (*(eq->eq_sat) == 1) {
	    egalite = true;
	    *(eq->eq_sat) = -1;
	    pv3 = vect_dup(eq->vecteur);
	    vect_chg_sgn(pv3);
	    (void) vect_multiply(pv3,den);
	    value_product(den,eq->denominateur);
	    (void) vect_multiply(pvsom,eq->denominateur);
	    pvsom2 = vect_add(pvsom,pv3);
	    vect_rm(pv3);
	    vect_rm(pvsom);
	    pvsom = pvsom2;
	}
    }
    if (egalite) {
	som2 = (Psommet)MALLOC(sizeof(Ssommet),SOMMET,"eq_in_ineq");
	som2->eq_sat =(int *)MALLOC(sizeof(int),INTEGER,"eq_in_ineq");
	*(som2->eq_sat) = -1;
	som2->denominateur =den ;
	som2->vecteur = pvsom;
	som2->succ = NULL;
	sommet_add(&sys2,som2,nb_som);
    }

#ifdef TRACE
    ps1 = som_sys_conv(sys2);

    sc_fprint(stdout,ps1,*variable_default_name);
    printf (" -- variables de base :");
    for (pvec = *lvbase;pvec!= NULL;pvec=pvec->succ)
    {
	printf(" (0x%x,", (unsigned int) pvec->var);
	print_Value(pvec->val);
	printf("), ");
    }
    printf("\n");
    sc_rm(ps1);

#endif

    sommets_normalize(sys2);
    return(sys2);

}

/* Psommet var_ecart_sup(Psommet sys, int nb_som, Pvecteur * lvbase,
 *                       int * nbvars, Pbase *b):
 * ajout des variables d'ecart necessaires a la transformation du systeme
 * A . X <= B en A'. X' = B .
 *
 *  resultat retourne par la fonction :
 *
 *  Psommet        : systeme lineaire initial modifie
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 *  int     nbvars : nombre de variables du systeme
 *  Pbase b  : liste des  variables du systeme 
 */
Psommet var_ecart_sup(sys,nb_som,lvbase,nbvars,b)
Psommet sys;
int nb_som;
Pvecteur *lvbase;
int *nbvars;
Pbase *b;
{
    Psysteme ps1=NULL;
    Psommet ineg;
    Psommet sys2 = sys;
    Variable nv;

#ifdef TRACE
    Pvecteur pvec;

    printf(" ** ajout des variables d'ecart \n");
#endif
    if (sys) {
	ps1 = som_sys_conv(sys);
	if (ps1 != NULL) {
	    ps1->dimension = *nbvars;
	    ps1->base = base_dup(*b);
		
	    for (ineg = sys2;ineg!= NULL;ineg=ineg->succ) {
		if (*(ineg->eq_sat) == -1) {
		    nv = creat_new_var(ps1);
		    vect_add_elem(&(ineg->vecteur),nv,VALUE_ONE);
		    lvbase_add(nv,nb_som,lvbase);
		    *(ineg->eq_sat) = 1;
		    *b = vect_add_variable(*b,nv);
		    (*nbvars)++; 
		}
		nb_som --;
	    }

#ifdef TRACE
	    printf (" -- variables de base :");
	    for (pvec = *lvbase;pvec!= NULL;pvec=pvec->succ)
	    {
		printf(" (0x%x,",(unsigned int)pvec->var);
		print_Value(pvec->val);
		printf("), ");
	    }
	    printf("\n");
	    ps1 = som_sys_conv(sys2);
	    sc_fprint(stdout,ps1,*variable_default_name);
#endif
	    sc_rm(ps1);
	}
    }
    return(sys2);
}


Psommet add_var_sup(sys,nb_som,lvbase,lvsup,nbvars,b,fonct)
Psommet sys;
int nb_som;
Pvecteur *lvbase;
Pvecteur *lvsup;
int *nbvars;
Pbase *b;
Psommet fonct;
{
    Psysteme ps1=NULL;
    Psommet ineg;
    Psommet sys2 = sys;
    Variable nv;

#ifdef TRACE
    Pvecteur pvec;

    printf(" ** ajout des variables suupplementaire \n");
#endif
    if (sys) {
	ps1 = som_sys_conv(sys);
	if (ps1 != NULL) {
	    ps1->dimension = *nbvars;
	    ps1->base = base_dup(*b);
		
	    for (ineg = sys2;ineg!= NULL;ineg=ineg->succ) {
		if (*(ineg->eq_sat) == -1) {
		    nv = creat_new_var(ps1);
		    vect_add_elem(&(ineg->vecteur),nv,VALUE_ONE);
		    *b = vect_add_variable(*b,nv);
		    (*nbvars) ++;
		    if (value_pos_p(vect_coeff(TCST,ineg->vecteur))) {
			nv = creat_new_var(ps1);
			vect_add_elem(&(ineg->vecteur),nv,VALUE_MONE);
			vect_add_elem(&(fonct->vecteur),nv,
				      int_to_value(999)); /* ??? 999 ??? */
			lvbase_add(nv,nb_som,lvsup);
		    }
		    lvbase_add(nv,nb_som,lvbase);

		    *(ineg->eq_sat) = 1;
		    *b = vect_add_variable(*b,nv);
		    (*nbvars) ++;
		}
		nb_som --;
	    }

#ifdef TRACE
	    printf (" -- variables de base :");
	    for (pvec = *lvbase;pvec!= NULL;pvec=pvec->succ)
	    {
		printf(" (0x%x,",(unsigned int)pvec->var);
		print_Value(pvec->val);
		printf("), ");
	    }
	    printf("\n");
	    ps1 = som_sys_conv(sys2);
	    sc_fprint(stdout,ps1,*variable_default_name);
#endif

	    sc_rm(ps1);
	}
    }
    return(sys2);
}
