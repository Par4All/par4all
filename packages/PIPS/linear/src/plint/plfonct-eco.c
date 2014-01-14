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

/*
 *  Creation d'une fonction economique de base. 
 *
 * L'utilisateur devra creer la fonction de minimisation (Pvecteur) qu'il
 * desire et initialiser le champ VECTEUR de la fonction economique de base.
 */
Psommet fonct_init()
{
    Psommet fonct_eco;

#ifdef TRACE
    printf(" ** creation fonction economique de base \n");
#endif
    fonct_eco = sommet_new();
    fonct_eco->denominateur =VALUE_ONE;
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_init");
    *(fonct_eco->eq_sat) = 0;
    return (fonct_eco);
}


/*
 * Creation d'une fonction economique permettant de calculer la valeur minimum
 * du premier element de la liste des variables du systeme
 *
 *  Les parametres de la fonction :
 *
 *  int nbvars 	   : nombre de variables du systeme (unused)
 *  Pbase    b   : liste des variables du systeme
 */
/*ARGSUSED*/
Psommet fonct_min(nbvars,b)
int nbvars;
Pbase b;
{
    Psommet fonct_eco;

    fonct_eco = sommet_new();
    fonct_eco->denominateur = VALUE_ONE;
    fonct_eco->vecteur = vect_new(vecteur_var(b),VALUE_ONE);
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_min");
    *(fonct_eco->eq_sat) = 0;
    return (fonct_eco);
}


Psommet fonct_max(nbvars,b)
int nbvars;
Pbase b;
{
    Psommet fonct_eco;

    fonct_eco = sommet_new();
    fonct_eco->denominateur = VALUE_ONE;
    fonct_eco->vecteur = vect_new(vecteur_var(b),VALUE_MONE);
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_max");
    *(fonct_eco->eq_sat) = 0;
    return (fonct_eco);
}




/*
 * creation d'une fonction economique permettant de calculer le minimum de la
 * somme de toutes les variables du systeme
 *
 *  Les parametres de la fonction :
 *
 *  int nbvars 	   : nombre de variables du systeme
 *  Pbase b   : liste des  variables du systeme
 */

Psommet fonct_min_all(nbvars,b)
int nbvars;
Pbase b;
{
    Psommet fonct_eco;
    register int i=0;
    Pvecteur pv;

#ifdef TRACE
    printf(" ** creation fonction economique \n");
#endif
    fonct_eco =sommet_new();
    fonct_eco->denominateur = VALUE_ONE;
    fonct_eco->vecteur = vect_new(vecteur_var(b),VALUE_ONE);
    for (i = 1, pv= b->succ;i< nbvars && !VECTEUR_NUL_P(pv); i++, pv=pv->succ)
	vect_add_elem (&(fonct_eco->vecteur),vecteur_var(pv),VALUE_ONE);
	
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),
				      INTEGER,"fonct_min_all");
    *(fonct_eco->eq_sat) = 0;
    return (fonct_eco);
}

/*
 * creation d'une fonction economique permettant de calculer le maximum de la
 * somme de toutes les variables du systeme
 *
 *  Les parametres de la fonction :
 *
 *  int nbvars 	   : nombre de variables du systeme
 *  Pbase b   : liste des  variables du systeme
 */



Psommet fonct_max_all(nbvars,b)
int nbvars;
Pbase b;
{
	Psommet fonct_eco;
	Pvecteur pv = VECTEUR_NUL;
#ifdef TRACE
	printf(" ** creation fonction economique \n");
#endif
	fonct_eco =sommet_new();
	fonct_eco->denominateur = VALUE_ONE;
	fonct_eco->vecteur = vect_new(vecteur_var(b),VALUE_MONE);
	for (pv= b->succ;!VECTEUR_NUL_P(pv);pv=pv->succ)
	    vect_add_elem (&(fonct_eco->vecteur),vecteur_var(pv),VALUE_MONE);
	fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),
					  INTEGER,"fonct_max_all");
	*(fonct_eco->eq_sat) = 0;
	return (fonct_eco);
}



/*
 * Creation d'une fonction economique permettant de calculer le minimum de la
 * difference des deux premieres variables du systeme
 *
 *  Les parametres de la fonction :
 *
 *  int nbvars 	   : nombre de variables du systeme ( unused)
 *  Pbase b   : liste des  variables du systeme
 */
/*ARGSUSED*/
Psommet fonct_min_d(nbvars,b)
int nbvars;
Pbase b;
{
	Psommet fonct_eco;

	fonct_eco =sommet_new();
	fonct_eco->denominateur = VALUE_ONE;
	fonct_eco->vecteur = vect_new(vecteur_var(b),VALUE_ONE);
	vect_add_elem(&(fonct_eco->vecteur),vecteur_var(b->succ),VALUE_MONE);
	fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_min_d");
	*(fonct_eco->eq_sat) = 0;
	return (fonct_eco);
}

/*
 * Creation d'une fonction economique permettant de calculer le maximum de la
 * difference des deux premieres variables du systeme
 *
 *  Les parametres de la fonction :
 *
 *  int nbvars 	   : nombre de variables du systeme
 *  Pbase b   : liste des  variables du systeme
 */
/*ARGSUSED*/
Psommet fonct_max_d(nbvars,b)
int nbvars;
Pbase b;
{
    Psommet fonct_eco;

    fonct_eco =sommet_new();
    fonct_eco->denominateur = VALUE_ONE;
    fonct_eco->vecteur = vect_new(vecteur_var(b),VALUE_MONE);
    vect_add_elem(&(fonct_eco->vecteur),vecteur_var(b->succ),VALUE_ONE);
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_max_d");
    *(fonct_eco->eq_sat) = 0;
    return (fonct_eco);
}

/*
 * Lecture au clavier de la fonction economique que l'on veut creer
 *
 *  Les parametres de la fonction :
 *
 *  int nbvars 	   : nombre de variables du systeme
 *  Pbase b   : liste des  variables du systeme
 */

Psommet fonct_read(nbvars,b)
int nbvars;
Pbase b;
{
	Psommet fonct_eco;
	register int i=0;
	Value d;
Pvecteur pv;

	fonct_eco =sommet_new();
	fonct_eco->denominateur = VALUE_ONE;
	printf (" *** creation d'une fonction economique \n");
	printf (" pour la premiere variable : 1 ou -1 ");
	scan_Value(&d);
	fonct_eco->vecteur = vect_new(vecteur_var(b),d);
	for (i = 1 ,pv= b->succ;i< nbvars && !VECTEUR_NUL_P(pv); 
	     i++,pv=pv->succ)
	{
		printf (" pour l'indice %d : 1 ou -1 ",i+1);
		scan_Value(&d);
		vect_add_elem (&(fonct_eco->vecteur),vecteur_var(pv),d);
	}
	printf ("\n");

	fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_read");
	*(fonct_eco->eq_sat) = 0;
	return (fonct_eco);
}
