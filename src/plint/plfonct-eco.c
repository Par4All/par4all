 /* package plint */

#include <stdio.h>
#include <malloc.h>

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
    fonct_eco->denominateur =1;
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_init");
    *(fonct_eco->eq_sat) = NULL;
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
    fonct_eco->denominateur =1;
    fonct_eco->vecteur = vect_new(vecteur_var(b),1);
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_min");
    *(fonct_eco->eq_sat) = NULL;
    return (fonct_eco);
}


Psommet fonct_max(nbvars,b)
int nbvars;
Pbase b;
{
    Psommet fonct_eco;

    fonct_eco = sommet_new();
    fonct_eco->denominateur =1;
    fonct_eco->vecteur = vect_new(vecteur_var(b),-1);
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_max");
    *(fonct_eco->eq_sat) = NULL;
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
    fonct_eco->denominateur =1;
    fonct_eco->vecteur = vect_new(vecteur_var(b),1);
    for (i = 1, pv= b->succ;i< nbvars & !VECTEUR_NUL_P(pv); i++, pv=pv->succ)
	vect_add_elem (&(fonct_eco->vecteur),vecteur_var(pv),1);
	
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),
				      INTEGER,"fonct_min_all");
    *(fonct_eco->eq_sat) = NULL;
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
	register int i=0;
Pvecteur pv = VECTEUR_NUL;
#ifdef TRACE
	printf(" ** creation fonction economique \n");
#endif
	fonct_eco =sommet_new();
	fonct_eco->denominateur =1;
	fonct_eco->vecteur = vect_new(vecteur_var(b),-1);
	for (pv= b->succ;!VECTEUR_NUL_P(pv);pv=pv->succ)
		vect_add_elem (&(fonct_eco->vecteur),vecteur_var(pv),-1);
	fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),
					  INTEGER,"fonct_max_all");
	*(fonct_eco->eq_sat) = NULL;
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
	fonct_eco->denominateur =1;
	fonct_eco->vecteur = vect_new(vecteur_var(b),1);
	vect_add_elem(&(fonct_eco->vecteur),vecteur_var(b->succ),-1);
	fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_min_d");
	*(fonct_eco->eq_sat) = NULL;
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
    fonct_eco->denominateur =1;
    fonct_eco->vecteur = vect_new(vecteur_var(b),-1);
    vect_add_elem(&(fonct_eco->vecteur),vecteur_var(b->succ),1);
    fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_max_d");
    *(fonct_eco->eq_sat) = NULL;
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
	int d;
Pvecteur pv;

	fonct_eco =sommet_new();
	fonct_eco->denominateur =1;
	printf (" *** creation d'une fonction economique \n");
	printf (" pour la premiere variable : 1 ou -1 ");
	scanf( "%d",&d);
	fonct_eco->vecteur = vect_new(vecteur_var(b),d);
	for (i = 1 ,pv= b->succ;i< nbvars && !VECTEUR_NUL_P(pv); i++,pv=pv->succ)
	{
		printf (" pour l'indice %d : 1 ou -1 ",i+1);
		scanf( "%d",&d);
		vect_add_elem (&(fonct_eco->vecteur),vecteur_var(pv),d);
	}
	printf ("\n");


	fonct_eco->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"fonct_read");
	*(fonct_eco->eq_sat) = NULL;
	return (fonct_eco);
}
