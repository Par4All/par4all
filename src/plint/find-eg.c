 /* package plint */

#include <stdio.h>
#include <malloc.h>

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

/* Psysteme find_eg(Psysteme sys):
 * Recherche des egalites du systeme en nombres entiers sys
 * parmi ses inegalites
 *
 * Le systeme initial est modifie.
 *
 * Les parametres de la fonction :
 *
 * !Psysteme ps    : systeme lineaire 
 */
Psysteme find_eg(sys)
Psysteme sys;
{
    Psysteme sys2=NULL;
    Psysteme sys3=NULL;
    Psommet som=NULL;
    Psommet ps = NULL ;
    Psommet fonct = fonct_init();
    Psolution sol1 = NULL;
    Pvecteur pv;
    int nbvars;
    int exnbv;
    int nbsom;
    /*
     * respectivement la valeur de la constante de l'equation
     * et la valeur de la constante de la fonction economique
     */
    int b0,f0;

    sys = sc_elim_redond(sys);
    if (sys != (Psysteme) NULL) {
	nbvars = sys->dimension;
	ps = sys_som_conv(sys,&nbsom);
	exnbv = nbvars;
	for (som = ps;((nbsom != 0) && (som!= NULL));som=som->succ)

	{
	    if (*(som->eq_sat) == -1) {
		/*
		 * on choisit une inequation A.x = b0
		 */

		nbvars = exnbv;
		/*
		 *on cree une fonction economique identique au
		 * membre gauche de l'inequation
		 *  fonct->vecteur = A.x
		 */
		fonct->denominateur = 1;
		fonct->vecteur = vect_dup(som->vecteur);
		vect_chg_coeff(&(fonct->vecteur),TCST,0);
		b0 = - vect_coeff(TCST,som->vecteur);
		pv = vect_dup(som->vecteur);

		/*
		 * on enleve l'inegalite choisie du systeme
		 */
		vect_rm(som->vecteur);
		som->vecteur = VECTEUR_NUL;

		if ((sys2 = som_sys_conv(ps)) != (Psysteme) NULL) {
		    sys2->dimension = sys->dimension;
		    sys2->base = vect_dup(sys->base);
		}

		/*
		 * on calcule le nouveau programme linaire
		 * ainsi forme
		 */
		sys3 = plint(sys2,fonct,&sol1);
		/*
		 * si la solution entiere trouvee donne pour
		 * minimum a la fonction economique la valeur
		 * de la constante b0 ==> cette inegalite peut
		 * etre transformee en egalite
		 */
		if (sys3 != NULL && fonct != NULL) {
		    f0 =  vect_coeff(TCST,fonct->vecteur)/fonct->denominateur;
		    if (b0 == f0)
			*(som->eq_sat) = 1;
		}

		som->vecteur = pv;
		if (fonct!= NULL) {
		    vect_rm(fonct->vecteur);
		    fonct->vecteur = NULL;
		}
		sc_rm(sys2);
		sc_rm (sys3);
		sys2 = NULL;
		sys3 = NULL;
	    }
	}
	
	if ((sys2 = som_sys_conv(ps)) != (Psysteme) NULL) {
	    sys2->dimension = sys->dimension;
	    sys2->base = vect_dup(sys->base);
	}
	sommets_rm(ps);
    }
    return(sys2);
}

