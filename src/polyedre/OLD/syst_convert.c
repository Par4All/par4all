/* package sommet: passage d'un systeme de contraintes a une liste de sommets
 * et d'une liste de sommets a un systeme de contraintes
 *
 * Corinne Ancourt
 */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"

#define MALLOC(s,t,f) malloc(s)

/* Psommet sys_som_conv(Psysteme sys, int * nb_som):
 * conversion d'un systeme lineaire sous forme d'un PSYSTEME en
 * une liste de sommets de type PSOMMET; le nombre de sommets alloues
 * pour constituer la liste, i.e. la longueur de la liste, est renvoye
 * par effet de bord via nb_som
 *
 * Chaque egalite et chaque inegalite definit un sommet dont les coordonnees
 * sont les coefficients de l'egalite ou de l'inegalite. Un champ saturation
 * de longueur 1 est aussi allouee. Il recoit la valeur -1 pour les inegalites
 * et la valeur 1 pour les egalites. 
 * 
 * Le systeme sys n'est pas modifie. Aucun sharing n'est introduit entre lui
 * et les sommets crees.
 */
Psommet sys_som_conv(sys,nb_som)
Psysteme sys;
int *nb_som;
{
    /* sommet courant */
    Psommet som1;
    /* tete de liste des sommets */
    Psommet som2=NULL;
    /* contrainte courante */
    Pcontrainte eq;

#ifdef TRACE
    (void) printf(" ***** conversion d'un systeme en sommet \n");
#endif
    *nb_som = 0;
    if (sys != NULL) {
	for (eq=sys->inegalites; eq!=NULL;eq=eq->succ)
	{
	    som1 = (Psommet)MALLOC(sizeof(Ssommet),SOMMET,"sys_som_conv");
	    som1->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"sys_som_conv");
	    *(som1->eq_sat) = -1;
	    som1->vecteur = vect_dup(eq->vecteur);
	    som1->denominateur = 1;
	    som1->succ = som2;
	    som2 = som1;
	    *nb_som = *nb_som + 1;
	};
	for (eq=sys->egalites; eq!=NULL;eq=eq->succ)
	{

	    som1 = (Psommet)MALLOC(sizeof(Ssommet),SOMMET,"sys_som_conv");
	    som1->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"sys_som_conv");
	    *(som1->eq_sat) = 1;
	    som1->vecteur =vect_dup(eq->vecteur);
	    som1->denominateur = 1;
	    som1->succ = som2;
	    som2 = som1;
	    *nb_som = *nb_som +1;
	};
    }
    return(som2);
}

/* Psysteme som_sys_conv(Psommet som1):
 * Conversion d'un systeme lineaire mis sous forme d'une liste de sommets
 * en un systeme lineaire de type Psysteme; les inegalites sont encodees
 * sous forme de sommets ayant un coefficient de saturation de -1; les
 * egalites sont encodees avec un coefficient de 1
 *
 * Voir la fonction inverse "sys_som-conv()"
 */
Psysteme som_sys_conv(som1)
Psommet som1;
{
    Psysteme sys = NULL;
    Pcontrainte eq;
    Psommet som2 = NULL;

#ifdef TRACE
    (void) printf(" ***** conversion d'un sommet en systeme \n");
#endif

    if (som1 != NULL) {
	sys = sc_new();

	for(som2 = som1;som2!=NULL;som2 = som2->succ) {
	    if (*(som2->eq_sat) == -1)
	    {
		eq = contrainte_new();
		/* contrainte_make() n'est pas utilisee pour ne pas risquer
		   un changement de signe */
		eq->vecteur = vect_dup(som2->vecteur);
		sc_add_inegalite(sys,eq);
	    }
	    else 
		if (*(som2->eq_sat) ==1)
		{
		    eq = contrainte_new();
		    eq->vecteur = vect_dup(som2->vecteur);
		    sc_add_egalite(sys,eq);
    		};
	};
    }
    return(sys);
}
