/* package plint */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"

#define MALLOC(s,t,f) malloc(s)

/* Psysteme sys_int_redond(Psysteme sys):
 * elimination des contraintes lineaires redondantes d'un systeme lineaire 
 * en nombres entiers par tests de faisabilite exacts. Chaque inegalite est
 * inversee tour a tour, et la faisabilite de chacun des systemes ainsi
 * obtenus est teste par sys_int_fais(), l'algorithme des congruences
 * decroissantes.
 */
Psysteme sys_int_redond(sys)
Psysteme sys;
{

    Pcontrainte eq,eq1;

    sys = sc_normalize(sc_dup(sys));
    if (sys && (sys->nb_ineq <= NB_INEQ_MAX2) && sys_int_fais(sys) && 
	sys != NULL) {
	for (eq = sys->inegalites;
	     eq != NULL && sys->nb_ineq > 1;
	     eq = eq1)
	{
	    eq1 = eq->succ;
	    /* inversion du sens de l'inegalite par multiplication */
	    /* par -1 du coefficient de chaque variable            */
	    vect_chg_sgn(eq->vecteur);
	    vect_add_elem(&(eq->vecteur),TCST,1);
	    /* test de faisabilite avec la nouvelle inegalite      */
	    if (sys_int_fais(sys) == FALSE)
	    {
		/* si le systeme est non faisable 
		   ==> inegalite redondante
		   ==> elimination de cette inegalite         */
		eq_set_vect_nul (eq);
		sc_rm_empty_constraints(sys,0);
	    }
	    else
	    {
		vect_add_elem (&(eq->vecteur),TCST,-1);
		vect_chg_sgn(eq->vecteur);
	    }
	}
    }
    return (sys);
}
