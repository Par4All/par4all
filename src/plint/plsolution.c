 /* package plint */

#include <stdio.h>
#include <sys/stdtypes.h>  /* for debug with dbmalloc */
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
#define FREE(s,t,f) free((char *)(s))

/* boolean sol_entiere(Psommet sys, Pvecteur lvbase, int nb_som):
 * Cette fonction teste si la solution est entiere
 *
 *  resultat retourne par la fonction :
 *
 *  boolean	   : TRUE si la solution est entiere
 * 		     FALSE sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 */
boolean sol_entiere(sys,lvbase,nb_som)
Psommet sys;
Pvecteur lvbase;
int nb_som;
{
    Psommet ps1=NULL;
    Variable var =NULL;
    boolean result = TRUE;
    int cv;
#ifdef TRACE
    printf(" ** la solution est elle entiere ? \n");
#endif

    for (ps1 = sys; ps1!= NULL && result;ps1 = ps1->succ) {
	var = coeff_no_ligne(lvbase,nb_som);
	if (var != NULL 
	    &&  (cv = vect_coeff(var,ps1->vecteur)) != 0 
	    && ((-vect_coeff(TCST,ps1->vecteur) % cv) != 0))
	    result = FALSE;
	nb_som --;
    }
    if (sys == NULL)
	result = FALSE;
    return (result);
}

/* boolean sol_positive(Psommet sys, Pvecteur lvbase, int nb_som):
 * Cette fonction teste si la solution est positive
 *
 * On deduit la solution du systeme lineaire a l'aide de la liste des variables de base 
 * du systeme. Chaque variable de base n'apparait que dans une des contraintes du systeme.
 * La valeur d'une variable de base dans le systeme est egale a la constante de la 
 * contrainte  divisee par le coefficient de la variable dans cette contrainte.
 *
 *  resultat retourne par la fonction :
 *
 *  boolean	   : TRUE si la solution est positive
 * 		     FALSE sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 */
boolean sol_positive(sys,lvbase,nb_som)
Psommet sys;
Pvecteur lvbase;
int nb_som;
{
    Psommet ps1=NULL;
    int in1;
    Variable var = NULL;
    int b;
    boolean result = TRUE;

#ifdef TRACE
    printf(" ** la solution est elle positive ? \n");
#endif

    for (ps1 = sys; ps1!= NULL && result;ps1 = ps1->succ) {
	var = coeff_no_ligne(lvbase,nb_som);
	if (var != NULL) {
	    int v1 = 0;  	
	    b = -vect_coeff(TCST,ps1->vecteur);
	    if ( (v1 = vect_coeff(var,ps1->vecteur)) != 0) 
		in1 = b /v1;
	    if (in1 <0) {
		result = FALSE; 
		printf ("sol. negative \n");
	    }
	}
	nb_som --;
    }
    if (sys == NULL)
	result = FALSE;
    return (result);
}


boolean sol_positive_simpl(sys,lvbase,lvsup,nb_som)
Psommet sys;
Pvecteur lvbase;
Pvecteur lvsup;
int nb_som;
{
    Psommet ps1=NULL;
    int in1;
    Variable var = NULL;
    int b;
    boolean result = TRUE;

#ifdef TRACE
    printf(" ** la solution est elle positive ? \n");
#endif

    for (ps1 = sys; ps1!= NULL && result;ps1 = ps1->succ) {
	var = coeff_no_ligne(lvbase,nb_som);
	if (var != NULL) {
	    int v1 = 0;  	

	    b = -vect_coeff(TCST,ps1->vecteur);
	    if ( (v1 = vect_coeff(var,ps1->vecteur)) != 0) 
		in1 = b /v1;
	    if ((in1 <0 && !vect_coeff(var,lvsup)) 
		 || (in1 >0 && vect_coeff(var,lvsup))  ) {
		result = FALSE; 
		printf ("sol. negative \n");
	    }
	}
	nb_som --;
    }
    if (sys == NULL)
	result = FALSE;
    return (result);
}
/* Psolution sol_finale(Psommet sys, Pvecteur lvbase, int nb_som):
 * Calcul de la solution finale du programme lineaire a partir du 
 * systeme final et de la liste  des variables de base 
 *
 *  resultat retourne par la fonction :
 *
 *  Psolution	   : solution finale
 *  NULL		   : si le systeme est infaisable
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 */
Psolution sol_finale(sys,lvbase,nb_som)
Psommet sys;
Pvecteur lvbase;
int nb_som;
{
    Psommet ps1=NULL;
    Psolution sol = NULL;
    Psolution sol1 = NULL;
    Variable var =NULL;

#ifdef TRACE
    printf(" ** la solution finale est: \n");

#endif

    for (ps1 = sys; ps1!= NULL;ps1 = ps1->succ) {
	var = coeff_no_ligne(lvbase,nb_som);

	if (var != TCST) {
	    sol1 = (Psolution)MALLOC(sizeof(Ssolution),
				     SOLUTION,"sol_finale");
	    sol1->var = var;
	    sol1->val = -vect_coeff(TCST,ps1->vecteur);
	    sol1->denominateur = vect_coeff(var,ps1->vecteur);
	    sol1->succ = sol;
	    sol = sol1;
#ifdef TRACE
	    (void) printf (" %s == %f; ",
			   noms_var(sol1->var),
			   (double)(sol1->val / sol1->denominateur));
#endif
	}
	nb_som --;
    }
#ifdef TRACE

    printf ("\n");
#endif
    return (sol);
}
