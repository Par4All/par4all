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
    double d1;
    int in1,b0;
    int lambda = 1;
    int sgn_op= 1;
    int d =1;

#ifdef TRACE
    printf(" ** Gomory - ajout d'une eq. verifiant les cond. de Gomory \n");
#endif

    if (eq != NULL && var != NULL)
    {
	/* calcul du determinant D	*/
	int p1 = ABS(vect_coeff(var,eq->vecteur));
	int p2 = ABS(vect_pgcd_all(eq->vecteur));

	d = p1/p2;

	/* calcul de la coupe de Gomory	*/
	ps1= (Psommet)MALLOC(sizeof(Ssommet),SOMMET,"gomory_trait_eq");
	ps1->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"gomory_trait_eq");
	pv1 = vect_dup(eq->vecteur);
	ps1->denominateur = 1;
	*(ps1->eq_sat) = -1;
	ps1->succ = NULL;
	vect_chg_coeff(&pv1,var,0);
	ps1->vecteur = pv1;
	sgn_op = (vect_coeff(var,eq->vecteur) <0)? (1):(- 1);
	if (d!=0) {
	    for (pv2=pv1;pv2 != NULL; pv2= pv2->succ) {
		pv2->val = ((sgn_op * pv2->val) % d);
		if (pv2->val < 0)
		    pv2->val += d;
	    }
	}
	b0 = ABS(vect_coeff(TCST,pv1));

	/* calcul de lambda = coefficient multiplicateur permettant
	   d'obtenir la coupe optimale		*/

	if (b0 != 0) {
	    d1 =(double)d/(double)b0;
	    in1 = d/b0;

	    lambda = (d1 == in1) ? (in1-1) : in1;
	    if (lambda == 0) 
		lambda = 1;
	}

	/* optimisation de la coupe de Gomory	*/
	if (lambda >1)
	    for (pv2 = pv1;pv2!= NULL; pv2=pv2->succ)
		pv2->val =(lambda *pv2->val) % d;

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
    int a;
    int max =0;
    Variable v1 = NULL;
    boolean borne = TRUE;

#ifdef TRACE
    printf(" **  Gomory - recherche d'une variable non_entiere \n");
#endif
    *var = NULL;

    for (eq = *sys ;eq != NULL && borne; eq=eq->succ) {
	int b0,den2;

	if (borne = test_borne(eq)) {
	    if ((v1 = find_vbase(eq,lvbase)) !=  NULL) {
		den2 = vect_coeff(v1,eq->vecteur);
		b0 = -vect_coeff(TCST,eq->vecteur);

		/* on a trouve une variable de base non entiere   */
		/* on choisit celle de plus grand denominateur     */

		if ((b0 % den2) != 0) { 
		    int p1=eq->denominateur;
		    int p2 = ABS(b0);
		    int p3 = pgcd(p2,p1);

		    if ((a =ABS(p1/p3)) >max) {
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
    if (max != 0)
	return (result);
    else
	return (NULL);
}
