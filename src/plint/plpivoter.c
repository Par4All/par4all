


#include <stdio.h>
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "matrix.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc(s)
char * malloc();

#define TRACE


/*
 * Remplacement de la variable VAR dans la contrainte EQ par sa valeur dans
 * la contrainte LIGNE.
 *
 *  resultat retourne par la fonction :
 *
 *  La contrainte EQ est modifiee.
 *
 *  Les parametres de la fonction :
 *
 *  Psommet eq     : contrainte du systeme
 *  Psommet ligne  : contrainte du systeme ( ligne pivot) 
 *  int     var    : variable pivot
 */


pivoter_pas(eq,ligne,var)
Psommet eq;
Psommet ligne;
Variable var;
{


    Pvecteur pvec = NULL;
    Pvecteur ligne2;

    int c1 = 0;
    int den;
    boolean cst = FALSE;
#ifdef TRACE
    printf(" --- pas - pivoter \n");
#endif
    if (ligne && eq) {
	Pvecteur pv3 = vect_dup(eq->vecteur);

	den = ligne->denominateur;
	cst = FALSE;
	if ((eq != ligne) && (c1 = vect_coeff(var,pv3)) != 0) {

	    ligne2 = vect_dup(ligne->vecteur);
	    eq->denominateur *= den;
	    for (pvec =pv3;pvec!= NULL; pvec=pvec->succ) {
		if (pvec->var == NULL) cst = TRUE;
		pvec->val =pvec->val* den - c1 * vect_coeff(pvec->var,
							    ligne->vecteur);
		vect_chg_coeff(&ligne2,pvec->var,0);
	    }

	    for (pvec=ligne2;pvec!= NULL;pvec = pvec->succ)
		if (pvec->var != TCST)
		    vect_add_elem(&pv3,pvec->var,
				  -c1*vect_coeff(pvec->var,ligne2));
	    if (!cst)
		vect_add_elem(&pv3,
			      TCST,-c1*vect_coeff(TCST,ligne->vecteur));
	}
	eq->vecteur = pv3;
    }
}

/*
 * Operation "pivot" avec VAR comme variable pivot et LIGNE comme ligne
 * pivot.
 *
 *  resultat retourne par la fonction :
 *
 *  Le systeme initial est modifie.
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Psommet fonct  : fonction economique du  programme lineaire
 *  Psommet ligne  : ligne pivot
 *  int     var    : variable pivot   
 */
pivoter(sys,ligne,var,fonct)
Psommet sys;
Psommet ligne;
Variable var;
Psommet fonct;

{

    Psommet sys1 = fonct;
    Psommet ps1 = NULL;
    int sgn_den = 1;
    int den;
#ifdef TRACE
    printf(" *** on effectue le pivot \n");
#endif
    if (ligne) {

	den = vect_coeff(var,ligne->vecteur);
	if (den <0)
	{
	    sgn_den = -1;
	    den *= sgn_den;
	}
	if (fonct != NULL)
	    fonct->succ = sys;
	else sys1 = sys;

	/* mise a jour du denominateur   */
	(void) vect_multiply(ligne->vecteur,sgn_den*ligne->denominateur);
	ligne->denominateur *= den;
	den = ligne->denominateur;
	for (ps1 = sys1; ps1!= NULL; ps1=ps1->succ)
	    pivoter_pas(ps1,ligne,var);
	sommets_normalize(sys1);

    }
    if (fonct)
	fonct->succ = NULL;

}

