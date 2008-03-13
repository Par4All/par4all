 /* package contrainte - NORMALISATION D'UN CONTRAINTE
  */

/*LINTLIBRARY*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

/* boolean contrainte_normalize(Pcontrainte c, boolean is_egalite): reduction
 * par le pgcd de ses coefficients d'une egalite (is_egalite==TRUE) ou
 * d'une inegalite (is_egalite==FALSE)
 *
 * Dans le cas des egalites, la faisabilite est testee et retournee
 *
 * Modifications:
 *  - double changement du signe du terme constant pour le traitement 
 *    des inegalites (Francois Irigoin, 30 octobre 1991) 
 *  - ajout d'un test pour les egalites du type 0 == k et les
 *    inegalites 0 <= -k quand k est une constante numerique entiere
 *    strictement positive (Francois Irigoin, 15 novembre 1995)
 */
boolean contrainte_normalize(
    Pcontrainte c,
    boolean is_egalite)
{
    /* is_c_norm: si is_egalite=TRUE, equation faisable */
    boolean is_c_norm = TRUE; 
    /* pgcd des termes non constant de c */
    Value a;
    /* modulo(abs(b0),a) */
    Value nb0 = VALUE_ZERO;

    if(c!=NULL && (c->vecteur != NULL))
    {
	a = vect_pgcd_except(c->vecteur,TCST);
	if (value_notzero_p(a)) {
	    Pvecteur v;
	    
	    nb0 = value_abs(vect_coeff(TCST,c->vecteur));
	    nb0 = value_mod(nb0,a);

	    if (is_egalite)	{
		if (value_zero_p(nb0)) {
		    (void) vect_div(c->vecteur,value_abs(a));
		    
		    /* si le coefficient du terme constant est inferieur 
		     * a ABS(a), on va obtenir un couple (TCST,coeff) avec 
		     * un coefficient qui vaut 0, ceci est contraire a nos 
		     * conventions
		     */
		    c->vecteur = vect_clean(c->vecteur);
		}
		else 
		    is_c_norm= FALSE;
	    }
	    
	    else {
		vect_chg_coeff(&(c->vecteur), TCST,
			       value_uminus(vect_coeff(TCST, c->vecteur)));
		(void) vect_div(c->vecteur,value_abs(a));
		c->vecteur= vect_clean(c->vecteur);
		vect_chg_coeff(&(c->vecteur), TCST, 
			       value_uminus(vect_coeff(TCST, c->vecteur)));
		/* mise a jour du resultat de la division C
		 * if ( b0 < 0 && nb0 > 0)
		 *	vect_add_elem(&(c->vecteur),0,-1);
		 * On n'en a plus besoin parce que vect_div utilise la
		 * division a reste toujours positif dont on a besoin
		 */
	    }
	    v=c->vecteur;
	    if(is_c_norm
	       && !VECTEUR_NUL_P(v)
	       && (vect_size(v) == 1)
	       && term_cst(v)) {
		if(is_egalite) {
		    assert(value_notzero_p(vecteur_val(v)));
		    is_c_norm = FALSE;
		}
		else { /* is_inegalite */
		    is_c_norm = value_negz_p(vecteur_val(v)) ;
		}
	    }
	}
    }
    
    return is_c_norm;
}

/* boolean egalite_normalize(Pcontrainte eg): reduction d'une equation
 * diophantienne par le pgcd de ses coefficients; l'equation est infaisable si
 * le terme constant n'est pas divisible par ce pgcd
 *
 * Soit eg == sum ai xi = b
 *             i
 * Soit k = pgcd ai
 *            i
 * eg := eg/k
 *
 * return b % k == 0 || all ai == 0 && b != 0;
 */
boolean egalite_normalize(eg)
Pcontrainte eg;
{
    return(contrainte_normalize(eg, TRUE));
}

/* boolean inegalite_normalize(Pcontrainte ineg): normalisation
 * d'une inegalite a variables entieres; voir contrainte_normalize;
 * retourne presque toujours TRUE car une inegalite n'ayant qu'un terme
 * constant est toujours faisable a moins qu'il ne reste qu'un terme 
 * constant strictement positif.
 *
 * Soit eg == sum ai xi <= b
 *             i
 * Soit k = pgcd ai
 *            i
 * eg := eg/k
 *
 * return TRUE unless all ai are 0 and b < 0
 */
boolean inegalite_normalize(ineg)
Pcontrainte ineg;
{
    return(contrainte_normalize(ineg ,FALSE));
}
