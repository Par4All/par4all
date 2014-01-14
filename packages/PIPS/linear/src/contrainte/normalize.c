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

 /* package contrainte - NORMALISATION D'UN CONTRAINTE
  */

/*LINTLIBRARY*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

/* bool contrainte_normalize(Pcontrainte c, bool is_egalite): reduction
 * par le pgcd de ses coefficients d'une egalite (is_egalite==true) ou
 * d'une inegalite (is_egalite==false)
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
bool contrainte_normalize(
    Pcontrainte c,
    bool is_egalite)
{
    /* is_c_norm: si is_egalite=true, equation faisable */
    bool is_c_norm = true; 
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
		    is_c_norm= false;
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
		    is_c_norm = false;
		}
		else { /* is_inegalite */
		    is_c_norm = value_negz_p(vecteur_val(v)) ;
		}
	    }
	}
    }
    
    return is_c_norm;
}

/* bool egalite_normalize(Pcontrainte eg): reduction d'une equation
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
bool egalite_normalize(eg)
Pcontrainte eg;
{
    return(contrainte_normalize(eg, true));
}

/* bool inegalite_normalize(Pcontrainte ineg): normalisation
 * d'une inegalite a variables entieres; voir contrainte_normalize;
 * retourne presque toujours true car une inegalite n'ayant qu'un terme
 * constant est toujours faisable a moins qu'il ne reste qu'un terme 
 * constant strictement positif.
 *
 * Soit eg == sum ai xi <= b
 *             i
 * Soit k = pgcd ai
 *            i
 * eg := eg/k
 *
 * return true unless all ai are 0 and b < 0
 */
bool inegalite_normalize(ineg)
Pcontrainte ineg;
{
    return(contrainte_normalize(ineg ,false));
}
