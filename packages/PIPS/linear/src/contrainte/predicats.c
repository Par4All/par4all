/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

 /* package contrainte - tests sur des contraintes
  */

/*LINTLIBRARY*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

/* bool eq_smg(Pcontrainte c1, Pcontrainte c2):
 * comparaison des coefficients de deux contraintes pour savoir si elles ont le
 * meme membre gauche.
 *
 * Note: this works for inequalities. Similar equations may differ
 *       by a factor of -1.
 */
bool eq_smg(c1,c2)
Pcontrainte c1,c2;
{
    bool result;

    if(c1==NULL && c2==NULL)
	result = true;
    else if(c1==NULL || c2==NULL)
	result = false;
    else
	result = vect_equal_except(c1->vecteur,c2->vecteur,TCST);

    return result;
}

/* bool inequalities_opposite_p(Pcontrainte c1, Pcontrainte c2):
 * True if the non-constant part of c1 is the opposite of
 * the non-constant part of c2.
 */
bool inequalities_opposite_p(c1,c2)
Pcontrainte c1,c2;
{
    bool result;

    if(c1==NULL && c2==NULL)
	result = true;
    else if(c1==NULL || c2==NULL)
	result = false;
    else
	result = vect_opposite_except(c1->vecteur,c2->vecteur,TCST);

    return result;
}

/* bool egalite_equal(Pcontrainte eg1, Pcontrainte eg2): teste
 * l'equivalence de deux egalites; leurs coefficients peuvent etre
 * tous egaux ou tous opposes; pour obtenir une meilleure equivalence
 * il faut commencer par reduire leurs coefficients par les PGCD
 *                                                                       
 * Soit eg1, sum a1i xi = b1, et eg2, sum a2i xi = b2.
 *            i                        i
 * return a1i == a2i || a1i == -a2i;
 *            i            i
 *
 * Note: 2x=2 est different de x=1
 */
bool egalite_equal(eg1,eg2)
Pcontrainte eg1;
Pcontrainte eg2;
{
    bool result;

    if(CONTRAINTE_UNDEFINED_P(eg1) && CONTRAINTE_UNDEFINED_P(eg2))
	result = true;
    else if(CONTRAINTE_UNDEFINED_P(eg1) || CONTRAINTE_UNDEFINED_P(eg2))
	result = false;
    else
	result = vect_equal(eg1->vecteur,eg2->vecteur) ||
	    vect_oppos(eg1->vecteur,eg2->vecteur);

    return(result);
}

/* bool contrainte_equal(Pcontrainte c1, Pcontrainte c2): test
 * d'egalite des contraintes c1 et c2; elles sont egales si tous
 * leurs coefficients et leur termes constants sont egaux; il faut les
 * avoir normalisees auparavant pour etre sur de leur egalite;
 *
 * La contrainte CONTRAINTE_UNDEFINED est assimilee a la contrainte nulle
 *
 * Ancien nom: ineg_same()
 *
 * Modifications:
 *  - utilisation de CONTRAINTE_UNDEFINED_P() et contrainte_vecteur()
 *    (FI, 08/12/89)
 */
bool contrainte_equal(c1,c2)
Pcontrainte c1,c2;
{
    register bool 
	undef1 = CONTRAINTE_UNDEFINED_P(c1),
	undef2 = CONTRAINTE_UNDEFINED_P(c2);

    if (undef1 || undef2) 
	return(undef1 && undef2);

    return(vect_equal(contrainte_vecteur(c1),
		      contrainte_vecteur(c2)));
}

/* bool contrainte_constante_p(Pcontrainte c): test de contrainte
 * triviale sans variables (ie du type 0<= K ou 0<=0 ou 0 == 0 ou 0 == K)
 * 
 * Les equations non-faisables ne sont pas detectees.
 * 
 * Modifications:
 *  - utilisation de CONTRAINTE_NULLE_P()
 * Bugs:
 *  - should assert !CONTRAINTE_UNDEFINED_P(c)
 */
bool contrainte_constante_p(c)
Pcontrainte c;
{

    if (CONTRAINTE_NULLE_P(c))
	return(true);
    else {
	return vect_constant_p(contrainte_vecteur(c));
    }
}

/* bool vect_constant_p(Pvecteur v): v contains only a constant term,
 * may be zero
 *
 * Bugs:
 *  - this function should not be in contrainte.dir; it should be moved into
 *    vecteur.dir with TCST...
 *  - should assert !VECTEUR_UNDEFINED_P(v)
 */
bool vect_constant_p(v)
Pvecteur v;
{
    return(VECTEUR_NUL_P(v) || (v->var == TCST && v->succ == NULL));
}

/* bool contrainte_verifiee(Pcontrainte ineg, bool eq_p): 
 * test de faisabilite d'inegalite (eq_p == false) ou d'egalite triviale
 *
 * Le test est different pour les egalites.
 *
 * Modifications:
 *  - test de l'absence d'un successeur dans vecteur (ajout d'un test
 *    succ == NULL)
 *  - changement de cote du terme constant (FI, 08/12/89)
 *  - ajout d'un assert pour pallier partiellement le bug 1 (FI, 08/12/89)
 *  - utilisation de la macro CONTRAINTE_NULLE_P() (FI, 08/12/89)
 * Bugs:
 *  - si on passe une inegalite non constante en argument, le resultat
 *    depend du premier terme du vecteur; il faudrait pouvoir retourner
 *    la valeur bottom pour les inegalites non constantes;
 *  - le nom devrait etre inegalite_verifiee()
 */
bool contrainte_verifiee(ineg,eq_p)
Pcontrainte ineg;
bool eq_p;
{
    Value v;
    assert(contrainte_constante_p(ineg));

    /* l'inegalite 0 <= 0 est representee par un vecteur nul */
    if (CONTRAINTE_NULLE_P(ineg))
	return(true);

    /* l'inegalite 0 <= K est representee par un vecteur a un element */
    v = val_of(ineg->vecteur);

    return (!eq_p && value_negz_p(v) && ineg->vecteur->succ==NULL)
	|| ( eq_p && value_zero_p(v) && ineg->vecteur->succ==NULL);
}

/* bool contrainte_oppos(Pcontrainte ineg1, Pcontrainte ineg2):
 * indique si 2 inegalites forment une egalite ou si deux egalites sont
 * equivalentes.
 *
 * return(ineg1 == -ineg2);
 */
bool contrainte_oppos(ineg1,ineg2)
Pcontrainte ineg1,ineg2;
{
    return(vect_oppos(ineg1->vecteur,ineg2->vecteur));
}


/* bool constraint_without_vars(c, vars)
 * Pcontrainte c;
 * Pbase vars;
 *
 *     IN: c, vars
 *    OUT: returned boolean
 * 
 * returns if the current constraint uses none of the variables in vars.
 *
 * (c) FC 16/05/94
 */
bool constraint_without_vars(c, vars)
Pcontrainte c;
Pbase vars;
{
    Pbase
	b = BASE_NULLE;

    for(b=vars;
	b!=BASE_NULLE;
	b=b->succ)
	if (vect_coeff(var_of(b), c->vecteur)!=(Value) 0) return(false);

    return(true);	    
}

/* bool constraints_without_vars(pc, vars)
 * Pcontrainte pc;
 * Pbase vars;
 *
 *     IN: c, vars
 *    OUT: returned boolean
 * 
 * returns true if none of the constraints use the variables in vars.
 */
bool constraints_without_vars(pc, vars)
Pcontrainte pc;
Pbase vars;
{
    Pcontrainte c;

    for (c=pc;
	 c!=NULL;
	 c=c->succ)
	if (!constraint_without_vars(c, vars)) return(false);

    return(true);
}	

/*
 *   that is all
 */
