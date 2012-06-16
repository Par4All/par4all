/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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

/* package sur les contraintes. 
 *
 * Une contrainte est une egalite ou une inegalite. Elle est representee
 * par un vecteur dont une coordonnee particuliere, TCST, represente
 * le terme constant.
 *
 * Les contraintes sont stockees sous forme de membres gauches, ce qui 
 * n'est utile a savoir que pour les inegalites:
 *
 *      sum a x + b <= 0
 *       i   i i
 * ou b est le terme constant.
 *
 * Les tableaux de saturations sont calcules en fonction de polyedre
 * (systeme generateur ou systeme de contraintes) et leurs dimensions
 * sont inconnues au niveau des contraintes. Ils sont ignores la
 * plupart du temps.
 *
 * Les contraintes sont traitees sous forme de liste de contraintes
 * (systeme d'egalites ou systeme d'inegalites) et possedent un champ
 * de chainage. Certaines des fonctions portent sur des listes de
 * contraintes et non sur des contraintes. Cette double semantique
 * complique beaucoup les choses.
 *
 * Le pointeur NULL represente implicitement l'une des contraintes toujours
 * faisables, 0 == 0 ou 0 <= 0. Au moins, la plupart du temps... car
 * il represente la valeur CONTRAINTE_UNDEFINED dans les routines de gestion
 * memoire.
 *
 * Il vaut mieux utiliser la macro CONTRAINTE_NULLE_P() qui verifie que
 * le vecteur associe est le VECTEUR_NUL.
 *
 * Les contraintes trivialement faisables et infaisables sont representees
 * par un unique terme constant: 0 == k, 0 <= k, 0 <= -k (k positif).
 * Il existe deux fonctions pour les tester.
 *
 * Autres packages a inclure:
 *  - types.h
 *  - boolean.h
 *  - vecteur.h
 *
 * Malik Imadache, Corinne Ancourt, Neil Butler, Francois Irigoin
 *
 * Bugs:
 *  - melange de la notion de chainage et de la notion de terme constant
 *  - definition directe de TCST: on ne peut pas decider dynamiquement
 *    de la variable representant le terme constant
 *  - definition de TCST au niveau contrainte, alors que TCST doit etre
 *    utilise au niveau vecteur (cf. macro term_cst())
 *
 * Modifications:
 *  - passage du terme constant a gauche (FI, 24/11/89)
 *  - deplacement de la definition du terme constant TCST et de la macro
 *    term_cst dans le package vecteur (PB, 06/06/90)
 */

#ifndef CONTRAINTE

/* constante associee a la structure de donnees "contrainte" */
#define CONTRAINTE 1005

typedef struct Scontrainte   {
        int *eq_sat;
	int *s_sat,*r_sat;
	Pvecteur vecteur;
	struct Scontrainte *succ;
	} Scontrainte,*Pcontrainte;

typedef Scontrainte Segalite, * Pegalite;

typedef Scontrainte Sinegalite, * Pinegalite;

/* MACROS ET CONSTANTES */

#define egalite_print(eg) egalite_fprint(stdout,eg)
/* FI: this macro requires an additional parameter or a default value
   as third parameter of inegalite_fprint() */
#define inegalite_print(ineg) inegalite_fprint(stdout,ineg)

/* passage au champ vecteur d'une contrainte "a la  Newgen" */
#define contrainte_vecteur(c) ((c)->vecteur)

#define contrainte_succ(c) ((c)->succ)

/* contrainte nulle (non contrainte 0 == 0 ou 0 <= 0) */
#define CONTRAINTE_NULLE_P(c) (VECTEUR_NUL_P(contrainte_vecteur(c)))

#define CONTRAINTE_UNDEFINED ((Pcontrainte) NULL)

#define CONTRAINTE_UNDEFINED_P(c) ((c)==CONTRAINTE_UNDEFINED)

/* int COEFF_CST(Pcontrainte c): terme constant d'une contrainte */
#define COEFF_CST(c) vect_coeff(TCST,(c)->vecteur)

/* the standard xxx_rm does not return a value */
#define contrainte_rm(c) (void) contrainte_free(c)

#define VERSION_FINALE
#ifndef VERSION_FINALE
#define CONTRAINTE_RM(rd,f) dbg_contrainte_rm(rd,f)
#else
#define CONTRAINTE_RM(rd,f) contrainte_rm(rd)
#endif


#endif /* CONTRAINTE */
