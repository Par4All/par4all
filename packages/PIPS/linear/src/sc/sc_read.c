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

 /* ensemble des fonctions utilisees dans le programme 'sc_gram.y'
  * permettant de construire la structure de donnees  'Psysteme '
  * a partir de la grammaire definie dans 'sc_gram.y'
  *
  * Ces fonction ne sont utilisables que dans ce contexte particulier
  * car elles utilisent des variables globales
  *
  *          * init_globals(): initialisation des variables globales
  *
  *          * new_ident (ps,s) : - introduction de la nouvelle variable s
  *                              dans la base
  *
  *          * rec_ident (ps,s) : - recherche du rang de la variable s
  *                              dans la base
  *
  *          * creer_eg (p)  : - ajout d'une egalite, dont les couples
  *                              (variable,coefficient) sont pointes
  *                              par p
  *
  *          * creer_ineg (p): - ajout d'une inegalite, dont les couples
  *                              (variable,coefficient) sont pointes
  *                              par p
  *
  * Corinne Ancourt
  *
  * Modifications:
  *  - introduction de la notion de base (FI, 3/1/90)
  */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "assert.h"
#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

 /* Variables globales malencontreusement utilisees pour rendre les modules
  * non reutilisables
  *
  * Introduites pour utilisation avec le module YACC gram.y
  */

 /* pointeur sur la derniere egalite          */
Pcontrainte p_eg_fin;

 /* pointeur sur la derniere inegalite        */
Pcontrainte p_ineg_fin;

 /* pointeur sur le predecesseur du couple courant */
Pvecteur p_pred;

Pvecteur p_membre_courant;

 /* pointeur sur le couple courant          */
Pvecteur cp;

char *nom_var[100];


 /* quelques  constantes symboliques */

#define OPINF 1
#define OPINFEGAL 2
#define OPEGAL 3
#define OPSUPEGAL 4
#define OPSUP 5
#define CHAINE 0

/* void init_globals: initialisation des variables globales */
void init_globals()
{
    p_eg_fin = NULL;
    p_ineg_fin = NULL;
    p_pred = NULL;
    cp = NULL;
}

/* void new_ident(Psysteme ps, char s[]): fonction introduisant un nouvel
 * identificateur dans la liste des variables du systeme en cours
 * de lecture
 * 
 * Modification:
 *  - utilisation des bases (FI, 13/12/89)
 *  - symbols handled in sc_lex.l (FC, 02/05/2000)
 */
void new_ident(Psysteme ps, Variable s)
{
    Pbase b = ps->base;

    if (!base_contains_variable_p(b, s)) {
	ps->base = vect_add_variable(b, s);
	ps->dimension ++;
    }
}

/* int rec_ident(Psysteme ps, char * s): cette fonction recherche
 *  dans la liste des variables du systeme la variable de nom s
 *
 * Modifications:
 *  - utilisation des bases (FI, 13/12/89)
 *  - parcours direct de la base avec strcmp pour eviter des problemes
 *    avec variable_equal() (FI, 3/1/90)
 */
Variable rec_ident(Psysteme ps, Variable s)
{
    Variable v;

    v = base_find_variable(ps->base, s);

    if(VARIABLE_UNDEFINED_P(v)) {
	(void) fprintf(stderr,
		       "Variable %s not declared. Add it to the VAR list!\n",
		       variable_default_name(s));
	exit(1);
    }

    return v;
}


/* void creer_eg(Psysteme ps,Pcontrainte peq): ajout de la contrainte (egalite)
 * peq au systeme ps.
 *
 * Inutilisable car elle utilise la variable globale p_eg_fin pour faire
 * des chainages rapides en FIN de liste 
 */
void creer_eg(ps,peq)
Psysteme ps;
Pcontrainte peq;
{
    ps->nb_eq++;

    if (ps->egalites != NULL) {
	p_eg_fin->succ = peq;
	p_eg_fin = peq;
    }
    else { 
	ps->egalites = peq;
	p_eg_fin = peq;
    }
}

/* void creer_ineg(Psysteme ps, Pcontrainte peq, int sens): ajout
 * d'une nouvelle inegalite peq dans la liste des inegalites du
 * systeme ps
 *
 * Si sens vaut -1, l'inegalite est multipliee par -1; ceci est utilise
 * pour convertir une inegalite de type >= en <=
 *
 * Inutilisable hors contexte car la variable globale p_ineg_fin est
 * utilisee pour effectuer rapidement des chainages en FIN de liste
 */
void creer_ineg(ps,peq,sens)
Psysteme ps;
Pcontrainte peq;
int sens;
{
    ps->nb_ineq +=1;

    /* multiplication par "-1" des coefficients de chaque variable
       si l'inegalite est "superieure"                              */
    if (sens == -1)
	vect_chg_sgn(peq->vecteur);

    if (ps->inegalites != NULL) {
	p_ineg_fin->succ = peq;
	p_ineg_fin = peq;
    }
    else {
	ps->inegalites = peq;
	p_ineg_fin = peq;
    }
}

/* Psysteme sc_reversal(Psysteme sc) */
Psysteme sc_reversal(sc)
Psysteme sc;
{

    if(!(SC_EMPTY_P(sc) || sc_empty_p(sc) || sc_rn_p(sc))) {
	Pcontrainte c = CONTRAINTE_UNDEFINED;

	for(c = sc_egalites(sc); !CONTRAINTE_UNDEFINED_P(c); c = c->succ) {
	    c = contrainte_reversal(c);
	}
	for(c = sc_inegalites(sc); !CONTRAINTE_UNDEFINED_P(c); c = c->succ) {
	    c = contrainte_reversal(c);
	}
    }

    return sc;
}

/* Pcontrainte contrainte_reversal(Pcontrainte c) */
Pcontrainte contrainte_reversal(c)
Pcontrainte c;
{
    Pvecteur v = contrainte_vecteur(c);
    Pvecteur rv = VECTEUR_NUL;

    rv = (Pvecteur) vect_reversal((Pbase) v);
    contrainte_vecteur(c) = rv;
    vect_rm(v);

    return c;
}
