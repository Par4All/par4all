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

/* package pour la structure de donnees ray_dte, qui represente un rayon ou
 * une droite de systeme generateur
 *
 * Francois Irigoin
 *
 * D'autres procedures sur ray_dte doivent etre dispersees dans les fichiers
 * de Malik
 *
 * Liste des procedures:
 *  - Pray_dte ray_dte_dup(Pray_dte): allocation d'une nouvelle structure
 *    et copie de l'ancienne, sauf pour le chainage
 *  - Pray_dte ray_oppose(Pray_dte): transforme un rayon en le rayon oppose
 *    (utilisation: remplacement d'une droite par deux rayons)
 *
 * Voir aussi ray_dte.h et poly.h
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

#include "ray_dte.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(s,t,f) free(s)

/* Pray_dte ray_dte_dup(Pray_dte rd_in): duplication (allocation et copie)
 * d'une structure ray_dte;
 *
 * Le chainage est mis a NULL; le tableau des saturations aussi car on
 * ne peut pas connaitre sa taille; le vecteur est copie pour ne pas
 * introduire de sharing
 */
Pray_dte ray_dte_dup(rd_in)
Pray_dte rd_in;
{
    Pray_dte rd_out;

    if(rd_in->eq_sat!=NULL) {
	(void) fprintf(stderr,
		       "ray_dte_dup: perte d'un tableau de saturation\n");
	abort();
    }

    rd_out = ray_dte_new();
    rd_out->vecteur = vect_dup(rd_in->vecteur);

    return rd_out;
}

/* Pray_dte ray_dte_new(): allocation d'une structure ray_dte;
 *
 * Le chainage est mis a NULL; le tableau des saturations aussi car on
 * ne peut pas connaitre sa taille; le vecteur est initialise a 
 * VECTEUR_UNDEFINED
 */
Pray_dte ray_dte_new()
{
    Pray_dte rd;

    rd = (Pray_dte) MALLOC(sizeof(Sray_dte),RAY_DTE,"ray_dte_new");
    rd->succ = NULL;
    rd->eq_sat = NULL;
    rd->vecteur = VECTEUR_UNDEFINED;

    return rd;
}

/* Pray_dte ray_dte_make(Pvecteur v): allocation et initialisation
 * d'une structure ray_dte;
 *
 * Le chainage est mis a NULL; le tableau des saturations aussi car on
 * ne peut pas connaitre sa taille; le vecteur n'est pas copie et
 * du sharing est introduit
 */
Pray_dte ray_dte_make(v)
Pvecteur v;
{
    Pray_dte rd;

    rd = ray_dte_new();
    rd->vecteur = v;

    return rd;
}

/* Pray_dte ray_oppose(Pray_dte r): transformation d'un rayon en son
 * oppose (effet de bord)
 *
 * ->    ->
 * r = - r
 */
Pray_dte ray_oppose(r)
Pray_dte r;
{
    vect_chg_sgn(r->vecteur);
    return r;
}

/* void ray_dte_rm(Pray_dte rm): desallocation complete d'une structure ray_dte
 */
void ray_dte_rm(rd)
Pray_dte rd;
{
    vect_rm(rd->vecteur);
    FREE((char *)rd,RAY_DTE,"ray_dte_rm");
}

/* void dbg_ray_dte(Pray_dte rd, char * f): desallocation complete d'une
 * structure ray_dte rd avec trace sur stderr; le nom de la fonction
 * demandant la desallocation est passe comme deuxieme argument, f.
 */
void dbg_ray_dte_rm(rd,f)
Pray_dte rd;
char *f;
{
    (void) fprintf(stderr,"destruction de R/D dans %s\n",f);
    vect_fprint(stderr, rd->vecteur, variable_debug_name);
    dbg_vect_rm(rd->vecteur,f);
    FREE((char *)rd,RAY_DTE,f);
}

/* void ray_dte_fprint(File * f, Pray_dte rd, char * (*nom_var)()):
 * impression d'un rayon ou d'une droite
 *
 * FI: tres maladroit; utilise la representation interne d'un vecteur
 * directement
 * Modification par YY (09/91)
 */
void ray_dte_fprint(f, rd, nom_var)
FILE * f;
Pray_dte rd;
char * (*nom_var)(Variable);
{   
    if(rd->vecteur==NULL) 
	(void) fprintf(f,"( vecteur nul )\n");
    else vect_fprint(f,rd->vecteur,nom_var);
	
}

/* void ray_dte_fprint_as_dense(File * f, Pray_dte rd, Pbase b)
 * impression d'un rayon ou d'une droite
 */
void ray_dte_fprint_as_dense(f, rd, b)
FILE * f;
Pray_dte rd;
Pbase b;
{   
    vect_fprint_as_dense(f,rd->vecteur, b);
}

/* void ray_dte_dump(Pray_dte rd): impression de debug de rd sur stderr,
 * utilisant variable_debug_name()
 */
void ray_dte_dump(rd)
Pray_dte rd;
{
    ray_dte_fprint(stderr, rd, variable_debug_name);
}

/* void fprint_lray_dte(FILE * f, Pray_dte listrd, char * (*nom_var)()):
 * impression d'une liste de rayons ou de droites
 */
void fprint_lray_dte(f,listrd, nom_var)
FILE * f;
Pray_dte listrd;
char * (*nom_var)(Variable);
{
    Pray_dte e;
    for (e = listrd; e != NULL; e = e->succ) {
	ray_dte_fprint(f, e, nom_var);
    }
}

/* void fprint_lray_dte_as_dense(FILE * f, Pray_dte listrd):
 * impression d'une liste de rayons ou de droites
 */
void fprint_lray_dte_as_dense(f, listrd, b)
FILE * f;
Pray_dte listrd;
Pbase b;
{
    Pray_dte e;
    for (e = listrd; e != NULL; e = e->succ) {
	ray_dte_fprint_as_dense(f, e, b);
    }
}

/*   void ray_dte_normalize(Pray_dte rd): normalisation des coordonnees
 * d'un rayon ou d'une droite rd (division par le pgcd leurs coefficients)
 *
 * Ancien nom: norm_rd()
 */
void ray_dte_normalize(rd)
Pray_dte    rd;
{
    vect_normalize(rd->vecteur);
}

/* bool rd_in_liste(Pray-dte rd, Pray_dte lrd): test si rd appartient
 * a la liste lrd
 *
 * Les coordonnees de rd et des elements de lrd sont supposees normalisees.
 *
 * Il faudrait distinguer le cas des droites qui peuvent avoir des vecteurs
 * directeurs opposes et etre neanmoins egales; a moins que le normalisation
 * soit effectuee de maniere a n'avoir qu'un vecteur directeur possible
 * (le lexico-positif par exemple).
 */
bool rd_in_liste(rd,lrd)
Pray_dte rd;
Pray_dte lrd;
{
    Pray_dte rd1;

    if (rd->vecteur == NULL) return(1);
    for (rd1=lrd;rd1!=NULL;rd1=rd1->succ) {
	if (vect_equal((rd1->vecteur),(rd->vecteur))) {
	    return(true);
	}
    }
    return(false);
}

/* bool egaliste_rd(Pray_dte l1, Pray_dte * l2): egalite de deux listes
 * de rayons ou de droites
 */
bool egaliste_rd(l1,ad_l2)
Pray_dte l1,*ad_l2;
{
    int egalite;
    Pray_dte eq1,eq2,eq21,eq23,*ad_aux;

    if (l1==(*ad_l2)) return(true);
    eq2 = *ad_l2;
    ad_aux = ad_l2;
    (*ad_l2) = NULL;
    for(eq1=l1;eq1!=NULL;eq1=eq1->succ)  {
	egalite = 0;
	for(eq21=eq2,eq23=eq2;eq21!=NULL;) {
	    if (vect_equal(eq21->vecteur,eq1->vecteur)) {
		if (eq21==eq2) {
		    eq2=eq2->succ;
		    eq21->succ = NULL;
		    (*ad_aux) = eq21;
		    ad_aux = &(eq21->succ);
		    eq21 = eq23 = eq2;
		}
		else {
		    eq23->succ = eq21->succ;
		    eq21->succ = NULL;
		    (*ad_aux) = eq21;
		    ad_aux = &(eq21->succ);
		    eq21 = eq23->succ;
		}
		egalite = 1;
		break;
	    }
	    else {
		eq23 = eq21;
		eq21 = eq21->succ;
	    }
	}
	if (egalite == 0) {
	    (* ad_aux) = eq2;
	    return(false);
	}
	else
	    egalite = 0;
    }
    if (eq2==NULL)
	return(true);
    else
	(*ad_aux) = eq2;
    return(false);
}

/* Pray_dte elim_null_vect(Pray_dte l, int * n):
 *  elimine les vecteurs nuls d'une liste l de rayons ou
 * de droites et renvoie un pointeur vers la nouvelle liste ainsi que le
 * nouveau nombre d'elements de la liste *n
 *
 * la nouvelle liste contient les elements non-nuls en ordre inverse
 */
Pray_dte elim_null_vect(l,n)
Pray_dte l; /* liste initiale, passee par valeur */
int * n; /* nombre d'elements non-nuls dans l */
{
    Pray_dte nl = (Pray_dte) NULL;	/* nouvelle liste */
    Pray_dte zero;			/* rayon ou droite nul a desallouer */

    *n = 0;

    while(l!=NULL) {
	if(l->vecteur==NULL) {
	    /* desallocation du rayon ou de la droite */
	    zero = l;
	    l = l->succ;
	    RAY_DTE_RM(zero,"elim_null_vect");
	}
	else {
	    Pray_dte ll;
	    (*n)++;
	    /* il faut le chainer a la nouvelle liste */
	    ll = l;
	    l = l->succ;
	    ll->succ = nl;
	    nl = ll;
	}
    }
    return nl;
}

/* void elim_tt_rd(Pray_dte listrd): suppression d'une liste de rayons
 * ou d'une liste de droites
 */
void elim_tt_rd(listrd)
Pray_dte listrd;
{
    Pray_dte rd,rd1;

    for (rd=listrd; rd!=NULL;) {
	rd1 = rd->succ;
	RAY_DTE_RM(rd,"elim_tt_rd");
	rd = rd1;
    }
}
