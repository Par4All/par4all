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

 /* package sc */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "arithmetique.h"
#include "assert.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#define MALLOC(s,t,f) malloc(s)

/* Psysteme sc_new():
 * alloue un systeme vide, initialise tous les champs avec des
 * valeurs nulles, puis retourne ce systeme en resultat.
 *
 * Attention, sc_new ne fabrique pas un systeme coherent comportant
 * une base. Un tel systeme s'obtient par appel a la fonction sc_creer_base,
 * apres avoir ajoute des equations et des inequations au systeme. La base
 * n'est pas entretenue lorsque le systeme est modifie.
 *
 * Ancien nom: init_systeme()
 */
Psysteme sc_new(void)
{
    Psysteme p = (Psysteme) malloc(sizeof(Ssysteme));

    assert(p);

    p->nb_eq = 0;
    p->nb_ineq = 0;
    p->dimension = 0;

    p->egalites = (Pcontrainte) NULL;
    p->inegalites = (Pcontrainte) NULL;
    p->base = BASE_NULLE;

    return p;
}

/* creation d'une base contenant toutes les variables
 * apparaissant avec des coefficients non-nuls
 *  dans les egalites ou les inegalites de ps
 */
Pbase sc_to_minimal_basis(Psysteme ps)
{
  linear_hashtable_pt seen;
  Pbase b = BASE_NULLE;
  Pcontrainte c;
  Pvecteur v;

  /* great optimization */
  if (!ps->egalites && !ps->inegalites)
    return BASE_NULLE;

  seen = linear_hashtable_make();

  for (c = ps->egalites; c!=NULL; c=c->succ) {
    for (v = c->vecteur; v!=VECTEUR_NUL; v=v->succ) {
      Variable var = var_of(v);
      if (var!=TCST && !linear_hashtable_isin(seen, var)) {
	linear_hashtable_put_once(seen, var, var);
	b = vect_chain(b, var, VALUE_ONE);
      }
    }
  }

  for (c = ps->inegalites; c!=NULL; c=c->succ) {
    for (v = c->vecteur; v!=VECTEUR_NUL; v=v->succ) {
      Variable var = var_of(v);
      if (var!=TCST && !linear_hashtable_isin(seen, var)) {
	linear_hashtable_put_once(seen, var, var);
	b = vect_chain(b, var, VALUE_ONE);
      }
    }
  }

  linear_hashtable_free(seen);

  return b;
}

/* void sc_creer_base(Psysteme ps): initialisation des parametres dimension
 * et base d'un systeme lineaire en nombres entiers ps, i.e. de la
 * base implicite correspondant aux egalites et inegalites du systeme;
 *
 * Attention, cette base ne reste pas coherente apres un ajout de nouvelles
 * egalites ou inegalites (risque d'ajout de nouvelles variables), ni apres
 * des suppressions (certaines variables de la base risque de n'apparaitre
 * dans aucune contrainte).
 *
 * dimension : nombre de variables du systeme (i.e. differentes de TCST, le
 *          terme constant)
 *
 * Modifications:
 *  - passage de num_var a base (FI, 13/12/89)
 */
void sc_creer_base(ps)
Psysteme ps;
{
    if (ps) {
	assert(ps->base == (Pbase) NULL);
	ps->base = sc_to_minimal_basis(ps);
	ps->dimension = vect_size(ps->base);
    }
}

/* fix system s for coherency of the base and number of things.
 */
void sc_fix(Psysteme s)
{
  if (s) {
    s->nb_eq = nb_elems_list(s->egalites);
    s->nb_ineq = nb_elems_list(s->inegalites);
    if (s->base) base_rm(s->base), s->base = NULL;
    sc_creer_base(s);
  }
}

/* Variable * sc_base_dup(int nbv, Variable * b):
 * duplication de la table des variables base, qui contient nbv elements
 *
 * Modifications:
 *  - on renvoie un pointeur NULL si le nombre de variables nbv est nul
 *  - changement de num_var en base; cette fonction perd tout interet
 *    et n'est conservee que pour faciliter la mise a jour des modules
 *    de plint (FI, 19/12/89)
 *
 * ancien nom: tab_dup
 */
Pbase sc_base_dup(nbv,b)
int nbv;
Pbase b;
{
    assert(nbv==base_dimension(b));

    return((Pbase) base_copy(b));
}

/* Psysteme sc_dup(Psysteme ps): should becomes a link
 *
 * Ancien nom (obsolete): cp_sc()
 *
 */
Psysteme sc_dup(Psysteme ps)
{
  return sc_copy(ps);
  /*
  Psysteme cp = SC_UNDEFINED;

  if (!SC_UNDEFINED_P(ps)) {
    Pcontrainte eq, eq_cp;
    cp = sc_new();

    for (eq = ps->egalites; eq != NULL; eq = eq->succ) {
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = vect_dup(contrainte_vecteur(eq));
      sc_add_egalite(cp, eq_cp);
    }

    for(eq=ps->inegalites;eq!=NULL;eq=eq->succ) {
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = vect_dup(contrainte_vecteur(eq));
      sc_add_inegalite(cp, eq_cp);
    }

    if(ps->dimension==0) {
      assert(VECTEUR_UNDEFINED_P(ps->base));
      cp->dimension = 0;
      cp->base = VECTEUR_UNDEFINED;
    }
    else {
      assert(ps->dimension==vect_size(ps->base));
      cp->dimension = ps->dimension;
      cp->base = base_dup(ps->base);
    }
  }

  return cp;
  */
}

/* Psysteme sc_copy(Psysteme ps): duplication d'un systeme (allocation
 * et copie complete des champs sans sharing)
 *
 * replace sc_dup(ps), which now becomes a link
 *
 * Ancien nom (obsolete): cp_sc()
 *
 * Modification: L'ordre des egalites, inegalites, la base et des
 * variables dans le syteme est recopie egalement. (CA,
 * 28/08/91),(DN,24/6/02)
 *
 * We can use contrainte_copy, contraintes_copy here If we test the
 * size of system here for debugging purposes, il may cost more time
 * ...
 *
 */
Psysteme sc_copy(Psysteme ps)
{
  Psysteme cp = SC_UNDEFINED;
  int i,j;

  if (!SC_UNDEFINED_P(ps)) {
    Pcontrainte eq, eq_cp;
    cp = sc_new();

    for (j=ps->nb_eq;j>0;j--) {
      for (eq = ps->egalites,i=1;i<j; eq = eq->succ,i++) {/**/}
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = vect_copy(contrainte_vecteur(eq));
      sc_add_egalite(cp, eq_cp);
    }

    for (j=ps->nb_ineq;j>0;j--) {
      for (eq = ps->inegalites,i=1;i<j; eq = eq->succ,i++) {/**/}
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = vect_copy(contrainte_vecteur(eq));
      sc_add_inegalite(cp, eq_cp);
    }

    if(ps->dimension==0) {
      cp->dimension = 0;
      cp->base = VECTEUR_UNDEFINED;
    }
    else {
      cp->dimension = ps->dimension;
      cp->base = base_copy(ps->base);
    }
  }

  return cp;
}
/* void sc_rm(Psysteme ps): liberation de l'espace memoire occupe par
 * le systeme de contraintes ps;
 *
 * utilisation standard:
 *    sc_rm(s);
 *    s = NULL;
 *
 * comme toujours, les champs pointeurs sont remis a NULL avant la
 * desallocation pour detecter au plus tot les erreurs dues a
 * l'allocation dynamique de memoire
 *
 */
void sc_rm(ps)
Psysteme ps;
{
    if (ps != NULL) {
	if (ps->inegalites != NULL) {
	    contraintes_free(ps->inegalites);
	    ps->inegalites = NULL;
	}

	if (ps->egalites != NULL) {
	    contraintes_free(ps->egalites);
	    ps->egalites = NULL;
	}

	if (!VECTEUR_NUL_P(ps->base)) {
	    vect_rm(ps->base);
	    ps->base = VECTEUR_UNDEFINED;
	}

	free((char *) ps);
    }
}

/* This function returns a new empty system which has been initialized
 * with the same dimension and base than sc.
 */
Psysteme sc_init_with_sc(sc)
Psysteme sc;
{

    Psysteme sc1= sc_new();
    sc1->dimension = sc->dimension;
    sc1->base = base_copy(sc->base);
    return(sc1);
}

/* Psysteme sc_empty(Pbase b): build a Psysteme with one unfeasible
 * constraint to define the empty subspace in a space R^n, where n is
 * b's dimension. b is shared by sc.
 *
 * The unfeasible constraint is the equations 0 == 1
 */
Psysteme sc_empty(b)
Pbase b;
{
    Psysteme sc = SC_UNDEFINED;
    Pvecteur v = vect_new(TCST, VALUE_ONE);
    Pcontrainte eq = contrainte_make(v);
    sc = sc_make(eq, CONTRAINTE_UNDEFINED);

    sc_base(sc) = b;
    sc_dimension(sc) = base_dimension(b);

    return sc;
}

/* Psysteme sc_rn(Pbase b): build a Psysteme without constraints to
 * define R^n, where n is b's dimension. b is shared by sc.
 */
Psysteme sc_rn(b)
Pbase b;
{
    Psysteme sc = sc_new();

    sc_base(sc) = b;
    sc_dimension(sc) = base_dimension(b);

    return sc;
}
/* bool sc_empty_p(Psysteme sc): check if the set associated to sc
 * is the constant sc_empty or not. More expensive tests like
 * sc_faisabilite() are necessary to handle the general case.
 */
bool sc_empty_p(sc)
Psysteme sc;
{
    bool empty = false;

    assert(!SC_UNDEFINED_P(sc));
    if(sc_nbre_inegalites(sc)==0 && sc_nbre_egalites(sc)==1) {
	Pvecteur eq = contrainte_vecteur(sc_egalites(sc));

	empty = vect_size(eq) == 1 && vecteur_var(eq) == TCST;
	if(empty)
	    assert(vecteur_val(eq)!=0);
    }
    return empty;
}

/* bool sc_rn_p(Psysteme sc): check if the set associated to sc is
   the whole space, rn
 */
bool sc_rn_p(sc)
Psysteme sc;
{
    assert(!SC_UNDEFINED_P(sc));
    return sc_nbre_inegalites(sc)==0 && sc_nbre_egalites(sc)==0;
}


/* void sc_add_egalite(Psysteme p, Pcontrainte e): macro ajoutant une
 * egalite e a un systeme p; la base n'est pas mise a jour; il faut faire
 * ensuite un appel a sc_creer_base(); il vaut mieux utiliser sc_make()
 *
 * sc_add_eg est (a peu pres) equivalent a sc_add_egalite, mais le
 * parametre e n'est utilise qu'une fois ce qui permet d'eviter
 * des surprises en cas de e++ et autres effects de bords a chaque
 * evaluation de e; sc_add_egalite est donc plus sur que sc_add_eg
 *
 * If the system basis should be updated, use sc_constraint_add()
 * and the two related function in sc_insert.c
 */
void sc_add_egalite(Psysteme p, Pcontrainte e)
{
  assert(p && e && !e->succ);

  e->succ = p->egalites;
  p->egalites = e;
  p->nb_eq++;
}

/* void sc_add_inegalite(Psysteme p, Pcontrainte i): macro ajoutant une
 * inegalite i a un systeme p; la base n'est pas mise a jour; il faut
 * ensuite faire un appel a sc_creer_base(); il vaut mieux utiliser
 * sc_make();
 *
 * sc_add_ineg est (a peu pres) equivalent a sc_add_inegalite; cf supra
 * pour l'explication des differences
 */
void sc_add_inegalite(Psysteme p, Pcontrainte i)
{
  assert(p && i && !i->succ);

  i->succ = p->inegalites;
  p->inegalites = i;
  p->nb_ineq++;
}

void sc_add_egalites(Psysteme p, Pcontrainte i)
{
  if(CONTRAINTE_UNDEFINED_P(sc_egalites(p))) {
    sc_egalites(p) = i;
  }
  else {
    Pcontrainte ineq = CONTRAINTE_UNDEFINED;
    for(ineq = sc_egalites(p);
	!CONTRAINTE_UNDEFINED_P(contrainte_succ(ineq));
	ineq = contrainte_succ(ineq)) {
    }
    contrainte_succ(ineq) = i;
  }
  /* Adjust the number of equalities */
  sc_nbre_egalites(p) = nb_elems_list(sc_egalites(p));
}

void sc_add_inegalites(Psysteme p, Pcontrainte i)
{
  if(CONTRAINTE_UNDEFINED_P(sc_inegalites(p))) {
    sc_inegalites(p) = i;
  }
  else {
    Pcontrainte ineq = CONTRAINTE_UNDEFINED;
    for(ineq = sc_inegalites(p);
	!CONTRAINTE_UNDEFINED_P(contrainte_succ(ineq));
	ineq = contrainte_succ(ineq)) {
    }
    contrainte_succ(ineq) = i;
  }
  /* Adjust the number of inequalities */
  sc_nbre_inegalites(p) = nb_elems_list(sc_inegalites(p));
}
