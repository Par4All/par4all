 /* package sur les systemes de contraintes sc
  *
  * Malik Imadache, Corinne Ancourt, Neil Butler, Francois Irigoin
  *
  */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>

#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#define MALLOC(s,t,f) malloc(s)

/* norm_syst(Psysteme): division des contraintes, egalites ou inegalites,
 * par le PGCD des coefficients de chaque contrainte; 
 *
 * la faisabilite ou la non-faisabilite n'est pas testee: le terme
 * constant ne devrait pas etre pris en compte dans norm_eq().
 */
void norm_syst(sc)
Psysteme sc;
{
    Pcontrainte eq;

    for (eq=sc->egalites;eq!=NULL;eq=eq->succ)
	norm_eq(eq);
    for (eq=sc->inegalites;eq!=NULL;eq=eq->succ)
	norm_eq(eq);
}

/* Psysteme sc_make(Pcontrainte leg, Pcontrainte lineg):
 * allocation et initialisation d'un systeme d'equations et inequations
 * lineaires a partir de deux listes de contraintes, une liste
 * d'egalites et une liste d'inegalites
 *
 * ATTENTION: les deux listes leg et lineq ne sont pas dupliquees; un appel
 * a cette fonction peut donc creer du sharing entre structures de donnees
 *
 * Ancien nom: mk_syst()
 *
 * Modifications:
 *  - ajout de l'initialisation de base et de dimension par un appel
 *    a sc_creer_base()
 */
Psysteme sc_make(leg,lineg)
Pcontrainte leg;
Pcontrainte lineg;
{
    Psysteme s;

    s = (Psysteme ) MALLOC(sizeof(Ssysteme),SYSTEME,"sc_make");
    s->egalites = leg;
    s->inegalites = lineg;
    s->nb_eq = nb_elems_list(leg);
    s->nb_ineq = nb_elems_list(lineg);
    s->base = VECTEUR_UNDEFINED;
    sc_creer_base(s);
    return(s);
}

/* Psysteme sc_translate(Psysteme s, Pbase b, char * (*variable_name)()):
 * reecriture du systeme s dans la base b basee sur les noms des vecteurs
 * de base; tous les vecteurs de base utilises dans s doivent avoir un
 * vecteur de base de meme nom dans b
 */
Psysteme sc_translate(s, b, variable_name)
Psysteme s;
Pbase b;
char * (*variable_name)();
{
    Pcontrainte e;
    Pcontrainte i;

    if(!SC_UNDEFINED_P(s)) {

	/* translate all equations */
	for(e=s->egalites; !CONTRAINTE_UNDEFINED_P(e); e = e->succ)
	    (void) contrainte_translate(e, b, variable_name);

	/* translate all inequalities */
	for(i=s->inegalites; !CONTRAINTE_UNDEFINED_P(i); i = i->succ)
	    (void) contrainte_translate(i, b, variable_name);

	/* update basis in s; its dimension should not change */
	s->base = vect_translate(s->base, b, variable_name);
    }
    return s;
}

/* Psysteme sc_variable_rename(Psysteme s, Variable v_old, Variable v_new):
 * reecriture du systeme s remplacant toutes les occurences de la coordonnees
 * v_old par des occurences de v_new
 */
Psysteme sc_variable_rename(s, v_old, v_new)
Psysteme s;
Variable v_old;
Variable v_new;
{
    Pcontrainte e, i;

    if(!SC_UNDEFINED_P(s)) {

	/* rename all equations */
	for(e=s->egalites; !CONTRAINTE_UNDEFINED_P(e); e = e->succ)
	    (void) contrainte_variable_rename(e, v_old, v_new);

	/* rename all inequalities */
	for(i=s->inegalites; !CONTRAINTE_UNDEFINED_P(i); i = i->succ)
	    (void) contrainte_variable_rename(i, v_old, v_new);

	/* update basis in s; its dimension should not change */
	s->base = vect_variable_rename(s->base, v_old, v_new);

	/* this is not safe as v_new may already be in the basis */
	assert(vect_check(s->base));
    }
    return s;
}

/* Psysteme sc_rename_variables(s, renamed_p, new_variable)
 * Psysteme s;
 * boolean (*renamed_p)(Variable);
 * Variable (*new_variable)(Variable);
 *
 * what: driven renaming of variables in s.
 * how: scans, decides and replaces.
 * input: Psysteme s, plus the decision and replacement functions
 * output: s is returned.
 * side effects:
 *  - the system is modified in place.
 * bugs or features:
 *  - was written by FC...
 */
Psysteme sc_rename_variables(s, renamed_p, new_variable)
Psysteme s;
boolean (*renamed_p)(/*Variable*/);
Variable (*new_variable)(/*Variable*/);
{
    Pcontrainte c;

    if(SC_UNDEFINED_P(s)) return(s);

    for(c=sc_egalites(s); c!=NULL; c=c->succ)
	(void) vect_rename_variables(contrainte_vecteur(c),
				     renamed_p, new_variable);

    for(c=sc_inegalites(s); c!=NULL; c=c->succ)
	(void) vect_rename_variables(contrainte_vecteur(c),
				     renamed_p, new_variable);

    (void) vect_rename_variables(sc_base(s), renamed_p, new_variable);

    return(s);
}

/* Psysteme sc_variables_rename(Psysteme s, Pvecteur pv_old, Pvecteur pv_new):
 * reecriture du systeme s remplacant toutes les occurences des coordonnees
 * de pv_old par des occurences de pv_new
 */
Psysteme sc_variables_rename(s, pv_old, pv_new,variable_name)
Psysteme s;
Pvecteur pv_old;
Pvecteur pv_new;
char * (*variable_name)();
{
    Pvecteur pv;
    for (pv = pv_old; !VECTEUR_UNDEFINED_P(pv); pv = pv->succ) {
	Variable var_new = base_find_variable_name(pv_new, vecteur_var(pv),
						   variable_name);
	if (!VARIABLE_UNDEFINED_P(var_new))
	    s = sc_variable_rename(s,pv->var,var_new);
    }  
    return s;
}

void sc_base_remove_variable(sc,v)
Psysteme sc;
Variable v;
{
    sc_base(sc) = base_remove_variable(sc_base(sc),v);
    sc_dimension(sc) --;
}


void sc_base_add_variable(sc,var)
Psysteme sc;
Variable var;
{

    Pbase b1 = sc->base; 

    if (!VECTEUR_NUL_P(b1)) {
	for(; !VECTEUR_NUL_P(b1) && !variable_equal(vecteur_var(b1), var);
	    b1 = b1->succ);
	if (VECTEUR_NUL_P(b1)) {
	    for (b1 = sc->base; !VECTEUR_NUL_P(b1->succ); b1=b1->succ);
	    b1->succ = vect_new(var, VALUE_ONE);
	    sc_dimension(sc)++;
	} 
    }
    else {
	sc->base = vect_new(var, VALUE_ONE);
	sc_dimension(sc)++; }
}

/* boolean sc_consistent_p(Psysteme sc): check that sc is well defined, that the
 * numbers of equalities and inequalities are consistent with the lists of
 * equalities and inequalities, and that every variable in the constraints is
 * in the base
 *
 * Francois Irigoin, 7 July 1993
 *
 * Note: 
 *  - it also checks that every variable in the basis is used with a non-zero
 * coefficient in at least one constraint (although there is no reason for that)
 *  - there is no explicit check of TCST in the basis; TCST should *not* be in the
 * basis
 */
boolean sc_consistent_p(sc)
Psysteme sc;
{
    boolean consistent;

    consistent = !SC_UNDEFINED_P(sc);

    if(consistent) {
	consistent = (sc->nb_eq == nb_elems_list(sc_egalites(sc)))
	    && 
		(sc->nb_ineq == nb_elems_list(sc_inegalites(sc)))
		    &&
			(sc_dimension(sc) == base_dimension(sc_base(sc)))
			    &&
				base_normalized_p(sc_base(sc));
    }
					      

    if(consistent) {
	Pcontrainte eq = CONTRAINTE_UNDEFINED;
	Pbase diagonale = BASE_UNDEFINED;
	Pvecteur pv = VECTEUR_UNDEFINED;
	Pbase diff = BASE_UNDEFINED;

	for(eq = sc->egalites; eq!= NULL; eq=eq->succ) {
	    for (pv = eq->vecteur;pv!= NULL;pv=pv->succ)
		if (pv->var != TCST)
		    vect_chg_coeff(&diagonale,pv->var, VALUE_ONE);
	}
	for(eq = sc->inegalites; eq!= NULL; eq=eq->succ) {
	    for (pv = eq->vecteur;pv!= NULL;pv=pv->succ)
		if (pv->var != TCST)
		    vect_chg_coeff(&diagonale,pv->var, VALUE_ONE);
	}
	diff = base_difference(diagonale, sc_base(sc));
	consistent = BASE_NULLE_P(diff);
    }

/* This assert is too bad ! I remove it.
 *
 * Alexis Platonoff, 31 january 1995 */
/*    assert(consistent); */

    return consistent;
}

/* check that sc is well defined, that the numbers of equalities and
 * inequalities are consistent with the lists of equalities and
 * inequalities, that the dimension is consistent with the basis, that the
 * basis itself is consistent (all coefficients must be 1), and that every
 * variable in the constraints is in the basis.
 *
 * Each component in the basis should only appear once thanks to the
 * specifications of Pvecteur (this is not checked).
 *
 * Francois Irigoin, 13 November 1995
 *
 * Note: 
 *  - there is no explicit check of TCST in the basis;
 * TCST should *not* be in the basis for some use of Psystemes, 
 * like transformers in PIPS.
 * */
boolean sc_weak_consistent_p(Psysteme sc)
{
    boolean weak_consistent;

    weak_consistent = !SC_UNDEFINED_P(sc);

    if(weak_consistent) {
	weak_consistent = (sc->nb_eq == nb_elems_list(sc_egalites(sc)))
	    && (sc->nb_ineq == nb_elems_list(sc_inegalites(sc)))
		    && (sc_dimension(sc) == base_dimension(sc_base(sc)));
    }

    if(weak_consistent && sc_dimension(sc) != 0) {
	base_normalized_p(sc_base(sc));
    }
					      

    if(weak_consistent) {
	Pcontrainte eq;
	Pvecteur pv;
	Pbase diagonale = BASE_NULLE;

	for(eq = sc->egalites; eq!= NULL; eq=eq->succ) {
	    for (pv = eq->vecteur;pv!= NULL;pv=pv->succ)
		if (pv->var != TCST)
		    vect_chg_coeff(&diagonale,pv->var, VALUE_ONE);
	}
	for(eq = sc->inegalites; eq!= NULL; eq=eq->succ) {
	    for (pv = eq->vecteur;pv!= NULL;pv=pv->succ)
		if (pv->var != TCST)
		    vect_chg_coeff(&diagonale,pv->var, VALUE_ONE);
	}
	weak_consistent =  base_included_p(diagonale, sc_base(sc));
	base_rm(diagonale);
    }

    /* assert(weak_consistent); */

    return weak_consistent;
}
