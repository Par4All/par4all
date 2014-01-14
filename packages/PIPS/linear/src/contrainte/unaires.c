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

 /* PACKAGE CONTRAINTE - OPERATIONS UNAIRES
  */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

/* norm_eq: normalisation d'une contrainte par le pgcd de TOUS les
 * coefficients, i.e. y compris le terme constant
 */
void norm_eq(nr)
Pcontrainte nr;
{
    vect_normalize(nr->vecteur);
}

/* void contrainte_chg_sgn(Pcontrainte eq): changement de signe d'une 
 * contrainte, i.e. multiplication par -1. Les equations ne sont pas
 * modifiees mais les inequations sont transformees.
 *
 * Ancien nom: ch_sgn
 */
void contrainte_chg_sgn(c)
Pcontrainte c;
{
    vect_chg_sgn(c->vecteur);
}


/* void contrainte_reverse(Pcontrainte eq): changement de signe d'une 
 * contrainte, i.e. multiplication par -1, et ajout de la constante 1.
 *
 */
void contrainte_reverse(c)
Pcontrainte c;
{
    contrainte_chg_sgn(c);
    vect_add_elem(&(c->vecteur), TCST, VALUE_ONE);
}

/* void_eq_set_vect_nul(Pcontrainte c): transformation d'une contrainte
 * en une contrainte triviale 0 == 0
 *
 * cette fonction est utile lorsque l'on veut eliminer plusieurs         
 * contraintes du systeme sans avoir a le restructurer apres chaque        
 * elimination.                                                          
 *
 * Pour eliminer toutes ces "fausses" contraintes on utilise a la fin la   
 * fonction "syst_elim_eq" (ou "sc_rm_empty_constraints"...)                                              
 */
void eq_set_vect_nul(c)
Pcontrainte c;
{
    if(!CONTRAINTE_UNDEFINED_P(c)) {
	vect_rm(contrainte_vecteur(c));
	contrainte_vecteur(c) = VECTEUR_NUL;
    }
}

/* Pcontrainte contrainte_translate(Pcontrainte c, Pbase b, 
 *                                  char * (*variable_name)()):
 * normalisation des vecteurs de base utilises dans c par rapport
 * a la base b utilisant les "noms" des vecteurs de base; en sortie
 * tous les vecteurs de base utilises dans c appartiennent a b;
 */
Pcontrainte contrainte_translate(c, b, variable_name)
Pcontrainte c;
Pbase b;
char * (*variable_name)();
{
    if(!CONTRAINTE_UNDEFINED_P(c))
	contrainte_vecteur(c) = vect_translate(contrainte_vecteur(c), b,
					       variable_name);

    return c;
}

/* Pcontrainte contrainte_variable_rename(Pcontrainte c, Variable v_old,
 *                                        Variable v_new):
 * rename the potential coordinate v_old in c as v_new
 */
Pcontrainte contrainte_variable_rename(c, v_old, v_new)
Pcontrainte c;
Variable v_old;
Variable v_new;
{
    if(!CONTRAINTE_UNDEFINED_P(c))
	contrainte_vecteur(c) = vect_variable_rename(contrainte_vecteur(c), 
						     v_old, v_new);

    return c;
}

/* void Pcontrainte_separate_on_vars(initial, vars, pwith, pwithout)
 * Pcontrainte initial;
 * Pbase vars;
 * Pcontrainte *pwith, *pwithout;
 *
 *     IN: initial, vars
 *    OUT: pwith, pwithout
 *
 * builds two Pcontraintes from the one given, using the
 * constraint_without_vars criterium.
 * 
 * (c) FC 16/05/94
 */
void Pcontrainte_separate_on_vars(initial, vars, pwith, pwithout)
Pcontrainte initial;
Pbase vars;
Pcontrainte *pwith, *pwithout;
{
    Pcontrainte
	c = (Pcontrainte) NULL,
	new = CONTRAINTE_UNDEFINED;

    for(c=initial, 
	*pwith=(Pcontrainte)NULL,
	*pwithout=(Pcontrainte)NULL; 
	c!=(Pcontrainte) NULL;
	c=c->succ)
	if (constraint_without_vars(c, vars))
	    new = contrainte_make(vect_dup(c->vecteur)),
	    new->succ = *pwithout, 
	    *pwithout = new;
	else
	    new = contrainte_make(vect_dup(c->vecteur)),
	    new->succ = *pwith, 
	    *pwith = new;
}

/* void constraints_for_bounds(var, pinit, plower, pupper)
 * Variable var;
 * Pcontrainte *pinit, *plower, *pupper;
 * IN: var, *pinit;
 * OUT: *pinit, *plower, *pupper;
 *
 * separate the constraints involving var for upper and lower bounds
 * The constraints are removed from the original system. 
 * everything is touched. Should be fast because there is no allocation.
 *
 * FC 28/11/94
 */
void constraints_for_bounds(var, pinit, plower, pupper)
Variable var;
Pcontrainte *pinit, *plower, *pupper;
{
    Value 
	v;
    Pcontrainte
        c, next,
	remain = NULL,
        lower = NULL,
	upper = NULL;

    for(c = *pinit, next=(c==NULL ? NULL : c->succ); 
	c!=NULL; 
        c=next, next=(c==NULL ? NULL : c->succ))
    {
	v = vect_coeff(var, c->vecteur);

	if (value_pos_p(v))
	    c->succ = upper, upper = c;
	else if (value_neg_p(v))
	    c->succ = lower, lower = c;
	else /* v==0 */
	    c->succ = remain, remain = c;
    }

    *pinit = remain,
    *plower = lower,
    *pupper = upper;
}

/* Pcontrainte contrainte_dup_extract(c, var)
 * Pcontrainte c;
 * Variable var;
 *
 * returns a copy of the constraints of c which contain var.
 *
 * FC 27/09/94
 */
Pcontrainte contrainte_dup_extract(c, var)
Pcontrainte c;
Variable var;
{
    Pcontrainte
	result = NULL,
	pc, ctmp;

    for (pc=c; pc!=NULL; pc=pc->succ)
	if ((var==NULL) || vect_coeff(var, pc->vecteur)!=0)
	    ctmp = contrainte_dup(pc),
	    ctmp->succ = result,
	    result = ctmp;
    
    return(result);
}

/* Pcontrainte contrainte_extract(pc, base, var)
 * Pcontrainte *pc;
 * Pbase base;
 * Variable var;
 *
 * returns the constraints of *pc of which the higher rank variable from base 
 * is var. These constraints are removed from *pc.
 *
 * FC 27/09/94
 */
Pcontrainte contrainte_extract(pc, base, var)
Pcontrainte *pc;
Pbase base;
Variable var;
{
    int
	rank = rank_of_variable(base, var);
    Pcontrainte
	ctmp = NULL,
	result = NULL,
	cprev = NULL,
	c = *pc;

    while (c!=NULL)
    {
	if (search_higher_rank(c->vecteur, base)==rank)
	{
	    /*
	     * c must be extracted
	     */
	    ctmp = c->succ,
	    c->succ = result,
	    result = c,
	    c = ctmp;
	    
	    if (cprev==NULL)
		*pc = ctmp;
	    else
		cprev->succ=ctmp;
	}
	else
	    c=c->succ, 
	    cprev=(cprev==NULL) ? *pc : cprev->succ;
    }

    return(result);
}

/* int level_contrainte(Pcontrainte pc, Pbase base_index)
 * compute the level (rank) of the constraint pc in the nested loops.
 * base_index is the index basis in the good order
 * The result corresponds to the rank of the greatest index in the constraint, 
 * and the sign of the result  corresponds to the sign of the coefficient of 
 * this  index  
 * 
 * For instance:
 * base_index :I->J->K ,
 *                 I - J <=0 ==> level -2
 *             I + J + K <=0 ==> level +3
 */
int level_contrainte(pc, base_index)
Pcontrainte pc;
Pbase base_index;
{
    Pvecteur pv;
    Pbase pb;
    int level = 0;
    int i;
    int sign=1;
    bool trouve = false;

    for (pv = pc->vecteur;
	 pv!=NULL;
	 pv = pv->succ)
    {
	for (i=1, trouve=false, pb=base_index;
	     pb!=NULL && !trouve;
	     i++, pb=pb->succ)
	    if (pv->var == pb->var)
	    {
		trouve = true;
		if (i>level)
		    level = i, sign = value_sign(pv->val);
	    }
    }
    return(sign*level);
}

/* it sorts the vectors as expected. FC 24/11/94
 */
void 
contrainte_vect_sort(c, compare)
Pcontrainte c;
int (*compare)(Pvecteur *, Pvecteur *);
{
    for (; c!=NULL; c=c->succ)
	vect_sort_in_place(&c->vecteur, compare);
}


/* Pcontrainte contrainte_var_min_coeff(Pcontrainte contraintes, Variable v,
 *                           int *coeff)
 * input    : a list of constraints (euqalities or inequalities), 
 *            a variable, and the location of an integer.
 * output   : the constraint in "contraintes" where the coefficient of
 *            "v" is the smallest (but non-zero).
 * modifies : nothing.
 * comment  : the returned constraint is not removed from the list if 
 *            rm_if_not_first_p is false.
 *            if rm_if_not_first_p is true, the returned contraint is
 *            remove only if it is not the first constraint.
 */
Pcontrainte 
contrainte_var_min_coeff(contraintes, v, coeff, rm_if_not_first_p) 
Pcontrainte contraintes;
Variable v;
Value *coeff;
bool rm_if_not_first_p;
{
    Value sc = VALUE_ZERO, cv = VALUE_ZERO;
    Pcontrainte result, eq, pred, eq1;

    if (contraintes == NULL) 
	return(NULL);

    result = pred = eq1 = NULL;
    
    for (eq = contraintes; eq != NULL; eq = eq->succ) {
	Value c, ca;
	c = vect_coeff(v, eq->vecteur);
	ca = value_abs(c);
	if ((value_lt(ca,cv) && value_pos_p(ca)) || 
	    (value_zero_p(cv) && value_notzero_p(c))) {
	    cv = ca;
	    sc = c;
	    result = eq;
	    pred = eq1;
	}
    }

    if (value_neg_p(sc))
	contrainte_chg_sgn(result);
    
    if (rm_if_not_first_p && pred != NULL) {
	pred->succ = result->succ;
	result->succ = NULL;
    }
  
    *coeff = cv;
    return result;
}

/*
 * Constraint sorting for prettyprinting
 *
 */

/* Required because qsort (and C) do no let us parametrize the
 * comparison function (no lambda closure).
 */
static int (* lexicographic_compare)(Pvecteur *, Pvecteur *) = NULL;

int
equation_lexicographic_compare(Pcontrainte c1, Pcontrainte c2, 
				 int (*compare)(Pvecteur*, Pvecteur*))
{
    /* it is assumed that constraints c1 and c2 are already
       lexicographically sorted */
    int cmp = 0;

    cmp = vect_lexicographic_compare(c1->vecteur, c2->vecteur, compare);

    return cmp;
}

static int
internal_equation_compare(Pcontrainte * pc1, Pcontrainte * pc2)
{
    int cmp = equation_lexicographic_compare(*pc1, *pc2, 
					       lexicographic_compare);
    return cmp;
}

int
inequality_lexicographic_compare(Pcontrainte c1, Pcontrainte c2, 
				 int (*compare)(Pvecteur*, Pvecteur*))
{
    /* it is assumed that constraints c1 and c2 are already
       lexicographically sorted */
    int cmp = 0;

    cmp = vect_lexicographic_compare2(c1->vecteur, c2->vecteur, compare);

    return cmp;
}

static int
internal_inequality_compare(Pcontrainte * pc1, Pcontrainte * pc2)
{
    int cmp = inequality_lexicographic_compare(*pc1, *pc2, 
					       lexicographic_compare);
    return cmp;
}


Pcontrainte 
equations_lexicographic_sort(Pcontrainte cl,
			       int (*compare)(Pvecteur*, Pvecteur*))
{
    Pcontrainte result = CONTRAINTE_UNDEFINED;

    result = constraints_lexicographic_sort_generic(cl, compare, true);

    return result;
}

Pcontrainte 
inequalities_lexicographic_sort(Pcontrainte cl,
			       int (*compare)(Pvecteur*, Pvecteur*))
{
    Pcontrainte result = CONTRAINTE_UNDEFINED;

    result = constraints_lexicographic_sort_generic(cl, compare, false);

    return result;
}

/* For historical reasons, equal to equations_lexicographic_sort() */
Pcontrainte 
constraints_lexicographic_sort(Pcontrainte cl,
			       int (*compare)(Pvecteur*, Pvecteur*))
{
    Pcontrainte result = CONTRAINTE_UNDEFINED;

    result = constraints_lexicographic_sort_generic(cl, compare, true);

    return result;
}

Pcontrainte 
constraints_lexicographic_sort_generic(Pcontrainte cl,
				       int (*compare)(Pvecteur*, Pvecteur*),
				       bool is_equation)
{
    int n = nb_elems_list(cl);
    Pcontrainte result = CONTRAINTE_UNDEFINED;
    Pcontrainte * table = NULL;
    Pcontrainte * elem = NULL;
    Pcontrainte ce;

    if ( n==0 || n==1 )
	return cl;

    lexicographic_compare = compare;

    /*  the temporary table is created and initialized
     */
    table = (Pcontrainte*) malloc(sizeof(Pcontrainte)*n);
    assert(table!=NULL);

    for (ce=cl, elem=table; ce!=CONTRAINTE_UNDEFINED; ce=ce->succ, elem++)
	*elem=ce;

    /*  sort!
     */
    if(is_equation)
	qsort((char *) table, n, sizeof(Pcontrainte),
	      (int (*)()) internal_equation_compare);
    else
	qsort((char *) table, n, sizeof(Pcontrainte),
	      (int (*)()) internal_inequality_compare);

    /*  the vector is regenerated in order
     */
    for (elem=table; n>1; elem++, n--)
	(*elem)->succ=*(elem+1);

    (*elem)->succ= CONTRAINTE_UNDEFINED;
    
    /*  clean and return
     */
    result = *table;
    free(table);
    return result;
}

/* returns whether a constraint is a simple equality: X == 12
 * the system is expected to be normalized?
 */
Variable contrainte_simple_equality(Pcontrainte e)
{
  Pvecteur v = e->vecteur;
  int size = vect_size(v);
  switch (size) {
  case 0: return NULL;
  case 1: if (v->var) return v->var; else return NULL;
  case 2: 
    if (v->var && !v->succ->var) return v->var;
    if (!v->var && v->succ->var) return v->succ->var;
  }
  return NULL;
}

/*    that is all
 */
