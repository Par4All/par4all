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

/* package vecteur - operations unaires */

/*LINTLIBRARY*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(p,t,f) free(p)

/* void vect_normalize(Pvecteur v): division de tous les coefficients de v
 * par leur pgcd; "normalisation" de vecteur directeur a coefficient entier
 *
 *    ->   ->      ->   ->
 * si v == 0 alors v := 0;
 * sinon
 *    pgcd = PGCD v[i];
 *	       i
 *    ->   ->
 *    v := v / pgcd
 *
 * Le pgcd est toujours positif.
 *
 * Ancien nom: vect_norm()
 */
void vect_normalize(Pvecteur v)
{
  Value gcd = vect_pgcd_all(v);
  if (value_notzero_p(gcd) && value_notone_p(gcd))
    (void) vect_div(v, gcd);
}

/* void vect_add_elem(Pvecteur * pvect, Variable var, Value val):
 * addition d'un vecteur colineaire au vecteur de base var au vecteur vect
 *
 * ----->    ----->       --->
 * *pvect := *pvect + val evar
 */
void vect_add_elem(Pvecteur * pvect,Variable var, Value val)
{
  if (val!=0)
  {
    Pvecteur vect;
    for (vect=(*pvect); vect!=NULL; vect=vect->succ)
    {
	    if (var_of(vect)==var)
      {
        value_addto(val_of(vect), val);
        if (value_zero_p(val_of(vect)))
          vect_erase_var(pvect, var_of(vect));
        return;
	    }
    }
    // else: le coefficient valait 0 et n'etait pas represente
    *pvect = vect_chain(*pvect, var, val);
  }
	// sinon, le vecteur est inchange et on ne fait rien
}

/* void vect_erase_var(Pvecteur * ppv, Variable v): projection du
 * vecteur *ppv selon la direction v (i.e. mise a zero de la
 * coordonnee v du vecteur pointe par ppv)
 *
 * Soit ev le vecteur de base correspondant a v:
 *
 *  --->    --->    --->   ->  ->
 *  *ppv := *ppv - <*ppv . ev> ev
 *
 * Note: cette routine ne fait pas l'hypothese que chaque coordonnee
 * n'apparait qu'une fois; on pourrait l'accelerer en forcant
 * pvcour a NULL des que la coordonnee est trouvee.
 */
void vect_erase_var(ppv, v)
Pvecteur *ppv;
Variable v;
{
    Pvecteur pvprec, pvcour;

    for (pvprec = NULL, pvcour = (*ppv); pvcour != NULL;) {
	/* A-t-on trouve la composante v? */
	if (pvcour->var == v) {
	    /* Si oui, est-il possible de la dechainer? */
	    if (pvprec != NULL) {
		/* elle n'est pas en tete de liste */
		Pvecteur pvprim = pvcour;
		pvcour = pvprec->succ = pvcour->succ;
		FREE((char *)pvprim, VECTEUR, "vect_erase_var");
	    }
	    else {
		/* Elle est en tete de liste; il faut modifier ppv */
		*ppv = pvcour->succ;
		FREE((char *)pvcour,VECTEUR,"vect_erase_var");
		pvcour = *ppv;
	    }
	}
	else {
	    /* Non, on passe a la composante suivante... */
	    pvprec = pvcour;
	    pvcour = pvcour->succ;
	}
    }
}

/* void vect_chg_coeff(Pvecteur *ppv, Variable var, Value val): mise
 * de la coordonnee var du vecteur *ppv a la valeur val
 *
 * --->   --->    --->   --->  --->       --->
 * *ppv = *ppv - <*ppv . evar> evar + val evar 
 */
void vect_chg_coeff(ppv,var,val)
Pvecteur   *ppv;
Variable var;
Value val;
{
    Pvecteur pvcour;

    if (val == 0) {
	vect_erase_var(ppv, var);
    }
    else {
	for (pvcour = (*ppv); pvcour != NULL; pvcour = pvcour->succ) {
	    if (pvcour->var == var) {
		pvcour->val = val;
		return;
	    }
	}
	/* on n'a pas trouve de composante var */
	*ppv = vect_chain(*ppv,var,val);
    }
}

/* void vect_chg_var(Pvecteur *ppv, Variable v_old, Variable v_new)
 * replace the variable v_old by v_new 
 */
void vect_chg_var(ppv,v_old,v_new)
Pvecteur *ppv;
Variable v_old,v_new;
{
    Pvecteur pvcour;

    for (pvcour = (*ppv); pvcour != NULL; pvcour = pvcour->succ) {
	if (pvcour->var == v_old){
	    pvcour->var = v_new;
	    return;
	}
    }
}

Variable vect_one_coeff_if_any(Pvecteur v)
{
  for (; v; v=v->succ)
    if (v->var && (value_one_p(v->val) || value_mone_p(v->val)))
      return v->var;
  return NULL;
}

/* Pvecteur vect_del_var(Pvecteur v_in, Variable var): allocation d'un
 * nouveau vecteur egal a la projection de v_in selon la direction var
 * (i.e. le coefficient de la coordonnee var est mis a 0)
 *
 * Soit evar le vecteur de base correspondant a var:
 *
 *           ---->
 *  allocate v_out;
 *
 *  ---->    --->    ---->   --->  --->
 *  v_out := v_in - <v_out . evar> evar
 *
 *        ---->
 * return v_out;
 *
 */
Pvecteur vect_del_var(v_in,var)
Pvecteur v_in;
Variable var;
{
    if(v_in!=NULL){
	Pvecteur v_out = vect_dup(v_in);
	vect_erase_var(&v_out,var);
	return(v_out);
    }
    else
	return(NULL);
}

/* Variable vect_coeff(Variable var, Pvecteur vect): coefficient
 * de coordonnee var du vecteur vect
 *      --->
 * Soit evar le vecteur de base de nom var:
 * 
 *         --->   --->
 * return <vect . evar>; (i.e. return vect[var])
 *
 */
Value vect_coeff(var,vect)
Variable var;
Pvecteur vect;
{
    for ( ; vect != NULL ; vect = vect->succ)
	if (var_of(vect) == var) {
	    assert(val_of(vect)!=VALUE_ZERO);
	    return(val_of(vect));
	}
    return VALUE_ZERO;
}

/* Value vect_coeff_sum(Pvecteur vect): coefficient sum
 * de tout les val de ce vecteur (devrait etre dans reduction? FC)
 * 
 * return Value
 * Lei Zhou    Mar.25, 91
 */
Value vect_coeff_sum(vect)
Pvecteur vect;
{
    Value val = VALUE_ZERO;

    if ( vect->var == TCST )
	return val;
    for (vect = vect; vect != NULL ; vect = vect->succ) {
	value_addto(val,vecteur_val(vect));
	assert(value_notzero_p(val_of(vect)));
    }
    return val;
}


/* Pvecteur vect_sign(Pvecteur v): application de l'operation signe au
 * vecteur v
 * 
 * ->         ->
 * v := signe(v );
 *        ->
 * return v ;
 */
Pvecteur vect_sign(v)
Pvecteur v;
{
    Pvecteur coord;

    for(coord = v; coord!=NULL; coord=coord->succ)
	val_of(coord) = int_to_value(value_sign(val_of(coord)));

    return v;
}


/* void vect_sort_in_place(pv, compare)
 * Pvecteur *pv;
 * int (*compare)(Pvecteur *, Pvecteur *);
 *
 * Sorts the vector in place. It is an interface to qsort (stdlib).
 * see man qsort about the compare function, which tells < == or >.
 *
 * FC 29/12/94
 */
void vect_sort_in_place(pv, compare)
Pvecteur *pv;
int (*compare)(Pvecteur *, Pvecteur *);
{
    int 
	n = vect_size(*pv);
    Pvecteur 
	v, 
	*table,
	*point;

    if ( n==0 || n==1 ) return;

    /*  the temporary table is created and initialized
     */
    table = (Pvecteur*) malloc(sizeof(Pvecteur)*n);

    for (v=*pv, point=table; v!=(Pvecteur)NULL; v=v->succ, point++)
	*point=v;

    /*  sort!
     */
    /* FI: I do not know how to cast compare() properly */
    /* qsort(table, n, sizeof(Pvecteur), int (* compare)()); */
    qsort(table, n, sizeof(Pvecteur),(int (*)()) compare);

    /*  the vector is regenerated in order
     */
    for (point=table; n>1; point++, n--)
	(*point)->succ=*(point+1);

    (*point)->succ=(Pvecteur) NULL;
    
    /*  clean and return
     */
    *pv=*table; free(table);
}

/* Pvecteur vect_sort(v, compare)
 * Pvecteur v;
 * int (*compare)();
 *
 *   --->           -->
 *   OUT  =  sorted IN
 */
Pvecteur vect_sort(v, compare)
Pvecteur v;
int (*compare)(Pvecteur *, Pvecteur *);
{
    Pvecteur
	new = vect_dup(v);

    vect_sort_in_place(&new, compare);
    return(new);
}

/*  for qsort, returns:
 *
 *     - if v1 < v2
 *     0 if v1 = v2
 *     + if v1 > v2
 */
int vect_compare(pv1, pv2)
Pvecteur *pv1, *pv2;
{
    return(strcmp((char *)&var_of(*pv1), (char *)&var_of(*pv2)));
}

/* void Pvecteur_separate_on_sign(v, pvpos, pvneg)
 * Pvecteur v, *pvpos, *pvneg;
 *
 *     IN: v
 *    OUT: pvpos, pvneg
 *
 * this function builds 2 vectors composed of the positive and negative
 * parts of the initial vector v which is not modified.
 * 
 * (c) FC 16/05/94
 */
void Pvecteur_separate_on_sign(v, pvpos, pvneg)
Pvecteur v, *pvpos, *pvneg;
{
    Pvecteur vc;
    Value val;
    Variable var;

    *pvneg = VECTEUR_NUL,
    *pvpos = VECTEUR_NUL;

    for(vc=v; vc; vc=vc->succ)
    {
	var = var_of(vc), 
	val = val_of(vc);
	if (value_neg_p(val))
	  vect_add_elem(pvneg, var, value_uminus(val));
	else
	  vect_add_elem(pvpos, var, val);
    }
}


/* bool vect_common_variables_p(Pvecteur v1, v2)    BA 19/05/94
 * input    : two vectors.
 * output   : true if they have at least one common variable, 
 *            false otherwise.
 * modifies : nothing.
 */
bool vect_common_variables_p(v1, v2)
Pvecteur v1, v2;
{
    Pvecteur ev;

    for(ev = v1; !VECTEUR_NUL_P(ev); ev = ev->succ) {
	if(vect_contains_variable_p(v2, vecteur_var(ev)))
	    return true;
    }
    return false;
}


/* bool vect_contains_variable_p(Pvecteur v, Variable var)    BA 19/05/94
 * input    : a vector and a variable
 * output   : true if var appears as a component of v, false otherwise.
 * modifies : nothing
 */
bool vect_contains_variable_p(v, var)
Pvecteur v;
Variable var;
{
    bool in_base;

    for(; !VECTEUR_NUL_P(v) && !variable_equal(vecteur_var(v), var); v = v->succ)
	;
    in_base = !VECTEUR_NUL_P(v);
    return(in_base);
}

/* qsort() is not safe if the comparison function is not antisymmetric.
 * It wanders out of the array to be sorted. It's a pain to debug.
 * Let's play safe.
 */
/* Version for equations */
int
vect_lexicographic_compare(Pvecteur v1, Pvecteur v2, 
			   int (*compare)(Pvecteur*, Pvecteur*))
{
    int cmp12 = 0;
    int cmp21 = 0;

    cmp12 = vect_lexicographic_unsafe_compare(v1, v2, compare);
    cmp21 = vect_lexicographic_unsafe_compare(v2, v1, compare);

    assert(cmp12 == -cmp21);

    return cmp12;
}

/* Version for inequalities */
int
vect_lexicographic_compare2(Pvecteur v1, Pvecteur v2,
			    int (*compare)(Pvecteur*, Pvecteur*))
{
    int cmp12 = 0;
    int cmp21 = 0;

    cmp12 = vect_lexicographic_unsafe_compare2(v1, v2, compare);
    cmp21 = vect_lexicographic_unsafe_compare2(v2, v1, compare);

    assert(cmp12 == -cmp21);

    return cmp12;
}

int
vect_lexicographic_unsafe_compare(Pvecteur v1, Pvecteur v2,
				  int (*compare)(Pvecteur*, Pvecteur*))
{
    int cmp = 0;

    cmp = vect_lexicographic_unsafe_compare_generic(v1, v2, compare, true);

    return cmp;
}

int vect_lexicographic_unsafe_compare2(Pvecteur v1, Pvecteur v2,
				       int (*compare)(Pvecteur*, Pvecteur*))
{
    int cmp = 0;

    cmp = vect_lexicographic_unsafe_compare_generic(v1, v2, compare, false);

    return cmp;
}

/* The two sparse vectors are assumed to have exactly the same
   structure, the same non-zero components in the same order. */
static int vect_lexicographic_coefficient_comparison(Pvecteur v1, Pvecteur v2)
{
  Pvecteur pv1, pv2;
  int cmp = 0;

  for(pv1 = v1, pv2 = v2;
      !VECTEUR_UNDEFINED_P(pv1) && !VECTEUR_UNDEFINED_P(pv2) && cmp == 0;
      pv1 = pv1->succ, pv2= pv2->succ) {
    Value c1 = vecteur_val(pv1);
    Value c2 = vecteur_val(pv2);
    if(value_gt(c1, c2))
      cmp = 1;
    else if(value_gt(c2, c1))
      cmp = -1;
  }
  return cmp;
}

/* This function is a trade-off between a real lexicographic sort
 * and a prettyprinting function also used for code generation.
 *
 * Its goal is to sort constraints used as loop bounds or obtained as
 * preconditions. Each constraint or vector is assumed internally
 * sorted using the compare function, e.g. 2i + 3j + 4 is correct,
 * while 3j + 4 + 2i is not if the alphabetical order is used and if
 * constants appear as last vector elements.
 *
 * When multiple constraints appear in a constraint system, we usually
 * want simpler constraints and vectors first but some order between
 * variables is still used. E.g. {1, i, i+1, i+j+1, j}. Here, we do
 * not take into account the vector length as a primary critarion, but
 * the alphabetical order. The above system is first reduced to a set
 * of "words" {"", "i", "ij", "j"} and is lexicographically sorted
 * [This is simplified: each letter in this example is in fact a work
 * in the general case]. In case, two "words" are identical, e.g. "ij"
 * and "ij", the length of the two underlying vectors, e.g. i+j,
 * i+j+1, i+2*j, are compared. If they are equal, the lexicographic
 * order of the coefficients is used to disambiguate the comparison,
 * e.g. i+j < i+2j.
 *
 * A lot of problems arise because 0 is not represented. It's
 * hard to compare I==0 and I==1 because they do not have the
 * same numbers of sparse components.
 *
 * Furthermore, vectors representing equalities and inequalities must not
 * be handled in the same way because only equalities can be multiplied by -1.
 *
 * Not satisfying for Transformations/Tiling.sub/tiling05: 0 constant
 * terms should be handled in a special way so as to have "i" be less
 * than "i-1". Since -1 is less than 0, the longest constraint comes
 * first.
 *
 * Not satisfying for Transformations/Tiling.sub/tiling04: scopes had
 * an impact on the comparisons that is not natural for users.
 */
int
vect_lexicographic_unsafe_compare_generic(Pvecteur v1, Pvecteur v2,
					  int (*compare)(Pvecteur*, Pvecteur*),
					  bool is_equation __attribute__ ((unused)))
{
    /* It is assumed that vectors v1 and v2 are already
       lexicographically sorted, according to the same lexicographic
       order.

       The constant term should always be the last one. But the
       lexicographic sort now move them ahead! So we have to skip them
       at each position but we assume that they are both either
       leading or trailing elements. */
  int cmp = 0;
    Pvecteur pv1 = term_cst(v1)? v1->succ : v1;
    Pvecteur pv2 = term_cst(v2)? v2->succ : v2;

    /* Lexicographic comparison on variable names */
    for(;
	!VECTEUR_UNDEFINED_P(pv1) && !VECTEUR_UNDEFINED_P(pv2) && cmp == 0
	&& !term_cst(pv1) && !term_cst(pv2);
	pv1 = pv1->succ, pv2= pv2->succ) {

	cmp = compare(&pv1, &pv2);
    }

    if(cmp==0) {
      if(VECTEUR_UNDEFINED_P(pv1)) {
	if(VECTEUR_UNDEFINED_P(pv2)) {
	  /* Use vector lengths as discriminator in case an initial
	     constant term makes a difference. */
	  int n1 = vect_size(v1);
	  int n2 = vect_size(v2);
	  if(n1>n2)
	    cmp = 1;
	  else if(n1<n2)
	    cmp = -1;
	  else {
	    /* Use lexicographic order on coefficients as discriminator */
	    cmp = vect_lexicographic_coefficient_comparison(v1, v2);
	  }
	}
	else
	  cmp = -1; // v2 is longer
      }
      else {
	if(VECTEUR_UNDEFINED_P(pv2))
	  cmp = 1; // v1 is longer
	else {
	  if(term_cst(pv1)) {
	    if(term_cst(pv2)) {
	      // Use only constant terms as differentiator
	      // cmp = value_comparison(vecteur_val(pv1), vecteur_val(pv2));
	      // To get lower bounds before upper bounds when only one variables is constrained
	      cmp = vect_lexicographic_coefficient_comparison(v1, v2);
	    }
	    else
	      cmp = -1;
	  }
	  else {
	    if(term_cst(pv2))
	      cmp = 1;
	    else
	      /* This point should not be reachable unless the two vectors
		 are identical and end with two constant terms */
	      cmp = 0;
	  }
	}
      }
    }

    /* We need a total order to avoid non-determinism, i.e. dependence
       on pointer values */
    assert(cmp!=0 || vect_equal(v1, v2));

    return cmp;
}


/*
 * that is all
 */
