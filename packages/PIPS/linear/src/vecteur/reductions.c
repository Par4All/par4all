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

 /* package vecteur - reductions */

/*LINTLIBRARY*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "boolean.h"
#include "assert.h"
#include "arithmetique.h"
#include "vecteur.h"

/* int vect_size(Pvecteur v): calcul du nombre de composantes non nulles
 * d'un vecteur
 * 
 * sum abs(sgn(v[i]))
 *  i
 */
int vect_size(v)
Pvecteur v;
{
    int nb_elem = 0;

    for (; v != NULL; v=v->succ)
	nb_elem++;

    return (nb_elem);
}

/* int vect_dimension(Pvecteur v): calcul du nombre de composantes non nulles
 * et non constantes d'un vecteur
 * 
 * sum abs(sgn(v[i]))
 *  i
 */
int vect_dimension(v)
Pvecteur v;
{
    Pvecteur el;
    int nb_elem = 0;

    for (el=v; el != NULL; el=el->succ)
	if(!term_cst(el)) 
	    nb_elem++;

    return (nb_elem);
}

/* Value vect_prod_scal(v1,v2): produit scalaire de v1 et de v2
 * 
 * sum v1[i] * v2[i]
 *  i
 */
Value vect_prod_scal(v1,v2)
Pvecteur v1, v2;
{
    Value result = VALUE_ZERO;

    if(v2==NULL) return result;

    for(; v1!=NULL; v1 = v1->succ)
    {
	Value tmp = vect_coeff(var_of(v1),v2);
	value_product(tmp, val_of(v1));
	value_addto(result, tmp);
    }

    return result;
}

/* Value vect_pgcd(Pvecteur v):
 * calcul du pgcd de tous les coefficients non nul d'un vecteur v
 *
 * return   PGCD  v[i]
 *           i
 *        v[i]!=0
 *
 * Renvoie 1 pour le vecteur nul (ca devrait etre +infinity)
 */
Value vect_pgcd_all(v)
Pvecteur v;
{
    Value d = (v!=NULL ? value_abs(val_of(v)) : VALUE_ONE);

    if (v!=NULL) {
	for (v=v->succ; v!=NULL && value_notone_p(d); v=v->succ)
	    d = pgcd(d, value_abs(val_of(v)));
    }
    return d;
}

/* Value vect_pgcd_except(Pvecteur v, Variable var):
 * calcul du pgcd de tous les coefficients non nul d'un vecteur v,
 * sauf le coefficient correspondant a la variable var
 *
 * return   PGCD  v[i]
 *         i!=var
 *        v[i]!=0
 *
 * Renvoie 1 pour le vecteur nul (ca devrait etre +infinity)
 */
Value vect_pgcd_except(v,var)
Pvecteur v;
Variable var;
{
    Value d;

    /* skip var's coeff if it comes first */
    if ((v!= NULL) && (var_of(v) == var))
	v = v->succ;


    if(v==NULL)
	d = VALUE_ONE;
    else {
	d = value_abs(val_of(v));
	for (v=v->succ; v!=NULL && value_notone_p(d); v=v->succ)
	    if (var_of(v) != var)
		d = pgcd(d,value_abs(val_of(v)));
    }

    return d;
}

/* Value vect_max0(Pvecteur v): recherche du coefficient maximum
 * d'un vecteur v; ce coefficient est toujours au moins egal a 0
 * car on ne dispose pas d'une base pour verifier que TOUS les
 * coefficients sont negatifs
 *
 * max(0,max v[i])
 *        i
 * 
 * Note: on evite le probleme du vecteur de dimension 0 dont le max
 * vaut moins l'infini
 */
Value vect_max0(v)
Pvecteur v;
{
    Value max = VALUE_ZERO;

    for(; v!= NULL; v= v->succ)
	max = value_max(val_of(v),max);

    return max;
}

/* Value vect_min0(Pvecteur v): recherche du coefficient minimum
 * d'un vecteur v; ce coefficient est toujours au moins egal a 0
 * car on ne dispose pas d'une base pour verifier que TOUS les
 * coefficients sont negatifs
 *
 * min(0,min v[i])
 *        i
 * 
 * Note: on evite le probleme du vecteur de dimension 0 dont le min
 * vaut moins l'infini
 */
Value vect_min0(v)
Pvecteur v;
{
    Value min = VALUE_ZERO;

    for(; v!= NULL; v= v->succ)
	min = value_min(val_of(v),min);

    return min;
}

/* Value vect_min(Pvecteur v): recherche du coefficient non nul minimum
 * d'un vecteur v; aborte sur le vecteur 0 puisqu'il faudrait renvoyer
 * plus l'infini.
 *
 *   min   v[i]
 *    i
 * v[i]!=0
 *
 * Note: changement de semantique puisque 0 etait renvoye auparavant
 * pour le vecteur 0
 */
Value vect_min(v)
Pvecteur v;
{
    if(v!=NULL) {
	Value min = val_of(v);
	for (v=v->succ; v!= NULL; v= v->succ)
	     min = value_min(val_of(v),min);

	return min;
    }
    else {
	vect_error("vect_min","ill. null vector as argument\n");
	return VALUE_NAN; /* just to avoid a gcc warning */
    }
}

/* Value vect_max(Pvecteur v): recherche du coefficient non nul maximum
 * d'un vecteur v; aborte sur le vecteur 0 puisqu'il faudrait renvoyer
 * plus l'infini.
 *
 *   max   v[i]
 *    i
 * v[i]!=0
 *
 * Note: changement de semantique puisque 0 etait renvoye auparavant
 * pour le vecteur 0
 *
 * Modifications:
 *  -  max = (val_of(v) < max) ? val_of(v) : max; I changed to:
 *  -  max = (val_of(v) > max) ? val_of(v) : max;
 *  L.Zhou    Apr. 4, 91
 */
Value vect_max(v)
Pvecteur v;
{
    if(v!=NULL) {
	Value max = val_of(v);
	for (v=v->succ; v!= NULL; v= v->succ)
	     max = value_max(val_of(v), max);
	return max;
    }
    else {
	vect_error("vect_max","ill. null vector as argument\n");
	return VALUE_NAN;
    }
}

/* Value vect_sum(Pvecteur v): somme des coefficients d'un vecteur
 * (i.e. produit scalaire avec le vecteur 1)
 *
 *   sum   v[i]
 *    i
 */
Value vect_sum(Pvecteur v)
{
    Value sum = VALUE_ZERO;

    for (; v!=NULL; v=v->succ)
	value_addto(sum, val_of(v));

    return sum;
}

/* bool vect_equal(Pvecteur v1, Pvecteur v2): test a egalite de
 * deux vecteurs
 * 
 *        ->    ->
 * return v1 == v2 ;
 *
 */
bool vect_equal(v1,v2)
Pvecteur v1,v2;
{
    /* Note: le test n'est pas optimal puisque v2 est parcouru et compare
     * a v1 meme si ces coefficients ont ete deja ete compare lors du
     * parcours de v1; mais cela evite le "marquage" des coefficients vus;
     *
     * shorter version, FC 28/09/94
     */
    Pvecteur v;
    register bool 
	result = true;

    if (!v1 || !v2)
	return(!v1 && !v2);

    /*   v1 must be preserved for the second loop: use v 
     */
    for (v=v1; 
	 v && result;
	 v=v->succ)
	result = value_eq(val_of(v),vect_coeff(var_of(v),v2));
    
    /*   now v2 may be lost: use v2
     */
    for (;
	 v2 && result;
	 v2=v2->succ) 
	result = value_eq(val_of(v2),vect_coeff(var_of(v2),v1));
    
    return result;
}

/* bool vect_equal_except(Pvecteur v1, Pvecteur v2, Variable var):
 * test a egalite des projections selon la coordonnees var de deux vecteurs
 *      ->
 * Soit e un vecteur de base quelconque:
 *         ->   ->     ->   ->
 * return <v1 . e> == <v2 . e>;
 *                e!=var
 */
bool vect_equal_except(v1,v2,var)
Pvecteur v1,v2;
Variable var;
{
    Pvecteur pv;
    /*
     * Note: le test n'est pas optimal puisque v2 est parcouru et compare
     * a v1 meme si ces coefficients ont ete deja ete compare lors du
     * parcours de v1; mais cela evite le "marquage" des coefficients vus;
     */
    bool result;

    if(v1==NULL && v2==NULL)
	result = true;
    else if(v1==NULL)
	result = v2->succ==NULL && var_of(v2)==var;
    else if(v2 == NULL)
	result = v1->succ==NULL && var_of(v1)==var;
    else {
	result = true;

	for (pv = v1; pv != NULL && result == true; pv = pv->succ)
	    if (var_of(pv) != var)
		result = value_eq(val_of(pv),vect_coeff(var_of(pv), v2));

	for (pv = v2; pv != NULL && result == true; pv = pv->succ)
	    if (var_of(pv) != var)
		result = value_eq(val_of(pv),vect_coeff(var_of(pv), v1));

    }

    return result;
}

/* bool vect_oppos(Pvecteur v1, Pvecteur v2): test de l'opposition de
 * deux vecteurs
 * 
 *        ->   ->    ->
 * return v1 + v2 == 0 ;
 *
 */
bool vect_oppos(v1,v2)
Pvecteur v1,v2;
{
    /*
     * Note: le test n'est pas optimal puisque v2 est parcouru et compare
     * a v1 meme si ces coefficients ont ete deja ete compare lors du
     * parcours de v1; mais cela evite le "marquage" des coefficients vus;
     */
    bool result;
    Pvecteur pv;

    if(v1==NULL && v2==NULL)
	result = true;
    else if(v1==NULL || v2 == NULL)
	result = false;
    else {
	result = true;

	for (pv = v1; pv != NULL && result == true; pv = pv->succ)
	    result = value_eq(val_of(pv),
			      value_uminus(vect_coeff(var_of(pv), v2)));

	for (pv = v2; pv != NULL && result == true; pv = pv->succ)
	    result = value_eq(val_of(pv),
			      value_uminus(vect_coeff(var_of(pv), v1)));

    }

    return result;
}

/* bool vect_opposite_except(Pvecteur v1, Pvecteur v2, Variable var):
 * test a egalite des projections selon la coordonnees var de deux vecteurs
 *      ->
 * Soit e un vecteur de base quelconque:
 *         ->   ->     ->   ->
 * return <v1 . e> == - <v2 . e>;
 *                e!=var
 */
bool vect_opposite_except(v1,v2,var)
Pvecteur v1,v2;
Variable var;
{
    Pvecteur pv;
    /*
     * Note: le test n'est pas optimal puisque v2 est parcouru et compare
     * a v1 meme si ces coefficients ont ete deja ete compare lors du
     * parcours de v1; mais cela evite le "marquage" des coefficients vus;
     */
    bool result;

    if(v1==NULL && v2==NULL)
	result = true;
    else if(v1==NULL)
	result = v2->succ==NULL && var_of(v2)==var;
    else if(v2 == NULL)
	result = v1->succ==NULL && var_of(v1)==var;
    else {
	result = true;

	for (pv = v1; pv != NULL && result == true; pv = pv->succ)
	    if (var_of(pv) != var)
		result = value_eq(val_of(pv),
				  value_uminus(vect_coeff(var_of(pv), v2)));

	for (pv = v2; pv != NULL && result == true; pv = pv->succ)
	    if (var_of(pv) != var)
		result = value_eq(val_of(pv),
				  value_uminus(vect_coeff(var_of(pv), v1)));

    }

    return result;
}

/* int vect_proport(Pvecteur v1, Pvecteur v2): test de la colinearite
 * de deux vecteurs et de leur direction.
 * 
 * return
 *    1    si les deux vecteurs sont colineaires et dans la
 *         meme direction c2 v1 == c1 v1, c1*c2 > 0
 *         c'est le cas si v1 ou v2 vaut le vecteur nul
 *   -1    si les deux vecteurs sont colineaires et dans des 
 *         directions opposees c2 v1 == c1 v1, c1*c2 < 0
 *    0    s'ils ne sont pas colineaires
 *
 *                  ->    ->    ->
 * Note: aborte pour v1 == v2 == 0  parce qu'il est impossible de decider entre
 * le retour de 1 et de -1
 *
 * Modifications:
 *  - introduction des variables temporaires t1 et t2 pour ne pas effectuer
 *    le test de proprotionalite uniquement sur la fin des vecteurs v1 et v2;
 *    Francois Irigoin, 26 mars 1991
 */
int vect_proport(Pvecteur v1, Pvecteur v2)
{
    int prop = 1;

    if (v1==NULL && v2==NULL)
	vect_error("vect_proport","ill. NULL v1 and v2 args\n");
    
    if (v1!=NULL && v2!=NULL) {
	Value c1, c2;
	Pvecteur t1;
	Pvecteur t2;

	c1 = val_of(v1);
	c2 = vect_coeff(var_of(v1),v2);
	prop = 1;

	for (t1 = v1->succ; (t1!=NULL) && (prop); t1=t1->succ)
	{
	    Value tmp1 = vect_coeff(var_of(t1),v2), tmp2;

	    value_product(tmp1,c1);
	    tmp2 = value_mult(c2,val_of(t1));
	    prop = value_eq(tmp1,tmp2);
	}

	for (t2 = v2; (t2!=NULL) && (prop != 0);t2=t2->succ) 
	{
	    Value tmp1 = vect_coeff(var_of(t2),v1), tmp2;

	    value_product(tmp1,c2);
	    tmp2 = value_mult(c1,val_of(t2));
	    prop = value_eq(tmp1,tmp2);
	}

	if(prop!=0) {
	    if (value_pos_p(value_mult(c1,c2)))
		prop = 1;
	    else
		prop = -1;
	}
    }

    return prop;
}

/* bool vect_colin_base(Pvecteur vec, Variable var): renvoie true si
 * -->     -->
 * vec = k var
 *
 * false sinon
 *
 * Attention: le vecteur nul est colineaire a tous les vecteurs de base
 */
bool vect_colin_base(vec,var)
Pvecteur vec;
Variable var;
{
    return(vec==NULL || (vec->succ==NULL && var_of(vec)==var));
}

/* bool vect_check(Pvecteur v): renvoie true si le vecteur v est
 * coherent avec les specifications du package; aucun des coefficients
 * effectivement conserves en memoire ne doit etre nul (la cellule aurait
 * du etre liberee) et aucune dimension (i.e. variable) ne peut apparaitre
 * deux fois.
 *
 * Ces conditions ne sont pas verifiees par Corinne dans ses routines
 * du package "sommet".
 *
 * new version to test linear_hashtable. better for large vectors,
 * but much worse for small ones I guess. FC.
 *
 * Especially for the NULL vector. FI.
 */
bool vect_check(Pvecteur cv)
{
  Pvecteur v = cv;
  register bool
    consistent = true,
    tcst_seen = false;
  linear_hashtable_pt seen = linear_hashtable_make();

  for(; v!=NULL && consistent; v=v->succ)
  {
    consistent = value_notzero_p(val_of(v));
    if (var_of(v))
    {
      if (linear_hashtable_isin(seen, var_of(v)))
	consistent = false;
      linear_hashtable_put(seen, var_of(v), (void*) 1);
    }
    else {
      if (tcst_seen) consistent = false;
      tcst_seen = true;
    }
  }

  linear_hashtable_free(seen);
  return consistent;
}

/* To ease retrieval of vect_check() */
bool vect_consistent_p(Pvecteur v) { return vect_check(v);}


/* @return whether one coef in v is greater than abs(val), but CST
 * @param v vecteur being scanned
 * @param val maximum absolute value allowed, or 0 to ignore
 */
bool vect_larger_coef_p(Pvecteur v, Value val)
{
  linear_assert("positive value", value_posz_p(val));
  if (value_zero_p(val)) return false;
  for (; v!=NULL; v=v->succ)
    if (var_of(v) &&
        (value_lt(val_of(v), value_uminus(val)) || value_gt(val_of(v), val)))
      return true;
  return false;
}
