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

 /* package vecteur - routines sur les bases
  *
  * Francois Irigoin
  *
  * The function variable_name should be inlined as much as possible to
  * improve performances. It has to be used to be generic over the
  * "Variable" type. For instance, variables represented by a character
  * string cannot be decided equal by a simple pointer comparison.
  *
  * Modifications:
  */

/*LINTLIBRARY*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

/* Pbase vect_add_variable(Pbase b, Variable v): add variable v as a new
 * dimension to basis b; if variable v is already in basis b, do nothing;
 * this is not clean but convenient to avoid a test;
 *
 * Note that basis b contains a pointer towards variable v and not a copy
 * of it. So some sharing is introduced.
 *
 * A routine to check variable equality, variable_equal(), is used.
 */
Pbase vect_add_variable(b, v)
Pbase b;
Variable v;
{
    Pbase b1 = b;

    for(; !VECTEUR_NUL_P(b1) && !variable_equal(vecteur_var(b1), v);
	b1 = b1->succ)
	;

    if(b1 == VECTEUR_NUL) {
	base_add_dimension(&b,v); 
    }

    return(b);
}

/* Pbase base_add_variable(Pbase b, Variable v): add variable v as a new
 * dimension to basis b at the end of the base list; if variable v is 
 * already in basis b, do nothing;
 * this is not clean but convenient to avoid a test;
 *
 * Note that basis b contains a pointer towards variable v and not a copy
 * of it. So some sharing is introduced.
 *
 * A routine to check variable equality, variable_equal(), is used.
 */
Pbase base_add_variable(b, var)
Pbase b;
Variable var;
{
    Pbase b1 = b;
    Pbase result = b;


    if (!VECTEUR_NUL_P(b1)) {
	for(; !VECTEUR_NUL_P(b1) && !variable_equal(vecteur_var(b1), var);
	    b1 = b1->succ);
	if (VECTEUR_NUL_P(b1)) {
	    for (b1 = b; !VECTEUR_NUL_P(b1->succ); b1=b1->succ);
	    b1->succ = vect_new(var, VALUE_ONE);
	}
    }
    else { result = vect_new(var, VALUE_ONE);
       }
    return(result);
}

Pbase make_base_from_vect(Pvecteur pv)
{
    Pbase b = (Pbase) NULL;
    for(;!VECTEUR_NUL_P(pv);pv=pv->succ)
	if (pv->var != TCST)
	    b = base_add_variable(b,pv->var);
    return(b);
}


/* Pbase base_remove_variable(b, v): remove basis vector relative to v
 * from b; abort if v is not in b;
 */
Pbase base_remove_variable(b, v)
Pbase b;
Variable v;
{
    assert(base_contains_variable_p(b, v));
    vect_erase_var(&b, v);
    return b;
}

/* bool base_contains_variable_p(Pbase b, Variable v): returns true if
 * variable v is one of b's elements;
 *
 * Based on variable_equal()
 */
bool base_contains_variable_p(b, v)
Pbase b;
Variable v;
{
    bool in_base;

    for(; !VECTEUR_NUL_P(b) && !variable_equal(vecteur_var(b), v); b = b->succ)
	;

    in_base = !VECTEUR_NUL_P(b);
    return(in_base);
}

/* Variable base_find_variable(Pbase b, Variable v): returns variable v if
 * variable v is one of b's elements (returns a pointer to the copy of v
 * that's pointed to by basis b); else returns VARIABLE_UNDEFINED
 *
 * Based on variable_equal()
 */
Variable base_find_variable(b, v)
Pbase b;
Variable v;
{
    for(; !VECTEUR_NUL_P(b) && !variable_equal(vecteur_var(b), v); b = b->succ)
	;

    return(VECTEUR_NUL_P(b)? VARIABLE_UNDEFINED : vecteur_var(b));
}

/* Variable base_find_variable_name(Pbase b, Variable v,
 *                                  char * (*variable_name)()):
 * returns the variable (i.e. coord) in b that has the same name as v;
 * else returns VARIABLE_UNDEFINED
 */
Variable base_find_variable_name(b, v, variable_name)
Pbase b;
Variable v;
char * (*variable_name)(Variable);
{
    char * nv;
    char * nb;
    bool equal;

    for(; !VECTEUR_NUL_P(b); b = b->succ) {
	nv = variable_name(v);
	nb = variable_name(vecteur_var(b));
	equal = !strcmp(nv, nb);
	if(equal) 
	    break;
    }

    return(VECTEUR_NUL_P(b)? VARIABLE_UNDEFINED : vecteur_var(b));
}

/* int base_find_variable_rank(Pbase b, Variable v,
 *                             char * (*variable_name)()): 
 * returns variable v's rank if it is in basis b, else -1
 */
int base_find_variable_rank(b, v, variable_name)
Pbase b;
Variable v;
char * (*variable_name)(Variable);
{
    char * nv;
    char * nb;
    bool equal;
    int rank;

    for(rank=1; !VECTEUR_NUL_P(b); b = b->succ, rank++) {
	nv = variable_name(v);
	nb = variable_name(vecteur_var(b));
	equal = !strcmp(nv, nb);
	if(equal) 
	    break;
    }

    return VECTEUR_NUL_P(b)? -1 : rank;
}

/* Pbase base_reversal(Pbase b_in): produces a basis b_out, having the
 * same basis vectors as b_in, but in reverse order. Basis b_in is not
 * touched.
 *
 * Example: b_in = { e1, e2, e3 } -> b_out = { e3, e2, e1}
 */
Pbase base_reversal(b_in)
Pbase b_in;
{
    Pbase b_out = VECTEUR_NUL;

    for( ; !VECTEUR_NUL_P(b_in); b_in = b_in->succ)
	vect_add_elem(&b_out, vecteur_var(b_in),  vecteur_val(b_in));

    return b_out;
}

/* Pvecteur vect_rename(Pvecteur v, Pbase b, char * (*variable_name)()):
 * modify vector v so that its coordinates are relative to basis b;
 * each basis vector is defined by a pointer of type Variable, but
 * different pointers can point to the same basis vector wrt variable_name;
 * these pointers are unified with respect to b and variable_name to let us
 * perform eq-type comparison in the library
 *
 * This function is identical to vect_translate, except that if a variable 
 * var of v does not appear in b, then the assert is not executed. 
 * 
 * Bugs:
 *  - TCST and VARIABLE_UNDEFINED are equal; a dirty test is performed
 *    to screen TCST terms; on top of that, the special variable TCST is not
 *    kept in bases!
 */
Pvecteur vect_rename(v, b, variable_name)
Pvecteur v;
Pbase b;
char * (*variable_name)(Variable);
{
    Pvecteur coord;

    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Variable var;

	if(VARIABLE_DEFINED_P(vecteur_var(coord))) {
	    var = base_find_variable_name(b, vecteur_var(coord),
					  variable_name);
	    if (!VARIABLE_UNDEFINED_P(var))
		vecteur_var(coord) = var;
	}
    }
    return v;
}

/* Pvecteur vect_rename_variables(v, renamed_p, new_variable)
 * Pvecteur v;
 * bool (*renamed_p)(Variable);
 * Variable (*new_variable)(Variable);
 *
 * what: driven renaming of variables in v.
 * how: scans the vector, decides and replaces.
 * input: Pvecteur v, decision and replacement functions.
 * output: v is returned (the same)
 * side effects:
 *  - the vector is modified in place.
 * bugs or features:
 *  - was written by FC...
 */
Pvecteur vect_rename_variables(
    Pvecteur v,
    bool (*renamed_p)(Variable),
    Variable (*new_variable)(Variable))
{
    Pvecteur i=v; /* initial vector is kept */
    Variable var;

    for(; v!=NULL; v=v->succ)
    {
	var = var_of(v);
	if (renamed_p(var)) var_of(v)=new_variable(var);
    }

    return(i);
}

/* Pvecteur vect_translate(Pvecteur v, Pbase b, char * (*variable_name)()):
 * modify vector v so that its coordinates are relative to basis b;
 * each basis vector is defined by a pointer of type Variable, but
 * different pointers can point to the same basis vector wrt variable_name;
 * these pointers are unified with respect to b and variable_name to let us
 * perform eq-type comparison in the library
 *
 * Bugs:
 *  - TCST and VARIABLE_UNDEFINED are equal; a dirty test is performed
 *    to screen TCST terms; on top of that, the special variable TCST is not
 *    kept in bases!
 */
Pvecteur vect_translate(v, b, variable_name)
Pvecteur v;
Pbase b;
char * (*variable_name)(Variable);
{
    Pvecteur coord;

    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Variable var;

	if(VARIABLE_DEFINED_P(vecteur_var(coord))) {
	    var = base_find_variable_name(b, vecteur_var(coord),
					  variable_name);
	    assert(!VARIABLE_UNDEFINED_P(var));
	    vecteur_var(coord) = var;
	}
    }
    return v;
}

/* Pvecteur vect_in_basis_p(Pvecteur v, Pbase b):
 * check that all coordinates in v are in b, i.e. vector v is a membre
 * of the space generated by b
 *
 * Bugs:
 *  - TCST and VARIABLE_UNDEFINED are equal; a dirty test is performed
 *    to screen TCST terms; on top of that, the special variable TCST is not
 *    kept in bases!
 */
bool vect_in_basis_p(v, b)
Pvecteur v;
Pbase b;
{
    Pvecteur coord;

    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {

	if(VARIABLE_DEFINED_P(vecteur_var(coord))) {
	    if(!base_contains_variable_p(b, vecteur_var(coord))) {
		return(false);
	    }
	}
	else {
	    /* I do not know what should be done for constant terms... */
	    abort();
	}
    }
    return true;
}

/* Pvecteur vect_variable_rename(Pvecteur v, Variable v_old, Variable v_new):
 * rename the potential coordinate v_old in v as v_new
 */
Pvecteur vect_variable_rename(v, v_old, v_new)
Pvecteur v;
Variable v_old;
Variable v_new;
{
    Pvecteur coord;

    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Variable var = vecteur_var(coord);

	if(var==v_old)
	    vecteur_var(coord) = v_new;
    }
    return v;
}

/* appends b2 to b1. modifies b1. b2 is not modified.
 */
void base_append(Pbase * pb1, Pbase b2)
{
  if (BASE_NULLE_P(*pb1))
    *pb1 = base_copy(b2);
  else
  {
    Pvecteur v;
    linear_hashtable_pt seen = linear_hashtable_make();
    for (v=*pb1; v; v=v->succ)
    {
      Variable var = var_of(v);
      if (var!=TCST) linear_hashtable_put_once(seen, var, var);
    }

    for (v=b2; v; v=v->succ)
    {
      Variable var = var_of(v);
      if (var!=TCST && !linear_hashtable_isin(seen, var))
      {
	linear_hashtable_put_once(seen, var, var);
	*pb1 = vect_chain(*pb1, var, VALUE_ONE);
      }
    }

    linear_hashtable_free(seen);
  }
}

/* Pbase base_union(Pbase b1, Pbase b2): compute a new basis containing
 * all elements of b1 and all elements of b2, in an unkown order
 *
 * b := b1 u b2;
 * return b;
 *
 * Bases b1 and b2 are not modified.
 * Basis vectors are compared for equality using variable_equal()
 *
 * Modifications:
 *  - Pbase b = (Pbase)vect_add((Pvecteur) b1, (Pvecteur) b2);
 *    This is the definition of b at the beginning. This ignored 
 *    one case that when addition of two values is zero, vect_add will call
 *    vect_add_elem, where there is vect_erase_var. We'll miss the variable
 *    Lei Zhou.   15/07/91
 */
Pbase base_union(Pbase b1, Pbase b2)
{
  Pbase b = BASE_NULLE;
  bool
    bn1 = BASE_NULLE_P(b1),
    bn2 = BASE_NULLE_P(b2);

  if (!bn1 && bn2)
    b = base_copy(b1);
  else if (bn1 && !bn2)
    b = base_copy(b2);
  else if (!bn1 && !bn2)
  {
    linear_hashtable_pt seen = linear_hashtable_make();
    Pvecteur v;
    Variable var;

    for (v = b1; v; v=v->succ)
    {
      var = var_of(v);
      if (var!=TCST)
      {
	linear_hashtable_put_once(seen, var, var);
	b = vect_chain(b, var, VALUE_ONE);
      }
    }

    for (v = b2; v; v=v->succ)
    {
      var = var_of(v);
      if (var!=TCST)
	if (!linear_hashtable_isin(seen, var))
	{
	  linear_hashtable_put_once(seen, var, var);
	  b = vect_chain(b, var, VALUE_ONE);
	}
    }

    linear_hashtable_free(seen);
  }

  return b;
}

/* Return variables/dimensions present in bases b1 and b2. Order is not preserved. */
Pbase base_intersection(Pbase b1, Pbase b2)
{
  Pbase b = BASE_NULLE;
  bool
    bn1 = BASE_NULLE_P(b1),
    bn2 = BASE_NULLE_P(b2);

  if(!bn1 && !bn2) {
    Pbase bc;
    for(bc=b1; !BASE_UNDEFINED_P(bc); bc = vecteur_succ(bc)) {
      Variable var = vecteur_var(bc);
      if(base_contains_variable_p(b2, var)) {
	b = base_add_variable(b, var);
      }
    }
  }

  return b;
}

/* this function returns the rank of the variable var in the base 
 * 0 encodes TCST, but I do not know why, TCST may be in base, sometimes
 * -1 encodes an error
 */
int rank_of_variable(base,var)
Pbase base;
Variable var;
{
    int rank=1;
    register Pvecteur pv;

    if (var!=TCST)
    {
	for(pv=base;
	    !VECTEUR_NUL_P(pv) && !(vecteur_var(pv) ==var); 
	    pv=pv->succ, rank++);
	if (VECTEUR_NUL_P(pv)) rank = -1; /* not found */
    } 
    else 
	rank = 0;

    return(rank);
}

/* Variable variable_of_rank():
 * this function returns the variable of rank "rank" 
 */
Variable variable_of_rank(base,rank)
Pbase base;
int rank;
{
    int i;
    register Pvecteur pv;

    if (rank ==0) return(TCST);
    else {
	for(pv=base, i=1;
	    !VECTEUR_NUL_P(pv) && i != rank; pv=pv->succ, i++);
	if (!VECTEUR_NUL_P(pv))
	    return(vecteur_var(pv)); 
	else return (TCST);
    }
}

/* int search_higher_rank(): 
 * this fonction returns the rank of the variable of higher rank in the 
 * vecteur
 */
int search_higher_rank(vect,base)
Pvecteur vect;
Pbase base;
{
    int rank_pv = 0;
    int rv=0;
    register Pvecteur pv;
 
    for (pv=vect;!VECTEUR_NUL_P(pv);pv=pv->succ){
	if ((rv =rank_of_variable(base,vecteur_var(pv))) > rank_pv) 
	    rank_pv =  rv;
    }
    return(rank_pv); 
}


/* this function returns the variable of higher rank, after the variable var,
 * in the vecteur pvect
 * 
 */
Variable search_var_of_higher_rank(pvect,base,var)
Pvecteur pvect;
Pbase base;
Variable var;
{
    int rv,rank_pv = 0;
    Variable higher_var=TCST;
    register Pvecteur pv;

    for (pv=pvect;!VECTEUR_NUL_P(pv);pv=pv->succ) 
	if ((vecteur_var(pv) != var) 
	    && 	((rv =rank_of_variable(base,vecteur_var(pv))) > rank_pv)) {
	    rank_pv =  rv;
	    higher_var = vecteur_var(pv);
	}
    
    return(higher_var); 
}

/* Pvecteur search_i_element():
 * recherche du i-ieme couple (var,val) dans la Pbase b
 */
Pvecteur search_i_element(b,i)
Pbase b;
int i;
{
    Pbase b1;
    int j;

    for (b1=b, j=1; j<i; b1=b1->succ,j++);
    return(b1);
}

Pbase base_normalize(b)
Pbase b;
{
    Pbase eb;

    for (eb = b ; !BASE_NULLE_P(eb) ; eb=eb->succ)
	vecteur_val(eb) = VALUE_ONE;
    return b;
}

bool base_normalized_p(b)
Pbase b;
{
    Pbase eb;

    for (eb = b ;
	 !BASE_NULLE_P(eb) && value_one_p(vecteur_val(eb));
	 eb=eb->succ)
	;
    return BASE_NULLE_P(eb) && vect_check((Pvecteur) b);
}

/* Pbase base_difference(Pbase b1, Pbase b2):
 * allocate b;
 * b = b1 - b2  -- with the set meaning
 * return b;
 */
Pbase base_difference(Pbase b1, Pbase b2)
{
  Pbase b = BASE_NULLE; 
  Pbase eb = BASE_UNDEFINED;
  
  for(eb = b1; !BASE_NULLE_P(eb); eb = eb->succ) {
    Variable v = vecteur_var(eb);
    
    if(!base_contains_variable_p(b2, v))
      b = vect_add_variable(b, v);
  }

  return b;
}

/* Pbase base_included_p(Pbase b1, Pbase b2):
 * include_p = b1 is included in b2  -- with the set meaning
 * return b;
 */
bool base_included_p(Pbase b1, Pbase b2)
{
  Pbase b;
  bool included_p = true;
  linear_hashtable_pt seen = linear_hashtable_make();
  
  for (b=b2; b; b=b->succ)
    if (var_of(b)!=TCST)
      linear_hashtable_put_once(seen, var_of(b), var_of(b));

  for (b=b1; b && included_p; b=b->succ)
    if (var_of(b)!=TCST && !linear_hashtable_isin(seen, var_of(b)))
      included_p = false;

  linear_hashtable_free(seen);
  
  return included_p;
}

/* Make sure that each dimension of b1 is the same dimension in b2 */
bool bases_strictly_equal_p(Pbase b1, Pbase b2)
{
  int s1 = base_dimension(b1);
  int s2 = base_dimension(b2);
  bool strictly_equal_p = true;

  if(s1==s2) {
    int i;
    for(i=1; i<= s1 && strictly_equal_p; i++) {
      Variable d1 = variable_of_rank(b1, i);
      Variable d2 = variable_of_rank(b2, i);
      strictly_equal_p = (d1==d2);
    }
  }
  else
    strictly_equal_p = false;

  return strictly_equal_p;
}
