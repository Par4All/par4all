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

 /*
  * CREATION, COPIE ET DESTRUCTION D'UN VECTEUR
  */

/*LINTLIBRARY*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(p,t,f) free(p)

/* Pvecteur vect_dup(Pvecteur v_in): duplication du vecteur v_in; allocation de
 * et copie dans v_out;
 *
 * allocate v_out;
 * v_out := v_in;
 */
Pvecteur vect_dup(v_in)
Pvecteur v_in;
{
  Pvecteur v_out;
  Pvecteur v;

  v_out = NULL;
  for(v=v_in; v!=NULL; v=v->succ) {
    v_out = vect_chain(v_out,var_of(v),val_of(v));
  }

  return v_out;
}

/* void vect_rm(Pvecteur v): desallocation des couples de v;
 *
 * Attention! La procedure appelante doit penser a modifier la
 * valeur de v apres l'appel pour ne pas pointer dans le vide:
 *   vect_rm(v);
 *   v = NULL;
 *
 * Il vaudrait mieux que vect_rm retourne NULL et qu'on puisse
 * ecrire:
 *  v = vect_rm(v);
 * ou que vect_rm prenne un Pvecteur * en argument:
 *  vect_rm(&v);
 */
void vect_rm(v)
Pvecteur v;
{
    Pvecteur elem = NULL;
    Pvecteur nelem = NULL;

    for(elem = v; elem!=NULL; elem = nelem) {
	nelem = elem->succ;
	free((char *)elem);
    }
    /*
    while (v != NULL) {
	Pvecteur nv = v->succ;
	free((char *) v);
	v = nv;
    }
    */
}

/* Pvecteur vect_new(Variable var,Value coeff):
 * allocation d'un vecteur colineaire
 * au vecteur de base var et de coefficient coeff (i.e. creation
 * d'un nouveau vecteur ne comportant qu'un seul couple (var,coeff))
 *
 *       -->
 * coeff var
 *
 * Pourrait etre remplace par un vect_chain(NULL,,)
 *
 * Modifications:
 *  - a 0 coeff generates a null vector; Francois Irigoin, 26 March 1991
 */
Pvecteur vect_new(Variable var, Value coeff)
{
  Pvecteur v;

  if(coeff!=0) {
    v = (Pvecteur) MALLOC(sizeof(Svecteur),VECTEUR,"vect_new");
    if (v == NULL) {
	    (void) fprintf(stderr,"vect_new: Out of memory space\n");
	    /* fprintf(stderr, "%10.3f MB",
	       (sbrk(0) - etext)/(double)(1 << 20)); // not portable */
	    abort();
	    /*exit(-1);*/
	}
	var_of(v) = var;
	val_of(v) = coeff;
	v->succ = NULL;
  }
  else
    v = NULL;

  return v;
}

/* void dbg_vect_rm(Pvecteur v, char * f): desallocation d'un vecteur
 * avec marquage de la fonction provoquant la desallocation
 *
 * Apparemment obsolete.
 */
/*ARGSUSED*/
void dbg_vect_rm(Pvecteur v,
		 char __attribute__((unused)) *f)
{
    Pvecteur v1,v2;
    v1 = v;
    while (v1!=NULL) {
	v2 = v1->succ;
	FREE((char *)v1,VECTEUR,f);
	v1 = v2;
    }
}

/* Pvecteur vect_make(v, [var, val,]* 0, val)
 * Pvecteur v; // may be NULL, use assigne anyway
 * Variable var;
 * Value val;
 *
 * Builds a vector from the list of arguments, by successive additions.
 * ends when a 0 Variable (that is TCST!) is encountered.
 *
 * Because of the var val order, this function cannot be called directly
 * with a va_list, but (va_list, 0) should be used, since the val argument
 * is expected, read and used anyway.
 *
 * CAUTION: the initial vector is modified by the process!
 */
Pvecteur vect_make(Pvecteur v, Variable var, Value val, ...)
{
  va_list the_args;

  // handle fist argument
  vect_add_elem(&v, var, val);

  // get others
  va_start(the_args, val);

  while (var != (Variable) 0)
  {
    var = va_arg(the_args, Variable);
    val = va_arg(the_args, Value);
    vect_add_elem(&v, var, val);
  }

  va_end (the_args);
  return v;
}

/* Allocate a new vector v whose coefficient are given by the list of
   values ad whose dimension is given by b. The number of constant
   values passed as argument is supposed to be equal to the dimension
   of b.

   Note: 0 is a normal value. I see no way to mark the last argument.

   FI: I add this function to check under gdb that a given point
   belongs to a constraint system. The manual verification is tedious
   and error prone. This is done for debugging in PIPS the
   linked_regions bug.
 */
Pvecteur vect_make_dense(Pbase b, Value val, ...)
{
    va_list the_args;
    Pvecteur v = VECTEUR_NUL;
    Variable var;
    int dim = (int) base_dimension(b);
    int i = 0; // current dimension

    /* handle first argument - the first element of a basis has rank 1 */
    var = variable_of_rank(b,1);
    vect_add_elem(&v, var, val);

    /* get others */
    va_start (the_args, val);

    for(i=2;i<=dim;i++) // i <= dim because of the way dimensions are counted
    {
	var = variable_of_rank(b,i);
	val = va_arg(the_args, Value);
	vect_add_elem(&v, var, val);
    }

    va_end (the_args);

    return v;
}

/* Generate a sparse vector a x + b TCST */
Pvecteur vect_make_1D(Value a, Variable x, Value b)
{
  Pvecteur v = vect_new(x, a);
  vect_add_elem(&v, TCST, b);
  return v;
}

/* direct duplication.
 * vect_dup() and vect_reversal() do the same thing :
 * duplicate the vector with the reversal order.
 * vect_copy duplicate the vector with the same order.
 * in use of sc_copy. (DN,24/6/02)
 * Does not change parameter b (DN,28/06/02)
 */
Pbase vect_copy(Pvecteur b)
{
  Pvecteur n = VECTEUR_NUL, p = VECTEUR_NUL, r = VECTEUR_NUL, tmp = b;

  for (; tmp!=VECTEUR_NUL; tmp=tmp->succ)
  {
    n = (Pvecteur) MALLOC(sizeof(Svecteur),VECTEUR,"vect_copy");
    if (n == NULL) {
      fprintf(stderr,"[vect_copy] out of memory space\n");
      abort();
    }
    var_of(n) = var_of(tmp);
    val_of(n) = val_of(tmp);
    n->succ = NULL;
    if (r==VECTEUR_NUL) r = n;
    if (p!=VECTEUR_NUL) p->succ = n;
    p = n;
  }

  return r;
}

/* Pbase base_dup(Pbase b) 
 * Note: this function changes the value of the pointer.
 * Use base_copy instead. Should become a link, not a function.
 * For the moment, it's a function, because of the sc.h cannot be updated
 * without installation, due to decision of integration of Janus or not? DN 12/5/03
 */
Pbase base_dup(Pbase b)
{
  /* return base_copy(b);*/

  Pbase n = BASE_NULLE, p = BASE_NULLE, r = BASE_NULLE;

  for (; b!=BASE_NULLE; b=b->succ)
  {
    n = (Pvecteur) MALLOC(sizeof(Svecteur),VECTEUR,"base_dup");
    if (n == NULL) {
      fprintf(stderr,"[base_dup] out of memory space\n");
      abort();
    }
    var_of(n) = var_of(b);
    val_of(n) = VALUE_ONE;
    n->succ = NULL;
    if (r==BASE_NULLE) r = n;
    if (p!=BASE_NULLE) p->succ = n;
    p = n;
  }
  return r;
}



/* Direct duplication. The initial Pbase is assumed to be valid.
 * Absolutely the same with base_dup, but base_up is the only function
 * that maintains the old order.
 * So recopy here for use with copy version including
 * vect_copy, contrainte_copy, contraintes_copy, sc_copy (DN,24/6/02)
 * Does not change the parameter. Did have a look at all copy version (DN,1/7/2002)
 */
Pbase base_copy(Pbase b)
{
  Pbase n = BASE_NULLE, p = BASE_NULLE, r = BASE_NULLE, tmp = b;

  for (; tmp!=BASE_NULLE; tmp=tmp->succ)
  {
    n = (Pvecteur) MALLOC(sizeof(Svecteur),VECTEUR,"base_copy");
    if (n == NULL) {
      fprintf(stderr,"[base_copy] out of memory space\n");
      abort();
    }
    var_of(n) = var_of(tmp);
    val_of(n) = VALUE_ONE;
    n->succ = NULL;
    if (r==BASE_NULLE) r = n;
    if (p!=BASE_NULLE) p->succ = n;
    p = n;
  }

  return r;
}
   
