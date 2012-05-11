/*

  $Id: sc_enumerate.c 1526 2012-04-26 08:14:32Z guelton $

  Copyright 1989-2011 MINES ParisTech

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

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include "linear.h"
/*#include "assert.h"*/
/*#include "boolean.h"*/
/*#include "arithmetique.h"*/
/*#include "vecteur.h"*/
/*#include "contrainte.h"*/
/*#include "sc.h"*/
/*#include "sommet.h"*/
/*#include "ray_dte.h"*/
/*#include "sg.h"*/
/*#include "polyedre.h"*/
/*#include "polynome.h"*/

/* IRISA/POLYLIB data structures.
 */
#include "polylib/polylib.h"

/* maximum number of rays allowed in chernikova... (was 20000)
 * it does not look a good idea to move the limit up, as it
 * makes both time and memory consumption to grow a lot.
 */
#define MAX_NB_RAYS (20000)

/* Irisa is based on int. We would like to change this to
 * some other type, say "long long" if desired, as VALUE may
 * also be changed. It is currently an int. Let us assume
 * that the future type will be be called "IRINT" (Irisa Int)
 */
/*
#define VALUE_TO_IRINT(val) VALUE_TO_INT(val)
#define IRINT_TO_VALUE(i) ((Value)i)
*/

#define VALUE_TO_IRINT(val) (val)
#define IRINT_TO_VALUE(i) (i)


static void my_Matrix_Free(Matrix * volatile * m)
{
  ifscdebug(9)
    fprintf(stderr, "[my_Matrix_Free] in %p\n", *m);

  Matrix_Free(*m);
  *m = NULL;

  ifscdebug(9)
    fprintf(stderr, "[my_Matrix_Free] out %p\n", *m);
}

static void my_Polyhedron_Free(Polyhedron * volatile * p)
{
  ifscdebug(9)
    fprintf(stderr, "[my_Polyhedron_Free] in %p\n", *p);

  Polyhedron_Free(*p);
  *p = NULL;

  ifscdebug(9)
    fprintf(stderr, "[my_Polyhedron_Free] out %p\n", *p);
}


/* Fonctions de conversion traduisant une Pcontrainte en une
 * ligne de la structure matrix de l'IRISA
 */
static void contrainte_to_matrix_ligne(
  Pcontrainte pc,
  Matrix *mat,
  int i,
  Pbase base)
{
  Pvecteur pv;
  int j;
  Value v;

  for (pv=base,j=1; !VECTEUR_NUL_P(pv); pv=pv->succ,j++)
  {
    v = value_uminus(vect_coeff(vecteur_var(pv),pc->vecteur));
    mat->p[i][j] = VALUE_TO_IRINT(v);
  }

  v = value_uminus(vect_coeff(TCST,pc->vecteur));
  mat->p[i][j]= VALUE_TO_IRINT(v);
}

/* Passage du systeme lineaire sc a une matrice matrix (structure Irisa)
 * Cette fonction de conversion est utilisee par la fonction
 * sc_to_sg_chernikova
 */
static void sc_to_matrix(Psysteme sc, Matrix *mat)
{
  int nbrows, i, j;
  Pcontrainte peq;
  Pvecteur pv;

  nbrows = mat->NbRows;

  // Differentiation equations and inequations
  for (i=0; i<=sc->nb_eq-1;i++)
    mat->p[i][0] =0;
  for (; i<=nbrows-1;i++)
    mat->p[i][0] =1;

  // Matrix initialisation
  for (peq = sc->egalites,i=0;
       !CONTRAINTE_UNDEFINED_P(peq);
       peq=peq->succ, i++)
    contrainte_to_matrix_ligne(peq,mat,i,sc->base);

  for (peq =sc->inegalites;
       !CONTRAINTE_UNDEFINED_P(peq);
       peq=peq->succ, i++)
    contrainte_to_matrix_ligne(peq,mat,i,sc->base);

  for (pv=sc->base,j=1; !VECTEUR_NUL_P(pv); pv=pv->succ, j++)
    mat->p[i][j] = 0;
  mat->p[i][j]=1;
}


#if 0
/* Evaluate  that the point defined by the value array N meets the
   sparse equality defined by v and b. See also contrainte_eval(). */
static Value eval_constraint_with_vertex(
  Pvecteur v,
  Pbase b,
  int d,
  Value * N)
{
  Pvecteur cv = VECTEUR_UNDEFINED;
  Value k = VALUE_ZERO;

  // debugging information - any equivalent to ifscdebug() for polyedre?
  // ifscdebug seems to be used although not in sc
  static int francois_debug=0;
  if(francois_debug) {
    int i;
    fprintf(stderr, "Constraint:\n");
    vect_dump(v);
    fprintf(stderr, "Basis:\n");
    vect_dump(b);
    fprintf(stderr, "Vertex:\n");
    for(i=0;i<d;i++)
      fprintf(stderr, "N[%d]=%ld, ", i, (long int) N[i]);
    fprintf(stderr, "\n");
  }

  for(cv=v; !VECTEUR_UNDEFINED_P(cv); cv = vecteur_succ(cv)) {
    Variable var = vecteur_var(cv);
    Value coeff = vecteur_val(cv);
    if(var==TCST) {
      value_addto(k, coeff);
    }
    else {
      int rank = rank_of_variable(b, var);
      assert(0<rank && rank<d-1);
      Value val = value_pdiv(N[rank],N[d-1]);
      value_addto(k, value_direct_multiply(coeff, val));
    }
  }

  return k;
}

/* Check that the point defined by the value array N meets the
   sparse equality defined by v and b */
static bool vertex_meets_equality_p(Pvecteur v, Pbase b, int d, Value * N)
{
  Value k = eval_constraint_with_vertex(v, b, d, N);
  bool meets_p = value_zero_p(k);
  return meets_p;
}

/* Check that the point defined by the value array N meets the
   sparse inequality defined by v and b */
static bool vertex_meets_inequality_p(Pvecteur v, Pbase b, int d, Value * N)
{
  Value k = eval_constraint_with_vertex(v, b, d, N);
  bool meets_p = value_negz_p(k);
  return meets_p;
}

/* Check that the point defined by the value array N meets the
   sparse inequality defined by v and b */
static bool vertex_strictly_meets_inequality_p
(Pvecteur v, Pbase b, int d, Value * N)
{
  Value k = eval_constraint_with_vertex(v, b, d, N);
  bool meets_p = value_neg_p(k);
  return meets_p;
}
#endif

#if 0
/* Check if the Value vector N is inside the set of integer points defined by sc
 *
 * Used to avoid adding redundant vertices.
 */
static bool redundant_vertex_p(Psysteme sc, int d, Value * N, bool strict_p)
{
  bool redundant_p = true;
  Pbase b = sc_base(sc);
  Pcontrainte eq = CONTRAINTE_UNDEFINED;

  assert(base_dimension(b)==d-2);

  for(eq=sc_egalites(sc);
      !CONTRAINTE_UNDEFINED_P(eq) && redundant_p;
      eq = contrainte_succ(eq)) {
    Pvecteur v = contrainte_vecteur(eq);
    redundant_p = vertex_meets_equality_p(v, b, d, N);
  }

  for(eq=sc_inegalites(sc);
      !CONTRAINTE_UNDEFINED_P(eq) && redundant_p;
      eq = contrainte_succ(eq)) {
    Pvecteur v = contrainte_vecteur(eq);
    if(strict_p)
      redundant_p = vertex_strictly_meets_inequality_p(v, b, d, N);
    else
      redundant_p = vertex_meets_inequality_p(v, b, d, N);
  }

  return redundant_p;
}
#endif


static Variable base_nth(Pbase b,size_t i)
{
  while(i--) {
    b=vecteur_succ(b);
  }
  return vecteur_var(b);
}

static Ppolynome evalue_to_polynome(evalue *, Pbase );

static Ppolynome enode_to_polynome(enode *e, Pbase ordered_base)
{
  int i;
  if (!e)
    return POLYNOME_UNDEFINED;
  Ppolynome p = POLYNOME_NUL;
  switch(e->type) {
  case evector:
    for (i=0; i<e->size; i++) {
      Ppolynome pp = evalue_to_polynome(&e->arr[i],ordered_base);
      polynome_add(&p,pp);
    }
    return p;
  case polynomial:
    for (i=e->size-1; i>=0; i--) {
      Ppolynome pp = evalue_to_polynome(&e->arr[i],ordered_base);
      pp=polynome_mult(pp,make_polynome(1,base_nth(ordered_base,e->pos-1),i));
      polynome_add(&p,pp);
    }
    return p;
  default:
    fprintf(stderr,"cannot represent periodic polynomials in linear\n");
    return POLYNOME_UNDEFINED;
  }
}

static Ppolynome evalue_to_polynome(evalue *e, Pbase ordered_base)
{
  Ppolynome p = NULL;
  if(value_notzero_p(e->d))
    p=make_polynome((float)e->x.n/(float)e->d,TCST,0);
  else
    p = enode_to_polynome(e->x.p,ordered_base);
  return p;
}

/* enumerate the systeme sc using base pb
 * pb contains the unknow variables
 * and sc all the constraints
 * ordered_sc must be order as follows:
 * elements from ordered_base appear last
 * variable_names can be provided for debugging purpose (name of bases) or set to NULL
 */
Ppolynome sc_enumerate(Psysteme ordered_sc, Pbase ordered_base, const char* variable_names[])
{
  // Automatic variables read in a CATCH block need to be declared volatile as
  // specified by the documentation
  Polyhedron * volatile A = NULL;
  Matrix * volatile a = NULL;
  Polyhedron * volatile C = NULL;
  Enumeration * volatile ehrhart = NULL;

  int nbrows = 0;
  int nbcolumns = 0;
  if( sc_dimension(ordered_sc) == 0 )
    return make_polynome(1,TCST,1);
  else
  {
    CATCH(any_exception_error)
    {
      if (A) my_Polyhedron_Free(&A);
      if (a) my_Matrix_Free(&a);
      if (C) my_Polyhedron_Free(&C);
      RETHROW();
    }
    TRY
    {
      assert(!SC_UNDEFINED_P(ordered_sc) && (sc_dimension(ordered_sc) != 0));
      nbrows = ordered_sc->nb_eq + ordered_sc->nb_ineq+1;
      nbcolumns = ordered_sc->dimension +2;
      a = Matrix_Alloc(nbrows, nbcolumns);
      sc_to_matrix(ordered_sc,a);

      A = Constraints2Polyhedron(a, MAX_NB_RAYS);
      my_Matrix_Free(&a);

      C = Universe_Polyhedron(base_dimension(ordered_base));

      ehrhart = Polyhedron_Enumerate(A, C, MAX_NB_RAYS, variable_names);
      // Value vals[2]= {0,0};
      // Value *val = compute_poly(ehrhart,&vals[0]);
      // printf("%lld\n",*val);
      my_Polyhedron_Free(&A);
      my_Polyhedron_Free(&C);
    } // end TRY

    UNCATCH(any_exception_error);
    return evalue_to_polynome(&ehrhart->EP,ordered_base);
  }
}
