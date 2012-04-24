/*

  $Id$

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
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "polyedre.h"
#include "polynome.h"

#undef CHERNIKOVA_DEPRECATED

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

#ifdef CHERNIKOVA_DEPRECATED
/* set timeout with signal and alarm
 * based on environment variable LINEAR_CONVEX_HULL_TIMEOUT
 */
#include <signal.h>

// name of environment variable to trigger local timeout
#define TIMEOUT_ENV "LINEAR_CONVEX_HULL_TIMEOUT"

// whether exception is raised by alarm.
static bool sc_convex_hull_timeout = false;

static int get_linear_convex_hull_timeout(void)
{
  static int linear_convex_hull_timeout = -1;
  // initialize once from environment
  if (linear_convex_hull_timeout==-1)
  {
    char * env = getenv(TIMEOUT_ENV);
    if (env)
    {
      // fprintf(stderr, "setting convex hull timeout from: %s\n", env);
      linear_convex_hull_timeout = atoi(env);
    }
    // set to 0 in case of
    if (linear_convex_hull_timeout<0)
      linear_convex_hull_timeout = 0;
  }
  return linear_convex_hull_timeout;
}

static void catch_alarm_sc_convex_hull(int sig  __attribute__ ((unused)))
{
  fprintf(stderr, "CATCH alarm sc_convex_hull !!!\n");
  //put inside CATCH(any_exception_error) alarm(0); //clear the alarm
  sc_convex_hull_timeout = true;
  THROW(timeout_error);
}
#endif

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

#ifdef CHERNIKOVA_DEPRECATED
/* Fonctions de conversion traduisant une ligne de la structure
 * Matrix de l'IRISA en un Pvecteur
 */
static Pvecteur matrix_ligne_to_vecteur(
  Matrix *mat,
  int i,
  Pbase base,
  int dim)
{
  int j;
  Pvecteur pv,pvnew = NULL;
  bool NEWPV = true;

  for (j=1,pv=base ;j<dim;j++,pv=pv->succ) {
    if (mat->p[i][j]) {
      if (NEWPV) {
        pvnew= vect_new(vecteur_var(pv),
                        IRINT_TO_VALUE(mat->p[i][j]));
        NEWPV =false;
      }
      else
        vect_add_elem(&pvnew,vecteur_var(pv),
                      IRINT_TO_VALUE(mat->p[i][j]));
    }
  }
  return pvnew;
}

/* Fonctions de conversion traduisant une ligne de la structure
 * Matrix de l'IRISA en une Pcontrainte
 */
static Pcontrainte matrix_ligne_to_contrainte(
  Matrix * mat,
  int i,
  Pbase base)
{
  Pcontrainte pc=NULL;
  int dim = vect_size(base) +1;

  Pvecteur pvnew = matrix_ligne_to_vecteur(mat,i,base,dim);
  vect_add_elem(&pvnew,TCST,IRINT_TO_VALUE(mat->p[i][dim]));
  vect_chg_sgn(pvnew);
  pc = contrainte_make(pvnew);
  return pc;
}


/*  Fonctions de conversion traduisant une ligne de la structure
 *  Polyhedron de l'IRISA en un Pvecteur
 */
static Pvecteur polyhedron_ligne_to_vecteur(
  Polyhedron *pol,
  int i,
  Pbase base,
  int dim)
{
  int j;
  Pvecteur pv,pvnew = NULL;
  bool NEWPV = true;

  for (j=1,pv=base ;j<dim;j++,pv=pv->succ) {
    if (pol->Ray[i][j]) {
	    if (NEWPV) { pvnew= vect_new(vecteur_var(pv),
                                   IRINT_TO_VALUE(pol->Ray[i][j]));
        NEWPV =false;
      }
	    else
        vect_add_elem(&pvnew,vecteur_var(pv),
                      IRINT_TO_VALUE(pol->Ray[i][j]));
    }
  }
  return pvnew;
}

/* Fonctions de conversion traduisant une Pray_dte en une
 * ligne de la structure matrix de l'IRISA
 */
static void ray_to_matrix_ligne(
  Pray_dte pr,
  Matrix *mat,
  int i,
  Pbase base)
{
  Pvecteur pb;
  unsigned int j;
  Value v;

  for (pb = base, j=1;
       !VECTEUR_NUL_P(pb) && j<mat->NbColumns-1;
       pb = pb->succ,j++)
  {
    v = vect_coeff(vecteur_var(pb),pr->vecteur);
    mat->p[i][j] = VALUE_TO_IRINT(v);
  }
}
#endif


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

#ifdef CHERNIKOVA_DEPRECATED
/* Fonction de conversion traduisant un systeme generateur sg
 * en une matrice de droites, rayons et sommets utilise'e par la
 * structure  de Polyhedron de l'Irisa.
 * Cette fonction de conversion est utilisee par la fonction
 * sg_to_sc_chernikova
 */
static void sg_to_polyhedron(Ptsg sg, Matrix * mat)
{
  Pray_dte pr;
  Psommet ps;
  int  nbC=0;
  int nbcolumns = mat->NbColumns;

  // Traitement des droites
  if (sg_nbre_droites(sg)) {
    pr = sg->dtes_sg.vsg;
    for (pr = sg->dtes_sg.vsg; pr!= NULL;
         mat->p[nbC][0] = 0,
           mat->p[nbC][nbcolumns-1] =0,
           ray_to_matrix_ligne(pr,mat,nbC,sg->base),
           nbC++,
           pr = pr->succ);
  }
  // Traitement des rayons
  if (sg_nbre_rayons(sg)) {
    pr =sg->rays_sg.vsg;
    for (pr = sg->rays_sg.vsg; pr!= NULL;
         mat->p[nbC][0] = 1,
           mat->p[nbC][nbcolumns-1] =0,
           ray_to_matrix_ligne(pr,mat,nbC,sg->base),
           nbC++,
           pr = pr->succ);
  }
  // Traitement des sommets
  if (sg_nbre_sommets(sg)) {
    for (ps = sg->soms_sg.ssg; ps!= NULL;
         mat->p[nbC][0] = 1,
           mat->p[nbC][nbcolumns-1] = VALUE_TO_IRINT(ps->denominateur),
           ray_to_matrix_ligne((Pray_dte) ps,mat,nbC,sg->base),
           nbC++,
           ps = ps->succ);
  }
}

/* Fonction de conversion traduisant une matrix structure Irisa
 * sous forme d'un syste`me lineaire sc. Cette fonction est
 * utilisee paar la fonction sg_to_sc_chernikova
 */
static void matrix_to_sc(Matrix *mat, Psysteme sc)
{
  Pcontrainte pce=NULL;
  Pcontrainte pci=NULL;
  Pcontrainte pc_tmp=NULL;
  bool neweq = true;
  bool newineq = true;
  int i,nbrows;
  int nbeq=0;
  int nbineq=0;

  // Premiere droite
  if ((nbrows= mat->NbRows)) {
    for (i=0; i<nbrows; i++) {
	    switch (mat->p[i][0]) {
	    case 0:
        nbeq ++;
        if (neweq) {
          pce= pc_tmp  =
            matrix_ligne_to_contrainte(mat, i, sc->base);
          neweq = false;}
        else {
          pc_tmp->succ =
            matrix_ligne_to_contrainte(mat, i, sc->base);
          pc_tmp = pc_tmp->succ;	}
        break;
	    case 1:
        nbineq++;
        if (newineq) {
          pci = pc_tmp = matrix_ligne_to_contrainte(mat, i,sc->base);
				  newineq = false;
        }
        else {
          pc_tmp->succ = matrix_ligne_to_contrainte(mat, i, sc->base);
          pc_tmp = pc_tmp->succ;
        }
        break;
	    default:
        printf("error in matrix interpretation in Matrix_to_sc\n");
        break;
	    }
    }
    sc->nb_eq = nbeq;
    sc->egalites = pce;
    sc->nb_ineq = nbineq;
    sc->inegalites = pci;
  }
}



/* Fonction de conversion traduisant un polyhedron structure Irisa
 * sous forme d'un syste`me ge'ne'rateur. Cette fonction est
 * utilisee paar la fonction sc_to_sg_chernikova
 */
static void polyhedron_to_sg(Polyhedron  *pol, Ptsg sg)
{
  Pray_dte ldtes_tmp=NULL,ldtes = NULL;
  Pray_dte lray_tmp=NULL,lray = NULL;
  Psommet lsommet_tmp=NULL,lsommet=NULL;
  Stsg_vects dtes,rays;
  Stsg_soms sommets;
  Pvecteur pvnew;
  unsigned int i;
  int nbsommets =0;
  int  nbrays=0;
  int dim = vect_size(sg->base) +1;
  bool newsommet = true;
  bool newray = true;
  bool newdte = true;

  for (i=0; i< pol->NbRays; i++) {
    switch (pol->Ray[i][0]) {
    case 0:
	    // Premiere droite
	    pvnew = polyhedron_ligne_to_vecteur(pol,i,sg->base,dim);
	    if (newdte) {
        ldtes_tmp= ldtes = ray_dte_make(pvnew);
        newdte = false;
	    } else {
        // Pour chaque droite suivante
        ldtes_tmp->succ = ray_dte_make(pvnew);
        ldtes_tmp =ldtes_tmp->succ;
	    }
	    break;
    case 1:
	    switch (pol->Ray[i][dim]) {
	    case 0:
        nbrays++;
        // premier rayon
        pvnew = polyhedron_ligne_to_vecteur(pol,i,sg->base,dim);
        if (newray) {
          lray_tmp = lray = ray_dte_make(pvnew);
          newray = false;
        } else {
          lray_tmp->succ= ray_dte_make(pvnew);
          lray_tmp =lray_tmp->succ;    }
        break;
	    default:
        nbsommets ++;
        pvnew = polyhedron_ligne_to_vecteur(pol,i,sg->base,dim);
        if (newsommet) {
          lsommet_tmp=lsommet=
            sommet_make(IRINT_TO_VALUE(pol->Ray[i][dim]),
                        pvnew);
          newsommet = false;
        } else {
          lsommet_tmp->succ=
            sommet_make(IRINT_TO_VALUE(pol->Ray[i][dim]),
                        pvnew);
          lsommet_tmp = lsommet_tmp->succ;
        }
        break;
	    }
	    break;

    default: printf("error in rays elements \n");
	    break;
    }
  }
  if (nbsommets) {
    sommets.nb_s = nbsommets;
    sommets.ssg = lsommet;
    sg->soms_sg = sommets;
  }
  if (nbrays) {
    rays.nb_v = nbrays;
    rays.vsg=lray;
    sg->rays_sg = rays;
  }
  if (pol->NbBid) {
    dtes.vsg=ldtes;
    dtes.nb_v=pol->NbBid;
    sg->dtes_sg = dtes;
  }
}

/* Fonction de conversion d'un systeme lineaire a un systeme generateur
 * par chenikova
 */
Ptsg  sc_to_sg_chernikova_polylib(Psysteme sc)
{
  // Automatic variables read in a CATCH block need to be declared volatile as
  // specified by the documentation
  Polyhedron * volatile A = NULL;
  Matrix * volatile a = NULL;
  Ptsg volatile sg = NULL;

  int nbrows = 0;
  int nbcolumns = 0;

  CATCH(any_exception_error)
  {
    if (A) my_Polyhedron_Free(&A);
    if (a) my_Matrix_Free(&a);
    if (sg) sg_rm(sg);
    RETHROW();
  }
  TRY
  {
    assert(!SC_UNDEFINED_P(sc) && (sc_dimension(sc) != 0));

    sg = sg_new();
    nbrows = sc->nb_eq + sc->nb_ineq + 1;
    nbcolumns = sc->dimension +2;
    sg->base = base_dup(sc->base);
    // replace base_dup, which changes the pointer given as parameter
    a = Matrix_Alloc(nbrows, nbcolumns);
    sc_to_matrix(sc,a);

    A = Constraints2Polyhedron(a, MAX_NB_RAYS);
    my_Matrix_Free(&a);

    polyhedron_to_sg(A,sg);
    my_Polyhedron_Free(&A);
  } // end TRY

  UNCATCH(any_exception_error);

  return sg;
}

/* Fonction de conversion d'un systeme generateur a un systeme lineaire.
 * par chernikova
 */
Psysteme sg_to_sc_chernikova_polylib(Ptsg sg)
{
  int nbrows = sg_nbre_droites(sg)+ sg_nbre_rayons(sg)+sg_nbre_sommets(sg);
  int nbcolumns = base_dimension(sg->base)+2;
  // Automatic variables read in a CATCH block need to be declared volatile as
  // specified by the documentation
  Matrix * volatile a = NULL;
  Psysteme volatile sc = NULL;
  Polyhedron * volatile A = NULL;

  CATCH(any_exception_error)
  {
    if (sc) sc_rm(sc);
    if (a) my_Matrix_Free(&a);
    if (A) my_Polyhedron_Free(&A);

    RETHROW();
  }
  TRY
  {
    sc = sc_new();
    sc->base = base_dup(sg->base);
    //replace base_dup, which changes the pointer given as parameter
    sc->dimension = vect_size(sc->base);

    if (sg_nbre_droites(sg)+sg_nbre_rayons(sg)+sg_nbre_sommets(sg))
    {
      a = Matrix_Alloc(nbrows, nbcolumns);
      sg_to_polyhedron(sg,a);

      A = Rays2Polyhedron(a, MAX_NB_RAYS);
      my_Matrix_Free(&a);

      a = Polyhedron2Constraints(A);
      my_Polyhedron_Free(&A);

      matrix_to_sc(a,sc);
      my_Matrix_Free(&a);

      sc=sc_normalize(sc);

      if (sc == NULL) {
        Pcontrainte pc = contrainte_make(vect_new(TCST, VALUE_ONE));
        sc = sc_make(pc, CONTRAINTE_UNDEFINED);
        sc->base = base_dup(sg->base);
        // replace base_dup, which changes the pointer given as parameter
        sc->dimension = vect_size(sc->base);
      }
    }
    else {
      sc->egalites = contrainte_make(vect_new(TCST,VALUE_ONE));
      sc->nb_eq ++;
    }
  } // end TRY

  UNCATCH(any_exception_error);

  return sc;
}

/* Check if the Value vector N is in the submatrix R[b:e-1]
 *
 * Used to avoid adding duplicate vertices.
 */
static bool duplicate_vertex_p(Value ** R, int b, int e, int d, Value * N)
{
  bool duplicate_p = true;
  int i, j;

  for(i=b;i<e && duplicate_p;i++)
    for(j=0; j< d && duplicate_p;j++)
      duplicate_p = (R[i][j]==N[j]);
  return duplicate_p;
}
#endif

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

#ifdef CHERNIKOVA_DEPRECATED
/* Returns the convex hull sc of two constraint systems, sc1 and
 * sc2. All these constraint systems are represented using a sparse
 * representation.
 *
 * Each constraint systems is first translated into a dense
 * representation and converted to its dual representation, the
 * generating systems a1 or a2, using a dense representation,
 * i.e. matrices. The union of a1 and a2 is performed using the matrix
 * representation and its result, the generating system a, is
 * converted back to constraints, using a dense representation, and
 * then is translated into a sparse representation sc. A (integer
 * based?) redundance elimination phase is then (unfortunately?) applied
 * (sc_normalize).
 *
 * Two extra dimensions are added: one for the constant terms and the
 * second one to transform any polyhedron into a zero-pointed cone so
 * as to have only rays to handle. See INRIA Tech Report 785 by Doran
 * K. Wilde for motivation and explanations.
 *
 * Overflows (and arithmetic errors) can occur during the process, but
 * they are caught and a correct result is returned.
 *
 * Note: sc1 and sc2 must have the exact same basis for the convex
 * hull and the conversion to the dense representation to be meaningfull.
 */
Psysteme sc_convex_hull_polylib(Psysteme sc1, Psysteme sc2)
{
  int nbrows1 = 0;
  int nbcolumns1 = 0;
  int nbrows2 = 0;
  int nbcolumns2 = 0;
  Pbase b1 = sc_base(sc1);
  Pbase b2 = sc_base(sc2);

  // Make sure that the two sparse constraint systems
  // are in the same space, with the same bases
  assert(bases_strictly_equal_p(b1, b2));

  // Automatic variables read in a CATCH block need to be declared volatile
  // as specified by the documentation
  Matrix * volatile a1 = NULL;
  Matrix * volatile a2 = NULL;
  Polyhedron * volatile A1 = NULL;
  Polyhedron * volatile A2 = NULL;
  Matrix * volatile a = NULL;
  Psysteme volatile sc = NULL;
  Polyhedron * volatile A = NULL;

  unsigned int i1, i2;
  int j;
  int Dimension,cp;

  CATCH(any_exception_error)
  {
    if (a) my_Matrix_Free(&a);
    if (a1) my_Matrix_Free(&a1);
    if (a2) my_Matrix_Free(&a2);
    if (A) my_Polyhedron_Free(&A);
    if (A1) my_Polyhedron_Free(&A1);
    if (A2) my_Polyhedron_Free(&A2);
    if (sc) sc_rm(sc);

    // clear the alarm if necessary
    if (get_linear_convex_hull_timeout())
      alarm(0);
    // There's maybe exceptions rethrown by polylib.
    // So clear alarm in catch_alarm_sc_convex_hull is not enough

    if (sc_convex_hull_timeout)
    {
      sc_convex_hull_timeout = false;

      //fprintf(stderr,"\n *** * *** Timeout from polyedre/chernikova : sc_convex_hull !!! \n");
	//We can change to print to stderr by using sc_default_dump(sc). duong

	/* need to install sc before to run, because of Production/Include/sc.h
        ifscprint(4) {
	char * filename;
	char * label;
	  // if  print to stderr
	  //fprintf(stderr, "Timeout [sc_convex_hull] considering:\n");
	  //sc_default_dump(sc1)
	  //sc_default_dump(sc2)
	  filename = "convex_hull_fail_sc_dump.out";
	  label = "LABEL - Timeout with sc_convex_hull considering: *** * *** SC ";
	  sc_default_dump_to_file(sc1,label,1,filename);
	  label = "                                                 *** * *** SC ";
	  sc_default_dump_to_file(sc2,label,2,filename);
	}
	*/
    }

      //fprintf(stderr,"\nThis is an exception rethrown from sc_convex_hull(): \n ");
      RETHROW();
    }
    TRY
    {
      assert(!SC_UNDEFINED_P(sc1) && (sc_dimension(sc1) != 0));
      assert(!SC_UNDEFINED_P(sc2) && (sc_dimension(sc2) != 0));

      //start the alarm
      if (get_linear_convex_hull_timeout())
      {
        signal(SIGALRM, catch_alarm_sc_convex_hull);
        alarm(get_linear_convex_hull_timeout());
      }

      ifscdebug(7) {
        fprintf(stderr, "[sc_convex_hull] considering:\n");
        sc_default_dump(sc1);
        sc_default_dump(sc2);
      }

      // comme on il ne faut pas que les structures irisa
      // apparaissent dans le fichier polyedre.h, une sous-routine
      // renvoyant un polyhedron n'est pas envisageable.
      // Le code est duplique

    nbrows1 = sc1->nb_eq + sc1->nb_ineq + 1;
    nbcolumns1 = sc1->dimension +2;
    a1 = Matrix_Alloc(nbrows1, nbcolumns1);
    sc_to_matrix(sc1,a1);

    nbrows2 = sc2->nb_eq + sc2->nb_ineq + 1;
    nbcolumns2 = sc2->dimension +2;
    a2 = Matrix_Alloc(nbrows2, nbcolumns2);
    sc_to_matrix(sc2,a2);

    ifscdebug(8) {
      fprintf(stderr, "[sc_convex_hull]\na1 =");
      Matrix_Print(stderr, "%4d",a1);
      fprintf(stderr, "\na2 =");
      Matrix_Print(stderr, "%4d",a2);
    }

    A1 = Constraints2Polyhedron(a1, MAX_NB_RAYS);

    my_Matrix_Free(&a1);

    A2 = Constraints2Polyhedron(a2, MAX_NB_RAYS);

    my_Matrix_Free(&a2);

    ifscdebug(8) {
      fprintf(stderr, "[sc_convex_hull]\nA1 (%p %p)=", A1, a1);
      Polyhedron_Print(stderr, "%4d",A1);
      fprintf(stderr, "\nA2 (%p %p) =", A2, a2);
      Polyhedron_Print(stderr, "%4d",A2);
    }

    sc = sc_new();
    sc->base = base_dup(sc1->base);
    //replace base_dup, which changes the pointer given as parameter
    sc->dimension = vect_size(sc->base);

    if (A1->NbRays == 0) {
      a = Polyhedron2Constraints(A2);
    } else  if (A2->NbRays == 0) {
      a = Polyhedron2Constraints(A1);
    } else {
      /* Keep track of the different kinds of generating elements copied in a:
       *
       * lines of A1 in [0:l1-1]
       * lines of A2 in [l1:l2-1]
       *
       * rays of A1 in [l2][r1-1]
       * rays of A2 in [r1][r2-1]
       *
       * vertices of A1 in [r2][v1-1]
       * vertices of A2 in [v1][v2-1]
       *
       * Each of these subsets may be empty.
       *
       * We have 0<=l1<=l2<=r1<=r2<=v1<=v2
       */
      int __attribute__((unused)) l1 = 0, l2 = 0,
        __attribute__((unused)) r1 = 0, r2 = 0,
        v1 = 0, __attribute__((unused)) v2 = 0;
      Dimension = A1->Dimension+2;
      a = Matrix_Alloc(A1->NbRays + A2->NbRays,Dimension);

      // Tri des contraintes de A1->Ray et A2->Ray, pour former
      // l'union de ces contraintes dans un meme format
      // Line , Ray , Vertex
      cp = 0;
      i1 = 0;
      i2 = 0;
      while (i1 < A1->NbRays && A1->Ray[i1][0] ==0) {
        for (j=0; j < Dimension ; j++)
          a->p[cp][j] = A1->Ray[i1][j];
        cp++; i1++;
      }
      l1 = cp;
      while (i2 < A2->NbRays && A2->Ray[i2][0] ==0) {
        for (j=0 ; j < Dimension ; j++)
          a->p[cp][j] = A2->Ray[i2][j];
        cp++; i2++;
      }
      l2 = cp;
      while (i1 < A1->NbRays && A1->Ray[i1][0] ==1
             && A1->Ray[i1][Dimension-1]==0) {
        for (j=0; j < Dimension ; j++)
          a->p[cp][j] = A1->Ray[i1][j];
        cp++; i1++;
      }
      r1 = cp;
      while (i2 < A2->NbRays && A2->Ray[i2][0] == 1
             && A2->Ray[i2][Dimension-1]==0) {
        for (j=0; j < Dimension ; j++)
          a->p[cp][j] = A2->Ray[i2][j];
        cp++; i2++;
      }
      r2 = cp;
      while (i1 < A1->NbRays && A1->Ray[i1][0] == 1
             && A1->Ray[i1][Dimension-1]!= 0) {
        // Insert vertex of sc1 if it is not strictly redundant with sc2
        if(true || !redundant_vertex_p(sc2, Dimension, A1->Ray[i1], true)) {
          for (j=0; j < Dimension ; j++)
            a->p[cp][j] = A1->Ray[i1][j];
          cp++;
        }
        i1++;
      }
      v1 = cp;
      while (i2 < A2->NbRays && A2->Ray[i2][0] == 1
             && A2->Ray[i2][Dimension-1]!=0) {
        // Insert vertex of sc2 if it is not also a vertex of s1 and
        // is not redundant with sc1
        // FC: what is the point of an "if (true || ...)" ?
        if(true ||
           (!duplicate_vertex_p(a->p, r2, v1, Dimension, A2->Ray[i2])
            && !redundant_vertex_p(sc1, Dimension, A2->Ray[i2], false))) {
          for (j=0; j < Dimension ; j++)
            a->p[cp][j] = A2->Ray[i2][j];
          cp++;
        }
        i2++;
      }
      v2 = cp;

      my_Polyhedron_Free(&A1);
      my_Polyhedron_Free(&A2);

      A = Rays2Polyhedron(a, MAX_NB_RAYS);
      my_Matrix_Free(&a);

      a = Polyhedron2Constraints(A);
      my_Polyhedron_Free(&A);
    }

    matrix_to_sc(a,sc);
    my_Matrix_Free(&a);

    sc = sc_normalize(sc);

    if (sc == NULL) {
      Pcontrainte pc = contrainte_make(vect_new(TCST, VALUE_ONE));
      sc = sc_make(pc, CONTRAINTE_UNDEFINED);
      sc->base = base_dup(sc1->base);
      //replace base_dup, which changes the pointer given as parameter
      sc->dimension = vect_size(sc->base);
    }

    } /* end TRY */

    // clear the alarm if necessary
    if (get_linear_convex_hull_timeout())
      alarm(0);

    UNCATCH(any_exception_error);

    return sc;
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
