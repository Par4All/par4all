/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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

/* Name     : utils.c
 * Package  : paf-util
 * Author   : Alexis Platonoff
 * Date     : july 1993
 *
 * Historic :
 * - 16 jul 93, change in paf_ri, AP
 * - 23 sep 93, remove some functions from paf_ri to placement, AP
 * - 29 sep 93, remove prefixes when they are not needed (adg, plc, ...), AP
 * - 9  nov 93, add merge_sort() and meld(), AP
 * - 9  dec 93, add rational_op_exp(), AP
 * - 21 fev 94, add prototype_var_subst(), AP
 * - 21 fev 94, add vecteur_mult(), AP
 * - 25 fev 94, add prototype_factorize(), modify vecteur_to_polynome(), AP
 * - 11 mar 94, modify polynome_to_sc(), AP
 * - 7  apr 94, add pu_constraints_with_sym_cst_to_matrices() and
 *		pu_matrices_to_constraints_with_sym_cst(), AP
 * - 29 jun 94, add simplify_minmax(), AP
 * -  9 nov 94, modify name : merge_sort() --> general_merge_sort()
 *              because it already exists in C3, AP
 * - 14 nov 94, add find_implicit_equation(), taken from prgm_mapping, AP
 * - 20 dec 94, add functions for the static control map, AP
 * - 19 jan 95, move make_rational_vect() in this file from scheduling and
 *              reindexing packages, AP
 * - 25 sep 95, moved stco_common_loops_of_statements() in this file from
 *              static_controlise/utils.c, AP
 *
 * Documents:
 *
 * Comments : This file contains useful functions used for the computation of
 * paf_ri.
 */

/* Ansi includes 	*/
#include <stdio.h>
#include <string.h>
/* #include <varargs.h> */ /* (not ANSI but SUN:-) */

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "paf_ri.h"
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;
#include "graph.h"
#include "dg.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"
#include "paf-util.h"
#include "static_controlize.h"

// Forward declaration to survive races in the compilation process
statement_mapping get_current_stco_map(void);

/* Macro functions	*/
#define STRING_FOUR_OPERATION_P(s) ( \
				    (strcmp(s,PLUS_OPERATOR_NAME) == 0) || \
				    (strcmp(s,MINUS_OPERATOR_NAME) == 0) || \
				    (strcmp(s,MULTIPLY_OPERATOR_NAME) == 0) || \
				    (strcmp(s,DIVIDE_OPERATOR_NAME) == 0) )
#define ENTITY_FOUR_OPERATION_P(s) (ENTITY_PLUS_P(s) || ENTITY_MINUS_P(s) || ENTITY_MULTIPLY_P(s) || ENTITY_DIVIDE_P(s))

/* Global variables	*/

/* Internal variables	*/

/* Local defines: FI, they are needed earlier in the file because
   newgen is now fully typed statically */
/*
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;
*/

/* begin MATRIX functions */

/*=========================================================================*/
/*
 * void matrix_scalar_multiply(A, nb)
 *
 * multiplies all elements of matrix A by the number nb.
 *
 */
/* never called
void matrix_scalar_multiply(A, nb)
Pmatrix   A;
int      nb;
{
  int i, j, m, n, d, p;

  m = MATRIX_NB_LINES(A);
  n = MATRIX_NB_COLUMNS(A);
  d = MATRIX_DENOMINATOR(A);
  p = pgcd(d, nb);

  MATRIX_DENOMINATOR(A) = d/p;
  for (i = 1; i <= m; i++)
    for (j = 1; j <= n; j++)
      MATRIX_ELEM(A,i,j) = (nb/p) * MATRIX_ELEM(A,i,j);
}
*/

/*=====================================================================*/
/*
 * void matrix_add(Pmatrix a, Pmatrix b, Pmatrix c)
 *
 * add rational matrix c to rational matrix b and store result
 * in matrix a
 *
 *      a is a (nxm) matrix, b a (nxm) and c a (nxm)
 *
 *      a = b + c ;
 *
 * Algorithm used is directly from definition, and space has to be
 * provided for output matrix a by caller. Matrix a is not necessarily
 * normalized: its denominator may divide all its elements
 * (see matrix_normalize()).
 *
 * Precondition:        n > 0; m > 0;
 * Note: aliasing between a and b or c is supported
 */
/* never called
void pu_matrix_add(a,b,c)
Pmatrix a;
Pmatrix b, c;
{
  int d1, d2, i, j, n, m;

  n = MATRIX_NB_LINES(a);
  m = MATRIX_NB_COLUMNS(a);
  pips_assert("matrix_add", (n > 0) && (m > 0));

  d1 = MATRIX_DENOMINATOR(b);
  d2 = MATRIX_DENOMINATOR(c);
  if (d1 == d2) {
    for (i = 1; i <= n; i++)
      for (j = 1; j <= m; j++)
        MATRIX_ELEM(a,i,j)=MATRIX_ELEM(b,i,j)+MATRIX_ELEM(c,i,j);
    MATRIX_DENOMINATOR(a) = d1;
  }
  else {
    int lcm = ppcm(d1,d2);
    d1 = lcm/d1;
    d2 = lcm/d2;
    for (i = 1; i <= n; i++)
      for (j = 1; j <= m; j++)
        MATRIX_ELEM(a,i,j)=MATRIX_ELEM(b,i,j)*d1+MATRIX_ELEM(c,i,j)*d2;
    MATRIX_DENOMINATOR(a) = lcm;
  }
}
*/

/* ======================================================================= */
/*
 * void constraints_with_sym_cst_to_matrices(Pcontrainte pc, Pbase ib, cb,
 *					     Pmatrix A B)
 *
 * constructs the matrices "A" and "B" corresponding to the linear
 * constraints "pc", so: A.ib + B1.cb + B2 = 0 <=> pc(ib, cb) = 0
 *
 *      B = ( B1 | B2 ), B2 of dimension (n,1).
 *
 * The basis "ib" gives the variables of the linear system.
 * The basis "cb" gives the symbolic constants of the linear system.
 *
 * WARNING: "cb" should not contain TCST.
 *
 * The matrices "A" and "B" are supposed to have been already allocated in
 * memory, respectively of dimension (n, m1) and (n, m2):
 *
 * n must be the exact number of constraints in "pc".
 * m1 must be the exact number of variables in "ib".
 * m2 must be the exact number of variables in "cb" PLUS ONE (TCST).
 */
/* never called
void pu_constraints_with_sym_cst_to_matrices(pc,ib,cb,A,B)
Pcontrainte pc;
Pbase ib,cb;
Pmatrix A, B;
{
    int i,j;
    Pcontrainte eq;
    Pvecteur pv;
    int n, m1, m2;

    for (eq = pc, n = 0; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ) { n++; }
    m1 = vect_size(ib);
    m2 = vect_size(cb) + 1;

    pips_assert("constraints_with_sym_cst_to_matrices",
		(MATRIX_NB_LINES(A) == n) && (MATRIX_NB_COLUMNS(A) == m1) &&
		(MATRIX_NB_LINES(B) == n) && (MATRIX_NB_COLUMNS(B) == m2));

    matrix_nulle(B);
    matrix_nulle(A);

    for (eq = pc,i=1; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,i++) {
        for(pv = ib, j=1; pv != NULL; pv = pv->succ, j++){
            MATRIX_ELEM(A,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
        }
        for(pv = cb, j=1; pv != NULL; pv = pv->succ, j++){
            MATRIX_ELEM(B,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
        }
        MATRIX_ELEM(B,i,m2) = vect_coeff(TCST,eq->vecteur);
    }
}
*/

/* ======================================================================= */
/*
 * void matrices_to_constraints_with_sym_cst(Pcontrainte *pc, Pbase ib, cb,
 *					     Pmatrix A, B)
 *
 * constructs the constraints "pc" corresponding to the matrices "A" and "B"
 * so: pc(ib, cb) = 0 <=> A.ib + B1.cb + B2 = 0
 *
 *      B = ( B1 | B2 ), B2 of dimension (n,1).
 *
 * The basis "ib" gives the variables of the linear system.
 * The basis "cb" gives the symbolic constants of the linear system.
 *
 * Matrices "A" and "B" are supposed to have been already allocated in
 * memory, respectively of dimension (n, m1) and (n, m2).
 *
 * n must be the exact number of constraints in "pc".
 * m1 must be the exact number of variables in "ib".
 * m2 must be the exact number of variables in "cb" PLUS ONE (TCST).
 *
 * "A" and "B" may be rationnal. In such case, we compute the least common
 * multiple of their denominators and multiply the system by it:
 *
 * Note: the formal parameter pc is a "Pcontrainte *". Instead, the resulting
 * Pcontrainte could have been returned as the result of this function.
 */
/* never called
void pu_matrices_to_constraints_with_sym_cst(pc,ib,cb,A,B)
Pcontrainte *pc;
Pbase ib,cb;
Pmatrix A, B;
{
    Pcontrainte newpc = NULL;
    int i, j, coeff, dena, denb, n, m1, m2, lcm;

    n  = MATRIX_NB_LINES(A);
    m1 = MATRIX_NB_COLUMNS(A);
    m2 = MATRIX_NB_COLUMNS(B);

    pips_assert("constraints_with_sym_cst_to_matrices",
		(MATRIX_NB_LINES(B) == n) &&
		(vect_size(ib) == m1) && ((vect_size(cb) + 1) == m2));

    dena = MATRIX_DENOMINATOR(A);
    denb = MATRIX_DENOMINATOR(B);
    lcm = ppcm(dena, denb);

    for (i=n;i>=1; i--) {
        bool found = false;
        Pcontrainte cp = contrainte_new();
        Pvecteur vect, pv = NULL;

        if ((coeff = MATRIX_ELEM(B,i,m2)) != 0) {
            pv = vect_new(TCST,  (lcm/denb) * coeff);
            found = true;
        }
        for (j=1, vect=ib;j<=m1;vect=vect->succ,j++) {
            if ((coeff = MATRIX_ELEM(A,i,j)) != 0)
                if (found)
                    vect_chg_coeff(&pv, vecteur_var(vect),(lcm/dena) * coeff);
                else {
                    pv = vect_new(vecteur_var(vect), (lcm/dena) * coeff);
                    found = true;
                }
        }
        for (j=1, vect=cb;j<=m2-1;vect=vect->succ,j++) {
            if ((coeff = MATRIX_ELEM(B,i,j)) != 0)
                if (found)
                    vect_chg_coeff(&pv, vecteur_var(vect),(lcm/denb) * coeff);
                else {
                    pv = vect_new(vecteur_var(vect), (lcm/denb) * coeff);
                    found = true;
                }
        }
        cp->vecteur = pv;
        cp->succ = newpc;
        newpc = cp;
    }
    *pc = newpc;
}
*/

/*============================================================================*/
/* void pu_matrices_to_contraintes(Pcontrainte *pc, Pbase b, matrice A B,
 *				   int n m):
 * constructs the constraints "pc" corresponding to the matrices "A" and "B"
 * so: pc(b) <=> Ab + B
 *
 * B represents the constant term.
 *
 * The base "b" gives the variables of the linear system.
 * The matrices "A" and "B" are respectively of dimension (n, m) and (n, 1).
 *
 * "n" will be the exact number of constraints contained in "pc".
 * "m" must be the exact number of variables contained in "b".
 */
void pu_matrices_to_contraintes(pc, b, A, B, n, m)
Pcontrainte *pc;
Pbase b;
matrice A, B;
int n, m;
{
  Pvecteur vect,pv=NULL;
  Pcontrainte cp, newpc= NULL;
  int i,j;
  Value cst,coeff,dena,denb;
  bool trouve ;

  dena = DENOMINATOR(A);
  denb = DENOMINATOR(B);

  for (i=n;i>=1; i--) {
    trouve = false;
    cp = contrainte_new();

    /* build the constant terme if it is null */
    if (value_notzero_p(cst = ACCESS(B,n,i,1))) {
      pv = vect_new(TCST,  value_mult(dena,cst));
      trouve = true;
    }

    for (vect = b,j=1;j<=m;vect=vect->succ,j++) {
      if (value_notzero_p(coeff = ACCESS(A,n,i,j))) {
	if (trouve)
	  vect_chg_coeff(&pv, vecteur_var(vect),
			 value_mult(denb,coeff));
	else {
	  /* build a new vecteur if there is a null constant term */
	  pv = vect_new(vecteur_var(vect), value_mult(denb,coeff));
	  trouve = true;
	}
      }
    }
    cp->vecteur = pv;
    cp->succ =  newpc;
    newpc = cp;
  }
  *pc = newpc;
}

/*============================================================================*/
/* void pu_contraintes_to_matrices(Pcontrainte pc, Pbase b, matrice A B,
 *				   int n m):
 * constructs the matrices "A" and "B" corresponding to the linear
 * constraints "pc", so: Ab + B <=> pc(b).
 *
 * The base "b" gives the variables of the linear system.
 *
 * The matrices "A" and "B" are supposed to have been already allocated in
 * memory, respectively of dimension (n, m) and (n, 1).
 *
 * "n" must be the exact number of constraints contained in "pc".
 * "m" must be the exact number of variables contained in "b".
 */
void pu_contraintes_to_matrices(pc, b, A, B, n, m)
Pcontrainte pc;
Pbase b;
matrice A, B;
int n;
int m;
{
  int i,j;
  Pvecteur pv;
  Pcontrainte eq;
  matrice_nulle(B,n,1);
  matrice_nulle(A,n,m);

  for(eq = pc,i=1; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,i++) {
    for(pv = b, j=1; pv != NULL; pv = pv->succ, j++){
      ACCESS(A,n,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
    }
    ACCESS(B,n,i,1) = vect_coeff(0,eq->vecteur);
  }
}


/* Creation de la matrice A correspondant au systeme lineaire et de la matrice
 * correspondant a la partie constante B
 * Le systeme peut contenir des constantes symboliques. Dans ce cas, la base
 * index_base ne doit contenir que les variables etant des indices de boucles
 * et la base  const_base les constantes symboliques. La matrice B
 * represente toutes les contraintes sur les constantes.
 *
 *  Les parametres de la fonction :
 *
 * Psysteme ps  : systeme lineaire
 *!int	A[]	:  matrice
 *!int	B[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m	: nombre de colonnes de la matrice
 */

void contraintes_with_sym_cst_to_matrices(pc,index_base,const_base,A,B,n,m1,m2)
Pcontrainte pc;
Pbase index_base,const_base;
matrice A;
matrice B;
int n,m1,m2;
{

    int i,j;
    Pcontrainte eq;
    Pvecteur pv;

    matrice_nulle(B,n,m2);
    matrice_nulle(A,n,m1);

    for (eq = pc,i=1; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,i++) {
	for(pv = index_base, j=1; pv != NULL; pv = pv->succ, j++){
	    ACCESS(A,n,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
	}
	for(pv = const_base, j=1; pv != NULL; pv = pv->succ, j++){
	    ACCESS(B,n,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
	}
	ACCESS(B,n,i,m2) = vect_coeff(TCST,eq->vecteur);
    }

}


/*
 * Creation d'un systeme lineaire  a partir de deux matrices. La matrice B
 * correspond aux termes constants de chacune des inequations appartenant
 * au systeme. La matrice A correspond a la partie lineaire  des expressions
 * des inequations le  composant.
 *
 * Le systeme peut contenir des constantes symboliques. Dans ce cas, la
 * matrice B represente toutes les contraintes sur les constantes.
 * La base index_base ne  contiendra que les variables etant des indices de
 * boucles et la base  const_base les constantes symboliques.
 *
 * L'ensemble des variables du nouveau systeme est initialise avec une base
 * d'indices que l'on donne en argument. Cette base peut etre vide (NULL).
 *
 * Des nouvelles variables sont creees si necessaire  si il n'y a pas assez
 * de variables dans la base fournie.
 *
 *  La matrice A correspond a la partie non constante du systeme lineaire.
 *  La matrice B correspond a la partie constante.
 *  Le syteme lineaire s'ecrit    A.x <= B.
 *
 *  Les parametres de la fonction :
 *
 *!Psysteme ps  : systeme lineaire
 * int	A[]	: matrice  de dimension (n,m)
 * int	B[]	: matrice  de dimension (n,m2)
 * int  n	: nombre de lignes de la matrice
 * int  m	: nombre de colonnes de la matrice A
 */
void matrices_to_contraintes_with_sym_cst(pc,index_base,const_base,A,B,n,m1,m2)
Pcontrainte *pc;
Pbase index_base,const_base;
matrice A,B;
int n,m1,m2;
{
  Pvecteur vect,pv=NULL;
  Pcontrainte cp,newpc= NULL;
  int i,j;
  Value cst,coeff,dena,denb;
  bool trouve ;

  dena = DENOMINATOR(A);
  denb = DENOMINATOR(B);

  for (i=n;i>=1; i--) {
    trouve = false;
    cp = contrainte_new();

    /* build the constant terme if it exists */
    if (value_notzero_p(cst = ACCESS(B,n,i,m2))) {
      pv = vect_new(TCST,  value_mult(dena,cst));
      trouve = true;
    }

    for (vect = base_union(index_base, const_base),j=1;
	 j<=m1;vect=vect->succ,j++) {
      if (value_notzero_p(coeff = ACCESS(A,n,i,j))) {
	if (trouve) {
	  vect_chg_coeff(&pv, vecteur_var(vect),
			 value_mult(denb, coeff));
	}
	else {
	  /* build a new vecteur if there is not constant term */
	  pv = vect_new(vecteur_var(vect), value_mult(denb, coeff));
	  trouve = true;
	}
      }
    }

    for (j=1;j<=m2-1;vect=vect->succ,j++) {
      if (value_notzero_p(coeff = ACCESS(B,n,i,j))) {
	if (trouve) {
	  vect_chg_coeff(&pv, vecteur_var(vect),
			 value_mult(denb, coeff));
	}
	else {
	  /* build a new vecteur if there is not constant term */
	  pv = vect_new(vecteur_var(vect),
			value_mult(denb, coeff));
	  trouve = true;
	}
      }
    }

    cp->vecteur = pv;
    cp->succ = newpc;
    newpc = cp;
  }
  *pc = newpc;
}


/*============================================================================*/
/* void pu_egalites_to_matrice(matrice a, int n m, Pcontrainte leg, Pbase b):
 * constructs the matrix "a" with the equalities contained in "leg". The base
 * "b" gives the column number of each variable. The first element of the
 * matrix is a(1,1), the ACCESS macro makes the conversion to the C array
 * numbering which begins at (0,0).
 *
 * The constant term is not included. The matrix "a" is supposed to have been
 * already allocated in memory.
 *
 * "n" must be the exact number of equalities contained in "leg".
 * "m" must be the exact number of variables contained in "b".
 */
/* never called
void pu_egalites_to_matrice(a, n, m, leg, b)
matrice a;
int n;
int m;
Pcontrainte leg;
Pbase b;
{
  Pvecteur v;
  Pcontrainte peq;
  int ligne = 1;

  matrice_nulle(a, n, m);

  for(peq = leg; peq != NULL; peq = peq->succ, ligne++) {
    pips_assert("pu_egalites_to_matrice",ligne<=n);

    for(v = peq->vecteur; !VECTEUR_UNDEFINED_P(v); v = v->succ) {
      int rank;
      if(vecteur_var(v) != TCST) {
         rank = base_find_variable_rank(base_dup(b), vecteur_var(v),
					pu_variable_name);
         pips_assert("pu_egalites_to_matrice", rank <= m);

         ACCESS(a, n, ligne, rank) = vecteur_val(v);
      }
    }
  }
}
*/
/* end MATRIX functions */



/*=======================================================================*/
/* vertex  in_dfg_vertex_list( (list) l, (vertex) v )            AL 30/06/93
 * Input  : A list l of vertices.
 *          A vertex v of a dependence graph.
 * Returns vertex_undefined if v is not in list l.
 * Returns v' that has the same statement_ordering than v.
 */
vertex in_dfg_vertex_list(list l, vertex v)
{
        vertex          ver;
	int             in;

	pips_debug(9, "doing \n");

	in = dfg_vertex_label_statement( (dfg_vertex_label)
					vertex_vertex_label(v) );
	for(;!ENDP(l); POP(l)) {
		int prov_i;
		ver = VERTEX(CAR( l ));
		prov_i = dfg_vertex_label_statement( (dfg_vertex_label)
					vertex_vertex_label(ver) );
		if ( prov_i == in ) return( ver );
	}

	return vertex_undefined;
}

/*=======================================================================*/
/* vertex  in_dg_vertex_list( (list) l, (vertex) v )            AL 30/06/93
 * Input  : A list l of vertices.
 *          A vertex v of a dependence graph.
 * Returns vertex_undefined if v is not in list l.
 * Returns v' that has the same statement_ordering than v.
 */
vertex in_dg_vertex_list(list l, vertex v)
{
        vertex          ver;
	int             in;

	pips_debug(9, "doing \n");

	in = dg_vertex_label_statement( (dg_vertex_label)
					vertex_vertex_label(v) );
	for(;!ENDP(l); POP(l)) {
		int prov_i;
		ver = VERTEX(CAR( l ));
		prov_i = dg_vertex_label_statement( (dg_vertex_label)
					vertex_vertex_label(ver) );
		if ( prov_i == in ) return( ver );
	}

	return vertex_undefined;
}

/*============================================================================*/
/* expression make_id_expression(string s): makes an expression with the
 * name of a variable. For this variable, we create a new entity if it does not
 * exist yet.
 */
expression make_id_expression(string s)
{
 entity new_ent;
 string exp_full_name;

 exp_full_name = strdup(concatenate(DFG_MODULE_NAME, MODULE_SEP_STRING,
				    s, (char *) NULL));

 new_ent = gen_find_tabulated(exp_full_name, entity_domain);

 if(new_ent == entity_undefined)
    new_ent = make_entity(exp_full_name,
			  make_type(is_type_variable,
				    make_variable(make_basic_int(4/*UU*/),
						  NIL, NIL)),
			  make_storage(is_storage_ram, ram_undefined),
			  make_value_unknown());

 return(make_expression(make_syntax(is_syntax_reference,
				    make_reference(new_ent,NIL)),
			normalized_undefined));
}

/*============================================================================*/
/* expression make_array_ref(list l): returns an expression which is a
 * reference to an array. The list "l" is composed of expressions, the first
 * one is the array itself and the others are the indices. In order to create
 * the desire expression, we only have to put the CDR of "l" into the list of
 * indices of the reference contained in the first expression of "l".
 */
expression make_array_ref(list l)
{
  expression new_exp;
  list array_inds;

  if(l == NIL)
    pips_internal_error("No args for array ref");

  new_exp = EXPRESSION(CAR(l));
  array_inds = CDR(l);

  if(! syntax_reference_p(expression_syntax(new_exp)))
    pips_internal_error("Array ref is not a reference");

  reference_indices(syntax_reference(expression_syntax(new_exp))) = array_inds;

  return(new_exp);
}

/*============================================================================*/
/* expression make_func_op(string func_name, list args): returns an expression
 * that represent the call to "func_name" with "args" as arguments.
 */
expression make_func_op(func_name, args)
string func_name;
list args;
{
 entity func_ent;

 func_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
                                                    func_name), entity_domain);

 if(func_ent == entity_undefined)
    pips_internal_error("Function unknown : %s", func_name);

 return(make_expression(make_syntax(is_syntax_call,
                                    make_call(func_ent, args)),
                        normalized_undefined));
}

/*============================================================================*/
/* expression lisp_exp_to_ri_exp(lisp_expression le): returns the
 * expression that represent the lisp_expression given in argument ("le"). A
 * lisp_expression is a Newgen structure that defines an expression as a list
 * with the operator as the first object and the arguments as the rest of the
 * list.
 *
 * There are a few cases. If the operator is "aref" or "aset" then this lisp
 * expression is a reference to an array. We use make_array_ref().
 * If the operator is not one of the four operations (+,-,*,/), then we use
 * make_func_op().
 * Else (the operator is one of the four operation) we use rational_op_exp().
 */
expression lisp_exp_to_ri_exp(le)
lisp_expression le;
{
 expression exp1, exp2;
 string exp_op = lisp_expression_operation(le);
 list exp_args = lisp_expression_args(le);

 if( (strncmp(exp_op, "aref", 4) == 0) || (strncmp(exp_op, "aset", 4) == 0) )
    return(make_array_ref(exp_args));

 if(! STRING_FOUR_OPERATION_P(exp_op))
    return(make_func_op(exp_op, exp_args));

 exp1 = EXPRESSION(CAR(exp_args));
 exp_args = CDR(exp_args);

 if(exp_args == NIL)
    pips_internal_error("Only 1 argument for a binary (or more) operation");

 for(; exp_args != NIL; exp_args = CDR(exp_args))
   {
    exp2 = EXPRESSION(CAR(exp_args));

    exp1 = rational_op_exp(exp_op, exp1, exp2);
   }
 return(exp1);
}


/*============================================================================*/
/* expression negate_expression(expression exp): returns the negation of
 * the expression given in argument "exp".
 *
 * In fact this computation is done only if the expression is linear with
 * integer coefficients. If so, we use the Pvecteur form. Else, we return the
 * duplication of the expression.
 */
expression negate_expression(exp)
expression exp;
{
 expression neg_exp;
 normalized nexp;

 nexp = NORMALIZE_EXPRESSION(exp);

 if(normalized_complex_p(nexp))
    neg_exp = copy_expression(exp);
 else
   {
    Pvecteur vexp, new_vec;

    vexp = (Pvecteur) normalized_linear(nexp);
    new_vec = vect_dup(vexp);
    vect_chg_sgn(new_vec);

    neg_exp = make_vecteur_expression(new_vec);
   }

 return(neg_exp);
}

/*============================================================================*/
/* predicate expressions_to_predicate(list exp_l): returns the predicate
 * that has the inequalities given as expressions in "exp_l". For example:
 * if A is an expresion of "exp_l" then we'll have the inequality A <= 0 in the
 * predicate.
 *
 * If an expression is not linear, we warn the user.
 *
 * Note: if "exp_l" is empty then we return an undefined predicate.
 */
predicate expressions_to_predicate(exp_l)
list exp_l;
{
 predicate new_pred;
 Psysteme new_sc;
 list l;

 if(exp_l == NIL)
    return(predicate_undefined);

 new_sc = sc_new();

 for(l = exp_l; l != NIL; l = CDR(l))
   {
    expression exp = EXPRESSION(CAR(l));
    normalized nexp = NORMALIZE_EXPRESSION(exp);

    if(normalized_linear_p(nexp))
      {
       Pvecteur new_vec = (Pvecteur) normalized_linear(nexp);
       sc_add_inegalite(new_sc, contrainte_make(new_vec));
      }
    else
      {
       printf("\nNon linear expression :");
       printf(" %s\n", words_to_string(words_expression(exp,NIL)));
      }
   }

 sc_creer_base(new_sc);
 new_pred = make_predicate(new_sc);

 return(new_pred);
}


/*============================================================================*/
/* int vertex_int_stmt(vertex v): returns the statement number contained
 * in the vertex. It is a "dfg" vertex.
 */
int vertex_int_stmt(v)
vertex v;
{
return(dfg_vertex_label_statement((dfg_vertex_label) vertex_vertex_label((v))));
}


/*============================================================================*/
/* void comp_exec_domain(graph g, STS): computes the execution domain of
 * each statement. The englobing loops are obtained through the static
 * control map. */
void comp_exec_domain(g, STS)
graph g;
hash_table STS;
{
  int stmt;
  list loop_l, dfg_edge_l;

  /* We update the graph global variable with the exec domain. */
  dfg_edge_l = graph_vertices(g);
  for(; dfg_edge_l != NIL; dfg_edge_l = CDR(dfg_edge_l)) {
    vertex v = VERTEX(CAR(dfg_edge_l));
    dfg_vertex_label dvl;
    Psysteme new_sc = sc_new();

    dvl = (dfg_vertex_label) vertex_vertex_label(v);
    stmt = vertex_int_stmt(v);
    loop_l = static_control_loops((static_control) hash_get(STS,
							    (char *) ((long)stmt)));

    for( ; loop_l != NIL; loop_l = CDR(loop_l))
      {
       Pvecteur vect_index, vect;
       normalized nub, nlb;
       /* loop aux_loop = ENTITY(CAR(loop_l)); */
       loop aux_loop = LOOP(CAR(loop_l));
       entity index_ent = loop_index(aux_loop);

       vect_index = vect_new((char *) index_ent, VALUE_ONE);
       nlb = NORMALIZE_EXPRESSION(range_lower(loop_range(aux_loop)));
       nub = NORMALIZE_EXPRESSION(range_upper(loop_range(aux_loop)));

       if (normalized_linear_p(nlb))
         {
          vect = vect_substract((Pvecteur) normalized_linear(nlb), vect_index);
          if(!VECTEUR_NUL_P(vect))
             sc_add_inegalite(new_sc, contrainte_make(vect));
         }
       if (normalized_linear_p(nub))
         {
          vect = vect_substract(vect_index, (Pvecteur) normalized_linear(nub));
          if(!VECTEUR_NUL_P(vect))
             sc_add_inegalite(new_sc, contrainte_make(vect));
         }
      }
    sc_creer_base(new_sc);

    dfg_vertex_label_exec_domain(dvl) = make_predicate(new_sc);
   }
}

/*============================================================================*/
/* Psysteme make_expression_equalities(list le): returns a Psysteme that
 * has equalities computed from "le", a list of expressions.
 */
Psysteme make_expression_equalities(list le)
{
 Psysteme new_sc;
 list l;

 new_sc = sc_new();

 for(l = le; l != NIL; l = CDR(l))
   {
    expression exp = EXPRESSION(CAR(l));
    normalized nexp = NORMALIZE_EXPRESSION(exp);

    if(normalized_linear_p(nexp))
      {
       Pvecteur new_vec = (Pvecteur) normalized_linear(nexp);
       sc_add_egalite(new_sc, contrainte_make(new_vec));
      }
    else
      {
       printf("\nNon linear expression :");
       printf(" %s\n", words_to_string(words_expression(exp,NIL)));
      }
   }
 sc_creer_base(new_sc);
 return(new_sc);
}

/*============================================================================*/
/* static list find_el_with_num(int stmt): returns the englobing loops list
 * corresponding to the statement number "stmt".
 *
 * This function uses an hash table, which contains the static_control
 * associated to each statement.  */
static list find_el_with_num(int stmt)
{
 hash_table scm = get_current_stco_map();

 static_control stct = (static_control) hash_get(scm, (char *) ((long)stmt));

 return(static_control_loops(stct));
}

/*============================================================================*/
/* Pbase make_base_of_nest(int stmt): returns the Pbase that contains the
 * indices of the englobing loops of "stmt".
 */
Pbase make_base_of_nest(int stmt)
{
 Pbase new_b = NULL;
 list el_l;

 for(el_l = find_el_with_num(stmt) ; el_l != NIL; el_l = CDR(el_l))
   vect_add_elem((Pvecteur *) &new_b,
		 (Variable) loop_index(LOOP(CAR(el_l))),
		 VALUE_ONE);

 return(new_b);
}

/*============================================================================*/
/* successor first_succ_of_vertex(vertex v): returns the first successor of
 * "v".
 */
successor first_succ_of_vertex(vertex v)
{
 return(SUCCESSOR(CAR(vertex_successors(v))));
}


/*============================================================================*/
/* dataflow first_df_of_succ(successor s): returns the first dataflow of
 * "s".
 */
dataflow first_df_of_succ(s)
successor s;
{
 return(DATAFLOW(CAR(dfg_arc_label_dataflows((dfg_arc_label)
successor_arc_label(s)))));
}

/*============================================================================*/
/* loop loop_dup(loop l): returns the duplication of "l".
 */
loop loop_dup(loop l)
{
  /*
 loop new_loop;

 new_loop = make_loop(loop_index(l), range_dup(loop_range(l)), loop_body(l),
		      loop_label(l), loop_execution(l), loop_locals(l));

 return(new_loop);
  */
  return copy_loop(l);
}


 /* package mapping : Alexis Platonoff, july 1993 */

/*============================================================================*/
/* list static_control_to_indices(static_control stct): returns the list of
 * the loop indices (entities) corresponding to the list of loops contained in
 * "stct". The list of indices is in the same order than the list of loop, i.e.
 * for example, if "stct" contains a list of loops like (LOOP1, LOOP3, LOOP5,
 * LOOP2) then the list of indices will be (I1, I3, I5, I2).
 */
list static_control_to_indices(stct)
static_control stct;
{
 list lo_l = static_control_loops(stct);
 list ind_l = NIL;

 for( ; lo_l != NIL; lo_l = CDR(lo_l))
   {
    loop lo = LOOP(CAR(lo_l));

    /* We keep the same order. */
    ind_l = gen_nconc(ind_l, CONS(ENTITY, loop_index(lo), NIL));
   }
 return(ind_l);
}




/* ======================================================================== */
/*
 * Pvecteur polynome_to_vecteur(Ppolynome pp)
 *
 * Translates a polynome "pp" into a vector. This is only possible if "pp"
 * is of degree one (1).
 */
Pvecteur polynome_to_vecteur(Ppolynome pp)
{
  Pvecteur new_pv = VECTEUR_NUL;
  Ppolynome ppp;

  for(ppp = pp; ppp != NULL; ppp = ppp->succ) {
    entity var;
    Value val;
    Pvecteur pv = (ppp->monome)->term;

    if(VECTEUR_NUL_P(pv))
      pips_internal_error("A null vector in a monome");
    else if(pv->succ != NULL)
      pips_internal_error("Polynome is not of degree one");

    var = (entity) pv->var;
    val = float_to_value((ppp->monome)->coeff);
    vect_add_elem(&new_pv, (Variable) var, val);
  }
  return(new_pv);
}

/* ======================================================================== */
/*
 * Pcontrainte polynome_to_contrainte(Ppolynome pp)
 *
 * Translates a polynome "pp" into a constraint (inequality or equality,
 * depends on its usage). This is only possible if "pp" is of degree one (1)
 * because we tranform it into a Pvecteur.
 *
 * If you want an inequality, it will be: pp <= 0
 */
Pcontrainte polynome_to_contrainte(pp)
Ppolynome pp;
{
 return(contrainte_make(polynome_to_vecteur(pp)));
}


/*============================================================================*/
/* Psysteme polynome_to_sc(Ppolynome pp, list l): returns a system of
 * equalities ("new_ps") computed from a polynome "pp" and a list of variables
 * "l".
 *
 * This list gives the variables of the polynome for which we need to nullify
 * the factor. Thus, the resulting system contains the equations that nullify
 * these factors (the degree of the polynome must be less or equal to two).
 *
 * When all these equations are computed, the remaining polynome, from each we
 * have removed all the occurences of these variables, is also nullify and the
 * equation added to the system (then, this remnant must be of degree 1).
 */
Psysteme old_polynome_to_sc(pp, l)
Ppolynome pp;
list l;
{
 Ppolynome aux_pp = polynome_dup(pp);
 Psysteme new_ps = sc_new();

 /* For each variable, we nullify its factor in the polynome. */
 for( ; l != NIL; l = CDR(l))
   {
    /* We get the current variable. */
    entity var = ENTITY(CAR(l));

    /* We get its factor in the polynome. */
    Ppolynome pp_fac = polynome_factorize(aux_pp, (Variable) var, 1);

    /* We add a new equality in the system. */
    sc_add_egalite(new_ps, polynome_to_contrainte(pp_fac));

    /* We delete the occurences of this variable in the polynome. */
    aux_pp = prototype_var_subst(aux_pp, (Variable) var, POLYNOME_NUL);
   }
 /* The remnant is added to the system. */
 sc_add_egalite(new_ps, polynome_to_contrainte(aux_pp));

 sc_creer_base(new_ps);
 return(new_ps);
}


/*============================================================================*/
/* Psysteme new_polynome_to_sc(Ppolynome pp, list l): returns a system of
 * equalities ("new_ps") computed from a polynome "pp" and a list of variables
 * "l".
 *
 * This list gives the variables of the polynome for which we need to nullify
 * the factor. Thus, the resulting system contains the equations that nullify
 * these factors (the degree of the polynome must be less or equal to two).
 *
 * When all these equations are computed, the remaining polynome, from each we
 * have removed all the occurences of these variables, is also nullify and the
 * equation added to the system (then, this remnant must be of degree 1).
 */
Psysteme polynome_to_sc(pp, l)
Ppolynome pp;
list l;
{
 Ppolynome aux_pp = polynome_dup(pp);
 Psysteme new_ps = sc_new();

 /* For each variable, we nullify its factor in the polynome. */
 for( ; l != NIL; l = CDR(l))
   {
    /* We get the current variable. */
    entity var = ENTITY(CAR(l));

    /* We get its factor in the polynome. */
    Pvecteur pv_fac = prototype_factorize(aux_pp, (Variable) var);

    if(!VECTEUR_NUL_P(pv_fac)) {
      sc_add_egalite(new_ps, contrainte_make(pv_fac));

      /* We delete the occurences of this variable in the polynome. */
      aux_pp = prototype_var_subst(aux_pp, (Variable) var, POLYNOME_NUL);
    }
   }
 /* The remnant is added to the system. */
 sc_add_egalite(new_ps, polynome_to_contrainte(aux_pp));

 sc_creer_base(new_ps);
 return(new_ps);
}


/*============================================================================*/
/* void substitute_var_with_vec(Psysteme ps, entity var, int val, Pvecteur vec):
 * Substitutes in a system ("ps") a variable ("var"), factor of a positive
 * value ("val"), by an expression ("vec").
 *
 * This substitution is done on all assertions of the system (equalities and
 * inequalities). For each assertion (represented by a vector Vold) we have:
 *
 *      Vold = c*var + Vaux
 *      val*var = vec
 *
 * Vnew represents the new assertion.  With: p = pgcd(c, val) >= 1, we have:
 *
 *	Vnew = (c/p)*vec + (val/p)*Vaux = (c/p)*vec + (val/p)*(Vold - c*var)
 *
 * Note: we have: Vold == 0 <=> (val/p)*Vold == 0
 *                Vold > 0 <=> (val/p)*Vold > 0
 *                ...
 *
 * because "val" is positive.
 */
void substitute_var_with_vec(ps, var, val, vec)
Psysteme ps;
entity var;
Value val;
Pvecteur vec;
{
 Variable Var = (Variable) var;
 Pcontrainte assert;

 ifdebug(7) {
fprintf(stdout, "\t\t\tAvant Sub: \n");
fprint_psysteme(stdout, ps);
fprintf(stdout, "\n");
 }

  /* "val" must be positive. */
  if(value_neg_p(val)) {
      value_oppose(val);
      vect_chg_sgn(vec);
  }

  /* Vnew = (c/p)*vec + (val/p)*Vaux = (c/p)*vec + (val/p)*(Vold - c*var) */
  for(assert = ps->egalites; assert != NULL; assert = assert->succ) {
    Pvecteur v_old = assert->vecteur;
    Value coeff = vect_coeff(Var, v_old);
    if(value_notzero_p(coeff)) {
	Value p = pgcd_slow(coeff, val);

	assert->vecteur = vect_cl2_ofl_ctrl
	    (value_div(coeff,p), vec,
	     value_div(val,p),
	     vect_cl2_ofl_ctrl(VALUE_ONE, v_old, VALUE_MONE,
			       vect_new(Var, coeff),
			       NO_OFL_CTRL),
	     NO_OFL_CTRL);
    }
  }
  for(assert = ps->inegalites; assert != NULL; assert = assert->succ) {
    Pvecteur v_old = assert->vecteur;
    Value coeff = vect_coeff(Var, v_old);
    if(value_notzero_p(coeff)) {
	Value p = pgcd_slow(coeff, val);

      assert->vecteur = vect_cl2_ofl_ctrl
	  (value_div(coeff,p), vec,
	   value_div(val,p),
	   vect_cl2_ofl_ctrl(VALUE_ONE, v_old, VALUE_MONE,
			     vect_new(Var, coeff),
			     NO_OFL_CTRL),
	   NO_OFL_CTRL);
    }
  }
  vect_rm((Pvecteur) ps->base);
  ps->base = (Pbase) NULL;
  sc_creer_base(ps);

  ifdebug(7) {
     fprintf(stdout, "\t\t\tApres Sub: \n");
     fprint_psysteme(stdout, ps);
     fprintf(stdout, "\n");
 }

}


/*============================================================================*/
/* bool is_entity_in_list_p(entity e, list l): returns true if entity "e" is
 * in the list of entities "l", false otherwise.
 */
/* FI: many similar functions. See ri-util/arguments.c which deals
   with entity lists, i.e. entities. */
bool is_entity_in_list_p(entity e, list l)
{
 bool is_in_list = false;
 for( ; (l != NIL) && (! is_in_list); l = CDR(l))
    if(same_entity_p(e, ENTITY(CAR(l))))
       is_in_list = true;
 return(is_in_list);
}

/*============================================================================*/
/* Psysteme elim_var_with_eg(Psysteme ps, list *init_l, list *elim_l):
 * Computes the elimination of variables in the system "ps" using its
 * equalities. All the used equalities are removed directly from "ps".
 *
 * However, we keep all these "used" equalities in "sc_elim". The return value
 * of this function is this system.
 *
 * Initially, "init_l" gives the list of variables that can be eliminated and
 * "elim_l" is empty. At the end, "init_l" contains the variables that are
 * not eliminated and "elim_l" contains the variables that are eliminated.
 *
 * At the end, to each equality of "sc_elim" will be associated a variable
 * of "elim_l". These lists are built so as to be able to match them directly.
 * Each variable of "elim_l" appears in one constraint of "sc_elim" and only
 * one. There will be as many equalities in "sc_elim" as variables in "elim_l"
 *
 * A variable can be eliminated using a given equality if it appears in this
 * equality, i.e. its associated coefficient is not equal to zero. Moreover, it
 * is easier to eliminate a variable with a value of 1 or -1. So, first we
 * try to find such a variable.
 *
 * Algo:
 * ----
 * BEGIN
 * vl = init_l;
 * el = NIL;
 * eqs = list of equalities of ps
 * sc_elim = system with no constraints
 * while eqs is not empty
 *   init_vec = vector of the first equality of eqs
 *   var_not_found = TRUE
 *   coeff_one_not_found = TRUE
 *   l = vl
 *   while coeff_one_not_found and l is not empty
 *     crt_var = first variable of l
 *     crt_val = its value in init_vec
 *     if crt_val is 1 or -1
 *       coeff_one_not_found = FALSE
 *       var_not_found = FALSE
 *       (var, val) = (crt_var, crt_val)
 *     else if crt_val is not 0 and var_not_found
 *       var_not_found = FALSE
 *       (var, val) = (crt_var, crt_val)
 *     end if
 *     l = CDR(l)
 *   end while
 *   if var_not_found is false (means that a variable has been found)
 *     (var, val) = variable and its value to eliminate in init_vec
 *     remove var from vl
 *     add var to el
 *     pv_elim = val*var - init_vec
 *     substitute val*var by pv_elim in ps
 *     substitute val*var by pv_elim in sc_elim
 *     add init_vec to sc_elim
 *     eqs = new list of equalities of ps
 *   else
 *     eqs = CDR(eqs)
 *   end if
 * end while
 * init_l = vl
 * elim_l = el
 * END
 *
 * Note: the substitution of val*var by pv_elim in a vector V uses the gcd:
 *	V = c*var + Vaux
 *      p = gcd(val,c)
 *      Vnew = (c/p)*pv_elim + (val/p)*Vaux
 *
 * BUG: reuse of freed memory.
 */
Psysteme elim_var_with_eg(Psysteme ps, list * init_l, list * elim_l)
{
    Psysteme sc_elim = sc_new();
    Pcontrainte eqs;
    list vl = *init_l,	/* We use "vl" during the computation, not *init_l */
    el = NIL,	/* We use "el" during the computation, not *elim_l */
    l;

    /* Nothing do if there is no variable to eliminate */
    if(!ENDP(vl)) {
	/* This elimination works only on equalities. While there remains
	 * equalities, we can eliminate variables.  */
	eqs = ps->egalites;
	while(eqs != NULL)
	{
	    bool coeff_one_not_found, var_found;
	    entity var = entity_undefined;
	    Value val = VALUE_ZERO;
	    Pvecteur init_vec, pv_elim;
	    Pcontrainte next = eqs->succ;

	    init_vec = vect_dup(eqs->vecteur);

	    /* We look, in vl (i.e. init_l), for a variable that we can
	     * eliminate in init_vec, i.e. with a coefficient not equal to
	     * 0. We prefer a coefficient * equal to 1 or -1, so we scan
	     * all the equality. We take the first * variable of "init_l"
	     * that has a coeff of 1 or -1. If there is no such *
	     * variable, we take the first with a coeff not equal to zero.
	     */
	    var_found = false;
	    coeff_one_not_found = true;

	    for(l = vl ; (l != NIL) && coeff_one_not_found; l = CDR(l))
	    {
		entity crt_var = ENTITY(CAR(l));
		Value crt_val = vect_coeff((Variable) crt_var, init_vec);

		if(value_one_p(crt_val) || value_mone_p(crt_val))
		{
		    coeff_one_not_found = false;
		    var_found = true;
		    var = crt_var;
		    val = crt_val;
		}
		else if((value_notzero_p(crt_val)) && !var_found)
		{
		    var_found = true;
		    var = crt_var;
		    val = crt_val;
		}
	    }

	    if(var_found) /* If we get such a variable, we eliminate it. */
	    {
		/* First, we remove it from "vl". */
		gen_remove(&vl, (void *) var);

		/* Then, we add it to "el". */
		el = CONS(ENTITY, var, el);

		/* We compute the expression (pv_elim) by which we are
		 * going to substitute our variable (var):
		 *
		 * We have: val*var = pv_elim
		 *
		 * The equality is: V = 0, with: V = val*var + Vaux
		 *
		 * So, we have: pv_elim = -Vaux, with: Vaux = V - val*var
		 *
		 * So: pv_elim = val*var - V
		 *
		 * ??? memory leak...
		 */
		pv_elim = vect_cl2_ofl_ctrl
		    (VALUE_MONE, vect_dup(init_vec),
		     VALUE_ONE,  vect_new((Variable) var, val),
		     NO_OFL_CTRL);

		/* substitute "val*var" by its value (pv_elim) in the system */
		substitute_var_with_vec(ps, var, val, vect_dup(pv_elim));
		/*ps = sc_normalize(ps);*/
		ps = sc_elim_db_constraints(ps);

		/* We substitute var by its value (pv_elim) in "sc_elim". */
		substitute_var_with_vec(sc_elim, var, val, vect_dup(pv_elim));

		/* The initial equality is added to "sc_elim". */
		sc_add_egalite(sc_elim, contrainte_make(vect_dup(init_vec)));

		/* We reinitialize the list of equalities. */
		eqs = ps->egalites;
	    }
	    /* Else, we try on the next equality. */
	    else
		eqs = next;

	    vect_rm(init_vec);
	}
    }
    *init_l = vl;
    *elim_l = el;
    sc_elim->base = NULL;
    sc_creer_base(sc_elim);

    return(sc_elim);
}


/*============================================================================*/
/* Psysteme better_elim_var_with_eg(Psysteme ps, list *init_l, list *elim_l):
 * Computes the elimination of variables in the system "ps" using its
 * equalities. All the used equalities are removed directly from "ps".
 *
 * However, we keep all these "used" equalities in "sc_elim". The return value
 * of this function is this system.
 *
 * Initially, "init_l" gives the list of variables that can be eliminated and
 * "elim_l" is empty. At the end, "init_l" contains the variables that are
 * not eliminated and "elim_l" contains the variables that are eliminated. We
 * keep the order of "init_l" in our process of elimination.
 *
 * At the end, to each equality of "sc_elim" will be associated a variable
 * of "elim_l". These lists are built so as to be able to match them directly.
 * Each variable of "elim_l" appears in one constraint of "sc_elim" and only
 * one. There will be as many equalities in "sc_elim" as variables in "elim_l"
 *
 * A variable can be eliminated using a given equality if it appears in this
 * equality, i.e. its associated coefficient is not equal to zero. Moreover,
 * for a given variable, we look for the equation in which it has the smallest
 * coefficient.
 *
 * Algo:
 * ----
 * BEGIN
 * vl = a copy of init_l;
 * el = NIL;
 * sc_elim = system with no constraints;
 * loop over the list of variables to eliminate vl
 *   v = current variable;
 *   eq = the equality of ps in which the coefficient of v is the smallest
 *   if eq is not NULL
 *     eq is taken off the list of equalities of ps
 *     loop over the list of equalities of ps
 *       eg = current equality
 *       substitute in eg the value of v given by eq
 *     end loop
 *     loop over the list of inequalities of ps
 *       eg = current inequality
 *       substitute in eg the value of v given by eq
 *     end loop
 *     loop over the list of equalities of sc_elim
 *       eg = current equality
 *       substitute in eg the value of v given by eq
 *     end loop
 *     loop over the list of inequalities of sc_elim
 *       eg = current inequality
 *       substitute in eg the value of v given by eq
 *     end loop
 *     add eq in the list of equalities of sc_elim
 *     remove v from init_l
 *     add v in el
 *   end if
 * end loop
 *
 * elim_l = el
 * END
 *
 * Note: Here is how we "substitute in eg the value of v given by eq":
 *	if eg and eq are as follows (Vaux and Vsub do not contained v):
 *		eg <=> c*v + Vaux = 0
 *      	eq <=> val*v = Vsub
 *      let p = gcd(val,c)
 *      after the substitution, we have:
 *		eg <=> (c/p)*Vsub + (val/p)*Vaux = 0
 */
Psysteme better_elim_var_with_eg(ps, init_l, elim_l)
Psysteme ps;
list *init_l, *elim_l;
{
  Psysteme sc_elim = sc_new();

  /* During the computation, we modify *init_l, so we duplicate it.
   * We use "el" not *elim_l, which should be  empty at the beginning.
   */
  list vl = gen_append(*init_l, NIL),
       el = NIL,
       l;
  Pcontrainte eq, eg;

  for(l = vl; !ENDP(l); POP(l)) {
    Variable v = (Variable) ENTITY(CAR(l));
    Value coeff;

    if ((eq = contrainte_var_min_coeff(ps->egalites,v, &coeff, true))
	!= NULL) {

 if(get_debug_level() > 7) {
fprintf(stderr, "System is :");
fprint_psysteme(stderr, ps);
fprintf(stderr, "\t\tElim var %s in equation:", entity_local_name((entity) v));
pu_vect_fprint(stderr, eq->vecteur);
fprintf(stderr, "\n");
 }

      if(!egalite_normalize(eq))
	pips_internal_error("Strange equality");

      sc_nbre_egalites(ps)--;
      if (eq == (ps->egalites)) ps->egalites = eq->succ;
      /* si eq etait en tete il faut l'enlever de la liste, sinon, eq a
         ete enleve par la fonction contrainte_var_min_coeff(). */

      for(eg = ps->egalites; eg != NULL; eg = eg->succ)
	(void) contrainte_subst_ofl_ctrl(v, eq, eg, true, NO_OFL_CTRL);
      for(eg = ps->inegalites; eg != NULL; eg = eg->succ)
	(void) contrainte_subst_ofl_ctrl(v, eq, eg, false, NO_OFL_CTRL);

      for(eg = sc_elim->egalites; eg != NULL; eg = eg->succ)
	(void) contrainte_subst_ofl_ctrl(v, eq, eg, true, NO_OFL_CTRL);
      for(eg = sc_elim->inegalites; eg != NULL; eg = eg->succ)
	(void) contrainte_subst_ofl_ctrl(v, eq, eg, false, NO_OFL_CTRL);

      sc_add_egalite(sc_elim, eq);
      gen_remove(init_l, (void *) v);
      el = CONS(ENTITY, (entity) v, el);
    }
  }

  *elim_l = el;
  sc_elim->base = NULL;
  sc_creer_base(sc_elim);

  ifdebug(7) {
fprintf(stderr, "[new_elim_var_with_eg] Results:\n");
fprintf(stderr, "Elim sys:\n");
fprint_entity_list(stderr, el);
fprint_psysteme(stderr, sc_elim);
fprintf(stderr, "Remnants sys:\n");
fprint_entity_list(stderr, *init_l);
fprint_psysteme(stderr, ps);
fprintf(stderr, "\n");
 }

  return(sc_elim);
}


/*============================================================================*/
/* bool single_var_vecteur_p(Pvecteur pv): returns true if the vector "pv"
 * contains only one element.
 *
 * Note: This element should not be a constant term (this is not tested).
 */
bool single_var_vecteur_p(Pvecteur pv)
{
 return(vect_size(pv) == 1);
}


/*============================================================================*/
/* list vecteur_to_list(Pvecteur v): translates a Pvecteur into a list of
 * entities, in the same order.
 */
/* FI: same comment as above: to be moved */
list vecteur_to_list(Pvecteur v)
{
 list l = NIL;

 for( ; v != NULL; v = v->succ)
   {
    entity var = (entity) v->var;
    if(var != (entity) TCST)
       l = gen_nconc(l, CONS(ENTITY, var, NIL));
   }

 return(l);
}




/*============================================================================*/
/* Ppolynome old_vecteur_to_polynome(Pvecteur vec): translates a Pvecteur into a
 * Ppolynome.
 */
/* FI: To be moved*/
Ppolynome old_vecteur_to_polynome(Pvecteur vec)
{
 Ppolynome pp_new = POLYNOME_NUL;

 for( ; vec != NULL; vec = vec->succ)
    polynome_add(&pp_new,
		 make_polynome(VALUE_TO_FLOAT(vec->val), vec->var, VALUE_ONE));

 return(pp_new);
}




/*============================================================================*/
/* list meld(list l1, list l2, bool (*compare_obj)()):
 */
list meld(l1, l2, compare_obj)
list l1, l2;
bool (*compare_obj)();
{
  if( ENDP(l1) ) {
    return(l2);
  }
  else if( ENDP(l2) ) {
    return(l1);
  }
  else if(compare_obj(CHUNK(CAR(l1)), CHUNK(CAR(l2)))) {
    return(gen_nconc(CONS(CHUNK, CHUNK(CAR(l1)), NIL),
		     meld(CDR(l1), l2, compare_obj)));
  }
  else {
    return(gen_nconc(CONS(CHUNK, CHUNK(CAR(l2)), NIL),
		     meld(l1, CDR(l2), compare_obj)));
  }
}


/*============================================================================*/
/* list general_merge_sort(list l, bool (*compare_obj)()): returns the
 * result of sorting the list "l" using the comparison function
 * "compare_obj". This bool function should retuns true if its first
 * argument has to be placed before its second argument in the sorted
 * list, else FALSE.
 *
 * This is a generic function that accepts any homogene list of newgen
 * objects. The comparison function must be coded by the user, its
 * prototype should be: bool my_compare_obj(chunk * obj1, chunk *
 * obj2);
 *
 * This function uses the merge sort algorithm which has a mean and worst
 * case complexity of n*log(n).  There are two steps.  First, we look for
 * sublists that are in the right order, each time we found one we put it
 * in a list of lists, a LIFO queue ("head" is the out point, "tail" is
 * the in point).  Second, we take the first two lists of our LIFO queue
 * (i.e. at the "head") and meld then into one list that we put again in
 * our LIFO (i.e.  at the "tail"). We continue until there remains only
 * one list in our LIFO queue, which in that case is the sorted list we
 * wanted.
 */

/*  hack to preserve the bool comparison while using qsort...
 */
static bool (*bool_compare_objects)();
static int compare_objects(p1, p2)
gen_chunk **p1, **p2;
{
    return(bool_compare_objects(*p2, *p1)-bool_compare_objects(*p1, *p2));
}

/* no way validate Prgm_mapping with this function...
 */
list new_general_merge_sort(l, compare_obj)
list l;
bool (*compare_obj)();
{
    bool (*current)();
    list lnew = gen_copy_seq(l);

    /* push
     */
    current = bool_compare_objects,
    bool_compare_objects = compare_obj;

    /* sort
     */
    gen_sort_list(lnew, compare_objects);

    /* pop
     */
    bool_compare_objects = current;

    return(lnew);
}

/*  I guess there is some kind of memory leaks here...
 *  I tried the one above, but I could not go thru the validation...
 *
 *  FC 02/01/94
 */
list general_merge_sort(l, compare_obj)
list l;
bool (*compare_obj)();
{
  list ch1, ch2, ch, ch_t, aux_l, head = NIL, tail = NIL;
  void * crt_obj, * prev_obj;

  pips_debug(9, "Debut\n");

  if( ENDP(l) || ENDP(CDR(l)) )
    return(l);

  /* First step */
  ch = l;       /* current sublist containing sorted objects */
  ch_t = ch;    /* tail of "ch" */
  prev_obj = CHUNK(CAR(l));
  for(aux_l = CDR(ch_t); !ENDP(aux_l); aux_l = CDR(ch_t), prev_obj = crt_obj) {
   crt_obj = CHUNK(CAR(aux_l));
    if(compare_obj(crt_obj, prev_obj)) {
      /* Current sublist stops here, we put it in our LIFO queue and ... */
      if(tail == NIL) {
	head = CONS(CONSP, ch, NIL);
	tail = head;
      }
      else {
	CDR(tail) = CONS(CONSP, ch, NIL);
	POP(tail);
      }

      /* ... we reinitialize our current sublist. */
      ch = CDR(ch_t);
      CDR(ch_t) = NIL;
      ch_t = ch;
    }
    else
      /* Else, current sublist increases. */
      POP(ch_t);
  }
  /* The last current sublist is put in our LIFO queue. */
  if(tail == NIL) {
    head = CONS(CONSP, ch, NIL);
    tail = head;
  }
  else {
    CDR(tail) = CONS(CONSP, ch, NIL);
    POP(tail);
  }

  /* Second step */
  for( ; ! ENDP(CDR(head)) ; ) {
    ch1 = CONSP(CAR(head));
    POP(head);
    ch2 = CONSP(CAR(head));
    POP(head);
    ch = meld(ch1, ch2, compare_obj);

    if(head == NIL) {
      head = CONS(CONSP, ch, NIL);
      tail = head;
    }
    else {
      CDR(tail) = CONS(CONSP, ch, NIL);
      POP(tail);
    }
  }
  pips_debug(9, "Fin\n");

  return(CONSP(CAR(head)));
}


/* ======================================================================== */
/* expression rational_op_exp(char *op_name, expression exp1 exp2):
 * Returns an expression containing the operation "op_name" between "exp1"
 * and "exp2".
 * "op_name" must be one of the four classic operations : +, -, * or /.
 *
 * If both expressions are integer constant values and the operation
 * result is an integer then the returned expression contained the
 * calculated result, but this calculus is a rational one, not an integer one
 * as in make_op_exp().
 *
 * Else, we treat five special cases :
 *       _ exp1 and exp2 are integer linear and op_name is + or -.
 *         This case is resolved by make_lin_op_exp().
 *       _ exp1 = 0
 *       _ exp1 = 1
 *       _ exp2 = 0
 *       _ exp2 = 1
 *
 * Else, we create a new expression with a binary call.
 *
 * Note: The function MakeBinaryCall() comes from Pips/.../syntax/expression.c
 *       The function int_to_expression() comes from ri-util.
 * Note: This function is almost equivalent to make_op_exp() but for the
 * rational calculus.
 */
/* FI: to be moved in ri-util/expression.c */
expression rational_op_exp(string op_name, expression exp1, expression exp2)
{
  expression result_exp = expression_undefined;
  entity op_ent, unary_minus_ent;

  pips_debug(7, "doing\n");
  op_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						   op_name), entity_domain);
  unary_minus_ent =
    gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
					    UNARY_MINUS_OPERATOR_NAME),
		       entity_domain);

  pips_debug(5, "begin OP EXP : %s  %s  %s\n",
	     words_to_string(words_expression(exp1,NIL)),
	     op_name,
	     words_to_string(words_expression(exp2,NIL)));

  if( ! ENTITY_FOUR_OPERATION_P(op_ent) )
    user_error("rational_op_exp", "operation must be : +, -, * or /");

  if( expression_constant_p(exp1) && expression_constant_p(exp2) ) {
    int val1, val2;

    pips_debug(6, "Constant expressions\n");
    val1 = expression_to_int(exp1);
    val2 = expression_to_int(exp2);

    if (ENTITY_PLUS_P(op_ent))
      result_exp = int_to_expression(val1 + val2);
    else if(ENTITY_MINUS_P(op_ent))
      result_exp = int_to_expression(val1 - val2);
    else if(ENTITY_MULTIPLY_P(op_ent))
      result_exp = int_to_expression(val1 * val2);
    else { /* ENTITY_DIVIDE_P(op_ent) */
      /* rational calculus */
      if((val1 % val2) == 0)
	result_exp = int_to_expression((int) (val1 / val2));
    }
  }
  else {
    /* We need to know the integer linearity of both expressions. */
    normalized nor1 = NORMALIZE_EXPRESSION(exp1);
    normalized nor2 = NORMALIZE_EXPRESSION(exp2);

    if((normalized_tag(nor1) == is_normalized_linear) &&
       (normalized_tag(nor2) == is_normalized_linear) &&
       (ENTITY_PLUS_P(op_ent) || ENTITY_MINUS_P(op_ent)) ) {
      pips_debug(6, "Linear operation\n");

      result_exp = make_lin_op_exp(op_ent, exp1, exp2);
    }
    else if(expression_equal_integer_p(exp1, 0)) {
      if (ENTITY_PLUS_P(op_ent))
	result_exp = exp2;
      else if(ENTITY_MINUS_P(op_ent))
	result_exp = MakeUnaryCall(unary_minus_ent, exp2);
      else /* ENTITY_MULTIPLY_P(op_ent) || ENTITY_DIVIDE_P(op_ent) */
	result_exp = int_to_expression(0);
    }
    else if(expression_equal_integer_p(exp1, 1)) {
      if(ENTITY_MULTIPLY_P(op_ent))
	result_exp = exp2;
    }
    else if(expression_equal_integer_p(exp2, 0)) {
      if (ENTITY_PLUS_P(op_ent) || ENTITY_MINUS_P(op_ent))
	result_exp = exp1;
      else if (ENTITY_MULTIPLY_P(op_ent))
	result_exp = int_to_expression(0);
      else /* ENTITY_DIVIDE_P(op_ent) */
	user_error("rational_op_exp", "division by zero");
    }
    else if(expression_equal_integer_p(exp2, 1)) {
      if(ENTITY_MULTIPLY_P(op_ent) || ENTITY_DIVIDE_P(op_ent))
	result_exp = exp1;
    }

    /* Both expressions are unnormalized because they might be reused in
     * an unnormalized expression. */
    unnormalize_expression(exp1);
    unnormalize_expression(exp2);
  }

  if(result_exp == expression_undefined)
    result_exp = MakeBinaryCall(op_ent, exp1, exp2);

  pips_debug(5, "end   OP EXP : %s\n",
	     words_to_string(words_expression(result_exp,NIL)));

  return (result_exp);
}


/*==================================================================*/
/* Pvecteur vect_var_subst(vect,var,new_vect): substitute in the
 * vector "vect", the variable "var" by new_vect.
 *  (vect) = (val)x(var) + (vect_aux)
 *                     => (vect) = (val)x(new_vect) + (vect_aux)
 *
 * AC 93/12/06
 */

Pvecteur vect_var_subst(vect,var,new_vect)

 Pvecteur    vect,new_vect;
 Variable    var;
{
 Pvecteur    vect_aux;
 Value       val;

 if ((val = vect_coeff(var,vect)) != 0)
    {
     vect_erase_var(&vect,var);
     vect_aux = vect_multiply(new_vect,val);
     vect = vect_add(vect, vect_aux);
    }

 return(vect);
}

/*==================================================================*/
/*
 * Ppolynome prototype_var_subst(Ppolynome pp, Variable var,
 *				 Ppolynome ppsubst)
 *
 * Substitutes in polynome "pp" variable "var" by polynome "ppsubst".
 * "pp" is a prototype polynome, i.e. a linear fonction with symbolic
 * constants. Thus, there is no squared variables.
 *
 * This function takes advantage of that fact only when ppsubst in equal
 * to zero, else it uses polynome_var_subst().
 */
Ppolynome prototype_var_subst(pp, var, ppsubst)
Ppolynome pp;
Variable var;
Ppolynome ppsubst;
{
  Ppolynome newpp;

  if(POLYNOME_NUL_P(ppsubst)) {
    Ppolynome ppp, p = POLYNOME_UNDEFINED;
    Pvecteur pv;

    newpp = polynome_dup(pp);
    for(ppp = newpp; ppp != NULL; ppp = ppp->succ) {
      entity first = entity_undefined,
             second = entity_undefined;
      pv = (ppp->monome)->term;
      for(; (pv != NULL) && (second == entity_undefined); pv = pv->succ) {
	second = first;
	first = (entity) pv->var;
      }
      if(pv != NULL)
	pips_internal_error("Vecteur should contains 2 var");
      else if( same_entity_p(first,  (entity) var) ||
	       same_entity_p(second, (entity) var)) {
	if(POLYNOME_UNDEFINED_P(p)) {
	  newpp = ppp->succ;
	}
	else
	  p->succ = ppp->succ;
      }
      else
	p = ppp;
    }
  }
  else
    newpp = polynome_var_subst(pp, var, ppsubst);
  return(newpp);
}


/* ======================================================================== */
/*
 * Ppolynome vecteur_mult(Pvecteur v1, v2)
 *
 * Does the multiplication of two vectors, the result is a polynome.
 */
Ppolynome vecteur_mult(Pvecteur v1, Pvecteur v2)
{
  Ppolynome pp = POLYNOME_NUL, new_pp;
  Pvecteur pv1, pv2, ppv;

  if(VECTEUR_NUL_P(v1) || VECTEUR_NUL_P(v2))
    return(POLYNOME_NUL);

  for(pv1 = v1; pv1 != NULL; pv1 = pv1->succ) {
    Variable var1 = pv1->var;
    Value val1 = pv1->val;
    for(pv2 = v2; pv2 != NULL; pv2 = pv2->succ) {
      Variable var2 = pv2->var;
      Value val2 = pv2->val;
      Value p = value_mult(val1,val2);
      float f = VALUE_TO_FLOAT(p);

      if(var1 == TCST)
	new_pp = make_polynome(f, var2, VALUE_ONE);
      else if(var2 == TCST)
	new_pp = make_polynome(f, var1, VALUE_ONE);
      else if(same_entity_p((entity) var1, (entity) var2))
	new_pp = make_polynome(f, var1, VALUE_CONST(2));
      else {
	new_pp = make_polynome(f, var1, VALUE_ONE);
	ppv = (new_pp->monome)->term;
	pips_assert("succ is NULL", ppv->succ == NULL);
	ppv->succ = vect_new(var2, VALUE_ONE);
      }
      polynome_add(&pp, new_pp);
    }
  }
  return(pp);
}


/* ======================================================================== */
/*
 * Pvecteur prototype_factorize(Ppolynome pp, Variable var)
 * returns the (linear) coefficient of var in polynomial pp.
 * "pp" is a prototype polynome, i.e. a linear fonction with symbolic
 * constants. Thus, there is no squared variables.
 *
 * This function takes plainly advantage of this property, and then is much
 * faster than polynome_factorize().
 */
Pvecteur prototype_factorize(Ppolynome pp, Variable var)
{
    Pvecteur pv = NULL;

    if(POLYNOME_NUL_P(pp))
	pv = VECTEUR_NUL;
    else if(var == TCST)
    {
	float f = polynome_TCST(pp);
	pv = vect_new(TCST, float_to_value(f));
    }
    else {
    Ppolynome ppp;

    for(ppp = pp; ppp != NULL; ppp = ppp->succ) {
      Variable newvar = VARIABLE_UNDEFINED;
      Value newval;
      Pvecteur vec, newpv;
      entity first = entity_undefined, second = entity_undefined;
      bool factor_found = true;

      vec = (ppp->monome)->term;
      for(; (vec != NULL) && (second == entity_undefined); vec = vec->succ) {
	second = first;
	first = (entity) vec->var;
      }
      if(vec != NULL)
	pips_internal_error("Vecteur should contains 2 var");
      else if(same_entity_p(first,  (entity) var))
	if(second == entity_undefined)
	  newvar = TCST;
	else
	  newvar = (Variable) second;
      else if(same_entity_p(second, (entity) var))
	newvar = (Variable) first;
      else
	factor_found = false;

      if(factor_found) {
	newval = float_to_value((ppp->monome)->coeff);
	newpv = vect_new(newvar, newval);
	newpv->succ = pv;
	pv = newpv;
      }
    }
  }

  return(pv);
}


#define MINMAX_REF_NAME "MMREF"

/*=================================================================== */
/*
 * Pcontrainte simplify_minmax_contrainte(Pcontrainte pc, Psysteme
 * ps_cont, int min_or_max) :
 *
 * Parameters :
 *   _ pc : a Pcontrainte, i.e. a list of vectors, noted (V1, ..., Vn)
 *   _ ps_cont : a system of equations, called the context.
 *   _ min_or_max : flag, says in which case we are (MIN or MAX)
 *
 * Result : list of vectors, a Pcontrainte
 *
 * Aims : simplify "pc", i.e. suppress one or more of its vectors. A
 * given vector can be suppressed if it surely smaller (case MAX) or
 * greater (case MIN) than one of the other. The case (MIN or MAX) is
 * given by "min_or_max". If two vectors can not be compared, both are
 * kept. The context allows the user to introduce relationship between
 * variables without which some vectors can not be eliminate.
 *
 * Algorithm : we use a function that eliminates redundances in a
 * system, sc_elim_redund(). This system corresponds to the context
 * augmented of following equations (i in {1, ...n}):
 *   _ Case MAX : Vi - X <= 0
 *   _ Case MIN : X - Vi <= 0
 * X is a special variable that represents the MAX or the MIN of our
 * vectors.  This elimination returns a new system in which we only
 * have to get the equations containing X and remove this variable to
 * obtain our simplified list of vectors.
 *
 * Note : "pc" is not changed, the returned list of vectors is a new one. */

Pcontrainte simplify_minmax_contrainte(pc, ps_cont, min_or_max)
Pcontrainte pc;
Psysteme ps_cont;
int min_or_max;
{
  Pcontrainte newnew_pc, new_pc, apc, pcc = CONTRAINTE_UNDEFINED,
  epc = CONTRAINTE_UNDEFINED;
  Psysteme psc, new_ps;
  entity ref_ent;
  int count;
  string exp_full_name;

  /* We get (or create) our special variable "X". */
  exp_full_name = strdup(concatenate(PAF_UTIL_MODULE_NAME,
				     MODULE_SEP_STRING,
				     MINMAX_REF_NAME, (char *) NULL));
  ref_ent = gen_find_tabulated(exp_full_name, entity_domain);
  if(ref_ent == entity_undefined)
    ref_ent = make_entity(exp_full_name,
			  make_type(is_type_variable,
				    make_variable(make_basic_int(4/*UU*/),
						  NIL, NIL)),
			  make_storage(is_storage_ram, ram_undefined),
			  make_value_unknown());


  /* We put our special variable in our vectors. */
  for(apc = pc, count = 0; apc != NULL; apc = apc->succ, count++) {
    Pvecteur pv;
    Pcontrainte aapc;

    pv = vect_dup(apc->vecteur);
    if(min_or_max == IS_MIN) {
      vect_chg_sgn(pv);
      vect_add_elem(&pv, (Variable) ref_ent, (Value) 1);
    }
    else {/* min_or_max == IS_MAX */
      vect_add_elem(&pv, (Variable) ref_ent, (Value) -1);
    }

    aapc = contrainte_make(pv);
    if(CONTRAINTE_UNDEFINED_P(pcc)) {
      pcc = aapc;
      epc = aapc;
    }
    else {
      epc->succ = aapc;
      epc = epc->succ;
    }
  }

  /* We add the context. */
  psc = sc_dup(ps_cont);
  epc->succ = psc->inegalites;
  psc->inegalites = pcc;
  psc->nb_ineq += count;
  psc->base = NULL;
  sc_creer_base(psc);

  new_ps = sc_elim_redund(psc);

  new_pc = new_ps->inegalites;

  /* We remove our MINMAX variable. */
  newnew_pc = CONTRAINTE_UNDEFINED;
  for(apc = new_pc; apc != NULL; apc = apc->succ) {
    Pvecteur pv;
    Pcontrainte aapc;
    Value xc;

    pv = vect_dup(apc->vecteur);
    xc = vect_coeff((Variable) ref_ent, pv);
    if(value_one_p(xc) && (min_or_max == IS_MIN)) {
      vect_erase_var(&pv, (Variable) ref_ent);
      vect_chg_sgn(pv);
      aapc = contrainte_make(pv);
      aapc->succ = newnew_pc;
      newnew_pc = aapc;
    }
    else if(value_mone_p(xc) && (min_or_max == IS_MAX)) {
      vect_erase_var(&pv, (Variable) ref_ent);
      aapc = contrainte_make(pv);
      aapc->succ = newnew_pc;
      newnew_pc = aapc;
    }
  }
  return(newnew_pc);
}


/*================================================================== */
list vectors_to_expressions(pc)
Pcontrainte pc;
{
  list lexp = NIL;
  Pcontrainte apc;

  for(apc = pc; apc != NULL; apc = apc->succ) {
    expression new_exp = make_vecteur_expression(apc->vecteur);
    ADD_ELEMENT_TO_LIST(lexp, EXPRESSION, new_exp);
  }
  return(lexp);
}


/*================================================================== */
Pcontrainte expressions_to_vectors(lexp)
list lexp;
{
  list ll;
  Pcontrainte epc = CONTRAINTE_UNDEFINED, pc = CONTRAINTE_UNDEFINED;

  for(ll = lexp; !ENDP(ll); POP(ll)){
    Pvecteur pv;
    expression cexp = EXPRESSION(CAR(ll));
    normalized cnor;

    cnor = NORMALIZE_EXPRESSION(cexp);
    if(normalized_tag(cnor) == is_normalized_complex)
      pips_internal_error("Expressions MUST be linear");

    pv = (Pvecteur) normalized_linear(cnor);

    if(CONTRAINTE_UNDEFINED_P(pc)) {
      pc = contrainte_make(pv);
      epc = pc;
    }
    else {
      Pcontrainte apc = contrainte_make(pv);
      epc->succ = apc;
      epc = epc->succ;
    }
  }
  return(pc);
}


/*================================================================== */
/* list simplify_minmax(list lexp, Psysteme ps_cont, int min_or_max)
 *
 * Parameters :
 *   _ lexp : list of LINEAR expressions
 *   _ ps_cont : system of equations, called the context
 *   _ min_or_max : flag, says in which case we are (MIN or MAX)
 *
 * Result : list of LINEAR expressions
 *
 * Aims : simplify "lexp", i.e. suppress one or more of its
 * expressions. A given expression can be suppressed if it surely
 * smaller (case MAX) or greater (case MIN) than one of the other. The
 * case (MIN or MAX) is given by "min_or_max". If two expressions can
 * not be compared, both are kept.The context allows the user to
 * introduce relationship between variables without which some vectors
 * can not be eliminate.
 *
 * Algorithm : see simplify_minmax_contrainte().
 *
 * Note : "lexp" is not changed, the returned list of expressions is a
 * new one. */
list simplify_minmax(lexp, ps_cont, min_or_max)
list lexp;
Psysteme ps_cont;
int min_or_max;
{
  list new_lexp = NIL;
  Pcontrainte pc, new_pc;

  pc = expressions_to_vectors(lexp);

  new_pc = simplify_minmax_contrainte(pc, ps_cont, min_or_max);

  new_lexp = vectors_to_expressions(new_pc);

  return(new_lexp);
}


/* ======================================================================== */
/*
 * Psysteme find_implicit_equation(Psysteme ps)
 *
 * Returns a system containing the implicit equations of the system "ps".
 *
 * Each inequality may hides an implicit equation which means that all
 * solutions verify this equation.  In order to find these implicit
 * equations, we have to test all the inequalities.
 *
 * Let Expr <= 0 be the current inequality, we replace it with Expr + 1 <=
 * 0.  If the system is infeasible then we have found an implicit
 * equation.
 *
 * Explanation: Expr + 1 <= 0 means Expr <= -1. Then, if the new system is
 * infeasible though the original one is feasible, then Expr = 0 is
 * verified by all the solutions of the original system (it is an implicit
 * equation).
 * */
Psysteme find_implicit_equation(Psysteme ps)
{
  Pcontrainte ineg, eg;
  Psysteme impl_ps, aux_ps;

  if(ps == NULL)
    return(NULL);

  /* We put the equalities of the system in our implicit system. */
  impl_ps = sc_new();
  for(eg = ps->egalites; eg != NULL; eg = eg->succ)
    sc_add_egalite(impl_ps, contrainte_dup(eg));

  /* We duplicate "ps" in order to keep it unchanged. */
  aux_ps = sc_dup(ps);

  /* We make the test on each inequality. We count them. */
  for(ineg = aux_ps->inegalites; ineg != NULL; ineg = ineg->succ) {
    Pvecteur expr;

    /* We replace the original inequality (Expr <= 0) by the modified
     * one (Expr + 1 <= 0).
     */
    expr = ineg->vecteur;
    ineg->vecteur = vect_add(expr, vect_new(TCST, VALUE_ONE));

    /* We test the feasibility. If it is not feasible, we add one more
     * implicit equation in our implicit system : Expr == 0.
     */
    if(! sc_rational_feasibility_ofl_ctrl(aux_ps, NO_OFL_CTRL, true)) {
      Pcontrainte pc_expr = contrainte_make(expr);

      if(get_debug_level() > 7) {
	fprintf(stderr, "Equation implicit : ");
	pu_egalite_fprint(stderr, pc_expr, entity_local_name);
	fprintf(stderr, "\n");
      }
      sc_add_egalite(impl_ps, pc_expr);
    }

    /* We put the old value back */
    ineg->vecteur = expr;
  }

  sc_creer_base(impl_ps);
  impl_ps = sc_normalize(impl_ps);

  return(impl_ps);
}



/* These three functions respectively initialize, return and reset the
   static map of the static control on the statements. */

static statement_mapping current_stco_map = hash_table_undefined;

/* ======================================================================== */
void set_current_stco_map(statement_mapping scm)
{
  pips_assert("current_stco_map is defined", current_stco_map ==
	      hash_table_undefined);

  current_stco_map = scm;
}

/* ======================================================================== */
statement_mapping get_current_stco_map(void)
{
  return current_stco_map;
}

/* ======================================================================== */
void reset_current_stco_map(void)
{
  current_stco_map = hash_table_undefined;
}

/* ======================================================================== */
static_control get_stco_from_current_map(statement s)
{
  statement_mapping STS = get_current_stco_map();

  return((static_control) GET_STATEMENT_MAPPING(STS,s));
}

/*======================================================================*/
/* expression make_rational_exp(v, d)
 *
 * From the vector v and the integer d creates the expression of the
 * following form: v/d .  AC 94/03/25
 *
 * Modification: this is an extension that verifies that v is not a
 * multiple of d, in which case it can be simplified, and no divide
 * operation is needed.  AP 94/08/19 */

expression make_rational_exp(v, d)
Pvecteur    v;
Value       d;
{
  expression e;

  if(VECTEUR_NUL_P(v))
    /* make a "zero" expression */
    e = int_to_expression(0);
  else if(value_zero_p(value_mod(vect_pgcd_all(v), value_abs(d))))
    /* divide "v" by "d", and make the expression with no denominator */
      e = make_vecteur_expression(vect_div(v, d));
  else {
      expression  e1, e2;
      entity      ent;
      list        le = NIL;

    /* build the denominator */
    e2 = Value_to_expression(d);
    le = CONS(EXPRESSION, e2, NIL);

    /* build the numerator */
    vect_normalize(v);
    e1 = make_vecteur_expression(v);
    le = CONS(EXPRESSION, e1, le);

    /* create the symbol of dividing */
    ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						  DIVIDE_OPERATOR_NAME),
			     entity_domain);

    /* make the expression */
    e = make_expression(make_syntax(is_syntax_call,
				    make_call(ent, le)),
			normalized_undefined);
  }

  return(e);
}



/* AP, sep 25th 1995 : I have added a function from
   static_controlise/utils.c */

/*=======================================================================*/
/* int stco_common_loops_of_statements(in_map, in_s, in_s2 )    AL 22/10/93
 * Input    : A statement mapping in_map wich associates a static_control
 *              to each statement, and two statements in_s and in_s2.
 * Output   : Number of same enclosing loops around ins_s and in_s2.
 */
int stco_common_loops_of_statements(in_map, in_s, in_s2 )
statement_mapping       in_map;
statement               in_s, in_s2;
{
        debug(9, "stco_common_loops_of_statements","doing\n");
        return( gen_length(stco_same_loops( in_map, in_s, in_s2 )) );
}
