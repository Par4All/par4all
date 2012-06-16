/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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

 /* package matrix */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"

/* void matrix_hermite(matrix MAT, matrix P, matrix H,
 *                      matrix Q, int *det_p, int *det_q):
 * calcul de la forme reduite de Hermite H de la matrice MAT avec
 * les deux matrices de changement de base unimodulaire P et Q, 
 * telles que H = P x MAT x Q 
 * et en le meme temp, produit les determinants des P et Q.
 * det_p = |P|, det_q = |Q|.
 *  
 *  (c.f. Programmation Lineaire. Theorie et algorithmes. tome 2. M.MINOUX. Dunod (1983))
 * FI: Quels est l'invariant de l'algorithme? MAT = P x H x Q?
 * FI: Quelle est la norme decroissante? La condition d'arret?
 * 
 *  Les parametres de la fonction :
 *
 * int 	MAT[n,m]:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice 
 * int 	P[n,n]	:  matrice
 * int 	H[n,m]  :  matrice reduite de Hermite
 * int 	Q[m,m]	:  matrice
 * int *det_p   :  determinant de P (nombre de permutation de ligne)
 * int *det_q   :  determinant de Q (nombre de permutation de colonne)
 *  
 * Les 3 matrices P(nxn), Q(mxm) et H(nxm) doivent etre allouees avant le 
 * calcul. 
 *
 * Note: les denominateurs de MAT, P, A et H ne sont pas utilises.
 *
 * Auteurs: Corinne Ancourt, Yi-qing Yang
 *
 * Modifications:
 *  - modification du profil de la fonction pour permettre le calcul du
 *    determinant de MAT
 *  - permutation de colonnes directe, remplacant une permutation
 *    par produit de matrices
 *  - suppression des copies entre H, P, Q et NH, PN, QN
 */
void matrix_hermite(MAT,P,H,Q,det_p,det_q)
Pmatrix MAT;
Pmatrix P;
Pmatrix H;
Pmatrix Q;
Value *det_p;
Value *det_q;
{
    Pmatrix PN=NULL;
    Pmatrix QN=NULL;
    Pmatrix HN = NULL;
    int n,m;
    /* indice de la ligne et indice de la colonne correspondant
       au plus petit element de la sous-matrice traitee */
    int ind_n,ind_m;   

    /* niveau de la sous-matrice traitee */
    int level = 0;    
    
    Value ALL;/* le plus petit element sur la diagonale */
    Value x=VALUE_ZERO;  /* la quotient de la division par ALL */
    bool stop = false;
    int i;
    
    *det_p = *det_q = VALUE_ONE;
    /* if ((n>0) && (m>0) && MAT) */
    n = MATRIX_NB_LINES(MAT);
    m=MATRIX_NB_COLUMNS(MAT);
    assert(n >0 && m > 0);
    assert(value_one_p(MATRIX_DENOMINATOR(MAT)));

    HN = matrix_new(n, m);
    PN = matrix_new(n, n);
    QN = matrix_new(m, m);
   
    /* Initialisation des matrices */

    matrix_assign(MAT,H);
    matrix_identity(PN,0);
    matrix_identity(QN,0);
    matrix_identity(P,0);
    matrix_identity(Q,0);
    matrix_nulle(HN);

    while (!stop) {  
	/* tant qu'il reste des termes non nuls dans la partie 
	   triangulaire superieure  de la matrice H,        
	   on cherche le plus petit element de la sous-matrice     */
	matrix_coeff_nnul(H,&ind_n,&ind_m,level);

	if (ind_n == 0 && ind_m == 0)
	    /* la sous-matrice restante est nulle, on a terminee */
	    stop = true; 
	else { 
	    /* s'il existe un plus petit element non nul dans la partie
	       triangulaire superieure de la sous-matrice,
	       on amene cet element en haut sur la diagonale.
	       si cet element est deja sur la diagonale, et que  tous les 
	       termes de la ligne correspondante sont nuls,
	       on effectue les calculs sur la sous-matrice de rang superieur*/
	       
	    if (ind_n > level + 1) {
		matrix_swap_rows(H,level+1,ind_n);
		matrix_swap_rows(PN,level+1,ind_n);
		*det_p = value_uminus(*det_p);
	    }

	    if (ind_m > level+1) {
		matrix_swap_columns(H,level+1,ind_m);
		matrix_swap_columns(QN,level+1,ind_m);	
		*det_q = value_uminus(*det_q);
	    }

	    if(matrix_line_el(H,level) != 0) {
		ALL = SUB_MATRIX_ELEM(H,1,1,level);
		for (i=level+2; i<=m; i++) {
		    x = value_div(MATRIX_ELEM(H,level+1,i),ALL);
		    matrix_subtraction_column(H,i,level+1,x);
		    matrix_subtraction_column(QN,i,level+1,x);
		}

	    }
	    else level++;
	}
    }

    matrix_assign(PN,P);
    matrix_assign(QN,Q);
    matrix_free(HN);
    matrix_free(PN);
    matrix_free(QN);
}

/* int matrix_hermite_rank(Pmatrix a): rang d'une matrice
 * en forme de hermite
 */
int matrix_hermite_rank(a)
Pmatrix a;
{
    int i;
    int r = 0;
    int n = MATRIX_NB_LINES(a);
    for(i=1; i<=n; i++, r++)
	if(MATRIX_ELEM(a, i, i) == 0) break;

    return r;
}


/* Calcul  de la dimension de la matrice de Hermite H.
 * La matrice de Hermite est une matrice tres particuliere, 
 * pour laquelle il est tres facile de calculer sa dimension. 
 * Elle est egale au nombre de colonnes non nulles de la matrice.
 */


int matrix_dim_hermite (H)
Pmatrix H;
{
    int i,j;
    bool trouve_j = false;
    int res=0;
    int n = MATRIX_NB_LINES(H);
    int m = MATRIX_NB_COLUMNS(H);
    for (j = 1; j<=m && !trouve_j;j++) {
	for (i=1; i<= n && MATRIX_ELEM(H,i,j) == 0; i++);
	if (i>n) {
	    trouve_j = true;
	    res= j-1;
	}
    }


    if (!trouve_j && m>=1) res = m;
    return (res);

}
