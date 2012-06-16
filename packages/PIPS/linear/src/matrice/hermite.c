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

 /* package matrice */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrice.h"

/* void matrice_hermite(matrice MAT, int n, int m, matrice P, matrice H,
 *                      matrice Q, int *det_p, int *det_q):
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
void matrice_hermite(MAT,n,m,P,H,Q,det_p,det_q)
Value *MAT;
int n,m;
Value *P;
Value *H;
Value *Q;
Value *det_p;
Value *det_q;

{
    Value *PN=NULL;
    Value *QN=NULL;
    Value *HN = NULL;

    /* indice de la ligne et indice de la colonne correspondant
       au plus petit element de la sous-matrice traitee */
    int ind_n,ind_m;   

    /* niveau de la sous-matrice traitee */
    int level = 0;    
    
    register Value ALL; /* le plus petit element sur la diagonale */
    register Value x;    /* la rest de la division par ALL */
    bool stop = false;
    int i;
    
    *det_p = VALUE_ONE;
    *det_q = VALUE_ONE;

    /* if ((n>0) && (m>0) && MAT) */
    assert((n > 0) && (m > 0));
    assert(value_one_p(DENOMINATOR(MAT)));

    HN = matrice_new(n, m);
    PN = matrice_new(n, n);
    QN = matrice_new(m, m);
   
    /* Initialisation des matrices */

    matrice_assign(MAT,H,n,m);
    matrice_identite(PN,n,0);
    matrice_identite(QN,m,0);
    matrice_identite(P,n,0);
    matrice_identite(Q,m,0);
    matrice_nulle(HN,n,m);

    while (!stop) {  
	/* tant qu'il reste des termes non nuls dans la partie 
	   triangulaire superieure  de la matrice H,        
	   on cherche le plus petit element de la sous-matrice     */
	mat_coeff_nnul(H,n,m,&ind_n,&ind_m,level);

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
		matrice_swap_rows(H,n,m,level+1,ind_n);
		matrice_swap_rows(PN,n,n,level+1,ind_n);
		value_oppose(*det_p);
	    }

	    if (ind_m > level+1) {
		matrice_swap_columns(H,n,m,level+1,ind_m);
		matrice_swap_columns(QN,m,m,level+1,ind_m);	
		value_oppose(*det_q);
	    }

	    if(mat_lig_el(H,n,m,level) != 0) {
		ALL = ACC_ELEM(H,n,1,1,level);
		for (i=level+2; i<=m; i++) {
		    x = ACCESS(H,n,level+1,i);
		    value_division(x,ALL);
		    matrice_soustraction_colonne(H,n,m,i,level+1,x);
		    matrice_soustraction_colonne(QN,m,m,i,level+1,x);
		}

	    }
	    else level++;
	}
    }

    matrice_assign(PN,P,n,n);
    matrice_assign(QN,Q,m,m);

    matrice_free(HN);
    matrice_free(PN);
    matrice_free(QN);
}

/* int matrice_hermite_rank(matrice a, int n, int m): rang d'une matrice
 * en forme de hermite
 */
int matrice_hermite_rank(matrice a, int n, int m __attribute__ ((unused)))
{
    int i;
    int r = 0;

    for(i=1; i<=n; i++, r++)
	if(value_zero_p(ACCESS(a, n, i, i))) break;

    return r;
}


/* Calcul  de la dimension de la matrice de Hermite H.
 * La matrice de Hermite est une matrice tres particuliere, 
 * pour laquelle il est tres facile de calculer sa dimension. 
 * Elle est egale au nombre de colonnes non nulles de la matrice.
 */


int dim_H (H,n,m)
matrice H;
int n,m;
{


    int i,j;
    bool trouve_j = false;
    int res=0;

    for (j = 1; j<=m && !trouve_j;j++) {
	for (i=1; i<= n && value_zero_p(ACCESS(H,n,i,j)); i++);
	if (i>n) {
	    trouve_j = true;
	    res= j-1;
	}
    }


    if (!trouve_j && m>=1) res = m;
    return (res);

}
