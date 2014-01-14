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

 /* package matrice */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrice.h"

#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(s,t,f) free((char *)(s))

/* void matrice_smith(matrice MAT, int n, int m, matrice P, matrice D, 
 *                    matrice Q):
 * Calcul de la forme reduite de Smith D d'une matrice entiere MAT et des deux
 * matrices de changement de base unimodulaires associees, P et Q.
 *
 * D est la forme reduite de Smith, P et Q sont des matrices unimodulaires;
 * telles que D == P x MAT x Q
 *
 * (c.f. Programmation Lineaire. M.MINOUX. (83))
 *
 * int 	MAT[n,m]	:  matrice
 * int  n	: nombre de lignes de la matrice MAT
 * int  m 	: nombre de colonnes de la matrice MAT
 * int 	P[n,n]	: matrice
 * int 	D[n,m]	: matrice (quasi-diagonale) reduite de Smith
 * int 	Q[m,m]	: matrice
 *  
 * Les 3 matrices P(nxn), Q(mxm) et D(nxm) doivent etre allouees avant le 
 * calcul. 
 *
 * Note: les determinants des matrices MAT, P, Q et D ne sont pas utilises.
 *
 */
void matrice_smith(MAT,n,m,P,D,Q)
matrice MAT;
int n,m;
matrice P;
matrice D;
matrice Q;
{
    int n_min,m_min;
    int level = 0;
    register Value ALL;        /* le plus petit element sur la diagonale */
    register Value x;          /* le rest de la division par ALL */
    int i;
    
    bool stop = false;
    bool next = true;

    /* precondition sur les parametres */
    assert(m > 0 && n >0);
    matrice_assign(MAT,D,n,m);
    assert(value_one_p(DENOMINATOR(D)));

    matrice_identite(P,n,0);
    matrice_identite(Q,m,0);
 
    while (!stop) {
	mat_min(D,n,m,&n_min,&m_min,level);
		
	if ((n_min == 0) && (m_min == 0))
	    stop = true;
	else {
	    /* le transformation n'est pas fini. */
	    if (n_min > level +1) {
		matrice_swap_rows(D,n,m,level+1,n_min);
		matrice_swap_rows(P,n,n,level+1,n_min);
	    }
#ifdef TRACE
	    (void) printf (" apres alignement du plus petit element a la premiere ligne \n");
	    matrice_print(D,n,m);
#endif
	    if (m_min >1+level) {
		matrice_swap_columns(D,n,m,level+1,m_min);
		matrice_swap_columns(Q,m,m,level+1,m_min);
	    }
#ifdef TRACE
	    (void) printf (" apres alignement du plus petit element"
			   " a la premiere colonne\n");
	    matrice_print(D,n,m);
#endif
	   
	    ALL = ACC_ELEM(D,n,1,1,level);
	    if (mat_lig_el(D,n,m,level) != 0) 
		for (i=level+2; i<=m; i++) {
		    x = ACCESS(D,n,level+1,i);
		    value_division(x,ALL);
		    matrice_soustraction_colonne(D,n,m,i,level+1,x);
		    matrice_soustraction_colonne(Q,m,m,i,level+1,x);
		    next = false;
		}
	    if (mat_col_el(D,n,m,level) != 0) 
		for(i=level+2;i<=n;i++) {
		    x = ACCESS(D,n,i,level+1);
		    value_division(x,ALL);
		    matrice_soustraction_ligne(D,n,m,i,level+1,x);
		    matrice_soustraction_ligne(P,n,n,i,level+1,x);
		    next = false;
		}
#ifdef TRACE
		(void) printf("apres division par A(%d,%d) des termes de la %d-ieme ligne et de la %d-em colonne \n",level+1,level+1,level+1,level+1);
		matrice_print(D,n,m);
#endif
	    if (next) level++;
	    next = true;
	}
    }

#ifdef TRACE
    (void) printf (" la  matrice D apres transformation est la suivante :");
    matrice_print(D,n,m);

    (void) printf (" la matrice P est \n");
    matrice_print(P,n,n);

    (void) printf (" la matrice Q est \n");
    matrice_print(Q,m,m);
#endif
}
	

