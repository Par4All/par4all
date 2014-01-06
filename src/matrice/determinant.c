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

#include <stdlib.h>
#include <stdio.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrice.h"

/* int matrice_determinant(matrice a, int n, int * result):  
 * calculate determinant of an (nxn) rational matrix a
 *
 * The result consists of two integers, a numerator result[1]
 * and a denominator result[0].
 * 
 * Algorithm: integer matrix triangularisation by Hermite normal form
 * (because the routine is available)
 *
 * Complexity: O(n**3) -- up to gcd computations...
 *
 * Authors: Francois Irigoin, September 1989, Yi-qing Yang, January 1990
 */
void matrice_determinant(a,n,result)
matrice	a;			        /* input */
int	n;				/* input */
Value	result[];			/* output */
{ 
    Value determinant_gcd;
    Value determinant_p;
    Value determinant_q;

    Value denominator = DENOMINATOR(a);
    int i;
    /* cette routine est FAUSSE car la factorisation de Hermite peut
       s'appuyer sur des matrice unimodulaires de determinant 1 ou -1.
       Il faudrait donc ecrire un code de calcul de determinant de
       matrice unimodulaire...
       
       Ou tout reprendre avec une triangularisation rapide et n'utilisant
       que des matrice de determinant +1. Voir mon code des US.
       Cette triangularisation donnera en derivation une routine
       d'inversion rapide pour matrice unimodulaire */

    assert(n > 0 && denominator != 0);

    switch(n) {

    case 1:	
	result[1] = ACCESS(a,n,1,1);
	result[0] = denominator;
	break;

    case 2:        
    {
	register Value v1, v2, a1, a2;
	a1 = ACCESS(a,n,1,1);
	a2 = ACCESS(a,n,2,2);
	v1 = value_mult(a1,a2);
	a1 = ACCESS(a,n,1,2);
	a2 = ACCESS(a,n,2,1);
	v2 = value_mult(a1,a2);
 
        result[1] = value_minus(v1,v2);
        result[0] = value_mult(denominator,denominator);

	break;
    }
    default: {
	matrice p = matrice_new(n,n);
	matrice h = matrice_new(n,n);
	matrice q = matrice_new(n,n);

	/* matrice_hermite expects an integer matrix */
	DENOMINATOR(a) = VALUE_ONE;
	matrice_hermite(a,n,n,p,h,q,&determinant_p,&determinant_q);
	DENOMINATOR(a) = denominator;

	/* product of diagonal elements */
	result[1] = ACCESS(h,n,1,1);
	result[0] = denominator;
	for(i=2; i <= n && result[1]!=0; i++) {
	    register Value a = ACCESS(h,n,i,i);
	    value_product(result[1], a);
	    value_product(result[0], denominator);
	    determinant_gcd = pgcd_slow(result[0],result[1]);
	    if(value_notone_p(determinant_gcd)) {
		value_division(result[0],determinant_gcd);
		value_division(result[1],determinant_gcd);
	    }
	}

	/* product of determinants of three matrixs P, H, Q */
        value_product(result[1],determinant_p);
        value_product(result[1],determinant_q);

	/* free useless matrices */
	matrice_free(p);
	matrice_free(h);
	matrice_free(q);
    }
    }
    /* reduce result */
    determinant_gcd = pgcd_slow(result[0],result[1]);\
    if(value_notone_p(determinant_gcd)) {
        value_division(result[0],determinant_gcd);
        value_division(result[1],determinant_gcd);
    }   
}

/* void matrice_sous_determinant(matrice a, int n,int i, int j, int result[])
 * calculate sub determinant of a matrix
 *
 * The sub determinant indicated by (i,j) is the one obtained
 * by deleting row i and column j from matrix of dimension n.
 * The result passed back consists of two integers, a numerator
 * result[1] and a denominator result[0]. 
 *
 * Precondition: n >= i > 0 && n >= j > 0
 *
 * Algorithm: put 0 in row i and in column j, and 1 in a[i,j], and then
 *            call matrice_determinant, restore a
 *
 * Complexity: O(n**3)
 */
void matrice_sous_determinant(a,n,i,j,result)
matrice	a;		        /* input matrix */
int	n;			/* dimension of matrix */
int	i,j;			/* coords of sub determinant */
Value	result[];		/* output */
{
    int r; 
    int c;
    matrice save_row = matrice_new(1,n);
    matrice save_column = matrice_new(1,n);

    /* save column j */
    for(r=1; r <= n; r++) {
	ACCESS(save_column,1,1,r) = ACCESS(a,n,r,j);
	ACCESS(a,n,r,j) = 0;
    }
    /* save row i, but for the (i,j) elements */
    for(c=1; c <= n; c++) {
	ACCESS(save_row,1,1,c) = ACCESS(a,n,i,c);
	ACCESS(a,n,i,c) = 0;
    }
    ACCESS(a,n,i,j) = VALUE_ONE;

    matrice_determinant(a,n,result);

    /* restore row i */
    for(c=1; c <= n; c++) {
	ACCESS(a,n,i,c) = ACCESS(save_row,1,1,c);
    }
    /* restore column j, with proper (i,j) element */
    for(r=1; r <= n; r++) {
	ACCESS(a,n,r,j) = ACCESS(save_column,1,1,r);
    }

    matrice_free(save_row);
    matrice_free(save_column);
}

