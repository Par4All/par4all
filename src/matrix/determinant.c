 /* package matrix */

#include <stdio.h>
#include <sys/stdtypes.h> /*for debug with dbmalloc */
#include <malloc.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"

/* int matrix_determinant(Pmatrix a, int * result):  
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
void  matrix_determinant(a,result)
Pmatrix	a;			        /* input */
int	result[];			/* output */
{

    int determinant_gcd;
    int determinant_p;
    int determinant_q;
    int	n = MATRIX_NB_LINES(a);   
    int denominator = MATRIX_DENOMINATOR(a);
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
	result[1] = MATRIX_ELEM(a,1,1);
	result[0] = denominator;
	break;

    case 2:
	result[1] = MATRIX_ELEM(a,1,1)*MATRIX_ELEM(a,2,2) -
	    MATRIX_ELEM(a,1,2)*MATRIX_ELEM(a,2,1);
	result[0] = denominator * denominator;
	break;

    default: {
	Pmatrix p = matrix_new(n,n);
	Pmatrix h = matrix_new(n,n);
	Pmatrix q = matrix_new(n,n);

	/* matrix_hermite expects an integer matrix */
	MATRIX_DENOMINATOR(a) = 1;
	matrix_hermite(a,p,h,q,&determinant_p,&determinant_q);
	MATRIX_DENOMINATOR(a) = denominator;

	/* product of diagonal elements */
	result[1] = MATRIX_ELEM(h,1,1);
	result[0] = denominator;
	for(i=2; i <= n && result[1]!=0; i++) {
	    result[1] *= MATRIX_ELEM(h,i,i);
	    result[0] *= denominator;
	    determinant_gcd = pgcd_slow(result[0],result[1]);
	    if(determinant_gcd!=1) {
		result[0] /= determinant_gcd;
		result[1] /= determinant_gcd;
	    }
	}

	/* product of determinants of three matrixs P, H, Q */
	result[1] *= determinant_p * determinant_q;
	
	    

	/* free useless matrices */
	matrix_free(p);
	matrix_free(h);
	matrix_free(q);
    }
    }
    /* reduce result */
    determinant_gcd = pgcd_slow(result[0],result[1]);
    if(determinant_gcd!=1) {
	result[0] /= determinant_gcd;
	result[1] /= determinant_gcd;
    }   
}

/* void matrix_sub_determinant(Pmatrix a,int i, int j, int result[])
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
 *            call matrix_determinant, restore a
 *
 * Complexity: O(n**3)
 */
void matrix_sub_determinant(a,i,j,result)
Pmatrix	a;		        /* input matrix */
int	i,j;			/* coords of sub determinant */
int	result[];		/* output */
{
    int n = MATRIX_NB_LINES(a);
    int r; 
    int c;
    Pmatrix save_row = matrix_new(1,n);
    Pmatrix save_column = matrix_new(1,n);

    /* save column j */
    for(r=1; r <= n; r++) {
	MATRIX_ELEM(save_column,1,r) = MATRIX_ELEM(a,r,j);
	MATRIX_ELEM(a,r,j) = 0;
    }
    /* save row i, but for the (i,j) elements */
    for(c=1; c <= n; c++) {
	MATRIX_ELEM(save_row,1,c) = MATRIX_ELEM(a,i,c);
	MATRIX_ELEM(a,i,c) = 0;
    }
    MATRIX_ELEM(a,i,j) = 1;

    matrix_determinant(a,result);

    /* restore row i */
    for(c=1; c <= n; c++) {
	MATRIX_ELEM(a,i,c) = MATRIX_ELEM(save_row,1,c);
    }
    /* restore column j, with proper (i,j) element */
    for(r=1; r <= n; r++) {
	MATRIX_ELEM(a,r,j) = MATRIX_ELEM(save_column,1,r);
    }

    matrix_free(save_row);
    matrix_free(save_column);
}

