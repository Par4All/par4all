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

 /* package matrix */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"

/* void matrix_transpose(Pmatrix a, Pmatrix a_t):
 * transpose an (nxm) rational matrix a into a (mxn) rational matrix a_t
 *
 *         t
 * a_t := a ;
 */
void matrix_transpose(a,a_t)
Pmatrix	a;
Pmatrix	a_t;
{
    int	loop1,loop2;
    /* verification step */
    int n = MATRIX_NB_LINES(a);
    int m = MATRIX_NB_COLUMNS(a);
    assert(n >= 0 && m >= 0);

    /* copy from a to a_t */
    MATRIX_DENOMINATOR(a_t) = MATRIX_DENOMINATOR(a);
    for(loop1=1; loop1<=n; loop1++)
	for(loop2=1; loop2<=m; loop2++) 
	    MATRIX_ELEM(a_t,loop2,loop1) = MATRIX_ELEM(a,loop1,loop2);
}

/* void matrix_multiply(Pmatrix a, Pmatrix b, Pmatrix c):
 * multiply rational matrix a by rational matrix b and store result in matrix c
 *
 *	a is a (pxq) matrix, b a (qxr) and c a (pxr)
 *
 *      c := a x b ;
 *
 * Algorithm used is directly from definition, and space has to be
 * provided for output matrix c by caller. Matrix c is not necessarily
 * normalized: its denominator may divide all its elements
 * (see matrix_normalize()).
 *
 * Precondition:	p > 0; q > 0; r > 0;
 *                      c != a; c != b; -- aliasing between c and a or b
 *                                      -- is not supported
 */
void matrix_multiply(a,b,c)
Pmatrix	a;			/* input */
Pmatrix b;			/* input */
Pmatrix	c;			/* output */
{
    register Value va, vb;
    int	loop1,loop2,loop3;
    Value i;


    /* validate dimensions */
    int	p= MATRIX_NB_LINES(a);	
    int	q = MATRIX_NB_COLUMNS(a);	
    int	r= MATRIX_NB_COLUMNS(b);	
    assert(p > 0 && q > 0 && r > 0);
    /* simplified aliasing test */
    assert(c != a && c != b);

    /* set denominator */
    MATRIX_DENOMINATOR(c) = 
	value_mult(MATRIX_DENOMINATOR(a),MATRIX_DENOMINATOR(b));

    /* use ordinary school book algorithm */
    for(loop1=1; loop1<=p; loop1++)
	for(loop2=1; loop2<=r; loop2++) {
	    i = VALUE_ZERO;
	    for(loop3=1; loop3<=q; loop3++) 
	    {
		va = MATRIX_ELEM(a,loop1,loop3), 
		vb = MATRIX_ELEM(b,loop3,loop2);
		value_addto(i, value_mult(va, vb));
	    }
	    MATRIX_ELEM(c,loop1,loop2) = i;
	}
}

/* void matrix_normalize(Pmatrix a)
 *
 *	A rational matrix is stored as an integer one with one extra
 *	integer, the denominator for all the elements. To normalise the
 *	matrix in this sense means to reduce this denominator to the 
 *	smallest positive number possible. All elements are also reduced
 *      to their smallest possible value.
 *
 * Precondition: MATRIX_DENOMINATOR(a)!=0
 */
void matrix_normalize(a)
Pmatrix	a;			/* input */
{
    int	loop1,loop2;
    Value	factor;
    int n = MATRIX_NB_LINES(a);
    int m = MATRIX_NB_COLUMNS(a);
    assert(MATRIX_DENOMINATOR(a) != 0);

    /* we must find the GCD of all elements of matrix */
    factor = MATRIX_DENOMINATOR(a);

    for(loop1=1; loop1<=n; loop1++)
	for(loop2=1; loop2<=m; loop2++) 
	    factor = pgcd(factor,MATRIX_ELEM(a,loop1,loop2));

    /* factor out */
    if (value_notone_p(factor)) {
	for(loop1=1; loop1<=n; loop1++)
	    for(loop2=1; loop2<=m; loop2++)
		value_division(MATRIX_ELEM(a,loop1,loop2), factor);
	value_division(MATRIX_DENOMINATOR(a),factor);
    }

    /* ensure denominator is positive */
    /* FI: this code is useless because pgcd()always return a positive integer,
       even if a is the null matrix; its denominator CANNOT be 0 */
    assert(value_pos_p(MATRIX_DENOMINATOR(a)));

/*
    if(MATRIX_DENOMINATOR(a) < 0) {
	MATRIX_DENOMINATOR(a) = MATRIX_DENOMINATOR(a)*-1;
	for(loop1=1; loop1<=n; loop1++)
	    for(loop2=1; loop2<=m; loop2++)
		MATRIX_ELEM(a,loop1,loop2) = 
		    -1 * MATRIX_ELEM(a,loop1,loop2);
    }*/
}

/* void matrix_normalizec(Pmatrix MAT):
 *  Normalisation des coefficients de la matrice MAT, i.e. division des 
 *  coefficients de la matrice MAT et de son denominateur par leur pgcd
 *
 *  La matrice est modifiee.
 *
 *  Les parametres de la fonction :
 *
 *!int MAT[]	: matrice  de dimension (n,m) 
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */
void matrix_normalizec(MAT)
Pmatrix MAT;
{
    int i;
    Value a;
    int n = MATRIX_NB_LINES(MAT);
    int m = MATRIX_NB_COLUMNS(MAT);
    assert( n>0 && m>0);

    a = MATRIX_DENOMINATOR(MAT);
    for (i = 1;(i<=n*m) && value_gt(a,VALUE_ONE);i++)
	a = pgcd(a,MAT->coefficients[i]);

    if (value_gt(a,VALUE_ONE)) {
	for (i = 0;i<=n*m;i++)
	    value_division(MAT->coefficients[i],a);
    }
}

/* void matrix_swap_columns(Pmatrix a, int c1, int c2):
 * exchange columns c1,c2 of an (nxm) rational matrix
 *
 *	Precondition:	n > 0; m > 0; 0 < c1 <= m; 0 < c2 <= m; 
 */
void matrix_swap_columns(a,c1,c2)
Pmatrix a;		/* input and output matrix */
int	c1,c2;			/* column numbers */
{
    int	loop1;
    Value	temp;
    int n = MATRIX_NB_LINES(a);
    int m = MATRIX_NB_COLUMNS(a);
    assert(n > 0 && m > 0);
    assert(0 < c1 && c1 <= m);
    assert(0 < c2 && c2 <= m);

    for(loop1=1; loop1<=n; loop1++) {
	temp = MATRIX_ELEM(a,loop1,c1);
	MATRIX_ELEM(a,loop1,c1) = MATRIX_ELEM(a,loop1,c2);
	MATRIX_ELEM(a,loop1,c2) = temp;
    }
}	

/* void matrix_swap_rows(Pmatrix a, int r1, int r2):
 * exchange rows r1 and r2 of an (nxm) rational matrix a
 *
 * Precondition: n > 0; m > 0; 1 <= r1 <= n; 1 <= r2 <= n
 */
void matrix_swap_rows(a,r1,r2)
Pmatrix a;		/* input and output matrix */
int	r1,r2;			/* row numbers */
{
    int	loop1;
    Value	temp;
    /* validation */
    int n = MATRIX_NB_LINES(a);
    int m = MATRIX_NB_COLUMNS(a);
    assert(n > 0);
    assert(m > 0);
    assert(0 < r1 && r1 <= n);
    assert(0 < r2 && r2 <= n);

    for(loop1=1; loop1<=m; loop1++) {
	temp = MATRIX_ELEM(a,r1,loop1);
	MATRIX_ELEM(a,r1,loop1) = MATRIX_ELEM(a,r2,loop1);
	MATRIX_ELEM(a,r2,loop1) = temp;
    }
}

/* void matrix_assign(Pmatrix A, Pmatrix B)
 * Copie de la matrice A dans la matrice B
 *
 *  Les parametres de la fonction :
 *
 * int 	A[]	:  matrice
 *!int 	B[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 *
 * Ancien nom: matrix_dup
 */
void matrix_assign(A,B)
Pmatrix  A,B;
{
    int i;
    int n = MATRIX_NB_LINES(A);
    int m = MATRIX_NB_COLUMNS(A);
    for (i = 0 ;i<= n*m;i++)
	B->coefficients[i] = A->coefficients[i];
}

/* bool matrix_equality(Pmatrix A, Pmatrix B)
 * test de l'egalite de deux matrices A et B; elles doivent avoir
 * ete normalisees au prealable pour que le test soit mathematiquement
 * exact
 *
 *  Les parametres de la fonction :
 *
 * int 	A[]	:  matrice
 * int 	B[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */
bool matrix_equality(A,B)
Pmatrix  A,B;
{
    int i;  
    int n = MATRIX_NB_LINES(A);
    int m = MATRIX_NB_COLUMNS(A);
    for (i = 0 ;i<= n*m;i++)
	if(B->coefficients[i] != A->coefficients[i])
	    return(false);
    return(true);
}

/* void matrix_nulle(Pmatrix Z):
 * Initialisation de la matrice Z a la valeur matrice nulle
 *
 * Post-condition: 
 *
 * QQ i dans [1..n]
 * QQ j dans [1..n]
 *    Z(i,j) == 0
 *
 *  Les parametres de la fonction :
 *
 *!int 	Z[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */
void matrix_nulle(Z)
Pmatrix Z;
{
    int i,j;
    int n = MATRIX_NB_LINES(Z);
    int m = MATRIX_NB_COLUMNS(Z);
    for (i=1;i<=n;i++)
	for (j=1;j<=m;j++)
	    MATRIX_ELEM(Z,i,j)=VALUE_ZERO;
    MATRIX_DENOMINATOR(Z) = VALUE_ONE;
}

/* bool matrix_nulle_p(Pmatrix Z):
 * test de nullite de la matrice Z
 *
 * QQ i dans [1..n]
 * QQ j dans [1..n]
 *    Z(i,j) == 0
 *
 *  Les parametres de la fonction :
 *
 * int 	Z[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */
bool matrix_nulle_p(Z)
Pmatrix Z;
{
    int i,j;
    int n = MATRIX_NB_LINES(Z);
    int m = MATRIX_NB_COLUMNS(Z);
    for (i=1;i<=n;i++)
	for (j=1;j<=m;j++)
	    if(MATRIX_ELEM(Z,i,j)!=0)
		return(false);
    return(true);
}

/* bool matrix_diagonal_p(Pmatrix Z):
 * test de nullite de la matrice Z
 *
 * QQ i dans [1..n]
 * QQ j dans [1..n]
 *    Z(i,j) == 0 && i != j || i == j
 *
 *  Les parametres de la fonction :
 *
 * int 	Z[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */
bool matrix_diagonal_p(Z)
Pmatrix Z;
{
    int i,j;
    int n = MATRIX_NB_LINES(Z);
    int m = MATRIX_NB_COLUMNS(Z);
    for (i=1;i<=n;i++)
	for (j=1;j<=m;j++)
	    if(i!=j && MATRIX_ELEM(Z,i,j)!=0)
		return(false);
    return(true);
}

/* bool matrix_triangular_p(Pmatrix Z, bool inferieure):
 * test de triangularite de la matrice Z
 *
 * si inferieure == true
 * QQ i dans [1..n]
 * QQ j dans [i+1..m]
 *    Z(i,j) == 0
 *
 * si inferieure == false (triangulaire superieure)
 * QQ i dans [1..n]
 * QQ j dans [1..i-1]
 *    Z(i,j) == 0
 *
 *  Les parametres de la fonction :
 *
 * int 	Z[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */
bool matrix_triangular_p(Z,inferieure)
Pmatrix Z;
bool inferieure;
{
    int i,j;
    int n = MATRIX_NB_LINES(Z);
    int m = MATRIX_NB_COLUMNS(Z);
    for (i=1; i <= n; i++)
	if(inferieure) {
	    for (j=i+1; j <= m; j++)
		if(MATRIX_ELEM(Z,i,j)!=0)
		    return(false);
	}
	else
	    for (j=1; j <= i-1; j++)
		if(MATRIX_ELEM(Z,i,j)!=0)
		    return(false);
    return(true);
}

/* bool matrix_triangular_unimodular_p(Pmatrix Z, bool inferieure)
 * test de la triangulaire et unimodulaire de la matrice Z.
 * si inferieure == true
 * QQ i dans [1..n]
 * QQ j dans [i+1..n]
 *    Z(i,j) == 0
 * i dans [1..n]
 *   Z(i,i) == 1
 *
 * si inferieure == false (triangulaire superieure)
 * QQ i dans [1..n]
 * QQ j dans [1..i-1]
 *    Z(i,j) == 0
 * i dans [1..n]
 *   Z(i,i) == 1
 * 
 * les parametres de la fonction :
 * matrice Z : la matrice entre
 * int n     : la dimension de la martice caree
 */
bool matrix_triangular_unimodular_p(Z,inferieure)
Pmatrix Z;
bool inferieure;
{
    bool triangulaire;
    int i;  
    int n = MATRIX_NB_LINES(Z);
    triangulaire = matrix_triangular_p(Z,inferieure);
    if (triangulaire == false)
	return(false);
    else{
	for(i=1; i<=n; i++)
	    if (value_notone_p(MATRIX_ELEM(Z,i,i)))
		return(false);
	return(true);
    }
}	    

/* void matrix_substract(Pmatrix a, Pmatrix b, Pmatrix c):
 * substract rational matrix c from rational matrix b and store result
 * in matrix a
 *
 *	a is a (nxm) matrix, b a (nxm) and c a (nxm)
 *
 *      a = b - c ;
 *
 * Algorithm used is directly from definition, and space has to be
 * provided for output matrix a by caller. Matrix a is not necessarily
 * normalized: its denominator may divide all its elements
 * (see matrix_normalize()).
 *
 * Precondition:	n > 0; m > 0;
 * Note: aliasing between a and b or c is supported
 */
void matrix_substract(a,b,c)
Pmatrix a;         /* output */
Pmatrix b, c;      /* input */
{
    Value d1,d2;   /* denominators of b, c */
    Value lcm;     /* ppcm of b,c */
    int i,j;

    /* precondition */
    int n= MATRIX_NB_LINES(a);
    int m = MATRIX_NB_COLUMNS(a);
    assert(n>0 && m>0);
    assert(value_pos_p(MATRIX_DENOMINATOR(b)));
    assert(value_pos_p(MATRIX_DENOMINATOR(c)));

    d1 = MATRIX_DENOMINATOR(b);
    d2 = MATRIX_DENOMINATOR(c);
    if (value_eq(d1,d2)){
	for (i=1; i<=n; i++)
	    for (j=1; j<=m; j++)
		MATRIX_ELEM(a,i,j) = 
		    value_minus(MATRIX_ELEM(b,i,j),MATRIX_ELEM(c,i,j));
	MATRIX_DENOMINATOR(a) = d1;
    }
    else{
	lcm = ppcm(d1,d2);
	d1 = value_div(lcm,d1);
	d2 = value_div(lcm,d2);
	for (i=1; i<=n; i++)
	    for (j=1; j<=m; j++)
		MATRIX_ELEM(a,i,j) = 
		    value_minus(value_mult(MATRIX_ELEM(b,i,j),d1),
				value_mult(MATRIX_ELEM(c,i,j),d2));
	MATRIX_DENOMINATOR(a) = lcm;
    }
}

void matrix_add(a,b,c)
Pmatrix a;         /* output */
Pmatrix b, c;      /* input */
{
    Value d1,d2;   /* denominators of b, c */
    Value lcm,t1,t2;     /* ppcm of b,c */
    int i,j;

    /* precondition */
    int n= MATRIX_NB_LINES(a);
    int m = MATRIX_NB_COLUMNS(a);
    assert(n>0 && m>0);
    assert(value_pos_p(MATRIX_DENOMINATOR(b)));
    assert(value_pos_p(MATRIX_DENOMINATOR(c)));

    d1 = MATRIX_DENOMINATOR(b);
    d2 = MATRIX_DENOMINATOR(c);
    if (value_eq(d1,d2)){
	for (i=1; i<=n; i++)
	    for (j=1; j<=m; j++)
		MATRIX_ELEM(a,i,j) = 
		    value_plus(MATRIX_ELEM(b,i,j),MATRIX_ELEM(c,i,j));
	MATRIX_DENOMINATOR(a) = d1;
    }
    else{
	lcm = ppcm(d1,d2);
	d1 = value_div(lcm,d1);
	d2 = value_div(lcm,d2);
	for (i=1; i<=n; i++)
	    for (j=1; j<=m; j++)
		t1 = value_mult(MATRIX_ELEM(b,i,j),d1),
		    t2 = value_mult(MATRIX_ELEM(c,i,j),d2),
		    MATRIX_ELEM(a,i,j) = value_plus(t1,t2);
	MATRIX_DENOMINATOR(a) = lcm;
    }
}

/* void matrix_subtraction_column(Pmatrix MAT,int c1,int c2,int x):
 * Soustrait x fois la colonne c2 de la colonne c1
 * Precondition: n > 0; m > 0; 0 < c1, c2 < m;
 * Effet: c1[0..n-1] = c1[0..n-1] - x*c2[0..n-1].
 *
 *  Les parametres de la fonction :
 *
 * int  MAT[]     :  matrice
 * int  c1        :  numero du colonne
 * int  c2        :  numero du colonne
 * int  x         : 
 */
void matrix_subtraction_column(MAT,c1,c2,x)
Pmatrix MAT;        
int c1,c2;
Value x;
{
    int i;
    int n = MATRIX_NB_LINES(MAT);
    for (i=1; i<=n; i++)
	value_substract(MATRIX_ELEM(MAT,i,c1),
			value_mult(x,MATRIX_ELEM(MAT,i,c2)));
}

/* void matrix_subtraction_line(Pmatrix MAT,int r1,int r2,int x):
 * Soustrait x fois la ligne r2 de la ligne r1 
 * Precondition: n > 0; m > 0; 0 < r1, r2 < n;
 * Effet: r1[0..m-1] = r1[0..m-1] - x*r2[0..m-1]
 *
 *  Les parametres de la fonction :
 *
 * int  MAT[]     :  matrice
 * int  n         :  nombre de lignes de la matrice
 * int  m         :  nombre de colonnes de la matrice
 * int  r1        :  numero du ligne
 * int  r2        :  numero du ligne
 * int  x         : 
 */
void matrix_subtraction_line(MAT,r1,r2,x)
Pmatrix MAT;
int r1,r2;
Value x;
{
    int i;
    int m = MATRIX_NB_COLUMNS(MAT);
    for (i=1; i<=m; i++)
	value_substract(MATRIX_ELEM(MAT,r1,i),
			value_mult(x,MATRIX_ELEM(MAT,r2,i)));
}

/* void matrix_uminus(A, mA)
 *
 * computes mA = - A
 *
 * input: A, larger allocated mA
 * output: none
 * modifies: mA
 */
void matrix_uminus(
    Pmatrix A,
    Pmatrix mA)
{
    int i,j;

    assert(MATRIX_NB_LINES(mA)>=MATRIX_NB_LINES(A) &&
	   MATRIX_NB_COLUMNS(mA)>=MATRIX_NB_COLUMNS(A));

    for (i=1; i<=MATRIX_NB_LINES(A); i++)
	for (j=1; j<=MATRIX_NB_COLUMNS(A); j++)
	    MATRIX_ELEM(mA, i, j) = value_uminus(MATRIX_ELEM(A, i, j));
}

/* that is all
 */
