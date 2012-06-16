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

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrice.h"

/* void matrice_transpose(matrice a, matrice a_t, int n, int m):
 * transpose an (nxm) rational matrix a into a (mxn) rational matrix a_t
 *
 *         t
 * a_t := a ;
 */
void matrice_transpose(a,a_t,n,m)
matrice	a;
matrice	a_t;
int	n;
int	m;
{
    int	loop1,loop2;

    /* verification step */
    assert(n >= 0 && m >= 0);

    /* copy from a to a_t */
    DENOMINATOR(a_t) = DENOMINATOR(a);
    for(loop1=1; loop1<=n; loop1++)
	for(loop2=1; loop2<=m; loop2++) 
	    ACCESS(a_t,m,loop2,loop1) = ACCESS(a,n,loop1,loop2);
}

/* void matrice_multiply(matrice a, matrice b, matrice c, int p, int q, int r):
 * multiply rational matrix a by rational matrix b and store result in matrix c
 *
 *	a is a (pxq) matrix, b a (qxr) and c a (pxr)
 *
 *      c := a x b ;
 *
 * Algorithm used is directly from definition, and space has to be
 * provided for output matrix c by caller. Matrix c is not necessarily
 * normalized: its denominator may divide all its elements
 * (see matrice_normalize()).
 *
 * Precondition:	p > 0; q > 0; r > 0;
 *                      c != a; c != b; -- aliasing between c and a or b
 *                                      -- is not supported
 */
void matrice_multiply(a,b,c,p,q,r)
matrice	a;			/* input */
matrice b;			/* input */
matrice	c;			/* output */
int	p;			/* input */
int	q;			/* input */
int	r;			/* input */
{
    int	loop1,loop2,loop3;
    register Value i,va,vb;

    /* validate dimensions */
    assert(p > 0 && q > 0 && r > 0);
    /* simplified aliasing test */
    assert(c != a && c != b);

    /* set denominator */
    DENOMINATOR(c) = value_mult(DENOMINATOR(a),DENOMINATOR(b));

    /* use ordinary school book algorithm */
    for(loop1=1; loop1<=p; loop1++)
	for(loop2=1; loop2<=r; loop2++) {
	    i = VALUE_ZERO;
	    for(loop3=1; loop3<=q; loop3++) 
	    {
		va = ACCESS(a,p,loop1,loop3);
		vb = ACCESS(b,q,loop3,loop2);
		value_addto(i,value_mult(va,vb));
	    }
	    ACCESS(c,p,loop1,loop2) = i;
	}
}

/* void matrice_normalize(matrice a, int n, int m)
 *
 *	A rational matrix is stored as an integer one with one extra
 *	integer, the denominator for all the elements. To normalise the
 *	matrix in this sense means to reduce this denominator to the 
 *	smallest positive number possible. All elements are also reduced
 *      to their smallest possible value.
 *
 * Precondition: DENOMINATOR(a)!=0
 */
void matrice_normalize(a,n,m)
matrice a;			/* input */
int	n;			/* input */
int	m;			/* output */
{
    int	loop1,loop2;
    Value factor;

    assert(value_notzero_p(DENOMINATOR(a)));

    /* we must find the GCD of all elements of matrix */
    factor = DENOMINATOR(a);

    for(loop1=1; loop1<=n; loop1++)
	for(loop2=1; loop2<=m; loop2++) 
	    factor = pgcd(factor,ACCESS(a,n,loop1,loop2));

    /* factor out */
    if (value_notone_p(factor)) {
	for(loop1=1; loop1<=n; loop1++)
	    for(loop2=1; loop2<=m; loop2++)
		value_division(ACCESS(a,n,loop1,loop2),factor);
	value_division(DENOMINATOR(a),factor);
    }

    /* ensure denominator is positive */
    /* FI: this code is useless because pgcd()always return a positive integer,
       even if a is the null matrix; its denominator CANNOT be 0 */
    assert(value_pos_p(DENOMINATOR(a)));
    /*
    if(DENOMINATOR(a) < 0) {
	DENOMINATOR(a) = DENOMINATOR(a)*-1;
	for(loop1=1; loop1<=n; loop1++)
	    for(loop2=1; loop2<=m; loop2++)
		value_oppose(ACCESS(a,n,loop1,loop2));
    }*/
}

/* void matrice_normalizec(matrice MAT, int n, int m):
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
void matrice_normalizec(MAT,n,m)
matrice MAT;
int n,m;
{
    int i;
    Value a;

    /* FI What an awful test! It should be an assert() */
    if (n && m) {
	a = MAT[0];
	for (i = 1;(i<=n*m) && value_gt(a,VALUE_ONE);i++)
	    a = pgcd(a,MAT[i]);

	if (value_gt(a,VALUE_ONE)) {
	    for (i = 0;i<=n*m;i++)
		value_division(MAT[i],a);
	}
    }
}

/* void matrice_swap_columns(matrice matrix, int n, int m, int c1, int c2):
 * exchange columns c1,c2 of an (nxm) rational matrix
 *
 *	Precondition:	n > 0; m > 0; 0 < c1 <= m; 0 < c2 <= m; 
 */
void matrice_swap_columns(matrix,n,m,c1,c2)
matrice	matrix;		/* input and output matrix */
int	n;			/* number of rows */
int	m;			/* number of columns */
int	c1,c2;			/* column numbers */
{
    int	loop1;
    register Value temp;

    /* validation step */
    /*
     * if ((n < 1) || (m < 1))   return(-208);
     * if ((c1 > m) || (c2 > m)) return(-209);
     */
    assert(n > 0 && m > 0);
    assert(0 < c1 && c1 <= m);
    assert(0 < c2 && c2 <= m);

    for(loop1=1; loop1<=n; loop1++) {
	temp = ACCESS(matrix,n,loop1,c1);
	ACCESS(matrix,n,loop1,c1) = ACCESS(matrix,n,loop1,c2);
	ACCESS(matrix,n,loop1,c2) = temp;
    }
}	

/* void matrice_swap_rows(matrice a, int n, int m, int r1, int r2):
 * exchange rows r1 and r2 of an (nxm) rational matrix a
 *
 * Precondition: n > 0; m > 0; 1 <= r1 <= n; 1 <= r2 <= n
 */
void matrice_swap_rows(a,n,m,r1,r2)
matrice a;		/* input and output matrix */
int	n;			/* number of columns */
int	m;			/* number of rows */
int	r1,r2;			/* row numbers */
{
    int	loop1;
    register Value temp;

    /* validation */
    assert(n > 0);
    assert(m > 0);
    assert(0 < r1 && r1 <= n);
    assert(0 < r2 && r2 <= n);

    for(loop1=1; loop1<=m; loop1++) {
	temp = ACCESS(a,n,r1,loop1);
	ACCESS(a,n,r1,loop1) = ACCESS(a,n,r2,loop1);
	ACCESS(a,n,r2,loop1) = temp;
    }
}

/* void matrice_assign(matrice A, matrice B, int n, int m)
 * Copie de la matrice A dans la matrice B
 *
 *  Les parametres de la fonction :
 *
 * int 	A[]	:  matrice
 *!int 	B[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 *
 * Ancien nom: mat_dup
 */
void matrice_assign(A,B,n,m)
matrice A,B;
int n,m;
{
    int i, size=n*m;
    for (i = 0 ;i<=size;i++)
	B[i] = A[i];
}

/* bool matrice_egalite(matrice A, matrice B, int n, int m)
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
bool matrice_egalite(A,B,n,m)
matrice A, B;
int n,m;
{
    int i;
    for (i = 0 ;i<= n*m;i++)
	if(value_ne(B[i],A[i]))
	    return(false);
    return(true);
}

/* void matrice_nulle(matrice Z, int n, int m):
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
void matrice_nulle(Z,n,m)
matrice Z;
int n,m;
{
    int i,j;

    for (i=1;i<=n;i++)
	for (j=1;j<=m;j++)
	    ACCESS(Z,n,i,j)=VALUE_ZERO;
    DENOMINATOR(Z) = VALUE_ONE;
}

/* bool matrice_nulle_p(matrice Z, int n, int m):
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
bool matrice_nulle_p(Z,n,m)
matrice Z;
int n,m;
{
    int i,j;

    for (i=1;i<=n;i++)
	for (j=1;j<=m;j++)
	    if(value_notzero_p(ACCESS(Z,n,i,j)))
		return(false);
    return(true);
}

/* bool matrice_diagonale_p(matrice Z, int n, int m):
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
bool matrice_diagonale_p(Z,n,m)
matrice Z;
int n,m;
{
    int i,j;

    for (i=1;i<=n;i++)
	for (j=1;j<=m;j++)
	    if(i!=j && value_notzero_p(ACCESS(Z,n,i,j)))
		return(false);
    return(true);
}

/* bool matrice_triangulaire_p(matrice Z, int n, int m, bool inferieure):
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
bool matrice_triangulaire_p(Z,n,m,inferieure)
matrice Z;
int n,m;
bool inferieure;
{
    int i,j;

    for (i=1; i <= n; i++)
        if(inferieure) {
	    for (j=i+1; j <= m; j++)
		if(value_notzero_p(ACCESS(Z,n,i,j)))
		    return(false);
	}
	else
	    for (j=1; j <= i-1; j++)
		if(value_notzero_p(ACCESS(Z,n,i,j)))
		    return(false);
    return(true);
}

/* bool matrice_triangulaire_unimodulaire_p(matrice Z, int n, bool inferieure)
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
bool matrice_triangulaire_unimodulaire_p(Z,n,inferieure)
matrice Z;
int n;
bool inferieure;
{
    bool triangulaire;
    int i;
    triangulaire = matrice_triangulaire_p(Z,n,n,inferieure);
    if (triangulaire == false)
	return(false);
    else{
	for(i=1; i<=n; i++)
	    if (value_notone_p(ACCESS(Z,n,i,i)))
		return(false);
	return(true);
    }
}	    

/* void matrice_substract(matrice a, matrice b, matrice c, int n, int m):
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
 * (see matrice_normalize()).
 *
 * Precondition:	n > 0; m > 0;
 * Note: aliasing between a and b or c is supported
 */
void matrice_substract(a,b,c,n,m)
matrice a;         /* output */
matrice b, c;      /* input */
int n,m;           /* input */
{
    register Value d1,d2;   /* denominators of b, c */
    register Value lcm;     /* ppcm of b,c */
    int i,j;

    /* precondition */
    assert(n>0 && m>0);
    assert(value_pos_p(DENOMINATOR(b)));
    assert(value_pos_p(DENOMINATOR(c)));

    d1 = DENOMINATOR(b);
    d2 = DENOMINATOR(c);
    if (d1 == d2){
	for (i=1; i<=n; i++)
	    for (j=1; j<=m; j++)
		ACCESS(a,n,i,j) = value_minus(ACCESS(b,n,i,j),
					      ACCESS(c,n,i,j));
	DENOMINATOR(a) = d1;
    }
    else{
	register Value v1,v2;
	lcm = ppcm(d1,d2);
	DENOMINATOR(a) = lcm;
	d1 = value_div(lcm,d1);
	d2 = value_div(lcm,d2);
	for (i=1; i<=n; i++)
	    for (j=1; j<=m; j++) 
	    {
		v1 = ACCESS(b,n,i,j);
		value_product(v1, d1);
		v2 = ACCESS(c,n,i,j);
		value_product(v2, d2);
		ACCESS(a,n,i,j) = value_minus(v1,v2);
	    }
    }
}

/* void matrice_soustraction_colonne(matrice MAT,int n,int m,int c1,int c2,int x):
 * Soustrait x fois la colonne c2 de la colonne c1
 * Precondition: n > 0; m > 0; 0 < c1, c2 < m;
 * Effet: c1[0..n-1] = c1[0..n-1] - x*c2[0..n-1].
 *
 *  Les parametres de la fonction :
 *
 * int  MAT[]     :  matrice
 * int  n         :  nombre de lignes de la matrice
 * int  m         :  nombre de colonnes de la matrice
 * int  c1        :  numero du colonne
 * int  c2        :  numero du colonne
 * int  x         : 
 */
void matrice_soustraction_colonne(matrice MAT,
				  int n,
				  int m __attribute__((unused)),
				  int c1,
				  int c2,
				  Value x)
{
    int i;
    register Value p;
    for (i=1; i<=n; i++)
    {
	p = ACCESS(MAT,n,i,c2);
	value_product(p,x);
	value_substract(ACCESS(MAT,n,i,c1),p);
    }
}

/* void matrice_soustraction_ligne(matrice MAT,int n,int m,int r1,int r2,int x):
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
void matrice_soustraction_ligne(MAT,n,m,r1,r2,x)
matrice MAT;        
int n,m; 
int r1,r2;
Value x;
{
    int i;
    register Value p;
    for (i=1; i<=m; i++)
    {
	p = ACCESS(MAT,n,r2,i);
	value_product(p,x);
	value_substract(ACCESS(MAT,n,r1,i),p);
    }
}
