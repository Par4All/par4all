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

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"

/* void matrix_unimodular_triangular_inversion(Pmatrix u ,Pmatrix inv_u,
 *  * bool infer)
 * u soit le matrice unimodulaire triangulaire.
 * si infer = true  (triangulaire inferieure),
 *    infer = false (triangulaire superieure). 
 * calcul de l'inversion de matrice u telle que I = U x INV_U .
 * Les parametres de la fonction :
 *
 * Pmatrix u     : matrice unimodulaire triangulaire     -- inpout
 * int n         : dimension de la matrice caree         -- inpout
 * bool infer : type de triangulaire                  -- input
 * matrice inv_u : l'inversion de matrice u              -- output
 */
void matrix_unimodular_triangular_inversion(u,inv_u,infer)
Pmatrix u;
Pmatrix inv_u;
bool infer;
{
    int i, j;
    Value x;
    int n = MATRIX_NB_LINES(u);
    /* test de l'unimodularite et de la trangularite de u */
    assert(matrix_triangular_unimodular_p(u,infer));

    matrix_identity(inv_u,0);
    if (infer){
	for (i=n; i>=1;i--)
	    for (j=i-1; j>=1; j--){
		x = MATRIX_ELEM(u,i,j);
		if (value_notzero_p(x))
		    matrix_subtraction_column(inv_u,j,i,x); 
	    }
    }
    else{
	for (i=1; i<=n; i++)
	    for(j=i+1; j<=n; j++){
		x = MATRIX_ELEM(u,i,j);
		if (value_notzero_p(x))
		    matrix_subtraction_column(inv_u,j,i,x);
	    }
    }
}

/* void matrix_diagonal_inversion(Pmatrix s,Pmatrix inv_s)
 * calcul de l'inversion du matrice en forme reduite smith.
 * s est un matrice de la forme reduite smith, inv_s est l'inversion 
 * de s ; telles que s * inv_s = I.
 *
 * les parametres de la fonction :
 * matrice s     : matrice en forme reduite smith     -- input
 * int n         : dimension de la matrice caree      -- input
 * matrice inv_s : l'inversion de s                   -- output
 */
void matrix_diagonal_inversion(s,inv_s)
Pmatrix s;
Pmatrix inv_s;
{
    Value d, d1; 
    Value gcd, lcm;
    int i;
    int n = MATRIX_NB_LINES(s);
    /* tests des preconditions */
    assert(matrix_diagonal_p(s));
    assert(matrix_hermite_rank(s)==n);
    assert(value_pos_p(MATRIX_DENOMINATOR(s)));

    matrix_nulle(inv_s);
    /* calcul de la ppcm(s[1,1],s[2,2],...s[n,n]) */
    lcm = MATRIX_ELEM(s,1,1);
    for (i=2; i<=n; i++)
	lcm = ppcm(lcm,MATRIX_ELEM(s,i,i));
    
    d = MATRIX_DENOMINATOR(s);
    gcd = pgcd(lcm,d);
    d1 = value_div(d,gcd);
    for (i=1; i<=n; i++) {
	Value tmp = value_div(lcm,MATRIX_ELEM(s,i,i));
	MATRIX_ELEM(inv_s,i,i) = value_mult(d1,tmp);
    }
    MATRIX_DENOMINATOR(inv_s) = value_div(lcm,gcd);
}
    
/* void matrix_triangular_inversion(Pmatrix h, Pmatrix inv_h,bool infer)
 * calcul de l'inversion du matrice en forme triangulaire.
 * soit h matrice de la reduite triangulaire; inv_h est l'inversion de 
 * h ; telle que : h * inv_h = I.
 * selon les proprietes de la matrice triangulaire:
 *    Aii = a11* ...aii-1*aii+1...*ann;
 *    Aij = 0     i>j  pour la matrice triangulaire inferieure (infer==true)
 *                i<j  pour la matrice triangulaire superieure (infer==false)
 *
 * les parametres de la fonction :
 * matrice h     : matrice en forme triangulaire        -- input
 * matrice inv_h : l'inversion de h                     -- output
 * int n         : dimension de la matrice caree        -- input
 * bool infer : le type de triangulaire              -- input
 */ 
void matrix_triangular_inversion(h,inv_h,infer)
Pmatrix h;
Pmatrix inv_h;
bool infer;
{
    Value deno,deno1;                    /* denominateur */
    Value determinant,sub_determinant;  /* determinant */
    Value gcd;
    int i,j;
    Value aij[2];
    
    /* tests des preconditions */
    int n = MATRIX_NB_LINES(h);
    assert(matrix_triangular_p(h,infer));
    assert(matrix_hermite_rank(h)==n);
    assert(value_pos_p(MATRIX_DENOMINATOR(h)));

    matrix_nulle(inv_h);
    deno = MATRIX_DENOMINATOR(h);
    deno1 = deno;
    MATRIX_DENOMINATOR(h) = VALUE_ONE;
    /* calcul du determinant de h */
    determinant = VALUE_ONE;
    for (i= 1; i<=n; i++)
	value_product(determinant, MATRIX_ELEM(h,i,i));

    /* calcul du denominateur de inv_h */
    gcd = pgcd(deno1,determinant);
    if (value_notone_p(gcd)){
	value_division(deno1,gcd);
	value_division(determinant,gcd);
    }
    if (value_neg_p(determinant)){
	value_oppose(deno1); 
	value_oppose(determinant);
    }
    MATRIX_DENOMINATOR(inv_h) = determinant;
    /* calcul des sub_determinants des Aii */
    for (i=1; i<=n; i++){
	sub_determinant = VALUE_ONE;
	for (j=1; j<=n; j++)
	    if (j != i)
		value_product(sub_determinant, MATRIX_ELEM(h,j,j));
	MATRIX_ELEM(inv_h,i,i) = value_mult(sub_determinant,deno1);
    }
    /* calcul des sub_determinants des Aij (i<j) */
    switch(infer) {
    case true:
	for (i=1; i<=n; i++)
	    for(j=i+1; j<=n;j++){
		matrix_sub_determinant(h,i,j,aij);
		assert(value_one_p(aij[0]));
		MATRIX_ELEM(inv_h,j,i) = value_mult(aij[1],deno1);
	    }
	break;
    case false:
	for (i=1; i<=n; i++)
	    for(j=1; j<i; j++){
		matrix_sub_determinant(h,i,j,aij);
		assert(value_one_p(aij[0]));
		MATRIX_ELEM(inv_h,j,i) = value_mult(aij[1],deno1);
	    }
	break;
   }
    MATRIX_DENOMINATOR(h) = deno;
}

/* void matrix_general_inversion(Pmatrix a; Pmatrix inv_a)
 * calcul de l'inversion du matrice general.
 * Algorithme : 
 *   calcul P, Q, H telque : PAQ = H ;
 *                            -1     -1
 *   si rank(H) = n ;        A  = Q H P .
 *
 * les parametres de la fonction : 
 * matrice a     : matrice general                ---- input
 * matrice inv_a : l'inversion de a               ---- output
 * int n         : dimensions de la matrice caree ---- input
 */
void matrix_general_inversion(a,inv_a)
Pmatrix a;
Pmatrix inv_a;
{
    int n = MATRIX_NB_LINES(a);
    Pmatrix p = matrix_new(n,n);
    Pmatrix q = matrix_new(n,n);
    Pmatrix h = matrix_new(n,n);
    Pmatrix inv_h = matrix_new(n,n);
    Pmatrix temp = matrix_new(n,n);
    Value deno;
    Value det_p, det_q;			/* ne utilise pas */

    /* test */
    assert(value_pos_p(MATRIX_DENOMINATOR(a)));

    deno = MATRIX_DENOMINATOR(a);
    MATRIX_DENOMINATOR(a) = VALUE_ONE;
    matrix_hermite(a,p,h,q,&det_p,&det_q);
    MATRIX_DENOMINATOR(a) = deno;
    if ( matrix_hermite_rank(h) == n){
	MATRIX_DENOMINATOR(h) = deno;
	matrix_triangular_inversion(h,inv_h,true);
	matrix_multiply(q,inv_h,temp);
	matrix_multiply(temp,p,inv_a);
    }
    else{
	fprintf(stderr," L'inverse de la matrice a n'existe pas!\n");
	exit(1);
    }
}

/* void matrix_unimodular_inversion(Pmatrix u, Pmatrix inv_u)
 * calcul de l'inversion de la matrice unimodulaire.
 * algorithme :
 * 1.  calcul la forme hermite de la matrice u :
 *          PUQ = H_U  (unimodulaire triangulaire);
 *      -1         -1
 * 2.  U  = Q (H_U) P.
 *
 * les parametres de la fonction :
 *    matrice u     : matrice unimodulaire     ---- input
 *    matrice inv_u : l'inversion de u         ---- output
 *    int n         : dimension de la matrice  ---- input
 */           
void matrix_unimodular_inversion(u,inv_u)
Pmatrix u;
Pmatrix inv_u;

{
    int n = MATRIX_NB_LINES(u);
    Pmatrix p = matrix_new(n,n);
    Pmatrix q = matrix_new(n,n);
    Pmatrix h_u = matrix_new(n,n);
    Pmatrix inv_h_u = matrix_new(n,n);
    Pmatrix temp = matrix_new(n,n);
    Value det_p,det_q;  /* ne utilise pas */
   
    /* test */
    assert(value_one_p(MATRIX_DENOMINATOR(u)));

    matrix_hermite(u,p,h_u,q,&det_p,&det_q);
    assert(matrix_triangular_unimodular_p(h_u,true));
    matrix_unimodular_triangular_inversion(h_u,inv_h_u,true);
    matrix_multiply(q,inv_h_u,temp);
    matrix_multiply(temp,p,inv_u);
}
    
    
	  

	   
 

