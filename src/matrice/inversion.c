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

/* void matrice_unimodulaire_triangulaire_inversion(matrice u ,matrice inv_u,
 * int n, * bool infer)
 * u soit le matrice unimodulaire triangulaire.
 * si infer = true  (triangulaire inferieure),
 *    infer = false (triangulaire superieure). 
 * calcul de l'inversion de matrice u telle que I = U x INV_U .
 * Les parametres de la fonction :
 *
 * matrice u     : matrice unimodulaire triangulaire     -- inpout
 * int n         : dimension de la matrice caree         -- inpout
 * bool infer : type de triangulaire                  -- input
 * matrice inv_u : l'inversion de matrice u              -- output
 */
void matrice_unimodulaire_triangulaire_inversion(u,inv_u,n,infer)
matrice u;
matrice inv_u;
int n;
bool infer;
{
    int i, j;
    Value x;

    /* test de l'unimodularite et de la trangularite de u */
    assert(matrice_triangulaire_unimodulaire_p(u,n,infer));

    matrice_identite(inv_u,n,0);
    if (infer){
	for (i=n; i>=1;i--)
	    for (j=i-1; j>=1; j--){
		x = ACCESS(u,n,i,j);
		if (value_notzero_p(x))
		   matrice_soustraction_colonne(inv_u,n,n,j,i,x); 
	    }
    }
    else{
	for (i=1; i<=n; i++)
	    for(j=i+1; j<=n; j++){
		x = ACCESS(u,n,i,j);
		if (value_notzero_p(x))
		    matrice_soustraction_colonne(inv_u,n,n,j,i,x);
	    }
    }
}

/* void matrice_diagonale_inversion(matrice s,matrice inv_s,int n)
 * calcul de l'inversion du matrice en forme reduite smith.
 * s est un matrice de la forme reduite smith, inv_s est l'inversion 
 * de s ; telles que s * inv_s = I.
 *
 * les parametres de la fonction :
 * matrice s     : matrice en forme reduite smith     -- input
 * int n         : dimension de la matrice caree      -- input
 * matrice inv_s : l'inversion de s                   -- output
 */
void matrice_diagonale_inversion(s,inv_s,n)
matrice s;
matrice inv_s;
int n;
{
    Value d, d1; 
    Value gcd, lcm;
    int i;

    /* tests des preconditions */
    assert(matrice_diagonale_p(s,n,n));
    assert(matrice_hermite_rank(s,n,n)==n);
    assert(value_pos_p(DENOMINATOR(s)));

    matrice_nulle(inv_s,n,n);
    /* calcul de la ppcm(s[1,1],s[2,2],...s[n,n]) */
    lcm = ACCESS(s,n,1,1);
    for (i=2; i<=n; i++)
	lcm = ppcm(lcm,ACCESS(s,n,i,i));
    
    d = DENOMINATOR(s);
    gcd = pgcd_slow(lcm,d);
    d1 = value_div(d,gcd);
    for (i=1; i<=n; i++) {
	Value tmp = value_div(lcm,ACCESS(s,n,i,i));
	ACCESS(inv_s,n,i,i) = value_mult(d1,tmp);
    }
    DENOMINATOR(inv_s) = value_div(lcm,gcd);
}
    
/* void matrice_triangulaire_inversion(matrice h, matrice inv_h, int n,bool infer)
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
void matrice_triangulaire_inversion(h,inv_h,n,infer)
matrice h;
matrice inv_h;
int n;
bool infer;
{
    Value deno,deno1;                    /* denominateur */
    Value determinant,sous_determinant;  /* determinant */
    Value gcd;
    int i,j;
    Value aij[2];
    register Value a;

    /* tests des preconditions */
    assert(matrice_triangulaire_p(h,n,n,infer));
    assert(matrice_hermite_rank(h,n,n)==n);
    assert(value_pos_p(DENOMINATOR(h)));

    matrice_nulle(inv_h,n,n);
    deno = DENOMINATOR(h);
    deno1 = deno;
    DENOMINATOR(h) = VALUE_ONE;
    /* calcul du determinant de h */
    determinant = VALUE_ONE;
    for (i= 1; i<=n; i++){
	a = ACCESS(h,n,i,i);
	value_product(determinant,a);
    }
    /* calcul du denominateur de inv_h */
    gcd = pgcd_slow(deno1,determinant);
    if (value_notone_p(gcd)){
	value_division(deno1,gcd);
	value_division(determinant,gcd);
    }
    if (value_neg_p(determinant)){
	value_oppose(deno1); 
	value_oppose(determinant);
    }
    DENOMINATOR(inv_h) = determinant;
    /* calcul des sous_determinants des Aii */
    for (i=1; i<=n; i++){
	sous_determinant = VALUE_ONE;
	for (j=1; j<=n; j++)
	    if (j != i) {
		a = ACCESS(h,n,j,j);
		value_product(sous_determinant,a) ;
	    }
	ACCESS(inv_h,n,i,i) = value_mult(sous_determinant,deno1);
    }
    /* calcul des sous_determinants des Aij (i<j) */
    if (infer) {
	for (i=1; i<=n; i++)
	    for(j=i+1; j<=n;j++){
		matrice_sous_determinant(h,n,i,j,aij);
		assert(aij[0] == VALUE_ONE);
		ACCESS(inv_h,n,j,i) = value_mult(aij[1],deno1);
	    }
    }
    else {
	for (i=1; i<=n; i++)
	    for(j=1; j<i; j++){
		matrice_sous_determinant(h,n,i,j,aij);
		assert(aij[0] == VALUE_ONE);
		ACCESS(inv_h,n,j,i) = value_mult(aij[1],deno1);
	    }
    }
    
    DENOMINATOR(h) = deno;
}

/* void matrice_general_inversion(matrice a; matrice inv_a; int n)
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
void matrice_general_inversion(a,inv_a,n)
matrice a;
matrice inv_a;
int n;
{
    matrice p = matrice_new(n,n);
    matrice q = matrice_new(n,n);
    matrice h = matrice_new(n,n);
    matrice inv_h = matrice_new(n,n);
    matrice temp = matrice_new(n,n);
    Value deno;
    Value det_p, det_q; /* ne utilise pas */

    /* test */
    assert(value_pos_p(DENOMINATOR(a)));

    deno = DENOMINATOR(a);
    DENOMINATOR(a) = VALUE_ONE;
    matrice_hermite(a,n,n,p,h,q,&det_p,&det_q);
    DENOMINATOR(a) = deno;
    if ( matrice_hermite_rank(h,n,n) == n){
	DENOMINATOR(h) = deno;
	matrice_triangulaire_inversion(h,inv_h,n,true);
	matrice_multiply(q,inv_h,temp,n,n,n);
	matrice_multiply(temp,p,inv_a,n,n,n);
    }
    else{
	printf(" L'inverse de la matrice a n'existe pas!\n");
	exit(1);
    }
}

/* void matrice_unimodulaire_inversion(matrice u, matrice inv_u, int n)
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
void matrice_unimodulaire_inversion(u,inv_u,n)
matrice u;
matrice inv_u;
int n;
{
    matrice p = matrice_new(n,n);
    matrice q = matrice_new(n,n);
    matrice h_u = matrice_new(n,n);
    matrice inv_h_u = matrice_new(n,n);
    matrice temp = matrice_new(n,n);
    Value det_p,det_q;  /* ne utilise pas */
   
    /* test */
    assert(value_one_p(DENOMINATOR(u)));

    matrice_hermite(u,n,n,p,h_u,q,&det_p,&det_q);
    assert(matrice_triangulaire_unimodulaire_p(h_u,n,true));
    matrice_unimodulaire_triangulaire_inversion(h_u,inv_h_u,n,true);
    matrice_multiply(q,inv_h_u,temp,n,n,n);
    matrice_multiply(temp,p,inv_u,n,n,n);
}
    
    
	  

	   
 
