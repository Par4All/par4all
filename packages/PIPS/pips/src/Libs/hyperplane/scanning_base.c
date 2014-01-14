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
/* 
 * package hyperplane.
 *
 * Build a change of basis matrix G compatible with a hyperplane direction h
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
/* #include <sys/stdtypes.h> */ /* for debug with dbmalloc */
/* #include "stdlib.h" */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "misc.h"


#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"

#include "hyperplane.h"

/* void scanning_base_hyperplane(int h[],int n,matrice U)
 * compute the  matrix U :
 *   ->   ->  ->
 *   U1 = h  (U1  the first column of U)
 *   determinant(U) = +(/-)1.
 * 
 * Let G the scanning basis, the relation between U and T is
 *          -1 T   
 *    G = (U  )
 * the parameters of the  function are:
 *
 * int h[]   :  hyperplan direction  ---- input
 * int n     :  iteration dimension  ---- input
 * matrice G :  matrix  ---- output
 *             
 */
void scanning_base_hyperplane(h,n,G)
Value h[];
int n;
matrice G;
{
    int k=1;

    if (value_zero_p(h[0])){ 
	/* search the first k / h[k]!=0 */
	while(k<n && value_zero_p(h[k]))
	    k++;
	if (k==n) user_error("scanning_base_hyperplane", "h is null vector\n");
	else{ /* permution de h[0] et h[k] */
	    h[0] = h[k];
	    h[k] = VALUE_ZERO;
	    base_G_h1_unnull(h,n,G);
	    matrice_swap_rows(G,n,n,1,k+1);
	}
    }
    else 
	base_G_h1_unnull(h,n,G);    
}
	    
/*void base_G_h1_unnull(int h[],int n,matrice G)	
 */
void base_G_h1_unnull(h, n, G)	
Value h[];
int n;
matrice G;
{
    matrice U = matrice_new(n,n);
    matrice G1 = matrice_new(n,n);
    Value det_Ui = VALUE_ZERO;
    Value det_Ui1 = VALUE_ZERO;	/* determinant of Ui and Ui-1*/
    Value Xi,Yi;
    int i,j;
    Value r;
   
    /* computation of matrix U */
    assert(n>0);
    assert(h[0]!=0);
    /* initialisation */
    matrice_nulle(U,n,n);
    ACCESS(U,n,1,1) = h[0];
    det_Ui1 = h[0];

    for(i=2; i<=n; i++){
	/* computation of Xi,Yi / Xi.det(Ui-1) - Yi.hi  = GCD(det(Ui-1),hi) */
	det_Ui = bezout_grl(det_Ui1,h[i-1],&Xi,&Yi);
	if (i ==2 || i%2 != 0)
	    value_oppose(Yi);
	/* make  Ui - the i-th line: U[i,1]=h[i-1],U[i,2..n-1] =0,U[i,n] = Xi*/
	ACCESS(U,n,i,1) = h[i-1];
	ACCESS(U,n,i,i) = Xi;
        /*                                            ->i-1 */
	/* the i-th column:U[1..n-1] = Yi/det(Ui-1) . h      */
	for (j=1; j<=i-1; j++){
	    r = value_div(h[j-1],det_Ui1);
	    ACCESS(U,n,j,i) = value_mult(Yi,r);
	}
	det_Ui1 = det_Ui;
    }
    for (i=1; i<=n; i++)
	/*         ->         ->*/
	/* divide U1 par GCD(h) */
	value_division(ACCESS(U,n,i,1),det_Ui);

    /* computation of  matrix G */
    matrice_general_inversion(U,G1,n);
    matrice_transpose(G1,G,n,n);
}
