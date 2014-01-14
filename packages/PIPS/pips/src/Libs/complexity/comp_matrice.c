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
/* comp_matrice.c
 *
 * matrice inversion for floating point. 20 Oct. 92 L. Zhou
 *
 * void lu_decomposition(a, n, indx, pd)
 *      Given an NxN matrix A, with physical dimension NP,
 * here we give NP the value MAX_CONTROLS_IN_UNSTRUCTURED ( that is 100 ) 
 * this routine 
 * replaces it by the LU decomposition of a rowwise permutation of itself.
 * A and N are input, A is output
 * INDX is output vector which recoreds the row permutation effected by the 
 * partial pivoting;
 * D is out put as +- 1 depending on whether the number of row interchanges 
 * was even or odd, respectively.
 *
 * void lu_back_substitution(a, n, indx, b)
 *  
 */

#include <stdio.h>
#include <math.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "matrice.h"
#include "complexity.h"


/* Added by AP, March 15th 93 */
#define A(i,j) a[n*i + j]

void lu_decomposition(a, n, indx, pd)
int n;

/* Modif by AP, March 15th 93
float a[n][n];
int indx[n];
*/
float *a;
int *indx;

int *pd;
{
    float aamax;

/* Added by AP, March 15th 93 */
    float vv[MAX_CONTROLS_IN_UNSTRUCTURED];
/* If using gcc to compile, vv can be declared as vv[n] LZ 29/10/92
    float vv[n];
*/

    int i,j,k;
    float sum,dum;
    int imax = 0;
    float tiny = 0.000001;
    
    *pd = 1;

    /* loop over rows to get the implicit scaling information */
    for (i=0; i<n; i++ ) {
	aamax = 0.0;
	for ( j=0; j<n; j++) {
	    if ( fabs(A(i,j)) > aamax ) {
		aamax = fabs(A(i,j));
	    }
	}

	if ( aamax == 0.0 ) {
	    fprintf(stderr, "Singular matrix -- No non-zero element\n");
	    user_error("lu_decomposition", "Module does not terminate\n");
	}
	vv[i] = 1./aamax;
    }

    /* loopover columns of Crout's method */
    for ( j=0; j<n; j++ ) {
	for ( i=0; i<=j-1; i++ ) {
	    sum = A(i,j);
	    for ( k=0; k<i; k++) {
		sum = sum - A(i,k)*A(k,j);
	    }
	    A(i,j) = sum;
	}

	aamax = 0.0;

	for ( i=j; (i<n) ; i++ ) {
	    sum = A(i,j);
	    for ( k=0; k<j; k++ ) {
		sum = sum - A(i,k)*A(k,j);
	    }
	    A(i,j) = sum;
	    dum = vv[i]*fabs(sum);
	    if ( dum >= aamax ) {
		imax = i;
		aamax = dum;
	    }
	}

	if ( j != imax ) {
	    for ( k=0; k<n; k++ ) {
		dum = A(imax,k);
		A(imax,k) = A(j,k);
		A(j,k) = dum;
	    }
	    *pd = - *pd;
	    vv[imax] = vv[j];
	}

	indx[j] = imax;
	if ( A(j,j) == 0.0 )
	    A(j,j) = tiny;
	if ( j != n-1 ) {
	    dum = 1./A(j,j);
	    for ( i=j+1; i<n; i++ ) {
		A(i,j) = A(i,j)*dum;
	    }
	}
    }
}

void lu_back_substitution(a, n, indx, b)
int n;
float *a;
int *indx;
float *b;
{
    int i,j;
    int ii = -1;
    float sum;
    int ll;

    for (i=0; i<n; i++ ) {
	ll = indx[i];
	sum = b[ll];
	b[ll] = b[i];
	if ( ii != -1 ) {
	    for ( j=ii; j<=i-1; j++ ) {
		sum = sum -A(i,j)*b[j];
	    }
	}
	else if ( sum != 0.0 ) {
	    ii = i;
	}
	b[i] = sum;
    }

    /* back substitution */
    for ( i=n-1; i>=0; i-- ) {
	sum = b[i];
	if ( i < n ) {
	    for ( j=i+1; j < n; j++ ) {
		sum = sum - A(i,j)*b[j];
	    }
	}
	b[i] = sum/A(i,i);
    }
}

/* Added by AP, March 23rd 93: allows the simulation of a two-dimensional array
 * with a mono-dimensional memory allocation.
 */
#define Y(i,j) y[n*i + j]

void float_matrice_inversion(a, n, indx, pd)
int n;
float *a;
int *indx;
int *pd;
{
    int i,j;
    float *y = (float *) malloc(sizeof(float) * n * n);

    /* set up identity matrix */
    for (i=0; i<n; i++ ) {
	for ( j=0; j<n; j++ )
	    Y(i,j) = 0.0;
	Y(i,i) = 1.0;
    }
		
    lu_decomposition(a, n, indx, pd);

    for ( j=0; j<n; j++ ) {
	lu_back_substitution(a, n, indx, &(Y(j,0)) );
    }
    for ( j=0; j<n; j++ )
	for ( i=0; i<n; i++ )
	    A(i,j) = Y(j,i);
}
