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
#include <stdio.h>

#include <math.h>

#define MAX_CONTROLS_IN_UNSTRUCTURED 100

void lu_decomposition(a, n, indx, pd)
int n;
float a[MAX_CONTROLS_IN_UNSTRUCTURED][MAX_CONTROLS_IN_UNSTRUCTURED];
int indx[];
int *pd;
{
    float aamax;
    float vv[MAX_CONTROLS_IN_UNSTRUCTURED];
    int i,j,k;
    float sum,dum;
    int imax;
    float tiny = 0.00;
    
    *pd = 1;

    /* loop over rows to get the implicit scaling information */
    for (i=0; i<n; i++ ) {
	aamax = 0.0;
	for ( j=0; j<n; j++) {
	    if ( fabs(a[i][j]) > aamax ) {
		aamax = fabs(a[i][j]);
	    }
	}

	if ( aamax == 0.0 )
	    fprintf(stderr, "Singular matrix -- No nonzero element\n");
	vv[i] = 1./aamax;
    }

    /* loopover columns of Crout's method */
    for ( j=0; j<n; j++ ) {
	for ( i=0; i<=j; i++ ) {
	    sum = a[i][j];
	    for ( k=0; k<i; k++) {
		sum = sum - a[i][k]*a[k][j];
	    }
	    a[i][j] = sum;
	}

	aamax = 0.0;

	for ( i=j+1; (i<n) ; i++ ) {
	    sum = a[i][j];
	    for ( k=0; k<j; k++ ) {
		sum = sum - a[i][k]*a[k][j];
	    }
	    a[i][j] = sum;
	    dum = vv[i]*fabs(sum);
	    if ( dum >= aamax ) {
		imax = i;
		aamax = dum;
	    }
	}

	if ( j != imax ) {
	    for ( k=0; k<n; k++ ) {
		dum = a[imax][k];
		a[imax][k] = a[j][k];
		a[j][k] = dum;
	    }
	    *pd = - *pd;
	    vv[imax] = vv[j];
	}

	indx[j] = imax;
	if ( a[j][j] == 0.0 )
	    a[j][j] = tiny;
	if ( j != n-1 ) {
	    dum = 1./a[j][j];
	    for ( i=j+1; i<n; i++ ) {
		a[i][j] = a[i][j]*dum;
	    }
	}
    }
}

void lu_back_substitution(a, n, indx, b)
float a[MAX_CONTROLS_IN_UNSTRUCTURED][MAX_CONTROLS_IN_UNSTRUCTURED];
int n;
int indx[];
float b[];
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
		sum = sum -a[i][j]*b[j];
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
		sum = sum - a[i][j]*b[j];
	    }
	}
	b[i] = sum/a[i][i];
    }
}

void float_matrice_inversion(a, n, indx, pd)
int n;
float a[MAX_CONTROLS_IN_UNSTRUCTURED][MAX_CONTROLS_IN_UNSTRUCTURED];
int indx[];
int *pd;
{
    int i,j;
    float y[MAX_CONTROLS_IN_UNSTRUCTURED][MAX_CONTROLS_IN_UNSTRUCTURED];

    /* set up identity matrix */
    for (i=0; i<n; i++ ) {
	for ( j=0; j<n; j++ )
	    y[i][j] = 0.0;
	y[i][i] = 1.0;
    }
		
    lu_decomposition(a, n, indx, pd);

    for ( j=0; j<n; j++ ) {
	lu_back_substitution(a, n, indx, y[j]);
    }
    for ( j=0; j<n; j++ )
	for ( i=0; i<n; i++ )
	    a[i][j] = y[j][i];
}

main()
{
    float a[MAX_CONTROLS_IN_UNSTRUCTURED][MAX_CONTROLS_IN_UNSTRUCTURED];
    int n=40;
    int indx[MAX_CONTROLS_IN_UNSTRUCTURED];
    int d;
    int i,j;

    for (i=0; i<n; i++) {
	a[i][i] = 2.0;
	a[i][i+1] = -2.0;
    }

    a[2][3] = a[2][37] = -1.;
    a[6][7] = a[6][33] = -1.;
    a[10][11] = a[10][29] = -1.;
    a[15][16] = a[15][17] = -1.;
    a[16][14] =  -2.;
    a[16][17] =  0.0;
    a[17][18] = a[17][19] = -1.0;
    a[18][14] = -2.0;
    a[18][21] = 0.0;
    a[21][22] = a[21][23] = -1.0;
    a[22][12] = -2.;
    a[22][23] = 0.0;
    a[23][24] = a[23][28] = -1.;
    
    a[27][28] = 0.0;

    a[28][12] = -2.0;
    a[28][29] = 0.0;

    a[29][30] = a[30][31] = -1.0;
    a[30][12] = -2.0;
    a[30][31] = 0.0;

    a[32][12] = -2.0;
    a[32][33] = 0.0;

    a[33][34] = a[33][35] = -1.0;

    a[34][8] = -2.0;
    a[34][35] = 0.0;

    a[36][8] = -2.0;
    a[36][37] = 0.0;

    a[37][38] = a[37][39] = -1.0;

    a[38][5] = -2.0;
    a[38][39] = 0.0;

    a[39][5] = -2.0;

    for ( i=0; i<n; i++ ) {
	for(j=0; j<n; j++ ) {
	    a[i][j] = a[i][j]/2.0;
	    printf("%-4.1f ",a[i][j]);
	}
	printf("\n");
    }
    
    float_matrice_inversion(a, n, indx, &d);

    printf("Final results -----------------\n");
    printf("indx[0] indx[1] are %d %d\n",indx[0], indx[1] );
    printf("d is %d\n",d);

    printf("a[0][0] a[0][1] %f %f\n", (a[0][0]), (a[0][1]) );
    printf("a[1][0] a[1][1] %f %f\n", (a[1][0]), (a[1][1]) );
}

