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

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"

/* void matrix_fprint(File * f, matrice a,n,m): print an (nxm) rational matrix
 *
 * Note: the output format is incompatible with matrix_fscan()
 */
void matrix_fprint(f,a)
FILE * f;
Pmatrix	a;

{
    int	loop1,loop2;
    int n =MATRIX_NB_LINES(a);
    int m = MATRIX_NB_COLUMNS(a);
    assert(MATRIX_DENOMINATOR(a)!=0);

    (void) fprintf(f,"\n\n");

    for(loop1=1; loop1<=n; loop1++) {
	for(loop2=1; loop2<=m; loop2++)
	    matrix_pr_quot(f, 
			   MATRIX_ELEM(a,loop1,loop2), 
			   MATRIX_DENOMINATOR(a));
	(void) fprintf(f,"\n\n");
    }
    (void) fprintf(f," ......denominator = ");
    (void) fprint_Value(f, MATRIX_DENOMINATOR(a));
    (void) fprintf(f, "\n");
}

/* void matrix_print(matrice a): print an (nxm) rational matrix
 *
 * Note: the output format is incompatible with matrix_fscan()
 *       this should be implemented as a macro, but it's a function for
 *       dbx's sake
 */
void matrix_print(a)
Pmatrix	a;
{
    matrix_fprint(stdout,a);
}

/* void matrix_fscan(FILE * f, matrice * a, int * n, int * m): 
 * read an (nxm) rational matrix in an ASCII file accessible 
 * via file descriptor f; a is allocated as a function of its number
 * of columns and rows, n and m.
 *
 * Format: 
 *
 * n m denominator a[1,1] a[1,2] ... a[1,m] a[2,1] ... a[2,m] ... a[n,m]
 *
 * After the two dimensions and the global denominator, the matrix as
 * usually displayed, line by line. Line feed can be used anywhere.
 * Example for a (2x3) integer matrix:
 *  2 3 
 *  1
 *  1 2 3
 *  4 5 6
 */
void matrix_fscan(f,a,n,m)
FILE * f;
Pmatrix  *a;
int * n;			/* column size */
int * m;			/* row size */
{
    int	r;
    int c;

    /* number of read elements */
    int n_read;

    /* read dimensions */
    n_read = fscanf(f,"%d%d", n, m);
    assert(n_read==2);
    assert(1 <= *n && 1 <= *m);

    /* allocate a */
    *a = matrix_new(*n,*m); 

    /* read denominator */
    n_read = fscan_Value(f,&(MATRIX_DENOMINATOR(*a)));
    assert(n_read == 1);
    /* necessaires pour eviter les divisions par zero */
    assert(value_notzero_p(MATRIX_DENOMINATOR(*a)));
    /* pour "normaliser" un peu les representations */
    assert(value_pos_p(MATRIX_DENOMINATOR(*a)));

    for(r = 1; r <= *n; r++) 
	for(c = 1; c <= *m; c++) {
	    n_read = fscan_Value(f, &MATRIX_ELEM(*a,r,c));
	    assert(n_read == 1);
	}
}

/* void matrix_pr_quot(FILE * f, int a, int b): print quotient a/b in a sensible way,
 * i.e. add a space in front of positive number to compensate for the
 * minus sign of negative number
 *
 * FI: a quoi sert le parametre b? A quoi sert la variable d? =>ARGSUSED
 */
/*ARGSUSED*/
void matrix_pr_quot(FILE * f,
		    Value a,
		    Value b __attribute__ ((unused)))
{	
    if (value_pos_p(a))
	fprintf(f, " ");
    fprintf(f, " ");
    fprint_Value(f, a);
    fprintf(f, " ");
}
