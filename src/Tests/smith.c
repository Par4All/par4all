/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>

#include "arithmetique.h"
#include "matrix.h"

int main(void)
{
  int m, n;
  Pmatrix A, P, D, Q;
  matrix_fscan(stdin, &A, &m, &n);
  P = matrix_new(m, m);
  D = matrix_new(m, n);
  Q = matrix_new(n, n);
  // compute Smith normal form: D = P.A.Q
  matrix_smith(A, P, D, Q);
  fprintf(stdout, "# A =\n");
  matrix_fprint(stdout, A);
  fprintf(stdout, "# P =\n");
  matrix_fprint(stdout, P);
  fprintf(stdout, "# D =\n");
  matrix_fprint(stdout, D);
  fprintf(stdout, "# Q =\n");
  matrix_fprint(stdout, Q);

  // discuss A x = 0 positive solutions
  int min_mn = m>n? n: m;
  int max_mn = m>n? m: n;
  int i;
  for (i = 1; i <= min_mn && MATRIX_ELEM(D, i, i) != VALUE_ZERO; i++);
  int nfree = max_mn-i+1;
  fprintf(stdout,
          "\n"
          "# A x = 0 strictly positive solutions:\n");
  if (nfree == 0) {
    fprintf(stdout, "# no free variable\n" "# no solution!\n" "# x* = 0\n");
  }
  else {
    fprintf(stdout, "# %d free variable%s\n", nfree, nfree>1? "s": "");
    for (int i=1; i<=n; i++) {
      fprintf(stdout, "# x%d = ", i);
      bool first = true;
      for (int j=0; j<nfree; j++) {
        Value v = matrix_elem(Q, i, n-j);
        if (v != VALUE_ZERO) {
          // should never be the case if no coupling
          if (!first) fprintf(stdout, "+ ");
          // value should be positive? if so, proof??
          fprint_Value(stdout, v);
          fprintf(stdout, " n%d", j);
          first = false;
        }
      }
      fprintf(stdout, "\n");
    }
  }
  return 0;
}
