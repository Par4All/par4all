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
  int n, m;
  Pmatrix A, P, D, Q;
  matrix_fscan(stdin, &A, &n, &m);
  P = matrix_new(n, n);
  D = matrix_new(n, m);
  Q = matrix_new(m, m);
  // D = P.A.Q
  matrix_smith(A, P, D, Q);
  matrix_fprint(stdout, A);
  matrix_fprint(stdout, P);
  matrix_fprint(stdout, D);
  matrix_fprint(stdout, Q);
  return 0;
}
