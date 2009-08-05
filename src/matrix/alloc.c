/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

 /* package matrix */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"

Pmatrix matrix_new(n,m)
int n,m;
{ 
    Pmatrix a = (Pmatrix) malloc(sizeof(Smatrix));
    a->denominator = VALUE_ONE;
    a->number_of_lines = n;
    a->number_of_columns = m;
    a->coefficients = (Value *) malloc(sizeof(Value)*((n*m)+1));
    return (a);
}
