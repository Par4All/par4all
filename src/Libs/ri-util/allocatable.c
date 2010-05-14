/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2009-2010 HPC Project

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
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "ri-util.h"



/***
 * Handling of ALLOCATABLE (Fortran95)
 *
 * Allocatable are represented internally as structure, for instance :
 *
 * integer, dimension (:,:,:), allocatable :: myarray
 *
 * would be represented as (pseudo code display here) :
 *
 * struct __pips_allocatable__2D myarray;
 *
 * with
 *
 * struct __pips_allocatable__2D {
 *  int lowerbound1;
 *  int upperbound1;
 *  int lowerbound2;
 *  int upperbound2;
 *  int data[lowerbound1:upperbound1][lowerbound2:upperbound2]
 * }
 *
 * The structure is dependent of the number of dimension and is created
 * dynamically when encounting an allocatable declaration.
 *
 * The prettyprint recognize the structure based on the special prefix and
 * display it as an allocatable array in Fortran95.
 *
 */












