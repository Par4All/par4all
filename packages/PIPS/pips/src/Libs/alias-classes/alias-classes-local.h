/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#include "points_to_private.h"
#include "effects.h"
#define SEQUENTIAL_POINTS_TO_SETS_SUFFIX ".pt"
#define USER_POINTS_TO_SETS_SUFFIX ".upt"

typedef set pt_map;
#define pt_map_undefined set_undefined
#define new_pt_map() set_generic_make(set_private, points_to_equal_p, points_to_rank)
#define assign_pt_map(x,y) set_assign(x, y)
