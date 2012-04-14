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

// FI: just in case another data structure would be more suitable or
// more efficient
// For instance, set of points-to could be a type declared in
// points_to_private.tex
typedef set pt_map;
#define pt_map_undefined set_undefined
#define pt_map_undefined_p(pt) ((pt)==set_undefined)
#define new_pt_map() set_generic_make(set_private, points_to_equal_p, points_to_rank)
#define assign_pt_map(x,y) set_assign(x, y)
#define clear_pt_map(pt) set_clear(pt)
#define free_pt_map(pt) set_free(pt)
#define print_pt_map(pt) print_points_to_set("",pt);
// FI: varargs; probably OK with gcc preprocessor
#define free_pt_maps sets_free
#define union_of_pt_maps(pt1, pt2, pt3) set_union(pt1, pt2, pt3)
#define difference_of_pt_maps(pt1, pt2, pt3) set_difference(pt1, pt2, pt3)
#define consistent_pt_map(s) consistent_points_to_set(s)
// FI: Not so sure we do not need a new name
#define source_in_pt_map_p(cell,set) source_in_set_p(cell,set)
#define add_arc_to_pt_map(a, s) set_add_element((set) s, (set) s, (void *) a)
