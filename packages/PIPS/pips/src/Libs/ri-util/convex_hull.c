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
 * Functions which should/could be in linear...
 */

#include <stdio.h>

#include "linear.h"
#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"

/* debug messages before calling the version in polyedre.
 */
Psysteme cute_convex_union(Psysteme s1, Psysteme s2)
{
  Psysteme s;

  ifdebug(9) {
    pips_debug(9, "IN\nS1 and S2:\n");
    sc_fprint(stderr, s1, (get_variable_name_t) entity_local_name);
    sc_fprint(stderr, s2, (get_variable_name_t) entity_local_name);
  }

  s = sc_cute_convex_hull(s1, s2);

  ifdebug(9) {
    pips_debug(9, "S =\n");
    sc_fprint(stderr, s, (get_variable_name_t) entity_local_name);
    pips_debug(9, "OUT\n");
  }

  return s;
}
