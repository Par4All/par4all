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
/* -- variables.c
 *
 * package atomizer :  Alexis Platonoff, juillet 91
 * --
 *
 * This file contains functions that creates the new temporary entities and
 * the new auxiliary entities, and functions that compute the "basic" of an
 * expression.
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "graph.h"
#include "dg.h"

#include "ri-util.h"
/* #include "constants.h" */
#include "misc.h"

/* AP: I removed this include because it no longer exists
#include "loop_normalize.h" */

#include "atomizer.h"

/* FI: I moved these procedures in ri-util/variable.c and ri-util/type.c */
