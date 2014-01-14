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
/*
 * stuff to include, local to pipsdbm.
 * SHOULD NOT include ri-util.h.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

typedef void * db_void;

#include "linear.h"

#include "resources.h"

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "properties.h"
#include "newgen.h" /* needed for statement_mapping */
#include "database.h" /* for obsolete functions... */
#include "pipsdbm.h"

