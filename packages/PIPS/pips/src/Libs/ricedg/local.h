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

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <setjmp.h>

#include "linear.h"

#include "genC.h"

#include "text.h"
#include "ri.h"
#include "effects.h"

#include "icfg.h"

#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "text-util.h"

#include "ri-util.h" /* linear.h is included in */
#include "effects-util.h" 
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "pipsdbm.h"
#include "semantics.h"
#include "pipsmake.h"

#include "constants.h"
#include "properties.h"
#include "resources.h"

#include "polyedre.h"

#include "ricedg.h" 

#include "expressions.h"
#include "alias_private.h"
#include "instrumentation.h"
#include "transformations.h"
#include "alias-classes.h"
