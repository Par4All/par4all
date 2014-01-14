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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "polyedre.h"

#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "database.h"

#include "misc.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "control.h"

#include "constants.h"
#include "properties.h"
#include "resources.h"

#include "chains.h"
#include "ricedg.h"
#include "rice.h"
#include "effects-simple.h"
