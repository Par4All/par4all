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
/* interface with pipsmake
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "constants.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "database.h"
#include "resources.h"

/* #include "generation.h" */

#include "transformations.h"

bool loop_interchange(const char* module_name)
{
  bool return_status = false;

  return_status =
    interactive_loop_transformation(module_name,
				    interchange_inner_outermost_loops);

  return return_status;
}
