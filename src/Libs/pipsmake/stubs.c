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
 * Stub functions for old passes that have been moved out.
 */

#include <stdio.h>
#include "genC.h"
#include "misc.h"

#define STUB(p)								     \
bool p(string module)							     \
{									     \
    pips_user_warning("pass on %s no more implemented or linked\n", module); \
    return false;							     \
}

/*
 * PAF-related stuff, moved out 16/04/1998 by FC.
 * No problem to move it back in! Just port to new newgen;-)
 */
STUB(array_dfg)
STUB(print_array_dfg)
STUB(scheduling)
STUB(print_bdt)
STUB(prgm_mapping)
STUB(print_plc)
STUB(reindexing)
STUB(print_parallelizedCMF_code)
STUB(print_parallelizedCRAFT_code)
/* FI: needed
STUB(static_controlize)
STUB(print_code_static_control)
*/
