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
/* Summary description of the target machine.
 *
 * Should be merged with the wp65 description and the complexity cost model by Lei Zhou
 *
 * Francois Irigoin, 16 January 1993
 */

#include <stdio.h>

#include "genC.h"


int get_processor_number(void)
{
    return 16;
}

int get_vector_register_length(void)
{
    return 64;
}

int get_vector_register_number(void)
{
    return 8;
}

#if 0
static int get_cache_line_size(void)
{
    return 1;
}
static int get_minimal_task_size(void)
{
    /* the unit is supposed to be consistent with the complexity cost tables used
     * that should be expressed in machine cycles
     */
    return 10000;
}
#endif
