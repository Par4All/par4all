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
/* #include <sys/stdtypes.h>*/
/* #include "stdlib.h" */

#include "genC.h"

#include "misc.h"

void pips_malloc_debug()
{
    debug_on("MALLOC_DEBUG_LEVEL");
#if __GNUC__foo || sunfoo
    if (get_debug_level()==9) {
	debug(9, "pips_malloc_debug", 
	      "malloc_debug level of error diagnosis is 2\n");
	malloc_debug(2);
    }
    else if (get_debug_level()>=5) {
	debug(5, "pips_malloc_debug", "malloc(50) returns %x\n", 
	      (int) malloc(50));
	debug(5, "pips_malloc_debug", "call to malloc_verify()\n");
	pips_assert("pips_malloc_debug", malloc_verify());
 	malloc_debug(1);
    }
    else if (get_debug_level()>=1) 
	malloc_debug(1);
    else
	malloc_debug(0);
#else
    debug(1, "pips_malloc_debug", "No malloc_debug on this system\n");
#endif
    debug_off();
}
