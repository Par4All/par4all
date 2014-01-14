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
 * a safe system call. abort if fails.
 * FC 09/95
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"

void safe_system(string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status)
	pips_internal_error("Failed (ret: %d, sig: %d) for %s", 
			    (status/0x100) & 0xff, status & 0xff, command);
}

int safe_system_no_abort(string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status == 127)
	pips_internal_error("Could not execute : '%s'", command);

    if (status) {
	/* For portability reasons, do not use pips_user_warning() here */
	pips_user_warning("Failed (ret: %d, sig: %d) for '%s'\n", 
			  (status/0x100) & 0xff, status & 0xff, command);
    }

    return (status / 256) & 255;
}

int safe_system_no_abort_no_warning(string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status == 127)
	pips_internal_error("Could not execute : '%s'", command);

    return (status / 256) & 255;
}

/* that is all
 */
