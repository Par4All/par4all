/* $RCSfile: system.c,v $ version $Revision$
 * ($Date: 1997/05/01 09:39:08 $, )
 *
 * a safe system call. abort if fails.
 * FC 09/95
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"

void
safe_system(
    string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status)
	pips_internal_error("Failed (ret: %d, sig: %d) for %s\n", 
			    (status/0x100) & 0xff, status & 0xff, command);
}

int
safe_system_no_abort(
    string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status == 127)
	pips_error("safe_system", "Could not execute : '%s'\n", command);

    if (status)
	pips_user_warning("Failed (ret: %d, sig: %d) for '%s'\n", 
			  (status/0x100) & 0xff, status & 0xff, command);

    return (status / 256) & 255;
}
/* that is all
 */
