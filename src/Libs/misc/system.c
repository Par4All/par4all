/* $RCSfile: system.c,v $ version $Revision$
 * ($Date: 1995/10/04 13:36:53 $, )
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
	pips_error("safe_system", "Failed (sig: %d, ret: %d) for %s\n", 
		   (status/0x100) & 0xff, status & 0xff, command);
}

int
safe_system_no_abort(
    string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status == 127)
	pips_error("safe_system", "Could not execute : %s\n", command);

    return (status / 256) & 255;
}
/* that is all
 */
