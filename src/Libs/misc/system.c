/* $RCSfile: system.c,v $ version $Revision$
 * ($Date: 1995/09/15 18:50:20 $, )
 *
 * a safe system call. abort if fails.
 * FC 09/95
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"

void safe_system(
    string command) /* the command to be executed */
{
    int status = system(command);

    if (status) 
	pips_error("safe_system", "failed (%d) for %s\n", status, command);
}

/* that is all
 */
