/* $RCSfile: system.c,v $ version $Revision$
 * ($Date: 1995/09/18 13:16:56 $, )
 *
 * a safe system call. abort if fails.
 * FC 09/95
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"

int safe_system(string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status == 127)
		pips_error("safe_system", "Could not execute : %s\n", command);

    return (status / 256) & 255;
}

/* that is all
 */
