/* $RCSfile: system.c,v $ version $Revision$
 * ($Date: 2003/12/22 15:56:12 $, )
 *
 * a safe system call. abort if fails.
 * FC 09/95
 *
 * $Log: system.c,v $
 * Revision 1.10  2003/12/22 15:56:12  irigoin
 * Function safe_system_no_abort_no_warning() added without restructuring to
 * fit the three system functions into one general frame.
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"

void safe_system(string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status)
	pips_internal_error("Failed (ret: %d, sig: %d) for %s\n", 
			    (status/0x100) & 0xff, status & 0xff, command);
}

int safe_system_no_abort(string command) /* the command to be executed */
{
    int status = system(command);
    
    if (status == 127)
	pips_internal_error("Could not execute : '%s'\n", command);

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
	pips_internal_error("Could not execute : '%s'\n", command);

    return (status / 256) & 255;
}

/* that is all
 */
