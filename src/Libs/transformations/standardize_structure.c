/*
 * STF()
 *
 * Guillaume Oget
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "text-util.h"

#include "boolean.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"

#include "arithmetique.h"

#include "transformations.h"

/* Top-level functions
 */

bool 
stf(char *mod_name)
{

#define MAX__LENGTH 256

    char tmpfile[MAX__LENGTH];
    char outline[MAX__LENGTH];
    FILE *ftmp;
    int status;
    string wdn = db_get_current_workspace_directory();

    extern int system(char*);
    extern int unlink(char*);
    extern char* mktemp(char*);

    debug_on("STF_DEBUG_LEVEL");

    strncpy (tmpfile,".stf-workspace-outputXXXXXX",MAX__LENGTH - 1);
    mktemp (tmpfile);

    debug (9,"stf", "temporary filename for output %s\n", tmpfile);

    if (!(*tmpfile))
	pips_error("stf","unable to make a temporary file\n");

    status = safe_system_no_abort(concatenate
				  ("stf-module ",
				   wdn,
				   "/",
				   db_get_memory_resource(DBR_SOURCE_FILE, mod_name, TRUE),
				   " > ",
				   tmpfile,
				   " 2>&1 ",
				   NULL));

    debug (9,"stf","status=%d\n",status);

    /* Print log info if any */
    if ((ftmp = fopen (tmpfile,"r")) != NULL)
    {
	while (!feof (ftmp))
	{
	    if (fgets(outline, MAX__LENGTH, ftmp))
		user_log ("[stf-log] %s", outline);
	    else
		break;
	}
	fclose (ftmp);
	unlink (tmpfile);
    }
    else
	user_warning ("stf","No ouput from command\n");

    if (!status) {
	debug (1,"stf", "ok for module %s\n", mod_name);
	/* Why did GO use a touch instead of a put? */
	if(!db_update_time (DBR_SOURCE_FILE, mod_name))
	    user_error ("stf",
			"Cannot find new source file for module %s\n",
			mod_name);
    }
    else if (status == 2)
	user_error ("stf",
		    "should clean up with toolpack command for module %s\n",
		    mod_name);
    else
	user_error ("stf",
		    "failed for module %s\n",
		    mod_name);

    debug_off ();

    free(wdn);

    return TRUE;
}
