/* Some modifications are made to save the current makefile (s.a. files
 * pipsmake/readmakefile.y openclose.h )
 * They only occure between following tags: 
 */
/**** Begin saved_makefile version ****/
/**** End saved_makefile version ****/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <sys/types.h>

/* Some modifications are made to save the current makefile (s.a. files
 * pipsmake/readmakefile.y pipsmake.h )
 * They only occure between following tags: 
 *
 * Bruno Baron
 */

#include <string.h>
#include <sys/param.h>

#include "genC.h"
#include "database.h"
#include "makefile.h"
#include "linear.h"
#include "ri.h"

#include "properties.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "misc.h"

extern makefile open_makefile();

/* returns the program makefile file name */
/* .pipsmake should be hidden in the .database
 * I move it to the .database
 * LZ 02/07/91
 * Next thing to do is to delete the prefix of .pipsmake
 * it's redundant. Done 04/07/91 
 */
char *
build_pgm_makefile(char *n)
{
    string dir_name = db_get_meta_data_directory(),
	res = strdup(concatenate(dir_name, "/pipsmake", 0));
    free(dir_name); return res;
}

string 
make_open_workspace(string name)
{
    if (db_open_workspace(name)) {
	open_properties();
	if (open_makefile(name) != makefile_undefined) {
	    pips_debug(7, "makefile opened\n");
	} else
	    pips_user_warning("No special makefile for this workspace "
			      "%s/%s.database\n", get_cwd(), name);
    } else
	pips_user_warning("No workspace %s to open\n", name);

    return db_get_current_workspace_name();
}

/* FI->GO: could be in top-level, no?
 */
bool 
make_close_workspace(void)
{
    bool res = TRUE;
    string name;

    if (db_get_current_module_name()) /* lazy... */
	db_reset_current_module_name();

    /* dup because freed in db_close_workspace */
    name = strdup(db_get_current_workspace_name()); 

    res &= close_makefile(name);
    save_properties();
    res &= db_close_workspace();

    if(res)
	user_log("Workspace %s closed.\n\n", name);
    else
	user_log("Failed to close workspace %s.\n\n", name);

    free(name);
    return res;
}

/* checkpoint the current workspace, i.e. save everything so
 * that it is possible to reopen it in case of failure. 
 */
void
checkpoint_workspace(void)
{
    if (db_get_current_workspace_name())
    {
	user_log("Checkpoint of workspace.\n");
	pips_debug(3, "\tdeleting obsolete resources...\n");
	delete_obsolete_resources();
	pips_debug(3, "\tsaving resources...\n");
	db_checkpoint_workspace();
	pips_debug(3, "\tproperties and makefile...\n");
	save_properties();
	save_makefile();
    }
}
