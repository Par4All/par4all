/* Some modifications are made to save the current makefile (s.a. files
 * pipsmake/readmakefile.y openclose.h )
 * They only occure between following tags: 
 */
/**** Begin saved_makefile version ****/
/**** End saved_makefile version ****/
#include <stdio.h>
#include <sys/types.h>
/* Some modifications are made to save the current makefile (s.a. files
 * pipsmake/readmakefile.y pipsmake.h )
 * They only occure between following tags: 
 *
 * Bruno Baron
 */
/**** Begin saved_makefile version ****/
/**** End saved_makefile version ****/
#include <string.h>
#include <sys/param.h>

#include "genC.h"
#include "database.h"
#include "makefile.h"
#include "ri.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "misc.h"

extern makefile open_makefile();
extern void close_makefile();

/**** Begin saved_makefile version ****/
static char pgm_makefile[MAXPATHLEN]="";

/* returns the program makefile file name */
/* .pipsmake should be hidden in the .database
 * I move it to the .database
 * LZ 02/07/91
 * Next thing to do is to delete the prefix of .pipsmake
 * it's redundant. Done 04/07/91 
 */
char *build_pgm_makefile(n)
char *n;
{
    return(strcpy(pgm_makefile, 
		  concatenate(get_cwd(), "/", n, ".database", 
			                 "/", "pipsmake", NULL)));
}
/**** End saved_makefile version ****/


string make_open_program(name)
string name;
{
    return make_open_workspace(name);
}

string make_open_workspace(name)
string name;
{
    if (open_makefile(name) == makefile_undefined)
	user_warning("make_open_workspace", 
		     "No special makefile for this workspace %s/%s.database\n", get_cwd(), name);
    else
	debug(7, "make_open_workspace", "makefile opened\n");

    db_open_workspace(name);

    return db_get_current_workspace_name();
}

void make_close_program()
{
    return make_close_workspace();
}

void make_close_workspace()
{
    string name;


    db_set_current_module_name(NULL);

    name = db_get_current_workspace_name();

    close_makefile(name);

    db_close_workspace();

    user_log("Workspace %s closed\n\n", name);
}

