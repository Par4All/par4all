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
/* Some modifications are made to save the current makefile (s.a. files
 * pipsmake/readmakefile.y openclose.h )
 * They only occure between following tags: 
 */
/**** Begin saved_makefile version ****/
/**** End saved_makefile version ****/

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
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
#include "linear.h"
#include "ri.h"

#include "properties.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "misc.h"

/* returns the program makefile file name
 */
#define PIPSMAKE_FILE "/pipsmake"
string build_pgm_makefile(const char* __attribute__ ((unused)) n)
{
    string dir_name = db_get_meta_data_directory(),
	res = strdup(concatenate(dir_name, PIPSMAKE_FILE, NULL));
    free(dir_name);
    return res;
}

string  make_open_workspace(const char* name)
{
    if (db_open_workspace(name)) 
    {
	if (open_properties())
	{
	    if (open_makefile(name) != makefile_undefined) 
	    {
		pips_debug(7, "makefile opened\n");
	    }
	    else
	    {
		/* should be an error? */
		pips_user_warning("No special makefile for this workspace "
				  "%s/%s.database\n", get_cwd(), name);
	    }
	}
	else
	{
	    pips_user_warning("Cannot read properties...\n");
	    db_close_workspace(true);
	}
    } else
	pips_user_warning("No workspace %s to open\n", name);

    return db_get_current_workspace_name();
}

/* FI->GO: could be in top-level, no?
 */
bool make_close_workspace(bool is_quit)
{
    bool res = true;
    string name;

    if (db_get_current_module_name()) /* lazy... */
	db_reset_current_module_name();

    /* dup because freed in db_close_workspace */
    name = strdup(db_get_current_workspace_name()); 

    res &= close_makefile(name);
    save_properties();
    res &= db_close_workspace(is_quit);

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
void checkpoint_workspace(void)
{
    if (db_get_current_workspace_name())
    {
	user_log("Checkpoint of workspace.\n");
	/* FC 25/06/2003
	 * this seems to break pipsmake internal data...
	pips_debug(3, "\tdeleting obsolete resources...\n");
	delete_obsolete_resources();
	*/
	pips_debug(3, "\tsaving resources...\n");
	db_checkpoint_workspace();
	pips_debug(3, "\tproperties and makefile...\n");
	save_properties();
	save_makefile(db_get_current_workspace_name());
    }
}
