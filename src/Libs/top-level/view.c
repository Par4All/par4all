/*
 * $Id$
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <dirent.h>

#include <sys/stat.h>
#include <ctype.h>
#include <setjmp.h>
#include <unistd.h>
#include <errno.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "properties.h"

#include "ri-util.h"
#include "pipsdbm.h"

#include "constants.h"
#include "resources.h"
#include "phases.h"

#include "property.h"
#include "pipsmake.h"
#include "pipsdbm.h"

#include "top-level.h"

/* returns the allocated full path name.
 * the database is added if it is relative...
 */
static string get_view_file(string print_type, bool displayable)
{
    string module_name = db_get_current_module_name();

   if(displayable && !displayable_file_p(print_type)) {
       pips_user_error("resource %s cannot be displayed\n", print_type);
   }

   if (module_name != NULL)
   {
      if (safe_make(print_type, module_name)) 
      {
         string file_name = 
	     db_get_file_resource(print_type, module_name, TRUE);

	 if (file_name[0]!='.' && file_name[0]!='/')
	 {
	     /* the file name is relative to the database.
	      * thus the workspace directory name is prefixed.
	      */
	     string pgm_wd, file_name_in_database;

	     pgm_wd = build_pgmwd(db_get_current_workspace_name());
	     file_name_in_database = 
		 strdup(concatenate(pgm_wd, "/", file_name, NULL));

	     free(pgm_wd); 
	     return file_name_in_database;
	 }
	 else
	 {
	     /* it is relative to pips pwd, or absolute.
	      */
	     return file_name;
	 }
      }
   }
   else {
      /* should be show_message */
      user_log("No current module; select a module\n");
   }
   return NULL;
}

string build_view_file(string print_type)
{
    return get_view_file(print_type, TRUE);
}

string get_dont_build_view_file(string print_type)
{
    return get_view_file(print_type, FALSE);
}
