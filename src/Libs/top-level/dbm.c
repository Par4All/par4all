#include <stdio.h>

#include "genC.h"

#include "database.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "makefile.h"
#include "pipsmake.h"
#include "misc.h"

#include "top-level.h"

void default_update_props()
{}

/* default assignment of pips_update_props_handler is default_update_props.
 * Some top-level (eg. wpips) may need a special update_props proceedure; they 
 * should let pips_update_props_handler point toward it.
 */
void (* pips_update_props_handler)() = default_update_props;

/* FI: should be called "initialize_workspace()"; a previous call to
   db_create_workspace() is useful to create the log file says RK */
bool create_workspace(pargc, argv)
int *pargc;
char *argv[];
{
    int i;
    string name;
    bool status = FALSE;

    /* since db_create_workspace() must have been called before... */
    pips_assert("create_workspace",
		db_get_current_workspace()!=database_undefined);

    open_log_file();

    for (i = 0; i < *pargc; i++) {
	status = process_user_file(argv[i]);
	if (status == FALSE)
	    break;
    }

    if (status) {
	(* pips_update_props_handler)();

	name = database_name(db_get_current_workspace());
	user_log("Workspace %s created and opened\n", name);

	status = open_module_if_unique();
    }
    else {
	/* FI: in fact, the whole workspace should be deleted!
	 The file and the directory should be removed, and the current
	 database become undefined... */
	close_log_file();
    }

    return status;
}

bool open_module_if_unique()
{
    char *module_list[ARGS_LENGTH];
    int  module_list_length = 0;
    bool status = TRUE;

    pips_assert("open_module_if_unique",
		db_get_current_workspace()!=database_undefined);

    /* First parse the makefile to avoid writing
       an empty one */
    (void) parse_makefile();

    status = db_get_module_list(&module_list_length, module_list);
    if (status) {
	if (module_list_length == 1) {
	    status = open_module(module_list[0]);
	}
	args_free(&module_list_length, module_list);
    }
    return status;
}

bool open_module(name)
char *name;
{
    bool status;

    pips_assert("open_module",
		db_get_current_workspace()!=database_undefined);

    status = db_set_current_module_name(name);
    reset_unique_variable_numbers();

    if (status)
	user_log("Module %s selected\n", name);
    else
	user_warning("open_module", "Could not open module %s\n", name);

    return status;
}


/* Do not open a module already opened : */
bool lazy_open_module(name)
char *name;
{
    bool status = TRUE;
    char *current_name = NULL;

    pips_assert("lazy_open_module",
		db_get_current_workspace()!=database_undefined);

    message_assert("cannot lazy_open no module", name != NULL);

    current_name = db_get_current_module_name();

    if (current_name == NULL || strcmp(current_name, name) != 0)
	status = open_module(name);
    else if (current_name != NULL)
	user_log ("Module %s already active\n", name);

    return status;
}
     
/* should be: success (cf wpips.h) */
bool open_workspace(name)
char *name;
{
    bool status;

    if (make_open_workspace(name) == NULL) {
	/* should be show_message */
	/* FI: what happens since log_file is not open? */
	user_log("Cannot open workspace %s\n", name);
	status = FALSE;
    }
    else {
	(* pips_update_props_handler)();

	open_log_file();

	user_log("Workspace %s opened\n", name);

	status = open_module_if_unique();
    }
    return status;
}

bool close_workspace()
{
    bool status;

    status = make_close_workspace();
    return status;
    /*clear_props();*/
}

bool delete_workspace(string wname)
{
    int status;
    
    /* FI: No check whatsoever about the current workspace, no information
       about deleting the non-current workspace vs deleting the current
       workspace... */

    if ((status=safe_system_no_abort (concatenate("Delete ", wname, NULL)))) {
	user_warning("delete_workspace",
		     "exit code for Delete is %d\n", status);
    }

    return !status;
}
