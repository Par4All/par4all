/*
 * $Id$
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"

#include "database.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "makefile.h"
#include "pipsmake.h"
#include "misc.h"

#include "preprocessor.h"

#include "top-level.h"

void default_update_props() {}

/* default assignment of pips_update_props_handler is default_update_props.
 * Some top-level (eg. wpips) may need a special update_props proceedure; they 
 * should let pips_update_props_handler point toward it.
 */
void (* pips_update_props_handler)() = default_update_props;

/* PIPS SRCPATH before openining the workspace, for restauration.
 */
static string saved_pips_src_path = NULL;

static void push_path(void)
{
    string dir;
    pips_assert("not set", !saved_pips_src_path);
    dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
    saved_pips_src_path = pips_srcpath_append(dir);
    free(dir);
}

static void pop_path(void)
{
    pips_assert("set", saved_pips_src_path);
    pips_srcpath_set(saved_pips_src_path);
    free(saved_pips_src_path), saved_pips_src_path = NULL;
}

/* tpips used to convert lower cases into upper cases for all module
   names, but this is no longer possible with C functions. To make it
   easier for the user and for the validation, an upper case version of
   name is open if name cannot be open. */
bool open_module(string name)
{
    bool success = FALSE;
    string upper_case_name = strupper(strdup(name), name);
    string module_name = string_undefined;

    if (!db_get_current_workspace_name())
      pips_user_error("No current workspace, open or create one first!\n");

    if (db_module_exists_p(name))
      module_name = name;
    else if(db_module_exists_p(upper_case_name)) {
      module_name = upper_case_name;
      pips_user_warning("Module %s selected instead of %s which was not found\n",
			module_name, name);
    }
    else
      module_name = NULL;

    if(module_name) {
      if (db_get_current_module_name()) /* reset if needed */
	db_reset_current_module_name();
      
      success = db_set_current_module_name(module_name);
    }

    if (success) {
      reset_unique_variable_numbers();
      user_log("Module %s selected\n", module_name);
    }
    else {
      if(strcmp(name, upper_case_name)==0)
	pips_user_warning("Could not open module %s\n", name);
      else
	pips_user_warning("Could not open module %s (nor %s)\n",
			  name, upper_case_name);
    }

    free(upper_case_name);
    return success;
}


bool open_module_if_unique()
{
    bool success = TRUE;
    gen_array_t a;

    pips_assert("some current workspace", db_get_current_workspace_name());

    /* First parse the makefile to avoid writing
       an empty one */
    (void) parse_makefile();

    a = db_get_module_list();
    if (gen_array_nitems(a)==1)
	success = open_module(gen_array_item(a, 0));
    gen_array_full_free(a);

    return success;
}

/* FI: should be called "initialize_workspace()"; a previous call to
 * db_create_workspace() is useful to create the log file between
 * the two calls says RK
 */
bool 
create_workspace(gen_array_t files)
{
    int i, argc = gen_array_nitems(files);
    string name, dir = db_get_current_workspace_directory();
    bool success = FALSE;

    /* since db_create_workspace() must have been called before... */
    pips_assert("some current workspace", db_get_current_workspace_name());

    open_log_file();
    open_warning_file(dir);
    free(dir);
    set_entity_to_size();

    /* pop_path() is too strict, let's push anyway since user errors
       are not caught below! */
    push_path();

    for (i = 0; i < argc; i++) 
    {
      /* FI: it would be nice to have a catch here on user_error()! */
	success = process_user_file(gen_array_item(files, i));
	if (success == FALSE)
	    break;
    }

    if (success) 
    {
	(* pips_update_props_handler)();
	name = db_get_current_workspace_name();
	user_log("Workspace %s created and opened.\n", name);
	success = open_module_if_unique();
	if (success) init_processed_include_cache();
	/* push_path(); */
    }
    else
    {
	/* FI: in fact, the whole workspace should be deleted!
	 The file and the directory should be removed, and the current
	 database become undefined... */
        /* DB: free the hash_table, otherwise core dump during the next
         call to create_workspace */
        reset_entity_to_size();
	close_log_file();
	close_warning_file();
	/* pop_path() is too strict, let's push anyway */
	/* push_path(); */
    }

    return success;
}

/* Do not open a module already opened : */
bool 
lazy_open_module(name)
char *name;
{
    bool success = TRUE;

    pips_assert("lazy_open_module", db_get_current_workspace_name());
    pips_assert("cannot lazy_open no module", name != NULL);

    if (db_get_current_module_name()) {
	char * current_name = db_get_current_module_name();
	if (strcmp(current_name, name) != 0)
	    success = open_module(name);
	else 
	    user_log ("Module %s already active.\n", name);
    } else
	success = open_module(name);

    return success;
}

/* should be: success (cf wpips.h) */
bool 
open_workspace(string name)
{
    bool success;

    if (db_get_current_workspace_name())
	pips_user_error("Some current workspace, close it first!\n");

    if (!workspace_exists_p(name))
	pips_user_error("Workspace %s does not exist!\n", name);

    if (!workspace_ok_p(name))
	pips_user_error("Workspace %s not readable!\n", name);

    if (make_open_workspace(name) == NULL) {
	/* should be show_message */
	/* FI: what happens since log_file is not open? */
	user_log("Cannot open workspace %s.\n", name);
	success = FALSE;
    }
    else {
	string dir = db_get_current_workspace_directory();
	(* pips_update_props_handler)();
	open_log_file();
	open_warning_file(dir);
	free(dir);
	set_entity_to_size();
	user_log("Workspace %s opened.\n", name);
	success = open_module_if_unique();
	if (success) init_processed_include_cache();
	push_path();	
    }
    return success;
}

bool close_workspace(bool is_quit)
{
    bool success;

    if (!db_get_current_workspace_name())
	pips_user_error("No workspace to close!\n");

    /* It is useless to save on disk some non up to date resources:
     */
    delete_some_resources();
    success = make_close_workspace(is_quit);
    close_log_file();
    close_processed_include_cache();
    reset_entity_to_size();
    close_warning_file();
    pop_path();
    return success;
    /*clear_props();*/
}

bool 
delete_workspace(string wname)
{
    int success = check_delete_workspace(wname,TRUE);

    return success;
}

bool 
check_delete_workspace(string wname, bool check)
{
    int failure;
    string current = db_get_current_workspace_name();

    /* Yes but at least close the LOGFILE if we delete the current
       workspace since it will fail on NFS because of the open file
       descriptor (creation of .nfs files). RK */

    if (check)
    {
	if (current && same_string_p(wname, current))
	pips_user_error("Cannot delete current workspace, close it first!\n");
    }
    else
    {
	string name = strdup(current); 
        (void) close_makefile(name);
	free(name);
	close_log_file();
	close_processed_include_cache();
	/* reset_entity_to_size(); */
	close_warning_file();
    }


    if ((failure=safe_system_no_abort(concatenate("Delete ", wname, NULL))))
	pips_user_warning("exit code for Delete is %d\n", failure);

    return !failure;
}
