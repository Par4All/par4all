#include <stdio.h>

#include "genC.h"

#include "database.h"
#include "ri.h"
#include "pipsdbm.h"
#include "makefile.h"
#include "pipsmake.h"
#include "misc.h"

#include "top-level.h"

void default_update_props()
{
}

/* default assignment of pips_update_props_handler is default_update_props.
 * Some top-level (eg. wpips) may need a special update_props proceedure; they 
 * should let pips_update_props_handler point toward it.
 */
void (* pips_update_props_handler)() = default_update_props;

void create_program(pargc, argv)
int *pargc;
char *argv[];
{
    int i;
    string name;

    for (i = 0; i < *pargc; i++)
	process_user_file(argv[i]);

    (* pips_update_props_handler)();

    name = database_name(db_get_current_program());
    user_log("Workspace %s created and opened\n", name);

    open_module_if_unique();
}

void open_module_if_unique()
{
    char *module_list[ARGS_LENGTH];
    int  module_list_length = 0;

    /* First parse the makefile to avoid writing
       an empty one */
    parse_makefile();

    db_get_module_list(&module_list_length, module_list);
    if (module_list_length == 1) {
	open_module(module_list[0]);
    }
    args_free(&module_list_length, module_list);
}

void open_module(name)
char *name;
{
    db_set_current_module_name(name);
    reset_unique_variable_numbers();

    user_log("Module %s selected\n", name);
}


/* Do not open a module already opened : */
void lazy_open_module(name)
char *name;
{
    char *current_name = db_get_current_module_name();
    if (current_name == NULL
	|| strcmp(current_name, name) != 0)
      open_module(name);
}
     
/* should be: success (cf wpips.h) */
bool open_program(name)
char *name;
{
    if (make_open_program(name) == NULL) {
	/* should be show_message */
	user_log("Cannot open workspace %s\n", name);
	return (FALSE);
    }
    else {
	(* pips_update_props_handler)();

	user_log("Workspace %s opened\n", name);

	open_module_if_unique();
	return (TRUE);
    }
}

void close_program()
{
    make_close_program();

    /*clear_props();*/
}
