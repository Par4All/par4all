
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

bool /* should be: success (cf wpips.h) */
open_program(name)
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
