#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>
#include <types.h>
#include <setjmp.h>

#include "genC.h"
#include "ri.h"
#include "makefile.h"
#include "database.h"

#include "misc.h"
#include "pipsdbm.h"
#include "top-level.h"
#include "wpips.h"

#include "resources.h"


success end_directory_notify(dir)
char *dir;
{
    char *s;

    if (dir != NULL) {
	if ((s = pips_change_directory(dir)) == NULL) {
	    user_log("Directory %s does not exist\n", dir);
	    /* FI: according to cproto and gcc */
	    /* prompt_user("Directory %s does not exist !", dir); */
	    prompt_user("Directory does not exist !");
	    return (FALSE);
	}
	else {
	    user_log("Directory %s selected\n", dir);
	}
    }

    show_directory();
    return (TRUE);
}



void start_directory_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
	if (db_get_current_workspace() != database_undefined)
		/* RK, 25/01/1993. */
		prompt_user("You have to close the current workspace before changing directory.");
	else
		start_query("Change Directory",
			"Enter directory path: ", 
			"ChangeDirectory",
			end_directory_notify);
}


static Menu_item create_pgm, open_pgm, close_pgm, module_item;

void start_create_program_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    /* It is not coherent to change directory while creating a program. 
     * Here should the menu item be set to MENU_INACTIVE
     */
    start_query("Create Workspace", 
		"Enter workspace name: ", 
		"CreateWorkspace",
		continue_create_program_notify);
}



success continue_create_program_notify(name)
char *name;
{
    char *fortran_list[ARGS_LENGTH];
    int  fortran_list_length = 0;
    extern jmp_buf pips_top_level;


    if( setjmp(pips_top_level) ) {
	xv_set(create_pgm, MENU_INACTIVE, FALSE, 0);
	xv_set(open_pgm, MENU_INACTIVE, FALSE, 0);
	return(FALSE);
    }
    else {
	xv_set(create_pgm, MENU_INACTIVE, TRUE, 0);
	xv_set(open_pgm, MENU_INACTIVE, TRUE, 0);

	pips_get_fortran_list(&fortran_list_length, fortran_list);

	if (fortran_list_length == 0) {
	    prompt_user("No Fortran files in this directory");
	    longjmp(pips_top_level, 1);
	}
	else {
	    db_create_program(name);
	    mchoose("Create Workspace", 
		    fortran_list_length, fortran_list, 
		    end_create_program_notify);
	    args_free(&fortran_list_length, fortran_list);
	    return(TRUE);
	}
    }
}


void end_create_program_notify(pargc, argv)
int *pargc;
char *argv[];
{
    create_program(pargc, argv);

    xv_set(close_pgm, MENU_INACTIVE, FALSE, 0);
    xv_set(module_item, MENU_INACTIVE, FALSE, 0);

    show_program();
    show_module();
}


void end_open_program_notify(name)
string name;
{
    schoose_close();

    if ( open_program(name) ) {
	xv_set(close_pgm, MENU_INACTIVE, FALSE, 0);
	xv_set(module_item, MENU_INACTIVE, FALSE, 0);
	show_program();
	show_module();
    }
    else {
	xv_set(create_pgm, MENU_INACTIVE, FALSE, 0);
	xv_set(open_pgm, MENU_INACTIVE, FALSE, 0);
    }
}

void cancel_open_program_notify()
{
    xv_set(create_pgm, MENU_INACTIVE, FALSE, 0);
    xv_set(open_pgm, MENU_INACTIVE, FALSE, 0);
}

void open_program_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    char *program_list[ARGS_LENGTH];
    int  program_list_length = 0;

    pips_get_program_list(&program_list_length, program_list);

    if (program_list_length == 0) {
		prompt_user("No workspace available in this directory.");
    }
    else {
		xv_set(create_pgm, MENU_INACTIVE, TRUE, 0);
		xv_set(open_pgm, MENU_INACTIVE, TRUE, 0);

		schoose("Select Workspace", 
			program_list_length, program_list, 
			end_open_program_notify,
			cancel_open_program_notify);

		/* FI/RK: too early; we are not sure to successfully open the workspace
		 * xv_set(module_item, MENU_INACTIVE, FALSE, 0);
		 */
    }
    args_free(&program_list_length, program_list);
}



void close_program_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    close_program();

    /* FI: according to cproto and gcc */
    /* edit_close_notify(); */
    edit_close_notify(menu, menu_item);

    show_program();
    show_module();

    xv_set(create_pgm, MENU_INACTIVE, FALSE, 0);
    xv_set(open_pgm, MENU_INACTIVE, FALSE, 0);
    xv_set(close_pgm, MENU_INACTIVE, TRUE, 0);

    xv_set(module_item, MENU_INACTIVE, TRUE, 0);
    hide_window(schoose_frame);
}


void end_select_module_notify(name)
string name;
{
    open_module(name);

    show_module();
}

void cancel_select_module_notify()
{
}

void select_module_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
    char *module_list[ARGS_LENGTH];
    int  module_list_length = 0;

    db_get_module_list(&module_list_length, module_list);

	if (module_list_length == 0)
	{
		/* If there is no module... RK, 23/1/1993. */
		prompt_user("No module available in this workspace");
	}
	else
		schoose("Select Module", 
			module_list_length, module_list, 
			end_select_module_notify,
			cancel_select_module_notify);

    args_free(&module_list_length, module_list);
}



void create_select_menu()
{
    Menu menu, pmenu;

    create_pgm = xv_create(NULL, MENUITEM, 
		      MENU_STRING, "Create",
		      MENU_NOTIFY_PROC, start_create_program_notify,
		      MENU_RELEASE,
		      NULL);

    open_pgm = xv_create(NULL, MENUITEM, 
		      MENU_STRING, "Open",
		      MENU_NOTIFY_PROC, open_program_notify,
		      MENU_RELEASE,
		      NULL);

    close_pgm = xv_create(NULL, MENUITEM, 
		      MENU_STRING, "Close",
		      MENU_NOTIFY_PROC, close_program_notify,
		      MENU_INACTIVE, TRUE,
		      MENU_RELEASE,
		      NULL);

    module_item = xv_create(NULL, MENUITEM, 
		      MENU_STRING, "Module",
		      MENU_NOTIFY_PROC, select_module_notify,
		      MENU_INACTIVE, TRUE,
		      MENU_RELEASE,
		      NULL);

		/* Exchange of the order of create_pgm & open_pgm on the screen
			for ergonomic reasons. :-) RK, 19/02/1993. */
    pmenu = 
	xv_create(NULL, MENU_COMMAND_MENU, 
		  MENU_APPEND_ITEM, open_pgm,
		  MENU_APPEND_ITEM, create_pgm,
		  MENU_APPEND_ITEM, close_pgm,
		  NULL);

		/* Exchange of the order of start_directory_notify &
			module_item on the screen for ergonomic reasons.
			:-) RK, 19/02/1993. */
    menu = 
	xv_create(NULL, MENU_COMMAND_MENU, 
		  MENU_APPEND_ITEM, module_item,
		  MENU_PULLRIGHT_ITEM, "Workspace", pmenu,
		  MENU_ACTION_ITEM, "Directory", start_directory_notify,
		  0);

    (void) xv_create(main_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Select",
		     PANEL_ITEM_MENU, menu,
		     NULL);
}
