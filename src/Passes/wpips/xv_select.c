#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/notice.h>
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


/* Try to select a main module (that is the PROGRAM in the Fortran
   stuff) if no one is selected: */
void
select_a_module_by_default()
{
    char *module_name = db_get_current_module_name();

    if (module_name == NULL) {
	/* Ok, no current module, then find a main module (PROGRAM): */
	string main_module_name = get_first_main_module();
      
	if (!string_undefined_p(main_module_name)) {
	    /* Ok, we got it ! Now we select it: */
	    module_name = main_module_name;
	    user_log("Main module PROGRAM \"%s\" found.\n", module_name);
	    end_select_module_notify(module_name);
	    /* GO: show_module() has already been called so return now */
	    return;
	}
    }

    /* Refresh the module name on the status window: */
    show_module();
}


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
		end_directory_notify,
		cancel_query_notify);
}


static Menu_item create_pgm, open_pgm, close_pgm, module_item;

void
start_create_program_notify(Menu menu,
                            Menu_item menu_item)
{
   /* It is not coherent to change directory while creating a program. 
    * Here should the menu item be set to MENU_INACTIVE
    */
   start_query("Create Workspace", 
               "Enter workspace name: ", 
               "CreateWorkspace",
               continue_create_program_notify,
               /* Pas la peine de faire quelque chose si on appuie
                  sur cancel : */
               cancel_create_program_notify);
}

void
cancel_create_program_notify(Panel_item item,
                             Event * event)
{
  /* Re'tablit le droit d'ouvrir ou de cre'er un autre worspace : */
  xv_set(create_pgm, MENU_INACTIVE, FALSE, 0);
  xv_set(open_pgm, MENU_INACTIVE, FALSE, 0);
  cancel_query_notify(item, event);
}

success
continue_create_program_notify(char * name)
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
         /* Code added to confirm for a database destruction before
            opening a database with the same name.
            RK 18/05/1993. */
         if (workspace_exists_p(name))
         {
	    int result;
            /* Send to emacs if we are in the emacs mode: */
            if (wpips_emacs_mode) 
               send_notice_prompt_to_emacs("The database",
                                           name,
                                           "already exists!",
                                           "Do you really want to remove it?",
                                           NULL);
	    result = notice_prompt(xv_find(main_frame, WINDOW, 0),
				   NULL,
				   NOTICE_MESSAGE_STRINGS,
				   "The database", name, "already exists!",
				   "Do you really want to remove it?",
				   NULL,
				   NOTICE_BUTTON_YES,  "Yes, remove the database",
				   NOTICE_BUTTON_NO,   "No, cancel",
				   NULL);
	    if (result == NOTICE_NO)
               return(FALSE);
         }

         db_create_program(name);
         open_log_file();
         display_memory_usage();
         mchoose("Create Workspace", 
                 fortran_list_length, fortran_list, 
                 end_create_program_notify);
         args_free(&fortran_list_length, fortran_list);

         return(TRUE);
      }
   }
}


void
end_create_program_notify(int * pargc, char * argv[])
{
   create_program(pargc, argv);

   xv_set(close_pgm, MENU_INACTIVE, FALSE, 0);
   xv_set(module_item, MENU_INACTIVE, FALSE, 0);

   show_program();
   select_a_module_by_default();
   display_memory_usage();
}


void
end_open_program_notify(string name)
{
   schoose_close();

   /* Around a bug in schoose... */
   if (db_get_current_program_name() != NULL
       && strcmp(db_get_current_program_name(), name) == 0)
      return;
    
   if ( open_program(name) ) {
      open_log_file();
      xv_set(close_pgm, MENU_INACTIVE, FALSE, 0);
      xv_set(module_item, MENU_INACTIVE, FALSE, 0);
      show_program();
      select_a_module_by_default();
   }
   else {
      xv_set(create_pgm, MENU_INACTIVE, FALSE, 0);
      xv_set(open_pgm, MENU_INACTIVE, FALSE, 0);
   }
   display_memory_usage();
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
			/* Choix initial sur le workspace courant si
                           possible : */
			db_get_current_program_name(),
			end_open_program_notify,
			cancel_open_program_notify);

		/* FI/RK: too early; we are not sure to successfully open the workspace
		 * xv_set(module_item, MENU_INACTIVE, FALSE, 0);
		 */
    }
    args_free(&program_list_length, program_list);
}



void
close_program_notify(Menu menu,
                     Menu_item menu_item)
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
   display_memory_usage();
}


/* To be used with schoose_create_abbrev_menu_with_text from the main
   panel: */
void
open_or_create_workspace(char * workspace_name)
{
   int i;
   char *program_list[ARGS_LENGTH];
   int  program_list_length = 0;

   if (db_get_current_program_name() != NULL)
      /* There is an open workspace: close it first: */
      close_program_notify((Menu) NULL, (Menu_item) NULL);

   /* To choose between open or create, look for the an existing
      workspace with the same name: */
   pips_get_program_list(&program_list_length, program_list);

   for(i = 0; i < program_list_length; i++)
      if (strcmp(workspace_name, program_list[i]) == 0) {
         /* OK, the workspace exist, open it: */
         end_open_program_notify(workspace_name);
         return;
      }
   
   /* The workspace does not exist, create it: */
   (void) continue_create_program_notify(workspace_name);
}


/* To use with schoose_create_abbrev_menu_with_text: */
Menu
generate_workspace_menu()
{
   Menu menu;
   int i;
   char *program_list[ARGS_LENGTH];
   int  program_list_length = 0;

   pips_get_program_list(&program_list_length, program_list);

   menu = xv_create(NULL, MENU,
                    MENU_TITLE_ITEM, " Select in the workspace list: ",
                    NULL);
   if (program_list_length == 0) {
      xv_set(menu, MENU_APPEND_ITEM,
             xv_create(XV_NULL, MENUITEM,
                       MENU_STRING,
                       "* No workspace available in this directory *",
                       MENU_RELEASE,
                       MENU_INACTIVE, TRUE,
                       NULL),
             NULL);
   }
   else {
      for(i = 0; i < program_list_length; i++)
         xv_set(menu, MENU_APPEND_ITEM,
                xv_create(XV_NULL, MENUITEM,
                          MENU_STRING, strdup(program_list[i]),
                          MENU_RELEASE,
                          /* The strdup'ed string will also be
                             freed when the menu is discarded: */
                          MENU_RELEASE_IMAGE,
                          NULL),
                NULL);
   }
   args_free(&program_list_length,
             program_list);
   
   return menu;
}


void
end_select_module_notify(string name)
{
   lazy_open_module(name);

   show_module();
   display_memory_usage();
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
			module_list_length,
			module_list,
			/* Affiche comme choix courant le module
			   courant (c'est utile si on ferme la fenêtre
			   module entre temps) : */
			db_get_current_module_name(),
			end_select_module_notify,
			cancel_select_module_notify);

    args_free(&module_list_length, module_list);
}


/* To use with schoose_create_abbrev_menu_with_text: */
Menu
generate_module_menu()
{
   Menu menu;
   int i;
   char *module_list[ARGS_LENGTH];
   int  module_list_length = 0;

   menu = xv_create(NULL, MENU,
                    MENU_TITLE_ITEM, " Select in the module list: ",
                    NULL);

   if (db_get_current_program_name() == NULL) {
      xv_set(menu, MENU_APPEND_ITEM,
             xv_create(XV_NULL, MENUITEM,
                       MENU_STRING, "* No workspace yet! *",
                       MENU_RELEASE,
                       MENU_INACTIVE, TRUE,
                       NULL),
             NULL);
   }
   else {
      db_get_module_list(&module_list_length, module_list);
   
      if (module_list_length == 0) {
         xv_set(menu, MENU_APPEND_ITEM,
                xv_create(XV_NULL, MENUITEM,
                          MENU_STRING,
                          "* No module available in this workspace *",
                          MENU_RELEASE,
                          MENU_INACTIVE, TRUE,
                          NULL),
                NULL);
      }
      else {
         for(i = 0; i < module_list_length; i++)
            xv_set(menu, MENU_APPEND_ITEM,
                   xv_create(XV_NULL, MENUITEM,
                             MENU_STRING, strdup(module_list[i]),
                             MENU_RELEASE,
                             /* The strdup'ed string will also be
                                freed when the menu is discarded: */
                             MENU_RELEASE_IMAGE,
                             NULL),
                   NULL);
      }
      args_free(&module_list_length,
                module_list);
   }
   
   return menu;
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
		  MENU_GEN_PIN_WINDOW, main_frame, "Workspace Menu",
		  MENU_APPEND_ITEM, open_pgm,
		  MENU_APPEND_ITEM, create_pgm,
		  MENU_APPEND_ITEM, close_pgm,
		  NULL);

		/* Exchange of the order of start_directory_notify &
			module_item on the screen for ergonomic reasons.
			:-) RK, 19/02/1993. */
    menu = 
	xv_create(NULL, MENU_COMMAND_MENU, 
		  MENU_GEN_PIN_WINDOW, main_frame, "Selection Menu",
		  MENU_APPEND_ITEM, module_item,
		  MENU_PULLRIGHT_ITEM, "Workspace", pmenu,
		  MENU_ACTION_ITEM, "Directory", start_directory_notify,
		  0);

    (void) xv_create(main_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Select",
		     PANEL_ITEM_MENU, menu,
		     NULL);
}
