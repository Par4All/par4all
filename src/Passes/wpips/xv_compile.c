/* 	%A% ($Date: 1995/09/22 17:25:08 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char wpips_xv_compile_c_vcid[] = "%A% ($Date: 1995/09/22 17:25:08 $, ) version $Revision$, got on %D%, %T% [%P%].\n École des Mines de Paris Proprietary.";
#endif /* lint */

#include <stdlib.h>
#include <stdio.h>
#include <xview/xview.h>
#include <xview/panel.h>

#include "genC.h"
#include "ri.h"
#include "makefile.h"
#include "pipsmake.h"
#include "phases.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "top-level.h"
#include "wpips.h"

static Menu compile_menu;

void
apply_on_each_compile_item(void (* function_to_apply_on_each_menu_item)(Menu_item))
{
   int i;

   for(i = (int) xv_get(compile_menu, MENU_NITEMS); i >= 1; i--) {
      Menu_item menu_item = (Menu_item) xv_get(compile_menu, MENU_NTH_ITEM, i);
      /* Skip the title item and the separator items: */
      if (!(bool) xv_get(menu_item, MENU_TITLE))
         function_to_apply_on_each_menu_item(menu_item);
   }
}


void
disable_compile_selection()
{
   apply_on_each_compile_item(disable_menu_item);
}


void
enable_compile_selection()
{
   apply_on_each_compile_item(enable_menu_item);
}


void
notify_hpfc_file_view(Menu menu,
                      Menu_item menu_item)
{
   int window_number;
   char * file_name = (char *) xv_get(menu_item, MENU_STRING);
   char * path_name = hpfc_generate_path_name_of_file_name(file_name);
    
   /* Is there an available edit_textsw ? */
   if ( (window_number = alloc_first_initialized_window())
        == NO_TEXTSW_AVAILABLE ) {
      prompt_user("None of the text-windows is available");
      return;
   }

   user_log("HPFC View of \"%s\" done.\n", file_name);
   wpips_file_view(path_name, file_name, "HPFC File", window_number, -1, "HPFC");
}


Menu
generate_a_menu_with_HPF_output_files(Menu_item menu_item,
                                      Menu_generate action)
{
   int i;
   Menu menu;
   char * file_names[ARGS_LENGTH];
   int file_number = 0;

   pips_debug(1, "Enter\n");
   
   menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
   
   switch(action) {
     case MENU_DISPLAY:
     {
        int return_code;
        char *hpfc_directory;
        
        pips_debug(1, "MENU_DISPLAY\n");
   
        /* Create a new menu with the content of the hpfc directory: */
        
        if (menu != NULL) {
           /* First, free the old menu if it exist: */
           /* We can remove all the menu it now: */
           for(i = (int) xv_get(menu, MENU_NITEMS); i > 0; i--) {
              xv_set(menu, MENU_REMOVE, i, NULL);
              xv_destroy(xv_get(menu, MENU_NTH_ITEM, i));
           }
           xv_destroy(menu);
        }
     

        return_code = hpfc_get_file_list(&file_number,
                                         file_names,
                                         &hpfc_directory);
        
        if (return_code == -1) {
           user_warning("generate_a_menu_with_HPF_output_files",
                        "Directory \"%s\" not found... \n"
                        " Have you run the HPFC compiler from the Compile menu?\n",
                        hpfc_directory);
           
           menu = (Menu) xv_create(NULL, MENU,
                                   MENU_TITLE_ITEM,
                                   "Are you sure you used the HPF compiler ?",
                                   MENU_ITEM, MENU_STRING, "*** No HPFC directory found ! ***", NULL,
                                   NULL);
        }
        else if (file_number == 0) {
           user_warning("generate_a_menu_with_HPF_output_files",
                        "No file found in the directory \"%s\"... \n"
                        " Have you run the HPFC compiler from the Compile menu?\n",
                        hpfc_directory);

           menu = (Menu) xv_create(NULL, MENU,
                                   MENU_TITLE_ITEM,
                                   "Are you sure you used the HPF compiler ?",
                                   MENU_ITEM, MENU_STRING, "*** No HPFC file found ! ***", NULL,
                                   NULL);
        }
        else {
           menu = (Menu) xv_create(NULL, MENU,
                                   MENU_TITLE_ITEM,
                                   " Select an HPFC file to view ",
                                   MENU_NOTIFY_PROC, notify_hpfc_file_view,
                                   NULL);
           
           for(i = 0; i < file_number; i++)
              xv_set(menu, MENU_APPEND_ITEM,
                     xv_create(XV_NULL, MENUITEM,
                               MENU_STRING, strdup(file_names[i]),
                               MENU_RELEASE,
                               /* The strdup'ed string will also be
                                  freed when the menu is discarded: */
                               MENU_RELEASE_IMAGE,
                               NULL),
                     NULL);

           args_free(&file_number,
                     file_names);
        }
        break;
     }
     
     case MENU_DISPLAY_DONE:
       /* We cannot remove the menu here since the notify
          procedure is called afterward. */
       menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
       debug(1, "generate_a_menu_with_HPF_output_files", "MENU_DISPLAY_DONE\n");
       break;

     case MENU_NOTIFY:
       /* Rely on the notify procedure. */
       menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
       debug(1, "generate_a_menu_with_HPF_output_files", "MENU_NOTIFY\n");
       break;

     case MENU_NOTIFY_DONE:
       menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
       debug(1, "generate_a_menu_with_HPF_output_files", "MENU_NOTIFY_DONE\n");
       break;

     default:
       pips_error("generate_a_menu_with_HPF_output_files",
                  "Unknown Menu_generate action: %d\n", action);
   }
   debug(1, "generate_a_menu_with_HPF_output_files", "Exit\n");
   return menu;
}

#define HPFC_COMPILE "Compile an HPF program"
#define HPFC_MAKE "Make an HPF program"

void
hpfc_notify(Menu menu,
            Menu_item menu_item)
{
    char *label, *modulename;
   
    modulename = db_get_current_module_name();
    if (!modulename)
    {
	prompt_user("No module selected");
	return;
    }

    label = (char *) xv_get(menu_item, MENU_STRING);
    if (same_string_p(label, HPFC_COMPILE))
	safe_apply(BUILDER_HPFC_INSTALL, modulename);
    else if (same_string_p(label, HPFC_MAKE))
	safe_apply(BUILDER_HPFC_MAKE, modulename);
    else
	pips_error("hpfc_notify", "Bad choice");
}



void
create_compile_menu()
{
   compile_menu = 
      xv_create(XV_NULL, MENU_COMMAND_MENU, 
                MENU_GEN_PIN_WINDOW, main_frame, "Compile Menu",
                MENU_TITLE_ITEM, "Compilation ",
                MENU_ACTION_ITEM, HPFC_COMPILE, hpfc_notify,
		MENU_ACTION_ITEM, HPFC_MAKE, hpfc_notify,
                /*
                MENU_ACTION_ITEM, "Display an HPF program", display_hpfc_file,
                */
                MENU_GEN_PULLRIGHT_ITEM,
                "View the HPF Compiler Output",
                generate_a_menu_with_HPF_output_files,
                NULL);
   
   (void) xv_create(main_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Compile",
                    PANEL_ITEM_MENU, compile_menu,
                    NULL);
}

