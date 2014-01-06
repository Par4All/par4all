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
#include <stdlib.h>
#include <stdio.h>
#include <xview/xview.h>
#include <xview/panel.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "phases.h"
#include "database.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "top-level.h"
#include "wpips.h"

static Menu compile_menu;

void
apply_on_each_compile_item(
    void (* function_to_apply_on_each_menu_item)(Menu_item))
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
   char * file_name = (char *) xv_get(menu_item, MENU_STRING);
   char * path_name = hpfc_generate_path_name_of_file_name(file_name);

   if (! wpips_emacs_mode) {
      /* Try to allocate an available edit_textsw in non-emacs mode: */
      (void) alloc_first_initialized_window(FALSE);
   }
   
   wpips_file_view(path_name, file_name, "HPFC File", "HPFC", "HPFC");
   user_log("HPFC View of \"%s\" done.\n", file_name);
}


Menu
generate_a_menu_with_HPF_output_files(
    Menu_item menu_item,
    Menu_generate action)
{
   int i;
   Menu menu;
   gen_array_t file_names = gen_array_make(0);
   int file_number = 0;

   pips_debug(2, "Enter\n");
   
   menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
   
   switch(action) {
     case MENU_DISPLAY:
     {
        int return_code;
        char *hpfc_directory;
        
        pips_debug(2, "MENU_DISPLAY\n");
   
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
     

        return_code = hpfc_get_file_list(file_names,
                                         &hpfc_directory);
        
        if (return_code == -1) {
           user_warning("generate_a_menu_with_HPF_output_files",
                        "Directory \"%s\" not found... \n"
                        " Have you run the HPFC compiler from the Compile menu?\n",
                        hpfc_directory);
           
           menu = (Menu) xv_create(NULL, MENU,
                                   MENU_TITLE_ITEM,
                                   "Are you sure you used the HPF compiler ?",
                                   MENU_ITEM, MENU_STRING, 
				   "*** No HPFC directory found ! ***", NULL,
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
                                   MENU_ITEM, MENU_STRING, 
				   "*** No HPFC file found ! ***", NULL,
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
                               MENU_STRING, 
			       strdup(gen_array_item(file_names, i)),
                               MENU_RELEASE,
                               /* The strdup'ed string will also be
                                  freed when the menu is discarded: */
                               MENU_RELEASE_IMAGE,
                               NULL),
                     NULL);

	   gen_array_full_free(file_names);
        }
        break;
     }
     
     case MENU_DISPLAY_DONE:
       /* We cannot remove the menu here since the notify
          procedure is called afterward. */
       menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
       pips_debug(2, "MENU_DISPLAY_DONE\n");
       break;

     case MENU_NOTIFY:
       /* Rely on the notify procedure. */
       menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
       pips_debug(2, "MENU_NOTIFY\n");
       break;

     case MENU_NOTIFY_DONE:
       menu = (Menu) xv_get(menu_item, MENU_PULLRIGHT);
       pips_debug(2, "MENU_NOTIFY_DONE\n");
       break;

     default:
       pips_internal_error("Unknown Menu_generate action: %d\n", action);
   }
   pips_debug(2, "Exit\n");
   return menu;
}

#define HPFC_COMPILE "Compile an HPF program"
#define HPFC_MAKE "Make an HPF program"
#define HPFC_RUN "Run an HPF program"

/* quick fix around pipsmake, FC, 23/10/95
 */
static bool wpips_hpfc_install_was_performed_hack = FALSE;

void 
initialize_wpips_hpfc_hack_for_fabien_and_from_fabien()
{
   wpips_hpfc_install_was_performed_hack = FALSE;
}

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

    /* I apply the installation only once, whatever...
     * Quick fix because the right dependences expressed for pipsmake
     * do not seem to work. It seems that the verification of up to date 
     * resources is too clever... FC.
     */
    if (!wpips_hpfc_install_was_performed_hack)
    {
	safe_apply(BUILDER_HPFC_INSTALL, modulename);
	wpips_hpfc_install_was_performed_hack = TRUE;
    }

    if (same_string_p(label, HPFC_COMPILE))
	;
    else if (same_string_p(label, HPFC_MAKE))
	safe_apply_outside_the_notifyer(BUILDER_HPFC_MAKE, modulename);
    else if (same_string_p(label, HPFC_RUN))
	safe_apply_outside_the_notifyer(BUILDER_HPFC_RUN,  modulename);
    else
	pips_internal_error("Bad choice: %s", label);
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
		MENU_ACTION_ITEM, HPFC_RUN, hpfc_notify,
                                /* Just a separator: */
                WPIPS_MENU_SEPARATOR,
                MENU_GEN_PULLRIGHT_ITEM,
                "View the HPF Compiler Output",
                generate_a_menu_with_HPF_output_files,
                NULL);
   
   (void) xv_create(main_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Compile",
                    PANEL_ITEM_MENU, compile_menu,
                    NULL);
}

