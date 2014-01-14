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

#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>

#if (defined(TEXT))
#undef TEXT
#endif

#if (defined(TEXT_TYPE))
#undef TEXT_TYPE
#endif

#include "genC.h"

#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "top-level.h"
#include "database.h"
#include "pipsmake.h"
#include "pipsdbm.h"
#include "wpips.h"

/* Include the label names: */
#include "wpips-labels.h"

#include "resources.h"
#include "phases.h"

wpips_transform_menu_layout_line wpips_transform_menu_layout[] = {
#include "wpips_transform_menu_layout.h"
   /* No more transformations */
   {
      NULL, NULL
   }
};


/* To pass arguments to execute_safe_apply_outside_the_notifyer(): */
string static execute_safe_apply_outside_the_notifyer_transformation_name_to_apply = NULL;
string static execute_safe_apply_outside_the_notifyer_module_name = NULL;


/* The transform menu: */
Menu transform_menu;

void
apply_on_each_transform_item(void (* function_to_apply_on_each_menu_item)(Menu_item))
{
   int i;
   /* Walk through items of  */
   for (i = (int) xv_get(transform_menu, MENU_NITEMS); i > 0; i--) {
      Menu_item menu_item = (Menu_item) xv_get(transform_menu,
                                               MENU_NTH_ITEM, i);
      /* Skip the title item: */
      if (!(bool) xv_get(menu_item, MENU_TITLE)
          && xv_get(menu_item, MENU_NOTIFY_PROC) != NULL)
          function_to_apply_on_each_menu_item(menu_item);
   }
}


void
disable_transform_selection()
{
   apply_on_each_transform_item(disable_menu_item);
}


void
enable_transform_selection()
{
   apply_on_each_transform_item(enable_menu_item);
}


void
execute_safe_apply_outside_the_notifyer()
{
   (void) safe_apply(execute_safe_apply_outside_the_notifyer_transformation_name_to_apply, execute_safe_apply_outside_the_notifyer_module_name);
   free(execute_safe_apply_outside_the_notifyer_transformation_name_to_apply);
   free(execute_safe_apply_outside_the_notifyer_module_name);

   /* The module list may have changed: */
   send_the_names_of_the_available_modules_to_emacs();
   display_memory_usage();
}


void
safe_apply_outside_the_notifyer(string transformation_name_to_apply,
                                string module_name)
{
   execute_safe_apply_outside_the_notifyer_transformation_name_to_apply =
      strdup(transformation_name_to_apply);
   execute_safe_apply_outside_the_notifyer_module_name =
      strdup(module_name);
   /* Ask to execute the execute_safe_apply_outside_the_notifyer(): */
   execute_main_loop_command(WPIPS_SAFE_APPLY);
   /* I guess the function above does not return... */
}

 
void static
transform_notify(Menu menu,
                 Menu_item menu_item)
{
   char * label = (char *) xv_get(menu_item, MENU_STRING);

   char * modulename = db_get_current_module_name();

   /* FI: borrowed from edit_notify() */
   if (modulename == NULL) {
      prompt_user("No module selected");
   }
   else {
      wpips_transform_menu_layout_line * current_transformation;
   
      /* Find the transformation to apply: */
      for (current_transformation = &wpips_transform_menu_layout[0];
           current_transformation->menu_entry_string != NULL;
           current_transformation++)
         if (strcmp(label, current_transformation->menu_entry_string) == 0)
            break;
      
      if (current_transformation->menu_entry_string != NULL)
         /* Apply the transformation: */
         safe_apply_outside_the_notifyer(current_transformation->transformation_name_to_apply, modulename);
      /* I guess the function above does not return... */
      else
         pips_error("transform_notify",
                    "What is this \"%s\" entry you ask for?",
                    label);
   }

   display_memory_usage();
}


void
create_transform_menu()
{
   wpips_transform_menu_layout_line * current_transformation;
   
   edit_menu_item = 
      xv_create(NULL, MENUITEM, 
                MENU_STRING, EDIT_VIEW,
                MENU_NOTIFY_PROC, edit_notify,
                MENU_RELEASE,
                NULL);
   
   transform_menu =
      xv_create(XV_NULL, MENU_COMMAND_MENU, 
                MENU_GEN_PIN_WINDOW, main_frame, "Transform Menu",
                MENU_TITLE_ITEM, "Apply a program transformation to a module ",
                NULL);
   
   /* Now add all the transformation entries: */
   for (current_transformation = &wpips_transform_menu_layout[0];
        current_transformation->menu_entry_string != NULL;
        current_transformation++) {
      if (strcmp(current_transformation->menu_entry_string,
                 WPIPS_MENU_SEPARATOR_ID) == 0)
         xv_set(transform_menu,
                                /* Just a separator: */
                WPIPS_MENU_SEPARATOR,
                NULL);
      else
         xv_set(transform_menu,
                MENU_ACTION_ITEM, current_transformation->menu_entry_string,
                transform_notify,
                NULL);
   }
   
   /* Add the Edit entry as the last one: */
   xv_set(transform_menu,
                                /* Just a separator: */
          WPIPS_MENU_SEPARATOR,
          MENU_APPEND_ITEM, edit_menu_item,
          NULL);

   (void) xv_create(main_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Transform/Edit",
                    PANEL_ITEM_MENU, transform_menu,
                    0);
}
