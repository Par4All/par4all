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
#include <sys/param.h>

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
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"

#include "misc.h"
#include "pipsdbm.h"

#include "resources.h"
#include "constants.h"
#include "top-level.h"

#include "wpips.h"

/* Include the label names: */
#include "wpips-labels.h"

static wpips_view_menu_layout_line wpips_view_menu_layout[] = {
#include "wpips_view_menu_layout.h"
   /* No more views */
   {
      NULL, NULL, NULL
   }
};



static Textsw edit_textsw[MAX_NUMBER_OF_WPIPS_WINDOWS];
static Panel_item check_box[MAX_NUMBER_OF_WPIPS_WINDOWS];
static bool dont_touch_window[MAX_NUMBER_OF_WPIPS_WINDOWS];
int number_of_wpips_windows = INITIAL_NUMBER_OF_WPIPS_WINDOWS;

static Menu_item current_selection_mi, 
                 close_menu_item,
sequential_view_menu_item;
Menu_item edit_menu_item;

/* The menu "View" on the main panel: */
Menu view_menu;


/* To pass the view name to
   execute_wpips_execute_and_display_something_outside_the_notifyer(): */
static wpips_view_menu_layout_line * execute_wpips_execute_and_display_something_outside_the_notifyer_menu_line = NULL;


void
edit_notify(Menu menu,
            Menu_item menu_item)
{
    char string_filename[SMALL_BUFFER_LENGTH],
    string_modulename[SMALL_BUFFER_LENGTH];
    char file_name_in_database[MAXPATHLEN*2];
    char * modulename = db_get_current_module_name();
    char * file_name;
    int win_nb;
    char * alternate_wpips_editor;

    if (modulename == NULL) {
	prompt_user("No module selected");
	return;
    }

    if (wpips_emacs_mode) {
	/* Rely on the standard EPips viewer: */
	wpips_view_menu_layout_line * current_view;
	char * label = (char *) xv_get(menu_item, MENU_STRING);
	/* Translate the menu string in a resource name: */
	for (current_view = &wpips_view_menu_layout[0];
	     current_view->menu_entry_string != NULL;
	     current_view++)
	    if (strcmp(label, current_view->menu_entry_string) == 0)
		break;

	pips_assert("Resource related to the menu entry not found",
		    current_view->menu_entry_string != NULL);
    
	wpips_execute_and_display_something_outside_the_notifyer(current_view);
    }
    else {
	file_name = db_get_file_resource(DBR_SOURCE_FILE, modulename, TRUE);
	sprintf(file_name_in_database, "%s/%s",
		build_pgmwd(db_get_current_workspace_name()),
		file_name);

	if ((alternate_wpips_editor = getenv("PIPS_WPIPS_EDITOR")) != NULL) {
	    char editor_command[MAXPATHLEN*2];
	    sprintf(editor_command, "%s %s &",
		    alternate_wpips_editor,
		    file_name_in_database);
	    system(editor_command);
	}
	else {
	    /* Is there an available edit_textsw ? */
	    if ( (win_nb=alloc_first_initialized_window(FALSE)) == NO_TEXTSW_AVAILABLE ) {
		prompt_user("None of the text-windows is available");
		return;
	    }

	    sprintf(string_filename, "File: %s", file_name);
	    sprintf(string_modulename, "Module: %s", modulename);

	    /* Display the file name and the module name. RK, 2/06/1993 : */
	    xv_set(edit_frame[win_nb], FRAME_LABEL, "Pips Edit Facility",
		   FRAME_SHOW_FOOTER, TRUE,
		   FRAME_LEFT_FOOTER, string_filename,
		   FRAME_RIGHT_FOOTER, string_modulename,
		   NULL);

	    xv_set(edit_textsw[win_nb], 
		   TEXTSW_FILE, file_name_in_database,
		   TEXTSW_BROWSING, FALSE,
		   TEXTSW_FIRST, 0,
		   NULL);

	    unhide_window(edit_frame[win_nb]);
   
	    xv_set(current_selection_mi, 
		   MENU_STRING, "Lasts",
		   MENU_INACTIVE, FALSE,
		   NULL);

	    xv_set(close_menu_item, MENU_INACTIVE, FALSE, NULL);
	}
    }
}


void current_selection_notify(menu, menu_item)
Menu menu;
Menu_item menu_item;
{
  int i;
  
  for(i = 0; i < number_of_wpips_windows; i++)
    unhide_window(edit_frame[i]);
}


#define DONT_TOUCH_WINDOW_ADDRESS 0

void dont_touch_window_notify(Panel_item item, int value, Event *event)
{
  bool *dont_touch_window = (bool *) xv_get(item, XV_KEY_DATA,
					    DONT_TOUCH_WINDOW_ADDRESS);
  *dont_touch_window = (bool) xv_get(item, PANEL_VALUE);
}


char *
compute_title_string(int window_number)
{
  char title_string_beginning[] = "Xview Pips Display Facility # ";
  static char title_string[sizeof(title_string_beginning) + 4];

  (void) sprintf(title_string, "%s%d",
		 title_string_beginning, window_number + 1);
  /* xv_set will copy the string. */
  return title_string;
}

/* Find the first free window if any. If called with TRUE, giive the
   same as the previous choosen one. */
int
alloc_first_initialized_window(bool the_same_as_previous)
{
   static int next = 0;
   static int candidate = 0;
   int i;

   if (the_same_as_previous)
      return candidate;
   
   for(i = next; i < next + number_of_wpips_windows; i++) {
      candidate = i % number_of_wpips_windows;
      /* Skip windows with modified text inside : */
      if ((bool)xv_get(edit_textsw[candidate], TEXTSW_MODIFIED))
         continue;
      /* Skip windows with a retain attribute : */
      if ((bool)xv_get(check_box[candidate], PANEL_VALUE))
         continue;
    
      next = candidate + 1;
      return candidate;
   }
   candidate = NO_TEXTSW_AVAILABLE;
   
   return candidate;
}    


/* Mark a wpips or epips window as busy: */
bool
wpips_view_marked_busy(char * title_module_name, /* The module name for example */
                       char * title_label, /* "Sequential View" for exemple */
                       char * icon_name,
                       char * icon_title)
{
   char busy_label[SMALL_BUFFER_LENGTH];
   
   if (! wpips_emacs_mode) {
      int window_number;
      /* Is there an available edit_textsw ? */
      if ((window_number = alloc_first_initialized_window(FALSE))
          == NO_TEXTSW_AVAILABLE) {
         prompt_user("None of the text-windows is available");
         return FALSE;
      }
      (void) sprintf(busy_label, "*Computing %s * ...", title_label);
      /* Display the file name and the module name. RK, 2/06/1993 : */
      xv_set(edit_frame[window_number],
             FRAME_LABEL, compute_title_string(window_number),
             FRAME_SHOW_FOOTER, TRUE,
             FRAME_LEFT_FOOTER, busy_label,
             FRAME_RIGHT_FOOTER, title_module_name,
             FRAME_BUSY, TRUE,
             NULL);

      set_pips_icon(edit_frame[window_number], icon_name, icon_title);

      unhide_window(edit_frame[window_number]);
   }
   display_memory_usage();

   return TRUE;
}


/* Display a file in a wpips or epips window: */
void
wpips_file_view(char * file_name,
                char * title_module_name, /* The module name for example */
                char * title_label, /* "Sequential View" for exemple */
                char * icon_name,
                char * icon_title)
{
   if (file_name == NULL) {
      /* Well, something got wrong... */
      prompt_user("Nothing available to display...");
      return;
   }
   
   if (! wpips_emacs_mode) {
      int window_number;
      
      /* Is there an available edit_textsw ? Ask for the same one
         allocated for wpips_view_marked_busy() */
      if ((window_number = alloc_first_initialized_window(TRUE))
          == NO_TEXTSW_AVAILABLE) {
         prompt_user("None of the text-windows is available");
         return;
      }
      /* Display the file name and the module name. RK, 2/06/1993 : */
      xv_set(edit_frame[window_number],
             FRAME_LABEL, compute_title_string(window_number),
             FRAME_SHOW_FOOTER, TRUE,
             FRAME_RIGHT_FOOTER, title_module_name,
             NULL);

      set_pips_icon(edit_frame[window_number], icon_name, icon_title);

      xv_set(edit_textsw[window_number], 
             TEXTSW_FILE, file_name,
             TEXTSW_BROWSING, TRUE,
             TEXTSW_FIRST, 0,
             NULL);

      xv_set(edit_frame[window_number],
             FRAME_LEFT_FOOTER, title_label,
             FRAME_BUSY, FALSE,
             NULL);
      
      unhide_window(edit_frame[window_number]);
   
      xv_set(current_selection_mi, 
	     MENU_STRING, "Lasts",
	     MENU_INACTIVE, FALSE, NULL);
      xv_set(close_menu_item, MENU_INACTIVE, FALSE, NULL);
   }
   else {
      /* The Emacs mode equivalent: */
      send_module_name_to_emacs(db_get_current_module_name());
      /* send_icon_name_to_emacs(icon_number); */
      send_view_to_emacs(title_label, file_name);
   }
   display_memory_usage();
}


/* Use daVinci to display a graph information: */
void
wpips_display_graph_file_display(wpips_view_menu_layout_line * menu_line)
{
    char * file_name;
    char a_buffer[SMALL_BUFFER_LENGTH];
    /* Exploit some parallelism between daVinci/Emacs and PIPS
       itself: */
    if (wpips_emacs_mode)
	ask_emacs_to_open_a_new_daVinci_context();

    file_name = build_view_file(menu_line->resource_name_to_view);

    user_log("Displaying in a \"daVinci\" window...\n");

    /* Preprocess the graph to be understandable by daVinci : */
    if (wpips_emacs_mode) {
	sprintf(a_buffer, "pips_graph2daVinci %s", file_name);
	system(a_buffer);
	/* Since Emacs may be with another current directory, use the
           full path name: */
	ask_emacs_to_display_a_graph(concatenate(get_cwd(),
						 "/", file_name, NULL));
    }
    else {
	(void) sprintf(a_buffer, "pips_graph2daVinci -launch_daVinci %s", 
		       file_name);
	system(a_buffer);
    }

    free(file_name);
}


/* Use some text viewer to display the resource: */
void
wpips_display_plain_file(wpips_view_menu_layout_line * menu_line)
{
    char title_module_name[SMALL_BUFFER_LENGTH];

    char * print_type = menu_line->resource_name_to_view;
    char * icon_name = menu_line->icon_name;
    char * label = menu_line->menu_entry_string;
    
    (void) sprintf(title_module_name, "Module: %s",
		   db_get_current_module_name());
    if (wpips_view_marked_busy(title_module_name, label, icon_name, 
			       db_get_current_module_name())) {
	char * file_name = build_view_file(print_type);

	wpips_file_view(file_name, title_module_name, label, icon_name, 
			db_get_current_module_name());

	free(file_name);
    }
}


/* Mainly a hack to display 2 files in only one method for WP65... */
void
wpips_display_WP65_file(wpips_view_menu_layout_line * menu_line)
{
    char title_module_name[SMALL_BUFFER_LENGTH];

    char * print_type = menu_line->resource_name_to_view;
    char * icon_name = menu_line->icon_name;
    char * label = menu_line->menu_entry_string;

    (void) sprintf(title_module_name, "Module: %s", 
		   db_get_current_module_name());
    if (wpips_view_marked_busy(title_module_name, label, icon_name, 
			       db_get_current_module_name())) 
    {
	char bank_view_name[SMALL_BUFFER_LENGTH];
	
	char * file_name = build_view_file(print_type);
	wpips_file_view(file_name, title_module_name, label, icon_name,  
			db_get_current_module_name());
	free(file_name);

	/* Now display the other file: */
	(void) sprintf(bank_view_name, "%s (bank view)", label);
	if (wpips_view_marked_busy(title_module_name, bank_view_name, 
				   "WP65_bank", db_get_current_module_name()))
	{
	    /* Assume the previous build_view_file built the both
               resources: */
	    file_name = get_dont_build_view_file(DBR_WP65_BANK_FILE);
      
	    wpips_file_view(file_name, title_module_name, 
			    bank_view_name, "WP65_bank", 
			    db_get_current_module_name());

	    free(file_name);
	}
    }
}


/* To execute something and display some Pips output with wpips or
   epips, called outside the notifyer: */
void
execute_wpips_execute_and_display_something_outside_the_notifyer()
{
   wpips_view_menu_layout_line * menu_line = execute_wpips_execute_and_display_something_outside_the_notifyer_menu_line;

   /* Execute the needed method: */
   menu_line->method_function_to_use(menu_line);
   
   /* The module list may have changed (well not very likely to
      happen, but...): */
   send_the_names_of_the_available_modules_to_emacs();
   display_memory_usage();
}


void
wpips_execute_and_display_something_outside_the_notifyer(wpips_view_menu_layout_line * menu_line)
{
   execute_wpips_execute_and_display_something_outside_the_notifyer_menu_line = menu_line;
   /* Ask to execute the
      execute_wpips_execute_and_display_something_outside_the_notifyer(): */
   execute_main_loop_command(WPIPS_EXECUTE_AND_DISPLAY);
   /* I guess the function above does not return... */
}


/* To execute something and display some Pips output with wpips or
   epips: */
void
wpips_execute_and_display_something(char * resource_name)
{
    char * module_name = db_get_current_module_name();
    wpips_view_menu_layout_line * current_view;

    if (module_name == NULL) {
	prompt_user("No module selected");
	return;
    }

    /* Translate the resource name in a menu entry descriptor: */
    for (current_view = &wpips_view_menu_layout[0];
	 current_view->menu_entry_string != NULL;
	 current_view++)
	if (strcmp(resource_name, current_view->resource_name_to_view) == 0)
	    break;

    pips_assert("Resource related to the menu entry not found",
		current_view->menu_entry_string != NULL);
    
    wpips_execute_and_display_something_outside_the_notifyer(current_view);
}


/* To execute something and display some Pips output with wpips or
   epips by knowing its alias: */
void
wpips_execute_and_display_something_from_alias(char * alias_name)
{
    char * module_name = db_get_current_module_name();
    wpips_view_menu_layout_line * current_view;

    if (module_name == NULL) {
	prompt_user("No module selected");
	return;
    }

    for (current_view = &wpips_view_menu_layout[0];
	 current_view->menu_entry_string != NULL;
	 current_view++)
	if (strcmp(alias_name, current_view->menu_entry_string) == 0)
	    break;

    pips_assert("Resource related to the menu entry not found",
		current_view->menu_entry_string != NULL);
    
    wpips_execute_and_display_something_outside_the_notifyer(current_view);
}


void
view_notify(Menu menu,
            Menu_item menu_item)
{
    /* Translate the menu string in a resource name: */
    char * label = (char *) xv_get(menu_item, MENU_STRING);
    wpips_execute_and_display_something_from_alias(label);
}


void
edit_close_notify(Menu menu,
                  Menu_item menu_item)
{
   int i;

   if (! wpips_emacs_mode) {
      for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++)
         if (! (bool)xv_get(edit_textsw[i], TEXTSW_MODIFIED))
            hide_window(edit_frame[i]);

      for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++)
         if ((bool)xv_get(edit_textsw[i], TEXTSW_MODIFIED)) {
            unhide_window(edit_frame[i]);
            prompt_user("File not saved in editor");
            return;
         }
  
      for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++)
         hide_window(edit_frame[i]);
   
      xv_set(current_selection_mi, 
	     MENU_STRING, "No Selection",
	     MENU_INACTIVE, TRUE, NULL);

      xv_set(close_menu_item, MENU_INACTIVE, TRUE, NULL);
   }
   display_memory_usage();
}


void
disable_menu_item(Menu_item item)
{
   xv_set(item, MENU_INACTIVE, TRUE, 0);
}


void
enable_menu_item(Menu_item item)
{
   xv_set(item, MENU_INACTIVE, FALSE, 0);
}


void
apply_on_each_view_item(void (* function_to_apply_on_each_menu_item)(Menu_item),
                        void (* function_to_apply_on_each_panel_item)(Panel_item))
{
   int i;

   /* Skip the "current_selection_mi" and "close" Menu_items: */
   for(i = (int) xv_get(view_menu, MENU_NITEMS); i > 0; i--) {
      Menu_item menu_item = (Menu_item) xv_get(view_menu, MENU_NTH_ITEM, i);
      /* Skip the title item: */
      if (!(bool) xv_get(menu_item, MENU_TITLE)
          && menu_item != current_selection_mi
          && menu_item != close_menu_item
          && xv_get(menu_item, MENU_NOTIFY_PROC) != NULL)
         function_to_apply_on_each_menu_item(menu_item);
   }

  /* Now walk through the options panel: */
   {
      Panel_item panel_item;

      PANEL_EACH_ITEM(options_panel, panel_item)
         /* Only on the PANEL_CHOICE_STACK: */
         if ((Panel_item_type) xv_get(panel_item, PANEL_ITEM_CLASS) ==
             PANEL_BUTTON_ITEM)
            function_to_apply_on_each_panel_item(panel_item);
      PANEL_END_EACH
         }
}


void
disable_view_selection()
{
   apply_on_each_view_item(disable_menu_item, disable_panel_item);
}


void
enable_view_selection()
{
   apply_on_each_view_item(enable_menu_item, enable_panel_item);
}


void create_edit_window()
{
  /* Xv_Window window; */
  int i;
  
  for(i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++) {
    Panel panel;
    
    edit_textsw[i] = xv_create(edit_frame[i], TEXTSW, 
			       TEXTSW_DISABLE_CD, TRUE,
			       TEXTSW_DISABLE_LOAD, TRUE,
			       0);
    window_fit(edit_textsw[i]);
    panel = xv_create(edit_frame[i], PANEL,
		      WIN_ROW_GAP, 1,
		      WIN_COLUMN_GAP, 1,
		      NULL);
    dont_touch_window[i] = FALSE;
    check_box[i] = xv_create(panel, PANEL_CHECK_BOX,
			     PANEL_CHOICE_STRINGS, "Retain this window", NULL,
			     PANEL_VALUE, dont_touch_window[i],
			     PANEL_ITEM_X_GAP, 1,
			     PANEL_ITEM_Y_GAP, 1,
			     PANEL_NOTIFY_PROC, dont_touch_window_notify,
			     XV_KEY_DATA, DONT_TOUCH_WINDOW_ADDRESS, &dont_touch_window[i],
			     NULL);
    window_fit_height(panel);
    window_fit(edit_frame[i]);
  }
}


void
create_edit_menu()
{
    wpips_view_menu_layout_line * current_view;
 
   current_selection_mi = 
      xv_create(NULL, MENUITEM, 
                MENU_STRING, "No Selection",
                MENU_NOTIFY_PROC, current_selection_notify,
                MENU_INACTIVE, TRUE,
                MENU_RELEASE,
                NULL);

   close_menu_item = 
      xv_create(NULL, MENUITEM, 
                MENU_STRING, "Close",
                MENU_NOTIFY_PROC, edit_close_notify,
                MENU_INACTIVE, TRUE,
                MENU_RELEASE,
                NULL);

   sequential_view_menu_item =
          xv_create(NULL, MENUITEM, 
                MENU_STRING, SEQUENTIAL_VIEW,
                MENU_NOTIFY_PROC, view_notify,
                NULL);

   view_menu = 
       xv_create(XV_NULL, MENU_COMMAND_MENU, 
		 MENU_GEN_PIN_WINDOW, main_frame, "View & Edit Menu",
		 MENU_TITLE_ITEM, "Viewing or editing a module ",
		 NULL);
   
   if (! wpips_emacs_mode) {
       /* Make sense only if we have XView edit windows: */
       xv_set(view_menu, MENU_APPEND_ITEM, current_selection_mi,
	      NULL);
   }
   /* Now add all the view entries: */
   for (current_view = &wpips_view_menu_layout[0];
        current_view->menu_entry_string != NULL;
        current_view++) {
      if (strcmp(current_view->menu_entry_string,
                 WPIPS_MENU_SEPARATOR_ID) == 0)
         xv_set(view_menu,
                                /* Just a separator: */
                WPIPS_MENU_SEPARATOR,
                NULL);
      else
         xv_set(view_menu,
                MENU_ACTION_ITEM, current_view->menu_entry_string,
                view_notify,
                NULL);
   }
   if (! wpips_emacs_mode) {
       /* Make sense only if we have XView edit windows: */
       xv_set(view_menu, 	  /* Just a separator: */
	      WPIPS_MENU_SEPARATOR,
	      MENU_APPEND_ITEM, close_menu_item,
	      NULL);
   }

   (void) xv_create(main_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "View",
                    PANEL_ITEM_MENU, view_menu,
                    NULL);

}
