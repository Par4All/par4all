/* 	%A% ($Date: 1996/10/11 17:20:00 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_xv_edit2[] = "%A% ($Date: 1996/10/11 17:20:00 $, ) version $Revision$, got on %D%, %T% [%P%].\n École des Mines de Paris Proprietary.";
#endif /* lint */

#include <stdlib.h>
#include <stdio.h>
#include <sys/param.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "makefile.h"
#include "database.h"

#include "misc.h"
#include "pipsdbm.h"

#include "resources.h"
#include "constants.h"
#include "top-level.h"

#include "wpips.h"

/* Include the label names: */
#include "wpips-labels.h"

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
string static execute_wpips_execute_and_display_something_outside_the_notifyer_view_name = NULL;


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
	char * label = (char *) xv_get(menu_item, MENU_STRING);
	/* Rely on the standard EPips viewer: */
	wpips_execute_and_display_something_outside_the_notifyer(label);
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
                       int icon_number,
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

      if (icon_number >= 0)
         set_pips_icon(edit_frame[window_number], icon_number, icon_title);

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
                int icon_number,
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

      if (icon_number >= 0)
         set_pips_icon(edit_frame[window_number], icon_number, icon_title);

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


void
display_graph_with_daVinci(string file_name)
{
    char a_buffer[SMALL_BUFFER_LENGTH];
    /* Exploit some parallelism between daVinci/Emacs and PIPS
       itself: */
    if (wpips_emacs_mode)
	ask_emacs_to_open_a_new_daVinci_context();

    file_name = build_view_file(DBR_GRAPH_PRINTED_FILE);

    user_log("Displaying in a \"daVinci\" window...\n");

    /* Preprocess the graph to be understandable by daVinci : */
    if (wpips_emacs_mode) {
	sprintf(a_buffer, "pips_graph2daVinci %s", file_name);
	system(a_buffer);
	ask_emacs_to_display_a_graph(file_name);
    }
    else {
	sprintf(a_buffer, "pips_graph2daVinci -launch_daVinci %s", file_name);
	system(a_buffer);
    }
}


/* To execute something and display some Pips output with wpips or
   epips, called outside the notifyer: */
void
execute_wpips_execute_and_display_something_outside_the_notifyer()
{
   char * file_name;
   char * module_name = db_get_current_module_name();

   string label = execute_wpips_execute_and_display_something_outside_the_notifyer_view_name;
   
   if (strcmp(label, SEQUENTIAL_GRAPH_VIEW) == 0)
       /* Use some graph viewer to display the resource: */
       display_graph_with_daVinci(file_name);
   else {
      /* Use some text viewer to display the resource: */
      char * print_type = NULL;
      char * print_type_2 = NULL;
      /* No icon image by default: */
      int icon_number = -1;
      int icon_number2 = -1;
      char title_module_name[SMALL_BUFFER_LENGTH];
      
      icon_number = icon_number2 = -1;
      if (strcmp(label, USER_VIEW) == 0) {
         print_type = DBR_PARSED_PRINTED_FILE;
         icon_number = user_ICON;
      }
      else if (strcmp(label, SEQUENTIAL_VIEW) == 0) {
         print_type = DBR_PRINTED_FILE;
         icon_number = sequential_ICON;
      }
      else if (strcmp(label, PARALLEL_VIEW) == 0) {
         print_type = DBR_PARALLELPRINTED_FILE;
         icon_number = parallel_ICON;
      }
      else if (strcmp(label, CALLGRAPH_VIEW) == 0) {
         print_type = DBR_CALLGRAPH_FILE;
         icon_number = callgraph_ICON;
      }
      else if (strcmp(label, ICFG_VIEW) == 0) {
         print_type = DBR_ICFG_FILE;
         icon_number = ICFG_ICON;
      }
      else if (strcmp(label, DISTRIBUTED_VIEW) == 0) {
         print_type = DBR_WP65_COMPUTE_FILE;
         icon_number = WP65_PE_ICON;
         print_type_2 = DBR_WP65_BANK_FILE;
         icon_number2 = WP65_bank_ICON;
      }
      else if (strcmp(label, DEPENDENCE_GRAPH_VIEW) == 0) {
         print_type = DBR_DG_FILE;
      }
      else if (strcmp(label, FLINT_VIEW) == 0) {
         print_type = DBR_FLINTED;
      }
      else if (strcmp(label, ARRAY_DFG_VIEW) == 0) {
         print_type = DBR_ADFG_FILE;
      }
      else if (strcmp(label, TIME_BASE_VIEW) == 0) {
         print_type = DBR_BDT_FILE;
      }
      else if (strcmp(label, PLACEMENT_VIEW) == 0) {
         print_type = DBR_PLC_FILE;
      }
      else if (strcmp(label, EDIT_VIEW) == 0) {
         print_type = DBR_SOURCE_FILE;
      }
      else {
         pips_error("view_notify", "bad label : %s\n", label);
      }

      sprintf(title_module_name, "Module: %s", module_name);
      if (wpips_view_marked_busy(title_module_name, label, icon_number, module_name)) {

         file_name = build_view_file(print_type);

         wpips_file_view(file_name, title_module_name, label, icon_number, module_name);
   
   
  
         if ( print_type_2 != NULL ) {
            char bank_view_name[SMALL_BUFFER_LENGTH];

	    /* I removed the "(bank view)" appended here so that 
	     * wpips/epips comms are ok for WP65...
	     * I could also have added a special view handler in epips...
	     * well... FC 30/11/95
	     */
            (void) sprintf(bank_view_name, "%s", label);
            if (wpips_view_marked_busy(title_module_name, bank_view_name, 
				       icon_number2, module_name)) {
               file_name = get_dont_build_view_file(print_type_2);
      
               wpips_file_view(file_name, title_module_name, 
			       bank_view_name, icon_number2, module_name);
            }
         }
      }
   }
   
   free(execute_wpips_execute_and_display_something_outside_the_notifyer_view_name);
   
   /* The module list may have changed (well not very likely to
      happen, but...): */
   send_the_names_of_the_available_modules_to_emacs();
   display_memory_usage();
}


void
wpips_execute_and_display_something_outside_the_notifyer(char * label)
{
   execute_wpips_execute_and_display_something_outside_the_notifyer_view_name = strdup(label);
   /* Ask to execute the
      execute_wpips_execute_and_display_something_outside_the_notifyer(): */
   execute_main_loop_command(WPIPS_EXECUTE_AND_DISPLAY);
   /* I guess the function above does not return... */
}


/* To execute something and display some Pips output with wpips or
   epips: */
void
wpips_execute_and_display_something(char * label)
{
   char * module_name = db_get_current_module_name();

   if (module_name == NULL) {
      prompt_user("No module selected");
      return;
   }

   wpips_execute_and_display_something_outside_the_notifyer(label);
}


void
view_notify(Menu menu,
            Menu_item menu_item)
{
   char * label = (char *) xv_get(menu_item, MENU_STRING);

   wpips_execute_and_display_something(label);
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
   xv_set(view_menu, MENU_APPEND_ITEM, sequential_view_menu_item,
	  /* The sequential_view_menu_item is the default item: */
	  MENU_DEFAULT_ITEM, sequential_view_menu_item,
	  MENU_ACTION_ITEM, USER_VIEW, view_notify,
	  MENU_ACTION_ITEM, SEQUENTIAL_GRAPH_VIEW, view_notify,
	  /* Just a separator: */
	  WPIPS_MENU_SEPARATOR,
	  MENU_ACTION_ITEM, DEPENDENCE_GRAPH_VIEW, view_notify,
	  /* Just a separator: */
	  WPIPS_MENU_SEPARATOR,
	  MENU_ACTION_ITEM, ARRAY_DFG_VIEW, view_notify,
	  MENU_ACTION_ITEM, TIME_BASE_VIEW, view_notify,
	  MENU_ACTION_ITEM, PLACEMENT_VIEW, view_notify,
	  /* Just a separator: */
	  WPIPS_MENU_SEPARATOR,
	  MENU_ACTION_ITEM, CALLGRAPH_VIEW, view_notify,
	  MENU_ACTION_ITEM, ICFG_VIEW, view_notify,
	  /* Just a separator: */
	  WPIPS_MENU_SEPARATOR,
	  MENU_ACTION_ITEM, DISTRIBUTED_VIEW, view_notify,
	  MENU_ACTION_ITEM, PARALLEL_VIEW, view_notify,
	  /* Just a separator: */
	  WPIPS_MENU_SEPARATOR,
	  MENU_ACTION_ITEM, FLINT_VIEW, view_notify,
	  NULL);   
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
