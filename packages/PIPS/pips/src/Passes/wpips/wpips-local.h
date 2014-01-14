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
#define SMALL_BUFFER_LENGTH 2560

#define MESSAGE_BUFFER_LENGTH 1024
#define TEXT_BUFFER_LENGTH 1024

/* How many display wondows can be opened : */
#define MAX_NUMBER_OF_WPIPS_WINDOWS 9
#define INITIAL_NUMBER_OF_WPIPS_WINDOWS 2
#define NO_TEXTSW_AVAILABLE -1 /* cannot be positive (i.e. a window number. */
extern int number_of_wpips_windows;


/* If we are in the Emacs mode, the log_frame is no longer really used: */
extern Frame main_frame, 
   schoose_frame, 
   mchoose_frame, 
   log_frame, 
   edit_frame[MAX_NUMBER_OF_WPIPS_WINDOWS], 
   help_frame, 
   query_frame,
   options_frame;

extern Panel main_panel,
   status_panel,
   query_panel,
   mchoose_panel,
   schoose_panel,
   help_panel;

typedef enum {PIPS_ICON, ICFG_ICON, WP65_PE_ICON, WP65_bank_ICON, callgraph_ICON,
		parallel_ICON, sequential_ICON, user_ICON, LAST_ICON} icon_list;


typedef bool success;


/* This variable is used to indicate wether wpips is in the Emacs
   mode: */
extern bool wpips_emacs_mode;


/* The type to describe the command to execute outside the notifyer: */
typedef enum {
   WPIPS_EXIT = 1647300,
      WPIPS_SAFE_APPLY,
      WPIPS_EXECUTE_AND_DISPLAY
} wpips_main_loop_command_type;


/* The type describing a Transform menu entry: */
typedef struct 
{
   char * menu_entry_string;
   char * transformation_name_to_apply;
}
wpips_transform_menu_layout_line;


/* The type describing a View menu entry: */
struct wpips_view_menu_layout_line_s
{
   char * menu_entry_string;
   char * resource_name_to_view;
   void (* method_function_to_use)(struct wpips_view_menu_layout_line_s *);
   char * icon_name;
};
typedef struct wpips_view_menu_layout_line_s wpips_view_menu_layout_line;


/* Define the menu separator: */
#define WPIPS_MENU_SEPARATOR MENU_ITEM, MENU_STRING, "", MENU_INACTIVE, TRUE, NULL
/* How it is specified in the layout .h: */
#define WPIPS_MENU_SEPARATOR_ID ""

/* Here are the X ressource stuff: */
/* The Log Window: */
#define WPIPS_LOG_WINDOW_WIDTH_RESSOURCE_NAME "wpips.logwindow.width"
#define WPIPS_LOG_WINDOW_WIDTH_RESSOURCE_CLASS "Wpips.Logwindow.Width"
#define WPIPS_LOG_WINDOW_WIDTH_DEFAULT DIALOG_WIDTH

#define WPIPS_LOG_WINDOW_HEIGHT_RESSOURCE_NAME "wpips.logwindow.height"
#define WPIPS_LOG_WINDOW_HEIGHT_RESSOURCE_CLASS "Wpips.Logwindow.Height"
#define WPIPS_LOG_WINDOW_HEIGHT_DEFAULT DIALOG_HEIGHT
