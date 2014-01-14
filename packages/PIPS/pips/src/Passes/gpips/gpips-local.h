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
#define MAX_NUMBER_OF_GPIPS_WINDOWS 9
#define INITIAL_NUMBER_OF_GPIPS_WINDOWS 2
#define NO_TEXTSW_AVAILABLE -1 /* cannot be positive (i.e. a window number. */
extern int number_of_gpips_windows;

// Gtk Windows (XV Frame -> GTK Window)
extern GtkWidget *main_window, *schoose_window, *mchoose_window, *log_window,
		*edit_window[MAX_NUMBER_OF_GPIPS_WINDOWS], *help_window, *query_dialog,
		*options_window;
extern GtkWidget *main_window_vbox;

// Gtk Frames (XV Panel -> GTK Frame)
extern GtkWidget *main_frame, *status_frame, *query_frame, *mchoose_frame,
		*help_frame;

extern GtkWidget *main_window_menu_bar;

typedef enum {
	PIPS_ICON,
	ICFG_ICON,
	GP65_PE_ICON,
	GP65_bank_ICON,
	callgraph_ICON,
	parallel_ICON,
	sequential_ICON,
	user_ICON,
	LAST_ICON
} icon_list;

typedef bool success;

/* The type to describe the command to execute outside the notifyer: */
typedef enum {
	GPIPS_EXIT = 1647300, GPIPS_SAFE_APPLY, GPIPS_EXECUTE_AND_DISPLAY
} gpips_main_loop_command_type;

/* The type describing a Transform menu entry: */
typedef struct {
	char * menu_entry_string;
	char * transformation_name_to_apply;
} gpips_transform_menu_layout_line;

/* The type describing a View menu entry: */
struct gpips_view_menu_layout_line_s {
	char * menu_entry_string;
	char * resource_name_to_view;
	void (* method_function_to_use)(struct gpips_view_menu_layout_line_s *);
	char * icon_name;
};
typedef struct gpips_view_menu_layout_line_s gpips_view_menu_layout_line;

/* Define the menu separator: */
#define GPIPS_MENU_SEPARATOR MENU_ITEM, MENU_STRING, "", MENU_INACTIVE, TRUE, NULL
/* How it is specified in the layout .h: */
#define GPIPS_MENU_SEPARATOR_ID ""

/* Here are the X ressource stuff: */
/* The Log Window: */
#define GPIPS_LOG_WINDOW_WIDTH_RESSOURCE_NAME "gpips.logwindow.width"
#define GPIPS_LOG_WINDOW_WIDTH_RESSOURCE_CLASS "Gpips.Logwindow.Width"
#define GPIPS_LOG_WINDOW_WIDTH_DEFAULT DIALOG_WIDTH

#define GPIPS_LOG_WINDOW_HEIGHT_RESSOURCE_NAME "gpips.logwindow.height"
#define GPIPS_LOG_WINDOW_HEIGHT_RESSOURCE_CLASS "Gpips.Logwindow.Height"
#define GPIPS_LOG_WINDOW_HEIGHT_DEFAULT DIALOG_HEIGHT
