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
#include <stdlib.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "constants.h"
#include "resources.h"
#include "pipsmake.h"

#include "top-level.h"

// imports gtk
#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>

#include "gpips.h"

// Gtk Windows (XV Frame -> GTK Window)
GtkWidget *main_window, *schoose_window, *mchoose_window, *log_window,
		*edit_window[MAX_NUMBER_OF_GPIPS_WINDOWS], *help_window, *query_dialog,
		*options_window;
GtkWidget * main_window_vbox;

// Gtk Frames (XV Panel -> GTK Frame)
GtkWidget *main_frame, *status_frame, *mchoose_frame, *help_frame;

GtkWidget *main_window_menu_bar;

/* The variables to pass information between inside and outside the
 XView notifyer: */

/* By default, exiting the notifyer is to exit gpips: */
static gpips_main_loop_command_type gpips_main_loop_command = GPIPS_EXIT;

/* To deal with argument parsing: */
static string workspace_name_given_to_gpips = NULL;
static string module_name_given_to_gpips = NULL;
static gen_array_t files_given_to_gpips = NULL;

static void create_main_window_menu_bar() {
	main_window_menu_bar = gtk_menu_bar_new();
	gtk_box_pack_start(GTK_BOX(main_window_vbox), main_window_menu_bar, FALSE,
			FALSE, 0);
	gtk_widget_show_all(main_window_menu_bar);
}

static void create_menus() {
	create_main_window_menu_bar();
	// create_select_menu();
	create_edit_menu();
	/*    create_analyze_menu();*/
	create_transform_menu();
	create_compile_menu();
	/* The option panel use the definition of the edit menu and so
	 needs to be create after it: */
	create_options_menu_and_window();
	/* Gone in create_menus_end(): ...No ! */
	create_help_menu();
	create_log_menu();
	create_quit_button();
}

/*
 static void
 create_menus_end()
 {
 create_help_menu();
 }
 */

//static int first_mapping = TRUE;

//void main_event_proc(window, event)
//	Xv_Window window;Event *event; {
//	if (first_mapping == TRUE && event_id(event) == 32526) {
//		first_mapping = FALSE;
//
//		//		/* we place all frames */
//		//		place_frames();
//	};
//}

void create_main_window() {
	//	main_panel = xv_create(main_frame, PANEL, NULL);
	main_frame = gtk_frame_new(NULL);
	main_window_vbox = gtk_vbox_new(FALSE, 0);
	gtk_container_add(GTK_CONTAINER(main_window), main_window_vbox);
	gtk_box_pack_start(GTK_BOX(main_window_vbox), main_frame, TRUE, FALSE, 0);
	gtk_widget_show_all(main_frame);
}

/*
static unsigned short pips_bits[] = {
#include "pips.icon"
		};
*/
//void create_icon() {
//	Server_image pips_image;
//	Icon icon;
//	Rect rect;
//
//	pips_image = (Server_image) xv_create(NULL, SERVER_IMAGE, XV_WIDTH, 64,
//			XV_HEIGHT, 64, SERVER_IMAGE_BITS, pips_bits, NULL);
//	icon = (Icon) xv_create(XV_NULL, ICON, ICON_IMAGE, pips_image, NULL);
//	rect.r_width = (int) xv_get(icon, XV_WIDTH);
//	rect.r_height = (int) xv_get(icon, XV_HEIGHT);
//	rect.r_left = 0;
//	rect.r_top = 0;
//
//	xv_set(main_frame, FRAME_ICON, icon, FRAME_CLOSED_RECT, &rect, NULL);
//}

/* Try to parse the gpips arguments.

 Should add a help and version option.
 */

static void gpips_parse_arguments(int argc, char * argv[]) {
	extern int optind; /* the one of getopt (3) */
	int iarg = optind;

	while (iarg < argc) {
		if (same_string_p(argv[iarg], "-workspace")) {
			argv[iarg] = NULL;
			workspace_name_given_to_gpips = argv[++iarg];
		} else if (same_string_p(argv[iarg], "-module")) {
			argv[iarg] = NULL;
			module_name_given_to_gpips = argv[++iarg];
		} else if (same_string_p(argv[iarg], "-files")) {
			argv[iarg] = NULL;

			if (!files_given_to_gpips) /* lazy init */
				files_given_to_gpips = gen_array_make(0);
			gen_array_append(files_given_to_gpips, argv[++iarg]);
		} else {
			if (argv[iarg][0] == '-') {
				fprintf(stderr, "Usage: %s ", argv[0]);
				fprintf(stderr, "[ X-Window options ]");
				fprintf(stderr, "[ -workspace name [ -module name ] ");
				fprintf(stderr, "[ -files file1.f file2.f ... ] ]\n");
				exit(1);
			}
		}
		iarg += 1;
	}
}

/* Execute some actions asked as option after XView initialization: */
static void execute_workspace_creation_and_so_on_given_with_options(void) {
	if (workspace_name_given_to_gpips != NULL) {
		if (files_given_to_gpips != NULL) {
			if (!db_create_workspace(workspace_name_given_to_gpips))
				/* It fails, Go on with the normal gpips behaviour... */
				return;

			if (!create_workspace(files_given_to_gpips))
				/* It fails, Go on with the normal gpips behaviour... */
				return;

			gen_array_free(files_given_to_gpips), files_given_to_gpips = NULL;
		} else {
			if (!open_workspace(workspace_name_given_to_gpips))
				/* It fails, Go on with the normal gpips behaviour... */
				return;
		}

		if (module_name_given_to_gpips != NULL) {
			end_select_module_callback(module_name_given_to_gpips);
		}
		enable_workspace_close();
		show_workspace();
		enable_module_selection();
		disable_change_directory();
		enable_workspace_create_or_open();
		display_memory_usage();
		show_module();
	}
}

/* How much to call the notifyer between each pipsmake phase: */
enum {
	GPIPS_NUMBER_OF_EVENT_TO_DEAL_DURING_PIPSMAKE_INTERPHASE = 10
};

/* Since XView is not called while pipsmake is running, explicitly run
 a hook from pipsmake to run the notifier such as to stop pipsmake: */
static bool deal_with_gpips_events_during_pipsmake() {
	int i;

	/* First, try to show we are working :-) */
	gpips_interrupt_button_blink();

	for (i = 0; i < GPIPS_NUMBER_OF_EVENT_TO_DEAL_DURING_PIPSMAKE_INTERPHASE; i++)
		/* Ask the XView notifier to deal with one event. */
		//		notify_dispatch();
		gtk_main_iteration_do(FALSE);

	/* Refresh the main frame: */
	//	XFlush((Display *) xv_get(main_frame, XV_DISPLAY));
	/* pipsmake not interrupted by default: */
	return TRUE;
}

/* To ask pipsmake to stop as soon as possible: */
//void gpips_interrupt_pipsmake(Panel_item item, Event * event) {
//	interrupt_pipsmake_asap();
//	user_log("PIPS interruption requested...\n");
//}

/* Try to inform the user about an XView error. For debug, use the
 GPIPS_DEBUG_LEVEL to have an abort on this kind
 of error. */
//static int gpips_xview_error(Xv_object object, Attr_avlist avlist) {
//	debug_on("GPIPS_DEBUG_LEVEL");
//
//	fprintf(stderr, "gpips_xview_error caught an error:\n%s\n",
//			xv_error_format(object, avlist));
//	/* Cannot use pips_assert since it uses XView, neither
//	 get_bool_property for the same reason: */
//	if (get_debug_level() > 0) {
//		fprintf(stderr,
//				"gpips_xview_error is aborting as requested since GPIPS_DEBUG_LEVEL > 0...\n");
//		abort();
//	}
//
//	debug_off();
//
//	return XV_OK;
//}

/* Exit the notify loop to execute a gpips command: */
void execute_main_loop_command(gpips_main_loop_command_type command) {
	gpips_main_loop_command = command;
	//	notify_stop();
	gtk_main_quit();
	/* I guess the function above does not return... */
}

/* The main loop that deals with command outside the XView notifier: */
static void gpips_main_loop() {
	//	xv_main_loop(frame_to_map_first);
	gtk_main();

	/* The loop to execute commands: */
	while (gpips_main_loop_command != GPIPS_EXIT) {
		debug(1, "gpips_main_loop", "gpips_main_loop_command = %d\n",
				gpips_main_loop_command);

		switch ((int) gpips_main_loop_command) {
		case GPIPS_SAFE_APPLY:
			execute_safe_apply_outside_the_notifier();
			break;

		case GPIPS_EXECUTE_AND_DISPLAY:
			execute_gpips_execute_and_display_something_outside_the_notifier();
			break;

		default:
			pips_assert(
					"gpips_main_loop does not understand the gpips_main_loop_command",
					0);
		}

		/* If the notifier happen to exit without a specified command,
		 just exit: */
		gpips_main_loop_command = GPIPS_EXIT;

		/* Restore the initial state of the blinking pips icon: */
		gpips_interrupt_button_restore();

		/* Wait again for something from X11 */
		//		notify_start();
		gtk_main();
	}
}

int gpips_main(int argc, char * argv[]) {
	pips_checks();

	pips_warning_handler = gpips_user_warning;
	pips_error_handler = gpips_user_error;
	pips_log_handler = gpips_user_log;
	pips_update_props_handler = update_options;
	pips_request_handler = gpips_user_request;

	initialize_newgen();
	initialize_sc((char*(*)(Variable)) entity_local_name);
	set_exception_callbacks(push_pips_context, pop_pips_context);

	debug_on("GPIPS_DEBUG_LEVEL");

	/* we parse command line arguments */
	/* XV_ERROR_PROC unset as we shifted to xview.3, Apr. 92 */
	//	xv_init(XV_INIT_ARGC_PTR_ARGV, &argc, argv, XV_ERROR_PROC,
	//			gpips_xview_error, 0);
	gtk_init(&argc, &argv);

	/* we parse remaining command line arguments */
	gpips_parse_arguments(argc, argv);

	/* we create all windows */
	create_windows();
	create_main_window();
	create_help_window();
	create_schoose_window();
	create_mchoose_window();
	create_query_window();
	create_edit_window();
	create_log_window();
	create_menus();
	create_status_subwindow();
	//create_icons(); TODO: icon mangement
	//	place_frames();
	gtk_widget_show_all(main_window);

	display_memory_usage();

	enable_workspace_create_or_open();
	enable_change_directory();
	disable_workspace_close();
	disable_module_selection();
	disable_view_selection();
	disable_transform_selection();
	disable_compile_selection();
	disable_option_selection();

	set_pipsmake_callback(deal_with_gpips_events_during_pipsmake);

	execute_workspace_creation_and_so_on_given_with_options();

	gpips_main_loop();

	reset_pipsmake_callback();

	close_log_file();

	debug_off();

	exit(0);
}
