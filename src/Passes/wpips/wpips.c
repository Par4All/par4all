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
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/svrimage.h>

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

#include "wpips.h"
#include "xv_sizes.h"

/* If we are in the Emacs mode, the log_frame is no longer really used: */
Frame main_frame, 
    schoose_frame, 
    mchoose_frame,
    log_frame, 
    edit_frame[MAX_NUMBER_OF_WPIPS_WINDOWS], 
    help_frame, 
    query_frame,
    options_frame;

Panel main_panel,
    status_panel,
    query_panel,
    mchoose_panel,
    schoose_panel,
    help_panel;


/* The variables to pass information between inside and outside the
   XView notifyer: */

/* By default, exiting the notifyer is to exit wpips: */
static wpips_main_loop_command_type wpips_main_loop_command = WPIPS_EXIT;

/* To deal with argument parsing: */
string static workspace_name_given_to_wpips = NULL;
string static module_name_given_to_wpips = NULL;
static gen_array_t files_given_to_wpips = NULL;
    


void static
create_menus()
{
   create_select_menu();
   create_edit_menu();
/*    create_analyze_menu();*/
   create_transform_menu();
   create_compile_menu();
   /* The option panel use the definition of the edit menu and so
      needs to be create after it: */
   create_options_menu_and_window();
   /* Gone in create_menus_end(): ...No ! */
   create_help_menu(); 
   /* In the Emacs mode, no XView log window: */
   /* In fact, create it but disabled to keep the same frame layout: */
   create_log_menu();
   create_quit_button();
}


/*
void static
create_menus_end()
{
   create_help_menu();
}
*/


static int first_mapping = TRUE;

void main_event_proc(window, event)
Xv_Window window;
Event *event;
{
    if (first_mapping == TRUE && event_id(event)==32526) {
		first_mapping = FALSE;

		/* we place all frames */
		place_frames();
    };
}


void
create_main_window()
{
    /* Xv_window main_window; */

    main_panel = xv_create(main_frame, PANEL,
			   NULL);

    /* The following mask is necessary to avoid that key events occure twice.
     */
    /*    main_window = (Xv_Window) xv_find(main_frame, WINDOW, 0);
	  xv_set(main_window,
	  WIN_IGNORE_EVENT, WIN_UP_ASCII_EVENTS, 
	  NULL);
	  */
    /* commented out as we shifted to xview.3, 92.04 */
/*    xv_set(canvas_paint_window(main_panel),
	   WIN_IGNORE_EVENT, WIN_UP_ASCII_EVENTS,
	   WIN_CONSUME_EVENT, 32526, 
	   WIN_EVENT_PROC, main_event_proc,
	   NULL);
*/
}


static unsigned short pips_bits[] = {
#include "pips.icon"
};

void create_icon()
{
    Server_image pips_image;
    Icon icon;
    Rect rect;

    pips_image = (Server_image)xv_create(NULL, SERVER_IMAGE, 
					 XV_WIDTH, 64,
					 XV_HEIGHT, 64,
					 SERVER_IMAGE_BITS, pips_bits, 
					 NULL);
    icon = (Icon)xv_create(XV_NULL, ICON, 
			   ICON_IMAGE, pips_image,
			   NULL);
    rect.r_width= (int)xv_get(icon, XV_WIDTH);
    rect.r_height= (int)xv_get(icon, XV_HEIGHT);
    rect.r_left= 0;
    rect.r_top= 0;

    xv_set(main_frame, FRAME_ICON, icon, 
	   FRAME_CLOSED_RECT, &rect,
	   NULL);
}


/* Try to parse the WPips arguments.
   
   The problem is that the -emacs option need to be known early but
   the workspace and other typical PIPS options need to be evaluate
   later... 
   
   Should add a help and version option.
*/

void static
wpips_parse_arguments(int argc,
                      char * argv[])
{
   extern int optind; /* the one of getopt (3) */
   int iarg = optind;

   while (iarg < argc) {
      if (same_string_p(argv[iarg], "-emacs")) {
         argv[iarg] = NULL;
         /* Wpips is called from emacs. RK. */
         wpips_emacs_mode = 1;
      }
      else if (same_string_p(argv[iarg], "-workspace")) {
         argv[iarg] = NULL;
         workspace_name_given_to_wpips = argv[++iarg];
      }
      else if (same_string_p(argv[iarg], "-module")) {
         argv[iarg] = NULL;
         module_name_given_to_wpips = argv[++iarg];
      }
      else if (same_string_p(argv[iarg], "-files")) {
         argv[iarg] = NULL;
	 
	 if (!files_given_to_wpips) /* lazy init */
	     files_given_to_wpips = gen_array_make(0);
	 gen_array_append(files_given_to_wpips, argv[++iarg]);
      }
      else {
         if (argv[iarg][0] == '-') {
            fprintf(stderr, "Usage: %s ", argv[0]);
            fprintf(stderr, "[ X-Window options ] [ -emacs ]");
            fprintf(stderr, "[ -workspace name [ -module name ] ");
            fprintf(stderr, "[ -files file1.f file2.f ... ] ]\n");
            exit(1);
         }
      }

      iarg += 1;
   }
}


/* Execute some actions asked as option after XView initialization: */
void static
execute_workspace_creation_and_so_on_given_with_options(void)
{
   if (workspace_name_given_to_wpips != NULL) 
   {
      if (files_given_to_wpips != NULL) 
      {
         if (! db_create_workspace(workspace_name_given_to_wpips))
            /* It fails, Go on with the normal WPips behaviour... */
            return;
         
         if (! create_workspace(files_given_to_wpips))
            /* It fails, Go on with the normal WPips behaviour... */
            return;
         
	 gen_array_free(files_given_to_wpips), files_given_to_wpips = NULL;
      } else {
         if (! open_workspace(workspace_name_given_to_wpips))
            /* It fails, Go on with the normal WPips behaviour... */
            return;
      }

      if (module_name_given_to_wpips != NULL) {
         end_select_module_notify(module_name_given_to_wpips);
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
   WPIPS_NUMBER_OF_EVENT_TO_DEAL_DURING_PIPSMAKE_INTERPHASE = 10
};

/* Since XView is not called while pipsmake is running, explicitly run
   a hook from pipsmake to run the notifyer such as to stop pipsmake: */
bool static
deal_with_wpips_events_during_pipsmake()
{
   int i;

   /* First, try to show we are working :-) */
   wpips_interrupt_button_blink();

   for (i = 0; i < WPIPS_NUMBER_OF_EVENT_TO_DEAL_DURING_PIPSMAKE_INTERPHASE; i++)
      /* Ask the XView notifyer to deal with one event. */
      notify_dispatch();

   /* Refresh the main frame: */
   XFlush((Display *) xv_get(main_frame, XV_DISPLAY));
   /* pipsmake not interrupted by default: */
   return TRUE;
}


/* To ask pipsmake to stop as soon as possible: */
void
wpips_interrupt_pipsmake(Panel_item item,
                         Event * event)
{
   interrupt_pipsmake_asap();
   user_log("PIPS interruption requested...\n");
}


/* Try to inform the user about an XView error. For debug, use the
   WPIPS_DEBUG_LEVEL to have an abort on this kind
   of error. */
int static
wpips_xview_error(Xv_object object,
                  Attr_avlist avlist)
{
   debug_on("WPIPS_DEBUG_LEVEL");

   fprintf(stderr, "wpips_xview_error caught an error:\n%s\n",
           xv_error_format(object, avlist));
   /* Cannot use pips_assert since it uses XView, neither
      get_bool_property for the same reason: */
   if (get_debug_level() > 0) {
      fprintf(stderr, "wpips_xview_error is aborting as requested since WPIPS_DEBUG_LEVEL > 0...\n");
      abort();
   }
   
   debug_off();
   
   return XV_OK;
}


/* Exit the notify loop to execute a WPips command: */
void
execute_main_loop_command(wpips_main_loop_command_type command)
{
   wpips_main_loop_command = command;
   notify_stop();
   /* I guess the function above does not return... */
}


/* The main loop that deals with command outside the XView notifyer: */
void static
wpips_main_loop(Frame frame_to_map_first)
{
   xv_main_loop(frame_to_map_first);

   /* The loop to execute commands: */
   while (wpips_main_loop_command != WPIPS_EXIT) {
      debug(1, "wpips_main_loop", "wpips_main_loop_command = %d\n", wpips_main_loop_command);
   
      switch((int) wpips_main_loop_command) {
        case WPIPS_SAFE_APPLY:
         execute_safe_apply_outside_the_notifyer();
         break;

        case WPIPS_EXECUTE_AND_DISPLAY:
         execute_wpips_execute_and_display_something_outside_the_notifyer();
         break;
         
        default:
         pips_assert("wpips_main_loop does not understand the wpips_main_loop_command", 0);
      }
      
      /* If the notifyer happen to exit without a specifyed command,
         just exit: */
      wpips_main_loop_command = WPIPS_EXIT;

      /* Restore the initial state of the blinking pips icon: */
      wpips_interrupt_button_restore();
         
      /* Wait again for something from X11 and emacs: */
      notify_start();
   }
}


int
wpips_main(int argc, char * argv[])
{
    pips_checks();

   pips_warning_handler = wpips_user_warning;
   pips_error_handler = wpips_user_error;
   pips_log_handler = wpips_user_log;
   pips_update_props_handler = update_options;
   pips_request_handler = wpips_user_request;

   initialize_newgen();
   initialize_sc((char*(*)(Variable))entity_local_name);
   set_exception_callbacks(push_pips_context, pop_pips_context);

   debug_on("WPIPS_DEBUG_LEVEL");

   /* we parse command line arguments */
   /* XV_ERROR_PROC unset as we shifted to xview.3, Apr. 92 */
   xv_init(XV_INIT_ARGC_PTR_ARGV, &argc, argv, 
           XV_ERROR_PROC, wpips_xview_error,
           0);

   /* we parse remaining command line arguments */
   wpips_parse_arguments(argc, argv);

   /* we create all frames */
   create_frames();

   create_main_window();

   create_help_window();

   create_schoose_window();

   create_mchoose_window();

   create_query_window();

   if (! wpips_emacs_mode)
      /* Create the edit/view windows only if we are not in the Emacs
         mode: */
      create_edit_window();

   /* In the Emacs mode, no XView log window but we need it to compute
      the size of some other frames... */
   create_log_window();

   create_menus();

   create_status_subwindow();

   /* create_menus_end(); */

   create_icons();
   /*    create_icon();*/

   /* Call added. RK, 9/06/1993. */
   place_frames();

   /* If we are in the emacs mode, initialize some things: */
   initialize_emacs_mode();
   
   display_memory_usage();

   enable_workspace_create_or_open();
   enable_change_directory();
   disable_workspace_close();
   disable_module_selection();
   disable_view_selection();
   disable_transform_selection();
   disable_compile_selection();
   disable_option_selection();

   set_pipsmake_callback(deal_with_wpips_events_during_pipsmake);

   execute_workspace_creation_and_so_on_given_with_options();
   
   wpips_main_loop(main_frame);
   
   reset_pipsmake_callback();

   close_log_file();
   
   debug_off();

   exit(0);
}
