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

#include "genC.h"
#include "misc.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gtk_sizes.h"
#include "gpips.h"

#if 0
#define BOUND(x,lb,ub) \
    ((x)<(lb)) ? (lb) :\
    ((x)>(ub)) ? (ub) : (x)
#define MAX(a, b) ((a)>(b) ? (a) : (b))
#endif

//static int display_width, display_height;

//void event_procedure(Xv_Window window, Event *event, Notify_arg arg) {
//	debug_on("GPIPS_EVENT_DEBUG_LEVEL");
//	debug(2, "event_procedure", "Event_id %d, event_action %d\n",
//			event_id(event), event_action(event));
//	switch (event_id(event)) {
//	case LOC_WINENTER:
//		win_set_kbd_focus(window, xv_get(window, XV_XID));
//		debug(2, "event_procedure", "LOC_WINENTER\n");
//		break;
//	}
//	debug_off();
//}

//void install_event_procedure(Xv_Window window) {
//	// au final uniquement pour du debug
//	// et changement de focus window par focus souris (à l'ancienne)
//	// on va comment tous les appels!
//	xv_set(window, WIN_EVENT_PROC, event_procedure, WIN_CONSUME_EVENTS,
//			LOC_WINENTER, NULL, NULL);
//}

//void place_frame(frame, l, t)
//	Frame frame;int l, t; {
//	Rect rect;
//
//	frame_get_rect(frame, &rect);
//
//	/* We need to estimate the size of the decor added by the widow
//	 manager, Y_WM_DECOR_SIZE & X_WM_DECOR_SIZE. RK, 9/10/1993. */
//	rect.r_top
//			= BOUND(t, 0, MAX(0,display_height-rect.r_height-Y_WM_DECOR_SIZE));
//	rect.r_left
//			= BOUND(l, 0, MAX(0,display_width-rect.r_width-X_WM_DECOR_SIZE));
//
//	frame_set_rect(frame, &rect);
//}

void create_windows() {
	guint i;

	main_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(main_window), "gpips");

	//	install_event_procedure(main_frame);

	log_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_transient_for(GTK_WINDOW(log_window), GTK_WINDOW(main_window));
	gtk_window_set_default_size(GTK_WINDOW(log_window), 300, 300);
//	gtk_window_set_default_size(log_window, defaults_get_integer(
//			GPIPS_LOG_WINDOW_WIDTH_RESSOURCE_NAME,
//			GPIPS_LOG_WINDOW_WIDTH_RESSOURCE_CLASS,
//			GPIPS_LOG_WINDOW_WIDTH_DEFAULT), defaults_get_integer(
//			GPIPS_LOG_WINDOW_HEIGHT_RESSOURCE_NAME,
//			GPIPS_LOG_WINDOW_HEIGHT_RESSOURCE_CLASS,
//			GPIPS_LOG_WINDOW_HEIGHT_DEFAULT));
	gtk_window_set_title(GTK_WINDOW(log_window), "gpips - Log Window");

	/* Footers added to edit window.
	 RK, 21/05/1993. */
	for (i = 0; i < MAX_NUMBER_OF_GPIPS_WINDOWS; i++) {
		//		edit_frame[i] = xv_create(main_frame, FRAME, XV_SHOW, FALSE,
		//				FRAME_DONE_PROC, hide_window, XV_WIDTH, EDIT_WIDTH, XV_HEIGHT,
		//				EDIT_HEIGHT, FRAME_SHOW_FOOTER, TRUE, FRAME_LEFT_FOOTER, "<",
		//				FRAME_RIGHT_FOOTER, ">", NULL);
		edit_window[i] = gtk_window_new(GTK_WINDOW_TOPLEVEL);
		gtk_window_set_transient_for(GTK_WINDOW(edit_window[i]), GTK_WINDOW(main_window));
		gtk_window_set_default_size(GTK_WINDOW(edit_window[i]), EDIT_WIDTH, EDIT_HEIGHT);
		// On bind "hide window" sur la fermeture d'une de ces fenêtres
		gtk_signal_connect(GTK_OBJECT(edit_window[i]), "delete-event",
				GTK_SIGNAL_FUNC(hide_window), NULL);
	}

	help_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	mchoose_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	schoose_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	query_dialog = gtk_dialog_new();
	options_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

	gtk_window_set_title(GTK_WINDOW(help_window), "pips online help facility");
	gtk_window_set_default_size(GTK_WINDOW(help_window), HELP_WIDTH, HELP_HEIGHT);
	gtk_window_set_default_size(GTK_WINDOW(query_dialog), QUERY_WIDTH, QUERY_HEIGHT);
	gtk_window_set_title(GTK_WINDOW(options_window), "gpips options");

	g_signal_connect(GTK_OBJECT(log_window), "delete-event",
				GTK_SIGNAL_FUNC(hide_window), NULL);
	g_signal_connect(GTK_OBJECT(help_window), "delete-event",
			GTK_SIGNAL_FUNC(hide_window), NULL);
	g_signal_connect(GTK_OBJECT(mchoose_window), "delete-event",
			GTK_SIGNAL_FUNC(hide_window), NULL);
	g_signal_connect(GTK_OBJECT(schoose_window), "delete-event",
			GTK_SIGNAL_FUNC(hide_window), NULL);
	g_signal_connect(GTK_OBJECT(query_dialog), "delete-event",
			GTK_SIGNAL_FUNC(hide_window), NULL);
	g_signal_connect(GTK_OBJECT(options_window), "delete-event",
			GTK_SIGNAL_FUNC(hide_window), NULL);
	g_signal_connect(GTK_OBJECT(main_window), "delete-event",
				GTK_SIGNAL_FUNC(quit_notify), NULL);

//	gtk_widget_show_all(help_window);
//	gtk_widget_show_all(mchoose_window);
//	gtk_widget_show_all(schoose_window);
//	gtk_widget_show_all(query_dialog);
//	gtk_widget_show_all(options_window);

	gtk_window_set_transient_for(GTK_WINDOW(help_window), GTK_WINDOW(main_window));
	gtk_window_set_transient_for(GTK_WINDOW(mchoose_window), GTK_WINDOW(main_window));
	gtk_window_set_transient_for(GTK_WINDOW(schoose_window), GTK_WINDOW(main_window));
	gtk_window_set_transient_for(GTK_WINDOW(query_dialog), GTK_WINDOW(main_window));
	gtk_window_set_transient_for(GTK_WINDOW(options_window), GTK_WINDOW(main_window));
}

//void place_frames() {
//	Rect rect;
//	int main_l, main_t, main_w, main_h;
//	int main_center_l, main_center_t;
//	int i;
//
//	Frame full_frame;
//	Xv_Screen screen;
//	Display *dpy;
//	int screen_no;
//
//	/* get the display dimensions */
//	full_frame = (Frame) xv_create(XV_NULL, FRAME, NULL);
//	dpy = (Display *) xv_get(full_frame, XV_DISPLAY);
//	screen = (Xv_Screen) xv_get(full_frame, XV_SCREEN);
//	screen_no = (int) xv_get(screen, SCREEN_NUMBER);
//	xv_destroy(full_frame);
//
//	display_width = DisplayWidth(dpy, screen_no);
//	display_height = DisplayHeight(dpy, screen_no);
//
//	/* warning: some window managers do NOT place the top frame (main_frame)
//	 themselves. In this case add this fonction call and modify the call
//	 to place_frames().
//
//	 place_frame(main_frame,
//	 (display_width-WPIPS_WIDTH)/2,
//	 (display_height-WPIPS_HEIGHT)/2);
//	 */
//	/* in the bottom left : */
//	place_frame(log_frame, 0, display_height);
//
//	frame_get_rect(log_frame, &rect);
//
//	main_t = rect.r_top;
//	main_w = rect.r_width;
//	main_h = rect.r_height;
//	main_l = rect.r_left;
//
//	place_frame(main_frame, 0, main_t - xv_get(main_frame, XV_HEIGHT)
//			- Y_WM_DECOR_SIZE);
//
//	frame_get_rect(main_frame, &rect);
//
//	main_t = rect.r_top;
//	main_w = rect.r_width;
//	main_h = rect.r_height;
//	main_l = rect.r_left;
//
//	main_center_t = main_t + main_h / 2;
//	main_center_l = main_l + main_w / 2;
//
//	/* Above the main frame : */
//	place_frame(options_frame, 0, main_t - xv_get(options_frame, XV_HEIGHT)
//			- Y_WM_DECOR_SIZE);
//
//	/* in the upper right corner, in the bottom right corner, etc : */
//	for (i = 0; i < MAX_NUMBER_OF_WPIPS_WINDOWS; i++)
//		/* According to the 2 least bits of the window number... */
//		place_frame(edit_frame[i], display_width * (1 - ((i >> 1) & 1)),
//				display_height * (i & 1));
//
//	/* in the upper */
//	place_frame(help_frame, main_l + (main_w - HELP_WIDTH) / 2, main_t
//			- HELP_HEIGHT);
//
//	/* in the upper left corner : */
//	place_frame(mchoose_frame, 0, 0);
//
//	/* in the upper left corner : */
//	place_frame(schoose_frame, 0, 0);
//
//	/* in the upper left corner */
//	place_frame(query_frame, main_l - QUERY_WIDTH + MIN(main_w, QUERY_WIDTH)
//			/ 3, main_t - QUERY_HEIGHT - Y_WM_DECOR_SIZE);
//}
