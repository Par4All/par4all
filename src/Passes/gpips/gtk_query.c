/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#include <stdarg.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "genC.h"
#include "misc.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

static GtkWidget * query_entry, * query_entry_label;
static GtkWidget * query_cancel_button;

static char *query_help_topic;
static success (*apply_on_query)(const char *);

void start_query(char * window_title, char * query_title, char * help_topic,
		success(* ok_func)(const char * ), void(* cancel_func)( GtkWidget*, gpointer)) {

	gtk_window_set_title(GTK_WINDOW(query_dialog), window_title);

	/*	     PANEL_NOTIFY_PROC, cancel_query_notify, */
	gtk_widget_set_sensitive(GTK_WIDGET(query_cancel_button), TRUE);
	if (cancel_func == NULL)
		/* No cancel button requested: */
		gtk_widget_set_sensitive(GTK_WIDGET(query_cancel_button), FALSE);
	else
		gtk_signal_connect(GTK_OBJECT(query_cancel_button), "clicked",
				GTK_SIGNAL_FUNC(cancel_func), NULL);

	gtk_label_set_text(GTK_LABEL(query_entry_label), query_title);
	gtk_entry_set_text(GTK_ENTRY(query_entry), "");

	query_help_topic = help_topic;
	apply_on_query = ok_func;

	gtk_widget_show_all(query_dialog);
}

// don't know the purpose of this code...

//void query_canvas_event_proc(window, event)
//	Xv_Window window;Event *event; {
//	debug_on("WPIPS_EVENT_DEBUG_LEVEL");
//	debug(2, "query_canvas_event_proc", "Event_id %d, event_action %d\n",
//			event_id(event), event_action(event));
//	debug_off();
//	switch (event_id(event)) {
//	case LOC_WINENTER:
//		/* enter_window(window); */
//		break;
//	case '\r':
//		/* ie. return key pressed */
//		if (event_is_up(event))
//			/* ie. key is released. It is necessary to use this event
//			 because notice_prompt() (in prompt_user() (in
//			 end_query_notify() )) also returns on up RETURN.
//			 This can cause the notice to return immediately when it is
//			 called on down RETURN.
//			 There schould be another possibility: put a mask to ignore
//			 key release events on the window which owns notice_prompt().
//			 This was done in create_main_window() but seems without
//			 effect.
//			 */
//			end_query_notify(NULL, event);
//		break;
//	default:
//		;
//	}
//}

void end_query_notify(GtkWidget * widget, gpointer data) {
	const char * s = gtk_entry_get_text(GTK_ENTRY(query_entry));
	if (s == NULL)
		s = strdup("");
	else
		s = strdup(s);

	if (apply_on_query(s))
		hide_window(query_dialog, NULL, NULL);
}

void help_query_notify(GtkWidget * widget, gpointer data) {
	display_help(query_help_topic);
}

/* hides a window... */
void cancel_query_notify(GtkWidget * widget, gpointer data) {
	hide_window(query_dialog, NULL, NULL);
}

/* Cancel clear the string value and return: */
void cancel_user_request_notify(GtkWidget * widget, gpointer data) {
	gtk_entry_set_text(GTK_ENTRY(query_entry), "");
	hide_window(query_dialog, NULL, NULL);
	/* Just return the "": */
	gtk_dialog_response(GTK_DIALOG(query_dialog), 1);
}

success end_user_request_notify(const char * the_answer) {
	hide_window(query_dialog, NULL, NULL);
	gtk_dialog_response(GTK_DIALOG(query_dialog), 1);
	return TRUE;
}

string gpips_user_request(const char * a_printf_format, va_list args) {

  /* char * the_answer; */

	static char message_buffer[SMALL_BUFFER_LENGTH];

	(void) vsprintf(message_buffer, a_printf_format, args);

	start_query("User Query", message_buffer, "UserQuery",
			end_user_request_notify, cancel_user_request_notify);

	user_log("User Request...\n");

	gtk_widget_show(query_dialog);
	gtk_window_set_modal(GTK_WINDOW(query_dialog), TRUE);
	gtk_dialog_run(GTK_DIALOG(query_dialog)); // On force l'attente de la réponse
	gtk_window_set_modal(GTK_WINDOW(query_dialog), FALSE);

	/* Log the answer for possible rerun through tpips: */
	user_log("%s\n\"%s\"\nEnd User Request\n", message_buffer, "" /*the_answer*/);

	return strdup(gtk_entry_get_text(GTK_ENTRY(query_entry)));
}

void create_query_window() {
	GtkWidget *help_button, *ok_button;
	//GtkWidget *action_area;
	GtkWidget *content_area;

	/* seems it has no use. RK, 9/11/93. */
	//	xv_set(canvas_paint_window(query_panel), WIN_CONSUME_EVENT, LOC_WINENTER,
	//			NULL,
	//			/*	   WIN_IGNORE_X_EVENT_MASK, KeyReleaseMask, */
	//			WIN_EVENT_PROC, query_canvas_event_proc, NULL);

	//action_area = gtk_dialog_get_action_area(query_dialog);
	content_area = gpips_gtk_dialog_get_content_area(GTK_DIALOG(query_dialog));

	GtkWidget * hbox = gtk_hbox_new(FALSE,0);
	query_entry_label = gtk_label_new(NULL);
	query_entry = gtk_entry_new_with_max_length(128);
	gtk_box_pack_start(GTK_BOX(hbox), query_entry_label, FALSE, FALSE, 5);
	gtk_box_pack_start(GTK_BOX(hbox), query_entry, FALSE, FALSE, 5);
	gtk_container_add(GTK_CONTAINER(content_area), hbox);
	gtk_widget_show_all(hbox);

	ok_button = gtk_button_new_with_label("OK");
	gtk_dialog_add_action_widget(GTK_DIALOG(query_dialog), ok_button, GTK_RESPONSE_OK);
	gtk_signal_connect(GTK_OBJECT(ok_button), "clicked", GTK_SIGNAL_FUNC(
			end_query_notify), NULL);
	gtk_window_set_default(GTK_WINDOW(query_dialog), ok_button);
	//gtk_container_add(GTK_CONTAINER(action_area), ok_button);

	help_button = gtk_button_new_with_label("Help");
	gtk_dialog_add_action_widget(GTK_DIALOG(query_dialog), help_button, GTK_RESPONSE_HELP);
	gtk_signal_connect(GTK_OBJECT(help_button), "clicked", GTK_SIGNAL_FUNC(
			help_query_notify), NULL);
	//gtk_container_add(GTK_CONTAINER(action_area), help_button);

	query_cancel_button = gtk_button_new_with_label("Cancel");
	gtk_dialog_add_action_widget(GTK_DIALOG(query_dialog), query_cancel_button, GTK_RESPONSE_CANCEL);
	//gtk_container_add(GTK_CONTAINER(action_area), query_cancel_button);

	gtk_widget_show_all(query_dialog);
	gtk_widget_hide(query_dialog);
}
