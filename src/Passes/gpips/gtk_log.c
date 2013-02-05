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
#include <errno.h>

#include <sys/time.h>
#include <sys/resource.h>

/* xview/newgen interaction
 */
#if (defined(TEXT))
#undef TEXT
#endif

#if (defined(TEXT_TYPE))
#undef TEXT_TYPE
#endif

#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "top-level.h"

#include "properties.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>

#include "gpips.h"

static GtkWidget *log_text_view;
static GtkWidget *open_or_front_menu_item, *clear_menu_item, *close_menu_item;
static GtkWidget *scrolled_window_vadjustment;

void prompt_user(string a_printf_format, ...) {
	va_list some_arguments;
	static char message_buffer[SMALL_BUFFER_LENGTH];

	va_start(some_arguments, a_printf_format);

	(void) vsprintf(message_buffer, a_printf_format, some_arguments);

	GtkWidget * dialog = gtk_message_dialog_new(GTK_WINDOW(log_window),
			GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_INFO, GTK_BUTTONS_OK,
						    "%s", message_buffer);
	gtk_dialog_run(GTK_DIALOG(dialog));
	gtk_widget_destroy(dialog);
}

static void insert_something_in_the_gpips_log_window(char * a_message) {
	GtkTextIter iter;
	GtkTextBuffer * buffer;

	buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(log_text_view));

	//int new_length;
	int message_length = strlen(a_message);

	/* insert at the end: */
	gtk_text_buffer_get_end_iter(GTK_TEXT_BUFFER(buffer), &iter);
	gtk_text_buffer_insert(GTK_TEXT_BUFFER(buffer), &iter, a_message,
			message_length);

	gtk_text_view_scroll_to_iter(GTK_TEXT_VIEW(log_text_view), &iter, 0.0,
			FALSE, 0.0, 0.0);
	gtk_adjustment_set_value(GTK_ADJUSTMENT(scrolled_window_vadjustment),
			gpips_gtk_adjustment_get_upper(GTK_ADJUSTMENT(
					scrolled_window_vadjustment)));

	gtk_widget_set_sensitive(clear_menu_item, TRUE);
}

void gpips_user_error_message(char error_buffer[]) {
	log_on_file(error_buffer);

	insert_something_in_the_gpips_log_window(error_buffer);

	show_message(error_buffer);
	gtk_widget_show(log_window);

	/* prompt_user("Something went wrong. Check the log window"); */

	/* terminate PIPS request */
	if (get_bool_property("ABORT_ON_USER_ERROR"))
		abort();

	THROW(user_exception_error);

	(void) exit(1);
}

void gpips_user_warning_message(char warning_buffer[]) {
	log_on_file(warning_buffer);

	insert_something_in_the_gpips_log_window(warning_buffer);
	/* François said a warning is not important enough...
	 gtk_widget_show(log_window);
	 */

	show_message(warning_buffer);
}

#define MAXARGS     100

void gpips_user_log(const char* fmt, va_list args) {
	static char log_buffer[SMALL_BUFFER_LENGTH];

	(void) vsprintf(log_buffer, fmt, args);

	log_on_file(log_buffer);

	if (get_bool_property("USER_LOG_P") == FALSE)
		return;

	insert_something_in_the_gpips_log_window(log_buffer);
	/* Display the "Message:" line in the main window */
	show_message(log_buffer);
}

void open_log_subwindow(GtkWidget * widget, gpointer data) {
	gpips_gtk_menu_item_set_label(open_or_front_menu_item, "Front");
	gtk_widget_set_sensitive(close_menu_item, TRUE);
	gtk_widget_show(log_window);
}

void clear_log_subwindow(GtkWidget * widget, gpointer data) {
	GtkTextIter start, end;
	GtkTextBuffer * buffer;
	buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(log_text_view));
	gtk_text_buffer_get_start_iter(GTK_TEXT_BUFFER(buffer), &start);
	gtk_text_buffer_get_end_iter(GTK_TEXT_BUFFER(buffer), &end);
	gtk_text_buffer_delete(GTK_TEXT_BUFFER(buffer), &start, &end);

	gtk_widget_set_sensitive(clear_menu_item, FALSE);
}

void close_log_subwindow(GtkWidget * widget, gpointer data) {
	gpips_gtk_menu_item_set_label(open_or_front_menu_item, "Open");
	gtk_widget_set_sensitive(close_menu_item, FALSE);
	hide_window(log_window, NULL, NULL);
}

void create_log_menu() {
	GtkWidget * log_menu;
	GtkWidget * log_menu_item;

	log_menu = gtk_menu_new();
	log_menu_item = gtk_menu_item_new_with_label("Log");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(log_menu_item), log_menu);

	open_or_front_menu_item = gtk_menu_item_new_with_label("Open");
	g_signal_connect(G_OBJECT(open_or_front_menu_item), "activate", G_CALLBACK(
			open_log_subwindow), NULL);
	gtk_menu_append(GTK_MENU(log_menu), open_or_front_menu_item);

	clear_menu_item = gtk_menu_item_new_with_label("Clear");
	g_signal_connect(G_OBJECT(clear_menu_item), "activate", G_CALLBACK(
			clear_log_subwindow), NULL);
	gtk_menu_append(GTK_MENU(log_menu), clear_menu_item);

	close_menu_item = gtk_menu_item_new_with_label("Close");
	g_signal_connect(G_OBJECT(close_menu_item), "activate", G_CALLBACK(
			close_log_subwindow), NULL);
	gtk_menu_append(GTK_MENU(log_menu), close_menu_item);

	gtk_widget_show(log_menu_item);
	gtk_widget_show_all(log_menu);
	gtk_menu_bar_append(GTK_MENU_BAR(main_window_menu_bar), log_menu_item);
}

/* This works but it is cleaner to use textsw_reset() instead...
 void
 recreate_log_window()
 {
 xv_destroy(log_textsw);
 log_textsw = (Xv_Window) xv_create(log_frame, TEXTSW, 0);
 }
 */

void create_log_window() {
	/* Xv_Window window; */

	GtkWidget * frame = gtk_frame_new("Log");

	GtkWidget * scrolled_window;

	scrolled_window = gtk_scrolled_window_new(NULL, NULL);
	scrolled_window_vadjustment = GTK_WIDGET(gtk_scrolled_window_get_vadjustment(
			GTK_SCROLLED_WINDOW(scrolled_window)));
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window),
			GTK_POLICY_AUTOMATIC, GTK_POLICY_ALWAYS);

	log_text_view = gtk_text_view_new();

	gtk_container_add(GTK_CONTAINER(log_window), frame);
	gtk_container_add(GTK_CONTAINER(frame), scrolled_window);
	gtk_container_add(GTK_CONTAINER(scrolled_window), log_text_view);

	gtk_widget_show_all(frame);
}
