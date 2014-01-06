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
/* 	%A% ($Date: 1998/04/16 14:45:19 $, ) version $Revision: 12279 $, got on %D%, %T% [%P%].\n Copyright (c) �cole des Mines de Paris Proprietary.	 */

#ifndef lint
char
		vcid_xv_edit2[] =
				"%A% ($Date: 1998/04/16 14:45:19 $, ) version $Revision: 12279 $, got on %D%, %T% [%P%].\n �cole des Mines de Paris Proprietary.";
#endif /* lint */

#include <stdlib.h>
#include <stdio.h>
#include <sys/param.h>

#include <sys/time.h>
#include <sys/resource.h>

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>

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

#include "gpips.h"

/* Include the label names: */
#include "gpips-labels.h"

static gpips_view_menu_layout_line gpips_view_menu_layout[] = {
#include "gpips_view_menu_layout.h"
		/* No more views */
		{ NULL, NULL, NULL } };

typedef struct {
	GtkTextView * view;
	GtkButton * save_button;
	GtkWidget * check_button;
	char * filename;
} EditedFile;

static EditedFile edited_file[MAX_NUMBER_OF_GPIPS_WINDOWS];
int number_of_gpips_windows = INITIAL_NUMBER_OF_GPIPS_WINDOWS;

static GtkWidget *current_selection_menu_item, *close_menu_item,
		*sequential_view_menu_item;
GtkWidget *edit_menu_item;

/* The menu "View" on the main panel: */
GtkWidget *view_menu, *view_menu_item;

/* To pass the view name to
 execute_gpips_execute_and_display_something_outside_the_notifyer(): */
static gpips_view_menu_layout_line
		* execute_gpips_execute_and_display_something_outside_the_notifyer_menu_line =
				NULL;

void edit_notify(GtkWidget * widget, gpointer data) {
	char string_filename[SMALL_BUFFER_LENGTH],
			string_modulename[SMALL_BUFFER_LENGTH];
	char file_name_in_database[MAXPATHLEN * 2];
	char * modulename = db_get_current_module_name();
	char * file_name;
	int win_nb;
	char * alternate_gpips_editor;

	if (modulename == NULL) {
		prompt_user("No module selected");
		return;
	}

	file_name = db_get_file_resource(DBR_SOURCE_FILE, modulename, TRUE);
	sprintf(file_name_in_database, "%s/%s", build_pgmwd(
			db_get_current_workspace_name()), file_name);

	if ((alternate_gpips_editor = getenv("PIPS_GPIPS_EDITOR")) != NULL) {
		char editor_command[MAXPATHLEN * 2];
		sprintf(editor_command, "%s %s &", alternate_gpips_editor,
				file_name_in_database);
		system(editor_command);
	} else {
		/* Is there an available edit_textsw ? */
		if ((win_nb = alloc_first_initialized_window(FALSE))
				== NO_TEXTSW_AVAILABLE) {
			prompt_user("None of the text-windows is available");
			return;
		}

		sprintf(string_filename, "File: %s", file_name);
		sprintf(string_modulename, "Module: %s", modulename);

		/* Display the file name and the module name. RK, 2/06/1993 : */
		gtk_window_set_title(GTK_WINDOW(edit_window[win_nb]), concatenate(
				"gpips edit facility - ", string_filename, " - ",
				string_modulename, NULL));

		// Charge le fichier file_name_in_database dans le text_view dispo
		gchar * file_in_mem;
		int size;
		GtkTextBuffer * txt_buff;
		GtkTextIter iter;

		edited_file[win_nb].filename = file_name_in_database;
		size = load_file(file_name_in_database, &file_in_mem);
		txt_buff = gtk_text_view_get_buffer(GTK_TEXT_VIEW(
				edited_file[win_nb].view));
		gtk_text_buffer_set_text(GTK_TEXT_BUFFER(txt_buff), file_in_mem, size);

		gtk_text_buffer_get_start_iter(GTK_TEXT_BUFFER(txt_buff), &iter);
		gtk_text_view_scroll_to_iter(GTK_TEXT_VIEW(edited_file[win_nb].view),
				&iter, 0.0, FALSE, 0.0, 0.0);
		gtk_widget_show_all(edit_window[win_nb]);

		g_free(file_in_mem);

		gtk_widget_set_sensitive(GTK_WIDGET(edited_file[win_nb].save_button),
				TRUE);

		gpips_gtk_menu_item_set_label(
				GTK_WIDGET(GTK_MENU_ITEM(current_selection_menu_item)), "Lasts");
		gtk_widget_set_sensitive(GTK_WIDGET(current_selection_menu_item), TRUE);

		gtk_widget_set_sensitive(GTK_WIDGET(close_menu_item), TRUE);
	}
}

void buffer_changed_callback(GtkWidget * widget, gpointer data) {
	EditedFile * f = (EditedFile *) data;
	// if the buffer changes, it means the document has been edited
	if ((!GTK_WIDGET_SENSITIVE(f->save_button)) && f->filename != NULL)
		gtk_widget_set_sensitive(GTK_WIDGET(f->save_button), TRUE);
}

static void save_edited_file(GtkWidget * widget, gpointer file) {
	EditedFile * f = (EditedFile *) file;
	char * filename = f->filename;
	GtkTextView * view = f->view;
	GtkTextBuffer * buff = gtk_text_view_get_buffer(GTK_TEXT_VIEW(view));

	GtkTextIter start, end;
	gtk_text_buffer_get_start_iter(GTK_TEXT_BUFFER(buff), &start);
	gtk_text_buffer_get_end_iter(GTK_TEXT_BUFFER(buff), &end);

	gchar * txt = gtk_text_buffer_get_text(GTK_TEXT_BUFFER(buff), &start, &end,
			TRUE);

	FILE * fd = fopen(filename, "w+");
	fprintf(fd, "%s", txt);
	fclose(fd);

	gtk_widget_set_sensitive(GTK_WIDGET(current_selection_menu_item), FALSE);
}

void current_selection_notify(GtkWidget * widget, gpointer data) {
	guint i;
	for (i = 0; i < number_of_gpips_windows; i++)
		gtk_widget_show(edit_window[i]);
}

char * compute_title_string(int window_number) {
	char title_string_beginning[] = "gpips display facility # ";
	static char title_string[sizeof(title_string_beginning) + 4];

	(void) sprintf(title_string, "%s%d", title_string_beginning, window_number
			+ 1);

	return title_string;
}

/* Find the first free window if any. If called with TRUE, give the
 same as the previous chosen one. */
int alloc_first_initialized_window(bool the_same_as_previous) {
	static int next = 0;
	static int candidate = 0;
	int i;

	if (the_same_as_previous)
		return candidate;

	for (i = next; i < next + number_of_gpips_windows; i++) {
		candidate = i % number_of_gpips_windows;
		/* Skip windows with modified text inside : */
		if (gpips_gtk_widget_get_sensitive(GTK_WIDGET(edited_file[candidate].save_button)))
			continue;
		/* Skip windows with a retain attribute : */
		if ((bool) gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(
				edited_file[candidate].check_button)))
			continue;

		next = candidate + 1;
		return candidate;
	}
	candidate = NO_TEXTSW_AVAILABLE;

	return candidate;
}

/* Mark a gpips window as busy: */
bool gpips_view_marked_busy(char * title_module_name, /* The module name for example */
char * title_label, /* "Sequential View" for exemple */
char * icon_name, char * icon_title) {
	char busy_label[SMALL_BUFFER_LENGTH];

	int window_number;
	/* Is there an available edit_textsw ? */
	if ((window_number = alloc_first_initialized_window(FALSE))
			== NO_TEXTSW_AVAILABLE) {
		prompt_user("None of the text-windows is available");
		return FALSE;
	}
	(void) sprintf(busy_label, "*Computing %s * ...", title_label);
	/* Display the file name and the module name. RK, 2/06/1993 : */
	gtk_window_set_title(GTK_WINDOW(edit_window[window_number]), concatenate(
			"gpips edit facility - ", busy_label, " - ", title_module_name,
			NULL));

	//set_pips_icon(edit_frame[window_number], icon_name, icon_title);

	//	unhide_window(edit_frame[window_number]);
	gtk_widget_show(edit_window[window_number]);

	display_memory_usage();

	return TRUE;
}

/* Display a file in a gpips window: */
void gpips_file_view(char * file_name, const char * title_module_name, /* The module name for example */
char * title_label, /* "Sequential View" for exemple */
char * icon_name, char * icon_title) {
	if (file_name == NULL) {
		/* Well, something got wrong... */
		prompt_user("Nothing available to display...");
		return;
	}

	int window_number;

	/* Is there an available edit_textsw ? Ask for the same one
	 allocated for gpips_view_marked_busy() */
	if ((window_number = alloc_first_initialized_window(TRUE))
			== NO_TEXTSW_AVAILABLE) {
		prompt_user("None of the text-windows is available");
		return;
	}
	/* Display the file name and the module name. RK, 2/06/1993 : */
	gtk_window_set_title(GTK_WINDOW(edit_window[window_number]), concatenate(
			compute_title_string(window_number), " - ", title_label, " - ",
			title_module_name, NULL));

	//	set_pips_icon(edit_window[window_number], icon_name, icon_title); TODO: Fix icones!

	// Charge le fichier file_name dans edited_file[window_number]
	gchar * file_in_mem;
	int size;
	GtkTextBuffer * txt_buff;
	GtkTextIter iter;

	edited_file[window_number].filename = file_name;
	size = load_file(file_name, &file_in_mem);
	txt_buff = gtk_text_view_get_buffer(GTK_TEXT_VIEW(
			edited_file[window_number].view));
	gtk_text_buffer_set_text(GTK_TEXT_BUFFER(txt_buff), file_in_mem, size);

	gtk_text_buffer_get_start_iter(GTK_TEXT_BUFFER(txt_buff), &iter);
	gtk_text_view_scroll_to_iter(
			GTK_TEXT_VIEW(edited_file[window_number].view), &iter, 0.0, FALSE,
			0.0, 0.0);
	gtk_widget_show_all(edit_window[window_number]);

	g_free(file_in_mem);

	gtk_widget_set_sensitive(
			GTK_WIDGET(edited_file[window_number].save_button), TRUE);

	gtk_widget_show(edit_window[window_number]);

	gpips_gtk_menu_item_set_label(current_selection_menu_item, "Lasts");
	gtk_widget_set_sensitive(current_selection_menu_item, TRUE);

	gtk_widget_set_sensitive(close_menu_item, TRUE);
	display_memory_usage();
}

/* Use daVinci to display a graph information: */
void gpips_display_graph_file_display(gpips_view_menu_layout_line * menu_line) {
	char * file_name;
	char a_buffer[SMALL_BUFFER_LENGTH];

	file_name = build_view_file(menu_line->resource_name_to_view);

	user_log("Displaying in a \"daVinci\" window...\n");

	/* Preprocess the graph to be understandable by daVinci : */

	(void) sprintf(a_buffer, "pips_graph2daVinci -launch_daVinci %s", file_name);
	system(a_buffer);

	free(file_name);
}

/* Use some text viewer to display the resource: */
void gpips_display_plain_file(gpips_view_menu_layout_line * menu_line) {
	char title_module_name[SMALL_BUFFER_LENGTH];

	char * print_type = menu_line->resource_name_to_view;
	char * icon_name = menu_line->icon_name;
	char * label = menu_line->menu_entry_string;

	(void) sprintf(title_module_name, "Module: %s",
			db_get_current_module_name());
	if (gpips_view_marked_busy(title_module_name, label, icon_name,
			db_get_current_module_name())) {
		char * file_name = build_view_file(print_type);

		gpips_file_view(file_name, title_module_name, label, icon_name,
				db_get_current_module_name());

		free(file_name);
	}
}

/* To execute something and display some Pips output with gpips, called outside the notifyer: */
void execute_gpips_execute_and_display_something_outside_the_notifier() {
	gpips_view_menu_layout_line * menu_line =
			execute_gpips_execute_and_display_something_outside_the_notifyer_menu_line;

	/* Execute the needed method: */
	menu_line->method_function_to_use(menu_line);

	display_memory_usage();
}

void gpips_execute_and_display_something_outside_the_notifyer(
		gpips_view_menu_layout_line * menu_line) {
	execute_gpips_execute_and_display_something_outside_the_notifyer_menu_line
			= menu_line;
	/* Ask to execute the
	 execute_gpips_execute_and_display_something_outside_the_notifyer(): */
	execute_main_loop_command(GPIPS_EXECUTE_AND_DISPLAY);
	/* I guess the function above does not return... */
}

/* To execute something and display some Pips output with gpips */
void gpips_execute_and_display_something(char * resource_name) {
	char * module_name = db_get_current_module_name();
	gpips_view_menu_layout_line * current_view;

	if (module_name == NULL) {
		prompt_user("No module selected");
		return;
	}

	/* Translate the resource name in a menu entry descriptor: */
	for (current_view = &gpips_view_menu_layout[0]; current_view->menu_entry_string
			!= NULL; current_view++)
		if (strcmp(resource_name, current_view->resource_name_to_view) == 0)
			break;

	pips_assert("Resource related to the menu entry not found",
			current_view->menu_entry_string != NULL);

	gpips_execute_and_display_something_outside_the_notifyer(current_view);
}

/* To execute something and display some Pips output with gpips by knowing its alias: */
void gpips_execute_and_display_something_from_alias(const char * alias_name) {
	char * module_name = db_get_current_module_name();
	gpips_view_menu_layout_line * current_view;

	if (module_name == NULL) {
		prompt_user("No module selected");
		return;
	}

	for (current_view = &gpips_view_menu_layout[0]; current_view->menu_entry_string
			!= NULL; current_view++)
		if (strcmp(alias_name, current_view->menu_entry_string) == 0)
			break;

	pips_assert("Resource related to the menu entry not found",
			current_view->menu_entry_string != NULL);

	gpips_execute_and_display_something_outside_the_notifyer(current_view);
}

void view_notify(GtkWidget * menu_item, gpointer data) {
	/* Translate the menu string in a resource name: */
	const char * label = gpips_gtk_menu_item_get_label(menu_item);
	gpips_execute_and_display_something_from_alias(label);
}

void edit_close_notify(GtkWidget * widget, gpointer data) {
	int i;

	for (i = 0; i < MAX_NUMBER_OF_GPIPS_WINDOWS; i++)
		if (!gpips_gtk_widget_get_sensitive(GTK_WIDGET(edited_file[i].save_button)))
			hide_window(edit_window[i], NULL, NULL);

	for (i = 0; i < MAX_NUMBER_OF_GPIPS_WINDOWS; i++)
		if (gpips_gtk_widget_get_sensitive(GTK_WIDGET(edited_file[i].save_button))) {
			gtk_widget_show(edit_window[i]);
			prompt_user("File not saved in editor");
			return;
		}

	for (i = 0; i < MAX_NUMBER_OF_GPIPS_WINDOWS; i++)
		hide_window(edit_window[i], NULL, NULL);

	gpips_gtk_menu_item_set_label(current_selection_menu_item, "No Selection");
	gtk_widget_set_sensitive(GTK_WIDGET(current_selection_menu_item), FALSE);

	gtk_widget_set_sensitive(GTK_WIDGET(close_menu_item), FALSE);

	display_memory_usage();
}

void disable_item(GtkWidget * item) {
	gtk_widget_set_sensitive(GTK_WIDGET(item), FALSE);
}

void enable_item(GtkWidget * item) {
	gtk_widget_set_sensitive(GTK_WIDGET(item), TRUE);
}

void apply_on_each_view_menu_item(GtkWidget * widget, gpointer _func) {
	void (*func)(GtkWidget *);
	func = (void(*)(GtkWidget *)) _func;
	if (widget != current_selection_menu_item && widget != close_menu_item)
		func(widget);
}

void apply_on_each_options_frame_button(GtkWidget * widget, gpointer _func) {
	void (*func)(GtkWidget *);
	func = (void(*)(GtkWidget *)) _func;
	if (GTK_IS_BUTTON(widget))
		func(widget);
}

void apply_on_each_view_item(void(* function_to_apply_on_each_menu_item)(
		GtkWidget *), void(* function_to_apply_on_each_panel_item)(GtkWidget *)) {
	//int i;

	/* Skip the "current_selection_mi" and "close" Menu_items: */
	gtk_container_foreach(GTK_CONTAINER(view_menu), (GtkCallback)
			apply_on_each_view_menu_item, function_to_apply_on_each_menu_item);
	//	for (i = (int) xv_get(view_menu, MENU_NITEMS); i > 0; i--) {
	//		Menu_item menu_item = (Menu_item) xv_get(view_menu, MENU_NTH_ITEM, i);
	//		/* Skip the title item: */
	//		if (!(bool) xv_get(menu_item, MENU_TITLE) && menu_item
	//				!= current_selection_menu_item && menu_item != close_menu_item
	//				&& xv_get(menu_item, MENU_NOTIFY_PROC) != NULL)
	//			function_to_apply_on_each_menu_item(menu_item);
	//	}

	/* Now walk through the options frame: */
	gtk_container_foreach(GTK_CONTAINER(options_frame), (GtkCallback)
			apply_on_each_options_frame_button,
			function_to_apply_on_each_panel_item);
}

void disable_view_selection() {
	apply_on_each_view_item(disable_item, disable_item);
}

void enable_view_selection() {
	apply_on_each_view_item(enable_item, enable_item);
}

void create_edit_window() {
	guint i;

	GtkWidget * frame;
	GtkWidget * text_buffer;
	GtkWidget * vbox;
	//GtkWidget * buttons_hbox;
	for (i = 0; i < MAX_NUMBER_OF_GPIPS_WINDOWS; i++) {

		vbox = gtk_vbox_new(FALSE, 0);
		//buttons_hbox = gtk_hbox_new(FALSE, 0);

		gtk_container_add(GTK_CONTAINER(edit_window[i]), vbox);

		edited_file[i].view = GTK_TEXT_VIEW(gtk_text_view_new());
		text_buffer = GTK_WIDGET(gtk_text_buffer_new(NULL));
		gtk_text_view_set_buffer(GTK_TEXT_VIEW(edited_file[i].view),
				GTK_TEXT_BUFFER(text_buffer));
		gtk_box_pack_start(GTK_BOX(vbox), GTK_WIDGET(edited_file[i].view), TRUE, FALSE, 0);

		g_signal_connect(G_OBJECT(text_buffer), "changed", G_CALLBACK(
				buffer_changed_callback), &edited_file[i]);

		edited_file[i].save_button = GTK_BUTTON(gtk_button_new_with_label("Save"));
		gtk_box_pack_start(GTK_BOX(vbox), GTK_WIDGET(edited_file[i].save_button), FALSE,
				FALSE, 0);
		gtk_widget_set_sensitive(GTK_WIDGET(edited_file[i].save_button), FALSE);
		g_signal_connect(G_OBJECT(edited_file[i].save_button), "clicked",
				G_CALLBACK(save_edited_file), &edited_file[i]);

		//gtk_box_pack_start(GTK_BOX(vbox), buttons_hbox, FALSE, FALSE, 0);
		frame = gtk_frame_new(NULL);
		gtk_box_pack_start(GTK_BOX(vbox), frame, FALSE, FALSE, 0);

		// A quoi sert ce check_button
		// TODO: trouver à quoi il sert
		edited_file[i].check_button = gtk_check_button_new_with_label(
				"Retain this window");
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(
				edited_file[i].check_button), FALSE);
		gtk_container_add(GTK_CONTAINER(frame), edited_file[i].check_button);
		//		gtk_box_pack_start(GTK_BOX(frame), edited_file[i].check_button,
		//				FALSE, FALSE, 0);

		edited_file[i].filename = NULL;

		gtk_widget_show_all(vbox);
	}
}

void create_edit_menu() {
	GtkWidget *menu_item;
	gpips_view_menu_layout_line * current_view;

	current_selection_menu_item = gtk_menu_item_new_with_label("No Selection");
	g_signal_connect(G_OBJECT(current_selection_menu_item), "activate",
			G_CALLBACK(current_selection_notify), NULL);
	gtk_widget_set_sensitive(GTK_WIDGET(current_selection_menu_item), FALSE);

	close_menu_item = gtk_menu_item_new_with_label("Close");
	g_signal_connect(G_OBJECT(close_menu_item), "activate", G_CALLBACK(
			edit_close_notify), NULL);
	gtk_widget_set_sensitive(GTK_WIDGET(close_menu_item), FALSE);

	sequential_view_menu_item = gtk_menu_item_new_with_label(SEQUENTIAL_VIEW);
	g_signal_connect(G_OBJECT(sequential_view_menu_item), "activate",
			G_CALLBACK(view_notify), NULL);

	view_menu = gtk_menu_new();
	gtk_menu_append(GTK_MENU(view_menu), current_selection_menu_item);

	/* Now add all the view entries: */
	for (current_view = &gpips_view_menu_layout[0]; current_view->menu_entry_string
			!= NULL; current_view++) {
		if (strcmp(current_view->menu_entry_string, GPIPS_MENU_SEPARATOR_ID)
				== 0) {
			gtk_menu_append(GTK_MENU(view_menu), gtk_separator_menu_item_new());
		} else {
			menu_item = gtk_menu_item_new_with_label(
					current_view->menu_entry_string);
			g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(
					view_notify), NULL);
			gtk_menu_append(GTK_MENU(view_menu), menu_item);
		}
	}
	gtk_menu_append(GTK_MENU(view_menu), gtk_separator_menu_item_new());
	gtk_menu_append(GTK_MENU(view_menu), close_menu_item);

	view_menu_item = gtk_menu_item_new_with_label("View & Edit Menu");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(view_menu_item), view_menu);

	gtk_widget_show_all(view_menu);
	gtk_widget_show(view_menu_item);
	gtk_menu_bar_append(GTK_MENU_BAR(main_window_menu_bar), view_menu_item);
}
