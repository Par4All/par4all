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

/* Multiple choices handling */

#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "genC.h"
#include "misc.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>

#include "gpips.h"

static GtkWidget *mchoices_label, *choices_list, *ok_button, *cancel_button,
		*help_button;
static GtkListStore *choices;

enum {
	M_AVAILABLE_CHOICES_COLUMN_ID, M_COLUMNS_NUMBER
};

static void (*apply_on_mchoices)( gen_array_t) = NULL;
static void (*cancel_on_mchoices)(void) = NULL;

#if 0
static void mchoose_help_notify(GtkWidget * widget __attribute__((unused)), gpointer data  __attribute__((unused))) {
	display_help("MultipleChoice");
}
#endif

static void mchoose_ok_notify(GtkWidget * widget  __attribute__((unused)), gpointer data  __attribute__((unused))) {
	gen_array_t mchoices_args = gen_array_make(0);

	GtkTreeSelection * selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(
			choices_list));

	gchar *buffer, *mchoices_notify_buffer;
	int mchoices_length = 0;
//	int nchoices;
	int len;
//	int item_is_in_the_list = FALSE;
	char * p;

//	nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, NULL);

	mchoices_notify_buffer = strdup(gtk_label_get_text(GTK_LABEL(mchoices_label)));
	/* Upperbound size for the scanf buffer: */
	buffer = (char *) malloc(strlen(mchoices_notify_buffer) + 1);

	p = mchoices_notify_buffer;
	while (sscanf(p, "%s%n", buffer, &len) == 1) {
		gen_array_dupaddto(mchoices_args, mchoices_length++, buffer);
//		item_is_in_the_list = FALSE;
//		for (int i = 0; i < nchoices; i++)
//			if (strcmp((char *) xv_get(choices, PANEL_LIST_STRING, i), buffer)
//					== 0) {
//				item_is_in_the_list = TRUE;
//				break;
//			}
//		if (item_is_in_the_list == FALSE)
//			break;
		p += len;
	}

	g_free(mchoices_notify_buffer);
	g_free(buffer);
	//
	//	/*	At least on item selected, and in the list.
	//	 RK, 21/05/1993.
	//	 */
	//	if (mchoices_length == 0 || item_is_in_the_list == FALSE) {
	//		char *s;
	//		s = mchoices_length == 0 ? "You have to select at least 1 item!"
	//				: "You have selected an item not in the choice list!";
	//		gen_array_full_free(mchoices_args);
	//		prompt_user(s);
	//		return;
	//	}
	if (gtk_tree_selection_count_selected_rows(selection) == 0) {
		prompt_user("You have to select at least 1 item !");
		return;
	}

	hide_window(mchoose_window, NULL, NULL);

	// ?
	//	xv_set(mchoices, PANEL_NOTIFY_PROC, NULL);

	(*apply_on_mchoices)(mchoices_args);
	gen_array_full_free(mchoices_args);

	/* Delay the graphics transformations. RK, 21/05/1993. */

	gtk_list_store_clear(choices);

	gtk_label_set_text(GTK_LABEL(mchoices_label), "");
}

static void mchoose_cancel_notify(GtkWidget * widget  __attribute__((unused)), gpointer data  __attribute__((unused))) {
	hide_window(mchoose_window, NULL, NULL);

	// The OK button becomes inactive through RETURN:
	gtk_window_set_default(GTK_WINDOW(mchoose_window), NULL);

	// ?
	//xv_set(mchoices, PANEL_NOTIFY_PROC, NULL);

	(*cancel_on_mchoices)();

	gtk_list_store_clear(choices);

	gtk_label_set_text(GTK_LABEL(mchoices_label), "");
}

/* Avoid the mchoose_frame destruction and act as cancel: */
static void mchoose_window_done_proc(GtkWidget * window  __attribute__((unused)), gpointer data  __attribute__((unused))) {
	mchoose_cancel_notify(NULL, NULL);
}

static void concat_labels(gpointer _row, gpointer _label) {
	GtkTreePath * path = (GtkTreePath *) _row;
	gchar ** label = (gchar **) _label;
	gchar * new_label;

	GValue path_associated_value = { 0, };
	GtkTreeIter iter;
	gtk_tree_model_get_iter(GTK_TREE_MODEL(choices), &iter, path);
	gtk_tree_model_get_value(GTK_TREE_MODEL(choices), &iter,
			M_AVAILABLE_CHOICES_COLUMN_ID, &path_associated_value);
	gchar * path_associated_value_string = (gchar *) g_value_get_string(
			&path_associated_value);

	new_label = strdup(concatenate(*label, path_associated_value_string, " ",
			NULL));
	g_free(*label);
	*label = new_label;
}

/* Function used to update the text panel according to the list panel: */
//int static mchoose_notify(Panel_item item, char * item_string,
//		Xv_opaque client_data, Panel_list_op op, Event * event, int row) {
static void mchoose_callback(GtkTreeSelection * selection, gpointer data  __attribute__((unused))) {
	GList * selected_rows;
	gchar * new_mchoices_label;

	selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(choices_list));
	new_mchoices_label = strdup("");

	/* Make the PANEL_VALUE of mchoices a string that is all the
	 names of the selected files: */
	selected_rows = gtk_tree_selection_get_selected_rows(GTK_TREE_SELECTION(
			selection), (GtkTreeModel**)&choices);

	g_list_foreach(selected_rows, concat_labels, &new_mchoices_label);
	g_list_free(selected_rows);

	gtk_label_set_text(GTK_LABEL(mchoices_label), new_mchoices_label);
	g_free(new_mchoices_label);
}

/* When we press on the "(De)Select" all button, select or deselect
 all the items. */
static void mchoose_de_select_all_notify(GtkWidget * widget  __attribute__((unused)), gpointer data  __attribute__((unused))) {
	GtkTreeSelection * selection;

	selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(choices_list));

	static bool select_all_when_press_this_button = TRUE;

	if (select_all_when_press_this_button) {
		gtk_tree_selection_select_all(GTK_TREE_SELECTION(selection));
	} else {
		gtk_tree_selection_unselect_all(GTK_TREE_SELECTION(selection));
	}

	/* Update the "Current choices": */
	mchoose_callback(selection, NULL);

	/* Next time we press this button, do the opposite: */
	select_all_when_press_this_button = !select_all_when_press_this_button;
}

void mchoose(char * title, gen_array_t array, void(*function_ok)( gen_array_t),
		void(*function_cancel)(void)) {
	int i, argc = gen_array_nitems(array);

	GtkTreeIter iter;
	gen_array_nitems(array);

	apply_on_mchoices = function_ok;
	cancel_on_mchoices = function_cancel;

	gtk_window_set_title(GTK_WINDOW(mchoose_window), title);
	gtk_list_store_clear(choices);

	for (i = 0; i < argc; i++) {
		string name = gen_array_item(array, i);
		gtk_list_store_append(GTK_LIST_STORE(choices), &iter);
		gtk_list_store_set(choices, &iter, M_AVAILABLE_CHOICES_COLUMN_ID, name,
				-1);
	}
	gtk_widget_show(mchoose_window);

	/* move the pointer to the center of the query window */
	// Sans vouloir remettre en question le design de la chose
	// A priori je ne traduirai pas ça à moins qu'on ne le
	// demande expressément...
	//pointer_in_center_of_frame(mchoose_frame);
	//?
	//	xv_set(mchoices, PANEL_NOTIFY_PROC, mchoose_ok_notify, NULL);
}

void create_mchoose_window() {
	guint i;
	GtkWidget *vbox, *buttons_hbox;
	GtkTreeIter iter;
	GtkTreeSelection * _select;
	GtkWidget *deselect_button;

	GtkWidget * scrolled_window;
	scrolled_window = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window),
			GTK_POLICY_AUTOMATIC, GTK_POLICY_ALWAYS);

	vbox = gtk_vbox_new(FALSE, 0);
	buttons_hbox = gtk_hbox_new(FALSE, 0);
	gtk_container_add(GTK_CONTAINER(mchoose_window), vbox);

	//	mchoose_panel = xv_create(mchoose_frame, PANEL, NULL);


	//	mchoices = xv_create(mchoose_panel, PANEL_TEXT, PANEL_LABEL_STRING,
	//			"Current choices", PANEL_VALUE_DISPLAY_LENGTH, 30,
	//			PANEL_VALUE_STORED_LENGTH, 12800,
	//			/* PANEL_NOTIFY_PROC, mchoose_ok_notify, */
	//			XV_X, xv_col(mchoose_panel, 0), NULL);
	gtk_box_pack_start(GTK_BOX(vbox), gtk_label_new("Current choices : "),
			FALSE, FALSE, 0);
	mchoices_label = gtk_label_new("");
	gtk_box_pack_start(GTK_BOX(vbox), mchoices_label, FALSE, FALSE, 0);
	gtk_widget_show(mchoices_label);

	mchoose_frame = gtk_frame_new("test");

	//	choices = xv_create(mchoose_panel, PANEL_LIST, PANEL_LABEL_STRING,
	//			"Available choices", PANEL_LIST_DISPLAY_ROWS, 5, PANEL_NOTIFY_PROC,
	//			mchoose_notify, PANEL_CHOOSE_ONE, FALSE, XV_X, xv_col(
	//					mchoose_panel, 0), XV_Y, xv_rows(mchoose_panel, 1), NULL);
	choices = gtk_list_store_new(M_COLUMNS_NUMBER, G_TYPE_STRING);
	for (i = 0; i < 5; i++) {
		gtk_list_store_append(GTK_LIST_STORE(choices), &iter);
		gtk_list_store_set(choices, &iter, M_AVAILABLE_CHOICES_COLUMN_ID, "",
				-1);
	}

	choices_list = gtk_tree_view_new_with_model(GTK_TREE_MODEL(choices));
	GtkCellRenderer * renderer = gtk_cell_renderer_text_new();
	GtkTreeViewColumn * column;
	column = gtk_tree_view_column_new_with_attributes("Available choices",
			renderer, "text", M_AVAILABLE_CHOICES_COLUMN_ID, NULL);
	gtk_tree_view_append_column(GTK_TREE_VIEW(choices_list), column);

	_select = gtk_tree_view_get_selection(GTK_TREE_VIEW(choices_list));
	gtk_tree_selection_set_mode(GTK_TREE_SELECTION(_select),
			GTK_SELECTION_MULTIPLE);
	g_signal_connect(_select, "changed", G_CALLBACK(mchoose_callback), NULL);

	gtk_container_add(GTK_CONTAINER(scrolled_window), choices_list);
	gtk_container_add(GTK_CONTAINER(mchoose_frame), scrolled_window);
	gtk_box_pack_start(GTK_BOX(vbox), mchoose_frame, TRUE, TRUE, 5);
	//	gtk_box_pack_start(GTK_BOX(window_vbox), choices_list, FALSE, FALSE, 0);
	gtk_widget_show(choices_list);

	//	ok = xv_create(mchoose_panel, PANEL_BUTTON, PANEL_LABEL_STRING, "OK",
	//			PANEL_NOTIFY_PROC, mchoose_ok_notify, XV_X,
	//			xv_col(mchoose_panel, 5), XV_Y, xv_rows(mchoose_panel, 5), NULL);
	ok_button = gtk_button_new_with_label("OK");
	gtk_signal_connect(GTK_OBJECT(ok_button), "clicked", GTK_SIGNAL_FUNC(
			mchoose_ok_notify), NULL);
	gtk_box_pack_start(GTK_BOX(buttons_hbox), ok_button, FALSE, FALSE, 5);

	//	(void) xv_create(mchoose_panel, PANEL_BUTTON, PANEL_LABEL_STRING,
	//			"(De)Select all", PANEL_NOTIFY_PROC, mchoose_de_select_all_notify,
	//			NULL);
	deselect_button = gtk_button_new_with_label("(De)Select all");
	gtk_signal_connect(GTK_OBJECT(deselect_button), "clicked", GTK_SIGNAL_FUNC(
			mchoose_de_select_all_notify), NULL);
	gtk_box_pack_start(GTK_BOX(buttons_hbox), deselect_button, FALSE, FALSE, 5);

	//	cancel = xv_create(mchoose_panel, PANEL_BUTTON, PANEL_LABEL_STRING,
	//			"Cancel", PANEL_NOTIFY_PROC, mchoose_cancel_notify, NULL);
	cancel_button = gtk_button_new_with_label("Cancel");
	gtk_signal_connect(GTK_OBJECT(cancel_button), "clicked", GTK_SIGNAL_FUNC(
			mchoose_cancel_notify), NULL);
	gtk_box_pack_start(GTK_BOX(buttons_hbox), cancel_button, FALSE, FALSE, 5);

	//	help = xv_create(mchoose_panel, PANEL_BUTTON, PANEL_LABEL_STRING, "Help",
	//			PANEL_NOTIFY_PROC, mchoose_help_notify, NULL);
	help_button = gtk_button_new_with_label("Help");
	gtk_signal_connect(GTK_OBJECT(help_button), "clicked", GTK_SIGNAL_FUNC(
			mchoose_ok_notify), NULL);
	gtk_box_pack_start(GTK_BOX(buttons_hbox), help_button, FALSE, FALSE, 5);

	gtk_box_pack_start(GTK_BOX(vbox), buttons_hbox, FALSE, FALSE, 5);
	//	xv_set(mchoose_frame, FRAME_DONE_PROC, mchoose_frame_done_proc, NULL);
	gtk_window_set_default(GTK_WINDOW(mchoose_window), ok_button);

	gtk_signal_connect(GTK_OBJECT(mchoose_window), "delete-event",
			GTK_SIGNAL_FUNC(mchoose_window_done_proc), NULL);

	gtk_widget_show_all(vbox);
}
