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
/* Single choice handling */

/* Difference with previous release:
 1/ schoose_close() must be called in order to close the schoose window.
 2/ schoose() has one more argument, because cancel button is created.
 bb 04.06.91
 */
/*
 * forked to gtk_schoose2.c
 * Edited by Johan GALL
 *
 */

#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "genC.h"
#include "misc.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

static GtkWidget *choice_label, *choices_list, *ok_button, *help,
		*cancel_button;
static GtkListStore *choices;

enum {
	SC2_AVAILABLE_CHOICES_COLUMN_ID, SC2_COLUMNS_NUMBER
};

static void (* apply_on_choice)(const char *);
static void (* apply_on_cancel)(void);

#if 0
static void schoose_help_notify(GtkWidget * widget, gpointer data) {
	display_help("SingleChoice");
}
#endif

/* called when the "ok button" is clicked to validate an entry or when you select
 * something with the menu associated to the beforementionned entry. */
static void schoose_ok_notify(GtkWidget * widget, gpointer data) {
	GtkTreeIter iter;
	GtkTreeModel ** model = NULL;
	gchar * gc_choice;

	GtkTreeSelection * selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(
			choices_list));

	if (!gtk_tree_selection_get_selected(selection, model, &iter)) {
		prompt_user("Choose one item or cancel");
		return;
	}

	//	gtk_tree_model_get(GTK_TREE_MODEL(*model), &iter,
	//			SC2_AVAILABLE_CHOICES_COLUMN_ID, &gc_choice, -1);

	gc_choice = strdup(gtk_label_get_text(GTK_LABEL(choice_label)));
	// ----

	//	curchoice = strdup((char *) xv_get(choice, PANEL_VALUE, 0));
	//
	//	/* Modified to verify that an correct item is selected.
	//	 RK, 21/05/1993. */
	//	nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);
	//	item_is_in_the_list = FALSE;
	//	for (i = 0; i < nchoices; i++)
	//		if (strcmp((char *) xv_get(choices, PANEL_LIST_STRING, i), curchoice)
	//				== 0) {
	//			item_is_in_the_list = TRUE;
	//			break;
	//		}
	//	if (item_is_in_the_list == FALSE)
	//		prompt_user("You have to choose one item of the list!");
	//	else {
	//		/* Normal case : */
	(*apply_on_choice)(gc_choice);
	g_free(gc_choice);
	schoose_close();
}

/* schoose_close() can be called even when schoose window is already closed.
 */
void schoose_close() {
	hide_window(schoose_window, NULL, NULL);
	gtk_list_store_clear(choices);
	gtk_label_set_text(GTK_LABEL(choice_label), "");
}

void schoose_cancel_notify(GtkWidget * widget, gpointer data) {
	schoose_close();
	(*apply_on_cancel)();
}

/* Function used to update the text panel according to the list panel: */
static void schoose_choice_callback(GtkTreeSelection * selection, gpointer data) {
	GtkTreeIter iter;
	GtkTreeModel * model;
	gchar * gc_choice;

	//	if (!gtk_tree_selection_get_selected(selection, &model, &iter)) {
	//		pips_assert("schoose_choice_notify: no item selected !", 0);
	//		return;
	//	}

	if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
		gtk_tree_model_get(GTK_TREE_MODEL(model), &iter,
				SC2_AVAILABLE_CHOICES_COLUMN_ID, &gc_choice, -1);
		gtk_label_set_text(GTK_LABEL(choice_label), gc_choice);
		g_free(gc_choice);
	}
}

/* Avoid the schoose_frame destruction and act as cancel: */
static void schoose_window_done_callback(GtkWidget * window, GdkEvent * ev,
		gpointer data) {
	(*apply_on_cancel)();
	hide_window(window, NULL, NULL);
}

void schoose(char * title, gen_array_t array, char * initial_choice,
		void(*function_for_ok)(const char *), void(*function_for_cancel)(void)) {
	guint i;
	string name;
	GtkTreeIter iter;
	GtkTreeSelection * selection;
	bool do_select = FALSE;
	int argc = gen_array_nitems(array);

	apply_on_choice = function_for_ok;
	apply_on_cancel = function_for_cancel;

	gtk_window_set_title(GTK_WINDOW(schoose_window), title);
	gtk_list_store_clear(choices);

	if (initial_choice != NULL) {
		do_select = TRUE;
		selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(choices_list));
	}

	for (i = 0; i < argc; i++) {
		name = gen_array_item(array, i);
		gtk_list_store_append(GTK_LIST_STORE(choices), &iter);
		gtk_list_store_set(choices, &iter, SC2_AVAILABLE_CHOICES_COLUMN_ID,
				name, -1);
		if (do_select) // Initialise au choix initial ou default le premier
			if (i == 0 || strcmp(initial_choice, name) == 0)
				gtk_tree_selection_select_iter(GTK_TREE_SELECTION(selection),
						&iter);
	}
	gtk_widget_show(schoose_window);

	/* move the pointer to the center of the query window */
	// Sans vouloir remettre en question le design de la chose
	// A priori je ne traduirai pas ça à moins qu'on ne le
	// demande expressément...
	//pointer_in_center_of_frame(schoose_frame);
}

void create_schoose_window() {
	guint i;
	GtkWidget *window_vbox, *buttons_hbox, *current_choice_hbox;
	GtkTreeIter iter;
	GtkTreeSelection * selection;

	window_vbox = gtk_vbox_new(FALSE, 0);
	buttons_hbox = gtk_hbox_new(FALSE, 0);
	current_choice_hbox = gtk_hbox_new(FALSE, 0);
	gtk_box_pack_start(GTK_BOX(window_vbox), current_choice_hbox, FALSE, FALSE,
			0);
	gtk_container_add(GTK_CONTAINER(schoose_window), window_vbox);

	gtk_box_pack_start(GTK_BOX(current_choice_hbox), gtk_label_new(
			"Current choice : "), FALSE, FALSE, 0);
	choice_label = gtk_label_new("");
	gtk_box_pack_start(GTK_BOX(current_choice_hbox), choice_label, FALSE,
			FALSE, 0);

	choices = gtk_list_store_new(SC2_COLUMNS_NUMBER, G_TYPE_STRING);
	for (i = 0; i < 5; i++) {
		gtk_list_store_append(GTK_LIST_STORE(choices), &iter);
		gtk_list_store_set(GTK_LIST_STORE(choices), &iter,
				SC2_AVAILABLE_CHOICES_COLUMN_ID, "", -1);
		//gtk_tree_store_set(GTK_TREE_STORE(choices), &iter, SC2_AVAILABLE_CHOICES, "");
	}

	// the "graphical" choice component is the GtkTreeView * choices_list
	choices_list = gtk_tree_view_new_with_model(GTK_TREE_MODEL(choices));
	GtkCellRenderer * renderer = gtk_cell_renderer_text_new();
	GtkTreeViewColumn * column;
	column = gtk_tree_view_column_new_with_attributes("Available choices",
			renderer, "text", SC2_AVAILABLE_CHOICES_COLUMN_ID, NULL);
	gtk_tree_view_append_column(GTK_TREE_VIEW(choices_list), column);

	selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(choices_list));
	gtk_tree_selection_set_mode(GTK_TREE_SELECTION(selection),
			GTK_SELECTION_SINGLE);
	g_signal_connect(selection, "changed", G_CALLBACK(schoose_choice_callback),
			NULL);

	gtk_box_pack_start(GTK_BOX(window_vbox), choices_list, TRUE, FALSE, 0);
	gtk_widget_show(choices_list);

	ok_button = gtk_button_new_with_label("OK");
	g_signal_connect(GTK_OBJECT(ok_button), "clicked", G_CALLBACK(
			schoose_ok_notify), NULL);
	gtk_box_pack_start(GTK_BOX(buttons_hbox), ok_button, FALSE, FALSE, 5);

	cancel_button = gtk_button_new_with_label("Cancel");
	g_signal_connect(GTK_OBJECT(cancel_button), "clicked", G_CALLBACK(
			schoose_cancel_notify), NULL);
	gtk_box_pack_start(GTK_BOX(buttons_hbox), cancel_button, FALSE, FALSE, 5);

	help = gtk_button_new_with_label("Help");
	g_signal_connect(GTK_OBJECT(help), "clicked",
			G_CALLBACK(schoose_ok_notify), NULL);
	gtk_box_pack_start(GTK_BOX(buttons_hbox), help, FALSE, FALSE, 5);

	gtk_box_pack_start(GTK_BOX(window_vbox), buttons_hbox, FALSE, FALSE, 5);

	gtk_window_set_default(GTK_WINDOW(schoose_window), ok_button);

	g_signal_connect(GTK_OBJECT(schoose_window), "delete_event", G_CALLBACK(
			schoose_window_done_callback), NULL);

	gtk_widget_show_all(window_vbox);
}
