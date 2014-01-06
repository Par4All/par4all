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
#include <sys/types.h>
#include <sys/param.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "genC.h"
#include "misc.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

extern char *getwd();

const char * gpips_gtk_menu_item_get_label(GtkWidget * w) {
	guint i;
	const char * label;
	GtkWidget * child;
	if (!GTK_IS_MENU_ITEM(w))
		return NULL;
	GList * children = gtk_container_get_children(GTK_CONTAINER(w));
	for (i = 0; i < g_list_length(children); i++) {
		child = (GtkWidget *) g_list_nth_data(children, i);
		if (!GTK_IS_LABEL(child))
			continue;
		label = gtk_label_get_text(GTK_LABEL(child));
	}
	g_list_free(children);
	return (label);
}

bool gpips_gtk_widget_get_sensitive(GtkWidget * w) {
	return GTK_WIDGET_SENSITIVE(w);
}

void gpips_gtk_menu_item_set_label(GtkWidget * w, gchar * text) {
	guint i;
	GtkWidget * child;
	if (!GTK_IS_MENU_ITEM(w))
		return;
	GList * children = gtk_container_get_children(GTK_CONTAINER(w));
	for (i = 0; i < g_list_length(children); i++) {
		child = (GtkWidget *) g_list_nth_data(children, i);
		if (!GTK_IS_LABEL(child))
			continue;
		gtk_label_set_text(GTK_LABEL(child), text);
	}
	g_list_free(children);
}

GtkWidget * gpips_gtk_dialog_get_content_area(GtkDialog *dialog) {
	g_return_val_if_fail(GTK_IS_DIALOG(dialog), NULL);
	return dialog->vbox;
}

gdouble gpips_gtk_adjustment_get_upper(GtkAdjustment *adjustment) {
	g_return_val_if_fail(GTK_IS_ADJUSTMENT(adjustment), 0.0);
	return adjustment->upper;
}

gint hide_window(GtkWidget *window, GdkEvent *ev __attribute__((unused)), gpointer data __attribute__((unused))) {
	gtk_widget_hide(GTK_WIDGET(window));
	return TRUE;
}

gint load_file(const gchar *filename, gchar ** data) {
	int size = 0;
	FILE *f = fopen(filename, "rb");
	if (f == NULL) {
		*data = NULL;
		return -1; // -1 means file opening fail
	}

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);

	*data = (char *) malloc(size + 1);

	if (size != fread(*data, sizeof(char), size, f)) {
		free(*data);
		return -2; // -2 means file reading fail
	}
	fclose(f);
	(*data)[size] = 0;
	return size;
}
