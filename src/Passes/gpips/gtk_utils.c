#include <stdio.h>
#include <sys/types.h>
#include <sys/param.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <gtk/gtk.h>

#include "genC.h"
#include "misc.h"

#include "gpips.h"

extern char *getwd();

gchar * gpips_gtk_menu_item_get_label(GtkWidget * w) {
	guint i;
	gchar * label;
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
