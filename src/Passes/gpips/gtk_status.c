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
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

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
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "gpips.h"

enum {
	DECALAGE_STATUS = 100
};

/* Max number of digits displayed in the status panel: */
enum {
	CPU_USAGE_LENGTH = 8
};

typedef struct {
	GtkWidget * label;
	GtkWidget * entry;
	GtkWidget * button;
} LabelEntryAndButton;

GtkWidget *directory_name_entry, *directory_name_entry_button;
GtkWidget *workspace_name_entry, *memory_name, *gmessage, *window_number;
GtkWidget *module_name_entry;
GtkWidget *cpu_usage_item;

//Server_image status_window_pips_image;

/* Strange, "man end" says that end is a function! */
extern char etext, edata, end;

void display_memory_usage() {
	char memory_string[17];
	char cpu_string[CPU_USAGE_LENGTH + 1];
	struct rusage an_rusage;

	/* etext, edata and end are only address symbols... */
	debug(6, "display_memory_usage",
			"_text %#x, _data %#x, _end %#x, _brk %#x\n", &etext, &edata, &end,
			sbrk(0));

	sprintf(memory_string, "%10.3f", ((intptr_t) sbrk(0) - (intptr_t) &etext)
			/ (double) (1 << 20));

	gtk_label_set_text(GTK_LABEL(memory_name), memory_string);

	if (getrusage(RUSAGE_SELF, &an_rusage) == 0) {
		double the_cpu_time = an_rusage.ru_utime.tv_sec
				+ an_rusage.ru_stime.tv_sec + (an_rusage.ru_utime.tv_usec
				+ an_rusage.ru_stime.tv_usec) * 1e-6;
		sprintf(cpu_string, "%*.*g", CPU_USAGE_LENGTH, CPU_USAGE_LENGTH - 2,
				the_cpu_time);
	} else
		/* getrusage() failed: */
		sprintf(cpu_string, "%*s", CPU_USAGE_LENGTH, "?");

	gtk_label_set_text(GTK_LABEL(cpu_usage_item), cpu_string);
}

//void window_number_notify(Panel_item item, int value, Event *event) {
//	number_of_gpips_windows = (int) xv_get(item, PANEL_VALUE);
//
//	display_memory_usage();
//}

void show_directory() {
	gtk_entry_set_text(GTK_ENTRY(directory_name_entry), get_cwd());
	display_memory_usage();
}

void show_workspace() {
	static char *none = "(* none *)";
	char *name = db_get_current_workspace_name();

	if (name == NULL)
		name = none;
	//
	//	xv_set(workspace_name_entry, PANEL_VALUE, name, 0);
	display_memory_usage();
}

void show_module() {
	static char *none = "(* none *)";
	char *module_name_content = db_get_current_module_name();

	if (module_name_content == NULL)
		module_name_content = none;
	//
	//	xv_set(module_name_entry, PANEL_VALUE, module_name_content, 0);
	display_memory_usage();
}

void gpips_interrupt_button_blink() {
	//	if ((Server_image) xv_get(status_window_pips_image, PANEL_LABEL_IMAGE)
	//			== gpips_negative_server_image)
	//		xv_set(status_window_pips_image, PANEL_LABEL_IMAGE,
	//				gpips_positive_server_image, NULL);
	//	else
	//		xv_set(status_window_pips_image, PANEL_LABEL_IMAGE,
	//				gpips_negative_server_image, NULL);

}

void gpips_interrupt_button_restore() {
	//	xv_set(status_window_pips_image, PANEL_LABEL_IMAGE,
	//			gpips_positive_server_image, NULL);
}

void show_message(string message_buffer /*, ...*/) {
	/* va_list some_arguments;
	 static char message_buffer[SMALL_BUFFER_LENGTH]; */

	/* va_start(some_arguments, a_printf_format); */

	/* (void) vsprintf(message_buffer, a_printf_format, some_arguments);*/

	//	xv_set(message, PANEL_VALUE, message_buffer, 0);
	display_memory_usage();
}

static void end_directory_notify_callback(GtkWidget * w, gpointer data __attribute__((unused))) {
	if (GTK_IS_ENTRY(w))
		end_directory_notify(gtk_entry_get_text(GTK_ENTRY(w)));
}

static void choose_dir_callback(GtkWidget * w __attribute__((unused)), gpointer data) {
	GtkWidget * file_chooser_dialog;

	file_chooser_dialog = gtk_file_chooser_dialog_new("Choose Directory",
			GTK_WINDOW(main_window), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
			GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL, GTK_STOCK_OPEN,
			GTK_RESPONSE_ACCEPT, NULL);

	if (gtk_dialog_run(GTK_DIALOG(file_chooser_dialog)) == GTK_RESPONSE_ACCEPT) {
		char * dirname = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(
				file_chooser_dialog));
		gtk_entry_set_text(GTK_ENTRY(directory_name_entry), dirname);
		g_free(dirname);
	}
	g_signal_emit_by_name(data, "activate");
	//end_directory_notify_callback((GtkWidget *) data, NULL);

	gtk_widget_destroy(file_chooser_dialog);
}

void open_or_create_workspace_callback(GtkWidget * w, gpointer data __attribute__((unused))) {
	//gtk_entry_set_text(GTK_ENTRY(workspace_name_entry), gtk_entry_get_text(query_entry)); TODO: FIX ICI pour transferer le workspace name du query_entry au workspace_enamer_entry
	open_or_create_workspace(gtk_entry_get_text(GTK_ENTRY(w)));
	return;
}
// crée dans la fenetre principale (dans la vbox en fait) un bloc
// Label : Entry [bouton] pour choisir un répertoire
// et quand un répertoire est choisi on applique "end_directory_notify" dessus
static GtkWidget * create_dir_choose_entry(GtkWidget * vbox) {
	GtkWidget * hbox = gtk_hbox_new(FALSE, 0);

	GtkWidget * label;
	GtkWidget * entry;
	GtkWidget * cd_button;

	label = gtk_label_new("Directory: ");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 5);

	entry = gtk_entry_new();
	gtk_box_pack_start(GTK_BOX(hbox), entry, TRUE, TRUE, 5);
	g_signal_connect(GTK_OBJECT(entry), "activate",
			G_CALLBACK(end_directory_notify_callback), NULL);
	gtk_widget_set_sensitive(entry, FALSE);

	cd_button = gtk_button_new_with_label("CD");
	gtk_box_pack_start(GTK_BOX(hbox), cd_button, FALSE, FALSE, 5);
	g_signal_connect(GTK_OBJECT(cd_button), "clicked", G_CALLBACK(choose_dir_callback),
			entry);

	gtk_widget_show_all(hbox);
	gtk_box_pack_start(GTK_BOX(vbox), hbox, TRUE, FALSE, 5);
	return entry;
}

static gboolean regenerate_workspace_menu_callback(GtkWidget * w,
		GdkEventButton * ev, gpointer data) {
	//GtkWidget * new_menu;
	GtkWidget * menu = (GtkWidget *) data;

	if (menu != NULL)
		gtk_menu_item_remove_submenu(GTK_MENU_ITEM(w));

	menu = generate_workspace_menu();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(w), menu);

	return FALSE;
}

static gboolean regenerate_module_menu_callback(GtkWidget * w,
		GdkEventButton * ev, gpointer data) {
	//GtkWidget * new_menu;
	GtkWidget * menu = (GtkWidget *) data;

	if (menu != NULL)
		gtk_menu_item_remove_submenu(GTK_MENU_ITEM(w));

	menu = generate_module_menu();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(w), menu);

	return FALSE;
}

// un bloc label / entry / menu, dans le menu la liste des actions possibles
// sur les workspace + la liste des workspace selon un algo a explorer
// entry "intelligente" (création ou ouverture de workspace)
static GtkWidget * create_workspace_entry(GtkWidget * vbox) {
	GtkWidget * hbox = gtk_hbox_new(FALSE, 0);

	GtkWidget * label;
	GtkWidget * entry;
	GtkWidget * menu;
	GtkWidget * menu_item;
	GtkWidget * menu_bar;

	label = gtk_label_new("Workspace: ");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 5);

	entry = gtk_entry_new();
	gtk_box_pack_start(GTK_BOX(hbox), entry, TRUE, TRUE, 5);
	g_signal_connect(GTK_OBJECT(entry), "activate",
			G_CALLBACK(open_or_create_workspace_callback), NULL);

	menu_item = gtk_menu_item_new_with_label("Workspace list");
	// regenerating the menu every time we try to access it
	menu = NULL; // (necessary initialization for the callback)
	g_signal_connect(G_OBJECT(menu_item), "button-press-event",
			G_CALLBACK(regenerate_workspace_menu_callback), menu);
	gtk_widget_show(menu_item);

	menu_bar = gtk_menu_bar_new();
	gtk_box_pack_start(GTK_BOX(hbox), menu_bar, FALSE, FALSE, 0);
	gtk_menu_bar_append(GTK_MENU_BAR(menu_bar), menu_item);
	gtk_widget_show(menu_bar);

	gtk_widget_show_all(hbox);
	gtk_box_pack_start(GTK_BOX(vbox), hbox, TRUE, FALSE, 5);
	return entry;
}

static GtkWidget * create_module_entry(GtkWidget * vbox) {
	GtkWidget * hbox = gtk_hbox_new(FALSE, 0);

	GtkWidget * label;
	GtkWidget * entry;
	GtkWidget * menu;
	GtkWidget * menu_item;
	GtkWidget * menu_bar;

	label = gtk_label_new("Module: ");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 5);

	entry = gtk_entry_new();
	gtk_box_pack_start(GTK_BOX(hbox), entry, TRUE, TRUE, 5);
	g_signal_connect(GTK_OBJECT(entry), "activate",
			G_CALLBACK(end_select_module_callback), NULL);

	menu_item = gtk_menu_item_new_with_label("Select module");
	// regenerating the menu every time we try to access it
	menu = NULL; // (necessary initialization for the callback)
	g_signal_connect(G_OBJECT(menu_item), "button-press-event",
			G_CALLBACK(regenerate_module_menu_callback), menu);
	gtk_widget_show(menu_item);

	menu_bar = gtk_menu_bar_new();
	gtk_box_pack_start(GTK_BOX(hbox), menu_bar, FALSE, FALSE, 0);
	gtk_menu_bar_append(GTK_MENU_BAR(menu_bar), menu_item);
	gtk_widget_show(menu_bar);

	gtk_widget_show_all(hbox);
	gtk_box_pack_start(GTK_BOX(vbox), hbox, TRUE, FALSE, 5);
	return entry;
}

void create_status_subwindow() {
	//Server_image pips_icon_server_image;
	status_frame = gtk_frame_new(NULL);
	gtk_box_pack_start(GTK_BOX(main_window_vbox), status_frame, FALSE, FALSE, 5);

	GtkWidget * status_frame_vbox = gtk_vbox_new(FALSE, 0);
	gtk_container_add(GTK_CONTAINER(status_frame), status_frame_vbox);

	gmessage = gtk_label_new("Message:");
	gtk_box_pack_start(GTK_BOX(status_frame_vbox), gmessage, FALSE, FALSE, 5);

	directory_name_entry = create_dir_choose_entry(status_frame_vbox);

	workspace_name_entry = create_workspace_entry(status_frame_vbox);

	module_name_entry = create_module_entry(status_frame_vbox);

	GtkWidget * data_hbox = gtk_hbox_new(FALSE, 0);
	gtk_box_pack_start(GTK_BOX(status_frame_vbox), data_hbox, FALSE, FALSE, 5);
	gtk_box_pack_start(GTK_BOX(data_hbox), gtk_label_new("Memory (MB):"),
			FALSE, FALSE, 5);
	memory_name = gtk_label_new("");
	gtk_box_pack_start(GTK_BOX(data_hbox), memory_name, FALSE, FALSE, 5);
	gtk_box_pack_start(GTK_BOX(data_hbox), gtk_label_new("CPU (s):"), FALSE,
			FALSE, 5);
	cpu_usage_item = gtk_label_new("");
	gtk_box_pack_start(GTK_BOX(data_hbox), cpu_usage_item, FALSE, FALSE, 5);
	gtk_widget_show_all(data_hbox);

	//	window_number = xv_create(main_panel, PANEL_NUMERIC_TEXT,
	//	/*PANEL_ITEM_X_GAP, DECALAGE_STATUS,
	//	 PANEL_VALUE_Y, xv_rows(main_panel, 4),*/
	//	PANEL_VALUE_X, xv_col(main_panel, 50), PANEL_VALUE_Y,
	//			xv_rows(main_panel, 4), PANEL_LABEL_STRING, "# windows:",
	//			PANEL_MIN_VALUE, 1, PANEL_MAX_VALUE, MAX_NUMBER_OF_GPIPS_WINDOWS,
	//			PANEL_VALUE, number_of_gpips_windows, PANEL_VALUE_DISPLAY_LENGTH,
	//			2, PANEL_NOTIFY_PROC, window_number_notify, NULL);

	//	pips_icon_server_image = create_status_window_pips_image();

	//	status_window_pips_image = xv_create(main_panel, PANEL_BUTTON,
	//			PANEL_LABEL_IMAGE, pips_icon_server_image, PANEL_NOTIFY_PROC,
	//			gpips_interrupt_pipsmake,
	//			/* Put the Pixmap above the Help button: */
	//			// quit_button a été remplacé par quit_menu_item
	//			XV_X, xv_get(quit_button, XV_X) + (xv_get(quit_button, XV_WIDTH)
	//					- xv_get(pips_icon_server_image, XV_WIDTH)) / 2, XV_Y,
	//			xv_rows(main_panel, 3) + 20, NULL);

	show_directory();
	show_workspace();
	show_module();
	display_memory_usage();

	gtk_widget_show_all(status_frame);
}
