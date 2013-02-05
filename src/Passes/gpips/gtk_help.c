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

#include <sys/time.h>
#include <sys/resource.h>

#include "genC.h"
#include "top-level.h"
#include "misc.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

#define min(a,b) ((a) < (b) ? (a) : (b))
#define HELP_LINES 32

/* The URL of the PIPS documentation at the �cole des Mines de Paris: */
#define PIPS_DOCUMENTATION_URL "http://www.cri.ensmp.fr/pips"
static GtkWidget *lines[HELP_LINES]; /* GRRRRRRRR FC. */
static gen_array_t help_list = (gen_array_t) NULL;

void display_help(char * topic) {
	int i, n;

	if(help_list != NULL)
		gen_array_free(help_list);
	//	if (!help_list) /* lazy */
	help_list = gen_array_make(0);

	get_help_topic(topic, help_list);
	n = gen_array_nitems(help_list);

	for (i = 0; i < min(HELP_LINES, n); i++) {
		gtk_label_set_text(GTK_LABEL(lines[i]), gen_array_item(help_list, i));
		//		xv_set(lines[i], PANEL_LABEL_STRING, gen_array_item(help_list, i), 0);
	}

	for (i = min(HELP_LINES, n); i < HELP_LINES; i++) {
		gtk_label_set_text(GTK_LABEL(lines[i]), "");
	}

	gtk_widget_show_all(help_window);
}

static void close_help_notify(GtkWidget *widget, gpointer *data) {
	gen_array_full_free(help_list), help_list = (gen_array_t) NULL;
	hide_window(help_window, NULL, NULL);
}

void create_help_window() {
	guint i;
	GtkWidget *vbox;
	GtkWidget *close_button;
	//	help_panel = xv_create(help_frame, PANEL, NULL);
	help_frame = gtk_frame_new(NULL);
	gtk_container_add(GTK_CONTAINER(help_window), help_frame);

	// Literal translation from XV
	// need to put in in a more gtk-like form
	vbox = gtk_vbox_new(FALSE, 0);
	for (i = 0; i < HELP_LINES; i++) {
		//		lines[i] = xv_create(help_panel, PANEL_MESSAGE, XV_X, 15, XV_Y, 15*
		//				(i +1),
		//				0);
		//			}
		lines[i] = gtk_label_new(NULL);
		gtk_box_pack_start(GTK_BOX(vbox), lines[i], FALSE, FALSE, 0);
		gtk_widget_show(lines[i]);
	}
	//	xv_create(help_panel, PANEL_BUTTON, PANEL_LABEL_STRING, "CLOSE", XV_X,
	//			HELP_WIDTH / 2 - 28, XV_Y, 15* (HELP_LINES +1),
	//			PANEL_NOTIFY_PROC, close_help_notify,
	//			0);
	close_button = gtk_button_new_with_label("CLOSE");
	gtk_box_pack_start(GTK_BOX(vbox), close_button, FALSE, FALSE, 15);
	gtk_widget_show(close_button);

	gtk_container_add(GTK_CONTAINER(help_frame), vbox);

	gtk_signal_connect(GTK_OBJECT(close_button), "clicked", GTK_SIGNAL_FUNC(
			close_help_notify), NULL);

	gtk_widget_show_all(help_frame);
}

static void help_notify(GtkWidget * widget, gpointer data) {
	//	display_help((char *) xv_get(menu_item, MENU_CLIENT_DATA));
	display_help((char *) data);
}

static void help_launch_pips_firefox(GtkWidget * widget, gpointer data) {
	system("firefox " PIPS_DOCUMENTATION_URL " &");
}
static char * menu_data[] = { "A few introductory words...", "Introduction",
		"A few words about \"Workspace\"...", "Workspace",
		"A few words about \"Module\"...", "Module",
		"A few words about \"Directory\"...", "Directory",
		"A few words about \"View\"...", "View",
		"A few words about \"Transform/Edit\"...", "Transform/Edit",
		"A few words about \"Compile\"...", "Compile",
		"A few words about \"Options\"...", "Options",
		"A few words about \"Log\"...", "Log" };

void create_help_menu() {
	guint i;
	GtkWidget *help_menu, *help_menu_item;
	help_menu = gtk_menu_new();
	help_menu_item = gtk_menu_item_new_with_label("Help");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(help_menu_item), help_menu);

	GtkWidget * menu_item;
	for (i = 0; i < 9; i++) {
		menu_item = gtk_menu_item_new_with_label(menu_data[i * 2]);
		g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(
				help_notify), menu_data[i * 2 + 1]);
		gtk_menu_append(GTK_MENU(help_menu), menu_item);
	}

	gtk_menu_append(GTK_MENU(help_menu), gtk_separator_menu_item_new());

	menu_item = gtk_menu_item_new_with_label(
			"The PIPS documentation on Internet with Firefox...");
	g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(
			help_launch_pips_firefox), NULL);
	gtk_menu_append(GTK_MENU(help_menu), menu_item);

	gtk_widget_show(help_menu_item);
	gtk_widget_show_all(help_menu);
	gtk_menu_bar_append(main_window_menu_bar, help_menu_item);
}
