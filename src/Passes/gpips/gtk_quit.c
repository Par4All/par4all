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
#include "database.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "top-level.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

#define QUICK_QUIT "Quit without saving"
#define CLOSE_QUIT "Close (save) the Workspace & Quit"
#define DELETE_QUIT "Delete the Workspace & Quit"
#define CD_HACK_QUIT "Change Directory (tcl/tk hack)"

GtkWidget * quit_menu_item;

void cd_notify(GtkWidget * widget, gpointer data) {
	direct_change_directory();
}

void quit_notify(GtkWidget * widget, gpointer data) {
	int result;
	string pn;

	if ((pn = db_get_current_workspace_name())) {
		string fmt = "Workspace %s not closed";
		char str[SMALL_BUFFER_LENGTH];
		string str1, str2;
        const char *menu_string;

		str2 = "Do you really want to quit PIPS?";
		menu_string = gpips_gtk_menu_item_get_label(widget);
		if (strcmp(menu_string, CLOSE_QUIT) == 0)
			str1 = " ";
		else
			str1 = "-=< Resources can get lost! >=-";

		sprintf(str, fmt, pn);

		GtkWidget * dialog = gtk_message_dialog_new(GTK_WINDOW(main_window),
				GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_INFO,
				GTK_BUTTONS_YES_NO,  "%s%s%s",str,str1,str2);
		result = gtk_dialog_run(GTK_DIALOG(dialog));

		if (result == GTK_RESPONSE_NO)
			return;
		else if (strcmp(menu_string, CLOSE_QUIT) == 0)
			close_workspace(TRUE);
		else if (strcmp(menu_string, DELETE_QUIT) == 0) {
			int win = 0;

			/* Destroy all the windows (close open files) */
			for (win = 0; win < number_of_gpips_windows; win++)
				gtk_widget_destroy(edit_window[win]);

			check_delete_workspace(pn, FALSE);
		}

	}

	/* Clear the log window to avoid the message about the edited
	 state:
	 clear_log_subwindow(NULL, NULL);
	 Does not work...
	 Quit:
	 xv_destroy[_safe](main_frame);
	 */
	/* Exit xv_main_loop() at top level: */
	gtk_main_quit();
}

void create_quit_button() {
	GtkWidget * quit_menu;

	quit_menu = gtk_menu_new();
	quit_menu_item = gtk_menu_item_new_with_label("Quit");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(quit_menu_item), quit_menu);

	GtkWidget * menu_item;

	menu_item = gtk_menu_item_new_with_label(CLOSE_QUIT);
	g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(quit_notify),
			NULL);
	gtk_menu_append(GTK_MENU(quit_menu), menu_item);

	menu_item = gtk_menu_item_new_with_label(QUICK_QUIT);
	g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(quit_notify),
			NULL);
	gtk_menu_append(GTK_MENU(quit_menu), menu_item);

	menu_item = gtk_menu_item_new_with_label(DELETE_QUIT);
	g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(quit_notify),
			NULL);
	gtk_menu_append(GTK_MENU(quit_menu), menu_item);

	menu_item = gtk_menu_item_new_with_label(CD_HACK_QUIT);
	g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(cd_notify),
			NULL);
	gtk_menu_append(GTK_MENU(quit_menu), menu_item);

	gtk_widget_show(quit_menu_item);
	gtk_widget_show_all(quit_menu);
	gtk_menu_bar_append(GTK_MENU_BAR(main_window_menu_bar), quit_menu_item);
}
