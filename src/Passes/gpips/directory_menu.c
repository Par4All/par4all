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
/* Generate a menu from the current directory to serve as a directory
 chooser. */

#ifndef lint
char vcid_directory_menu[] = "$Id$";
#endif /* lint */

/* To have SunOS 5.5 happy about MAXNAMELEN (in SunOS 4, it is already
 defined in sys/dirent.h): */
#include <sys/param.h>
#include <sys/stat.h>

#if __linux__ || __bsdi__ || __NetBSD__ || __OpenBSD__ || __FreeBSD__
/* Posix version: */
#define MAXNAMELEN NAME_MAX
#else
/* To have SunOS4 still working: */
#ifndef MAXNAMELEN
#define MAXNAMELEN MAXNAMLEN
#endif
#endif
#include "genC.h"
#include "misc.h"
#include "database.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"


enum {
	MENU_PATH_DATA_HANDLER = 54829,
	/* Maximum size of the directory menu of the main frame: */
	GPIPS_MAX_DIRECTORY_MENU_SIZE = 80
};

#if 0
/* Note the pedantic way to avoid the warning about unused file_name. :-) */
static bool accept_all_file_names(char * file_name __attribute__ ((unused))) {
	return TRUE;
}
#endif

static GtkWidget * directory_gen_pullright(GtkWidget * widget) {
	GtkWidget * parent_menu;

	parent_menu = gtk_widget_get_parent(widget);

	char directory[MAXNAMELEN + 1];
	const char * parent_directory;

	debug(2, "directory_gen_pullright", "widget = %#x (%s), parent = %#x\n",
			widget, gpips_gtk_menu_item_get_label(GTK_WIDGET(GTK_MENU_ITEM(widget))), parent_menu);

	/* First get the parent directory name that is the title: */
	parent_directory = gtk_menu_get_title(GTK_MENU(parent_menu));
	debug(2, "directory_gen_pullright", " parent_directory = %s\n",
			parent_directory);

	/* Build the new directory name: */
	(void) sprintf(directory, "%s/%s", parent_directory, gpips_gtk_menu_item_get_label(GTK_WIDGET(GTK_MENU_ITEM(widget))));

	if (gtk_menu_item_get_submenu(GTK_MENU_ITEM(widget)) != NULL) {
		/* Well, there is already a menu... we've been already here.
		 Free it first: */
		/* Free the associated directory name: */
		GtkWidget * submenu = gtk_menu_item_get_submenu(GTK_MENU_ITEM(widget));
		gtk_menu_item_set_submenu(GTK_MENU_ITEM(widget), NULL);
		gtk_widget_destroy(submenu);
	}

	/* Then initialize it with a new directory menu: */

	return generate_a_directory_menu(directory);
}

static void generate_a_directory_menu_notify(GtkWidget * widget, gpointer data) {
	GtkWidget * parent;
	const char * parent_path_name;
	const char * directory_name;
	char full_directory_name[MAXNAMELEN + 1];
	directory_name = gpips_gtk_menu_item_get_label(widget);
	parent = gtk_widget_get_parent(widget);
	if(parent == NULL || ! GTK_IS_MENU(parent))
		parent_path_name = "";
	else
		parent_path_name = gtk_menu_get_title(GTK_MENU(parent));

	(void) sprintf(full_directory_name, "%s/%s", parent_path_name,
			directory_name);
	(void) end_directory_notify(full_directory_name);
}

GtkWidget * generate_a_directory_menu(char * directory) {
	gen_array_t file_list;
	int file_list_length;
	int return_code;
	int i;

	GtkWidget * menu;

	menu = gtk_menu_new();
	gtk_menu_set_title(GTK_MENU(menu), directory);

//	menu = (Menu) xv_create((int) NULL, MENU,
//	/* The string is *not* copied in MENU_TITLE_ITEM: */
//	MENU_TITLE_ITEM, strdup(directory),
//	/* and furthermore MENU_TITLE_ITEM is write
//	 only, so add the info somewhere else: */
//	XV_KEY_DATA, MENU_PATH_DATA_HANDLER, strdup(directory),
//	/* Add its own notifying procedure: */
//	MENU_NOTIFY_PROC, generate_a_directory_menu_notify, NULL);

	pips_debug(2, "menu = %p (%s)\n", (void *) menu, directory);

	if (db_get_current_workspace_name()) {
		GtkWidget * please_close = gtk_menu_item_new_with_label(
				"* Close the current workspace before changing directory *");
		gtk_widget_set_sensitive(please_close, FALSE);
		gtk_menu_append(GTK_MENU(menu), please_close);
		pips_user_warning("Close the current workspace before changing "
			"directory.\n");
	} else {
		GtkWidget * file_item;
		/* Get all the files in the directory: */
		file_list = gen_array_make(0);
		return_code = safe_list_files_in_directory(file_list, directory, ".*",
				directory_exists_p);
		file_list_length = gen_array_nitems(file_list);

		if (return_code == -1 || file_list_length == 0) {
			file_item = gtk_menu_item_new_with_label(
					"* No file in this directory or cannot be open *");
			gtk_widget_set_sensitive(file_item, FALSE);
			gtk_menu_append(GTK_MENU(menu), file_item);
		} else if (file_list_length > GPIPS_MAX_DIRECTORY_MENU_SIZE) {
			file_item
					= gtk_menu_item_new_with_label(
							"* Too many files. Type directly in the Directory line of the main panel *");
			gtk_widget_set_sensitive(file_item, FALSE);
			gtk_menu_append(GTK_MENU(menu), file_item);
			user_warning("generate_a_directory_menu",
					"Too many files in the \"%s\" directory.\n", directory);
		} else
			/* Generate a corresponding entry for each file: */
			for (i = 0; i < file_list_length; i++) {
				string file_name = gen_array_item(file_list, i);
				/* Skip the "." directory: */
				if (strcmp(file_name, ".") != 0) {
					struct stat buf;
					char complete_file_name[MAXNAMELEN + 1];

					file_item = gtk_menu_item_new_with_label(file_name);
					gtk_menu_append(GTK_MENU(menu), file_item);

					(void) sprintf(complete_file_name, "%s/%s", directory,
							file_name);
					if (((stat(complete_file_name, &buf) == 0) && (buf.st_mode
							& S_IFDIR))) {
						/* Since a menu item cannot be selected as an item, add an
						 plain item with the same name. Not beautiful
						 hack... :-( */
						gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_item),
								directory_gen_pullright(file_item));
						g_signal_connect(G_OBJECT(file_item), "activate",
											G_CALLBACK(generate_a_directory_menu_notify), NULL);
						pips_debug(2, " file_item = %p (%s)\n",
								(void *) file_item, file_name);
					} else {
						gtk_widget_set_sensitive(file_item, FALSE);
					}
					/* And disable non-subdirectory entry: */
				}
			}
		gen_array_full_free(file_list);
	}
	gtk_widget_show_all(menu);
	return menu;
}
