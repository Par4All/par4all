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

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "phases.h"
#include "database.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "top-level.h"
#include "preprocessor.h"

// imports gtk
#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

#define HPFC_COMPILE "Compile an HPF program"
#define HPFC_MAKE "Make an HPF program"
#define HPFC_RUN "Run an HPF program"

static GtkWidget *compile_menu, *compile_menu_item;
static GtkWidget *dynamic_hpf_output_files_menu,
		*dynamic_hpf_output_files_menu_item;

void apply_on_each_compile_item(void(* function_to_apply_on_each_menu_item)(
		GtkWidget *)) {
	guint i;

	GtkWidget * child;
	GList * compile_menu_children = gtk_container_get_children(GTK_CONTAINER(
			compile_menu));
	for (i = 0; i < g_list_length(compile_menu_children); i++) {
		child = (GtkWidget *) g_list_nth_data(compile_menu_children, i);
		if (!GTK_IS_MENU_ITEM(child))
			continue;
		if (GTK_IS_SEPARATOR(child))
			continue;
		function_to_apply_on_each_menu_item(child);
	}
	g_list_free(compile_menu_children);
}

void disable_compile_selection() {
	apply_on_each_compile_item(disable_item);
}

void enable_compile_selection() {
	apply_on_each_compile_item(enable_item);
}

void notify_hpfc_file_view(GtkWidget * widget, gpointer data) {
	const char * file_name = gpips_gtk_menu_item_get_label(widget);
	char * path_name = hpfc_generate_path_name_of_file_name(file_name);

	(void) alloc_first_initialized_window(FALSE);

	gpips_file_view(path_name, file_name, "HPFC File", "HPFC", "HPFC");
	user_log("HPFC View of \"%s\" done.\n", file_name);
}

void generate_a_menu_with_HPF_output_files(GtkWidget * widget, gpointer data) {
	guint i;
	GtkWidget * menu_item;

	gen_array_t file_names = gen_array_make(0);
	int file_number = 0;

	int return_code;
	char *hpfc_directory;

	/* Create a new menu with the content of the hpfc directory: */

	if (dynamic_hpf_output_files_menu != NULL)
		/* First, free the old menu if it exist: */
		/* We can remove all the menu it now: */
		gtk_menu_item_remove_submenu(GTK_MENU_ITEM(
				dynamic_hpf_output_files_menu_item));

	return_code = hpfc_get_file_list(file_names, &hpfc_directory);

	dynamic_hpf_output_files_menu = gtk_menu_new();

	if (return_code == -1) {
		user_warning("generate_a_menu_with_HPF_output_files",
				"Directory \"%s\" not found... \n"
					" Have you run the HPFC compiler from the Compile menu?\n",
				hpfc_directory);

		menu_item = gtk_menu_item_new_with_label(
				"*** No HPFC directory found ! ***");
		gtk_menu_shell_append(GTK_MENU_SHELL(dynamic_hpf_output_files_menu),
				menu_item);
	} else if (file_number == 0) {
		user_warning("generate_a_menu_with_HPF_output_files",
				"No file found in the directory \"%s\"... \n"
					" Have you run the HPFC compiler from the Compile menu?\n",
				hpfc_directory);

		menu_item
				= gtk_menu_item_new_with_label("*** No HPFC file found ! ***");
		gtk_menu_shell_append(GTK_MENU_SHELL(dynamic_hpf_output_files_menu),
				menu_item);
	} else {
		for (i = 0; i < file_number; i++) {
			menu_item = gtk_menu_item_new_with_label(gen_array_item(file_names,
					i));
			gtk_menu_append(GTK_MENU(dynamic_hpf_output_files_menu), menu_item);
			g_signal_connect(G_OBJECT(menu_item), "activate", G_CALLBACK(
					notify_hpfc_file_view), NULL);
		}

		gen_array_full_free(file_names);
	}
	gtk_menu_item_set_submenu(
			GTK_MENU_ITEM(dynamic_hpf_output_files_menu_item),
			dynamic_hpf_output_files_menu);
}

/* quick fix around pipsmake, FC, 23/10/95
 */
static bool gpips_hpfc_install_was_performed_hack = FALSE;

void initialize_gpips_hpfc_hack_for_fabien_and_from_fabien() {
	gpips_hpfc_install_was_performed_hack = FALSE;
}

void hpfc_notify(GtkWidget * menu_item, gpointer data) {
	const char *label;
    char *modulename;

	modulename = db_get_current_module_name();
	if (!modulename) {
		prompt_user("No module selected");
		return;
	}

	label = gpips_gtk_menu_item_get_label(menu_item);

	/* I apply the installation only once, whatever...
	 * Quick fix because the right dependences expressed for pipsmake
	 * do not seem to work. It seems that the verification of up to date
	 * resources is too clever... FC.
	 */
	if (!gpips_hpfc_install_was_performed_hack) {
		safe_apply(BUILDER_HPFC_INSTALL, modulename);
		gpips_hpfc_install_was_performed_hack = TRUE;
	}

	if (same_string_p(label, HPFC_COMPILE))
		;
	else if (same_string_p(label, HPFC_MAKE))
		safe_apply_outside_the_notifyer(BUILDER_HPFC_MAKE, modulename);
	else if (same_string_p(label, HPFC_RUN))
		safe_apply_outside_the_notifyer(BUILDER_HPFC_RUN, modulename);
	else
		pips_internal_error("Bad choice: %s", label);
}

void create_compile_menu() {
	GtkWidget *hpfc_compile_menu_item, *hpfc_make_menu_item,
			*hpfc_run_menu_item;

	compile_menu = gtk_menu_new();

	hpfc_compile_menu_item = gtk_menu_item_new_with_label(HPFC_COMPILE);
	hpfc_make_menu_item = gtk_menu_item_new_with_label(HPFC_MAKE);
	hpfc_run_menu_item = gtk_menu_item_new_with_label(HPFC_RUN);

	g_signal_connect(G_OBJECT(hpfc_compile_menu_item), "activate",
			G_CALLBACK(hpfc_notify), NULL);
	g_signal_connect(G_OBJECT(hpfc_make_menu_item), "activate",
			G_CALLBACK(hpfc_notify), NULL);
	g_signal_connect(G_OBJECT(hpfc_run_menu_item), "activate",
			G_CALLBACK(hpfc_notify), NULL);

	gtk_menu_shell_append(GTK_MENU_SHELL(compile_menu), hpfc_compile_menu_item);
	gtk_menu_shell_append(GTK_MENU_SHELL(compile_menu), hpfc_make_menu_item);
	gtk_menu_shell_append(GTK_MENU_SHELL(compile_menu), hpfc_run_menu_item);
	gtk_menu_shell_append(GTK_MENU_SHELL(compile_menu),
			gtk_separator_menu_item_new());

	dynamic_hpf_output_files_menu = gtk_menu_new();
	dynamic_hpf_output_files_menu_item = gtk_menu_item_new_with_label(
			"View the HPF Compiler Output");
	gtk_menu_item_set_submenu(
			GTK_MENU_ITEM(dynamic_hpf_output_files_menu_item),
			dynamic_hpf_output_files_menu);

	gtk_menu_shell_append(GTK_MENU_SHELL(compile_menu),
			dynamic_hpf_output_files_menu_item);
	// each time the menu is accessed, it is regenerated
	g_signal_connect(G_OBJECT(dynamic_hpf_output_files_menu_item), "activate",
			G_CALLBACK(generate_a_menu_with_HPF_output_files), NULL);

	compile_menu_item = gtk_menu_item_new_with_label("Compile");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(compile_menu_item), compile_menu);

	gtk_menu_bar_append(GTK_MENU_BAR(main_window_menu_bar), compile_menu_item);
	gtk_widget_show(compile_menu_item);
	gtk_widget_show_all(compile_menu);
}

