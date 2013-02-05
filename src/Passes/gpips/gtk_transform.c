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

#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>

#if (defined(TEXT))
#undef TEXT
#endif

#if (defined(TEXT_TYPE))
#undef TEXT_TYPE
#endif

#include "genC.h"

#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "top-level.h"
#include "database.h"
#include "pipsmake.h"
#include "pipsdbm.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

/* Include the label names: */
#include "gpips-labels.h"

#include "resources.h"
#include "phases.h"

gpips_transform_menu_layout_line gpips_transform_menu_layout[] = {
#include "wpips_transform_menu_layout.h"
		/* No more transformations */
		{ NULL, NULL } };

/* To pass arguments to execute_safe_apply_outside_the_notifyer(): */
static string
		execute_safe_apply_outside_the_notifyer_transformation_name_to_apply =
				NULL;
static string execute_safe_apply_outside_the_notifyer_module_name = NULL;

/* The transform menu: */
GtkWidget *transform_menu, *transform_menu_item;

static void apply_on_each_transform_menu_item(GtkWidget * widget,
		gpointer _func) {
	void (*func)(GtkWidget *);
	func = (void(*)(GtkWidget *)) _func;
	func(widget);
}

void apply_on_each_transform_item(void(* function_to_apply_on_each_menu_item)(
		GtkWidget *)) {
	gtk_container_foreach(GTK_CONTAINER(transform_menu), (GtkCallback)
			apply_on_each_transform_menu_item,
			function_to_apply_on_each_menu_item);
}

void disable_transform_selection() {
	apply_on_each_transform_item(disable_item);
}

void enable_transform_selection() {
	apply_on_each_transform_item(enable_item);
}

void execute_safe_apply_outside_the_notifier() {
	(void) safe_apply(
			execute_safe_apply_outside_the_notifyer_transformation_name_to_apply,
			execute_safe_apply_outside_the_notifyer_module_name);
	free(execute_safe_apply_outside_the_notifyer_transformation_name_to_apply);
	free(execute_safe_apply_outside_the_notifyer_module_name);

	display_memory_usage();
}

void safe_apply_outside_the_notifyer(string transformation_name_to_apply,
		string module_name) {
	execute_safe_apply_outside_the_notifyer_transformation_name_to_apply
			= strdup(transformation_name_to_apply);
	execute_safe_apply_outside_the_notifyer_module_name = strdup(module_name);
	/* Ask to execute the execute_safe_apply_outside_the_notifyer(): */
	execute_main_loop_command(GPIPS_SAFE_APPLY);
	/* I guess the function above does not return... */
}

static void transform_notify(GtkWidget * menu_item, gpointer data __attribute__((unused))) {
	const char * label = gpips_gtk_menu_item_get_label(menu_item);

	char * modulename = db_get_current_module_name();

	/* FI: borrowed from edit_notify() */
	if (modulename == NULL) {
		prompt_user("No module selected");
	} else {
		gpips_transform_menu_layout_line * current_transformation;

		/* Find the transformation to apply: */
		for (current_transformation = &gpips_transform_menu_layout[0]; current_transformation->menu_entry_string
				!= NULL; current_transformation++)
			if (strcmp(label, current_transformation->menu_entry_string) == 0)
				break;

		if (current_transformation->menu_entry_string != NULL)
			/* Apply the transformation: */
			safe_apply_outside_the_notifyer(
					current_transformation->transformation_name_to_apply,
					modulename);
		/* I guess the function above does not return... */
		else
			pips_error("transform_notify",
					"What is this \"%s\" entry you ask for?", label);
	}

	display_memory_usage();
}

void create_transform_menu() {
	GtkWidget * menu_item;

	gpips_transform_menu_layout_line * current_transformation;

	edit_menu_item = gtk_menu_item_new_with_label(EDIT_VIEW);
	g_signal_connect(G_OBJECT(edit_menu_item), "activate", G_CALLBACK(
			edit_notify), NULL);

	transform_menu = gtk_menu_new();
	transform_menu_item = gtk_menu_item_new_with_label("Transform/Edit");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(transform_menu_item), transform_menu);
	gtk_menu_bar_append(GTK_MENU_BAR(main_window_menu_bar), transform_menu_item);

	/* Now add all the transformation entries: */
	for (current_transformation = &gpips_transform_menu_layout[0]; current_transformation->menu_entry_string
			!= NULL; current_transformation++) {
		if (strcmp(current_transformation->menu_entry_string,
				GPIPS_MENU_SEPARATOR_ID) == 0) {
			gtk_menu_append(GTK_MENU(transform_menu),
					gtk_separator_menu_item_new());
		} else {
			menu_item = gtk_menu_item_new_with_label(
					current_transformation->menu_entry_string);
			g_signal_connect(G_OBJECT(menu_item), "activate",
					G_CALLBACK(transform_notify), NULL);
			gtk_menu_append(GTK_MENU(transform_menu), menu_item);
		}
	}

	/* Add the Edit entry as the last one: */
	gtk_menu_append(GTK_MENU(transform_menu), gtk_separator_menu_item_new());

	gtk_menu_append(GTK_MENU(transform_menu), edit_menu_item);

	gtk_widget_show_all(transform_menu);
	gtk_widget_show(transform_menu_item);
}
