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

#ifndef lint
char vcid_xv_props[] = "$Id$";
#endif /* lint */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "top-level.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "complexity_ri.h"

#include "constants.h"
#include "resources.h"
#include "properties.h"

#undef test_undefined // also defined in glib included from gtk
#include <gtk/gtk.h>
#include "gpips.h"

typedef struct {
	char *name;
	list choices;
} option_type;

static GtkWidget * smenu_options;
static GtkWidget * options_menu, *options_menu_item;
static GtkWidget * options_frame_menu_item;
GtkWidget * options_frame;

static hash_table aliases;
static hash_table menu_item_to_combo_box;

static char display_options_panel[] = "Options panel...";
static char hide_options_panel[] = "Hide options panel";

enum {
	BSIZE_HERE = 256
};

typedef struct {
	const char * label; // label en court de traitement ds le frame option
	int position; // position du dit label dans la vbox du frame option
	GtkWidget * vbox; // Vbox du frame options
} LabelAndVbox;

/* Flag allowing update_options also to select a option : */
int verbose_update_options = 1;

/* returns the first key which value is svp. when not found, returns NULL */
string hash_get_key_by_value(hash_table htp, string svp) {
	HASH_MAP(kp, vp, {
				if (strcmp(vp, svp) == 0)
				return kp;
			}, htp);

	return HASH_UNDEFINED_VALUE;
}

void apply_on_each_options_menu_item(GtkWidget * widget, gpointer _func) {
	void (*func)(GtkWidget *);
	func = (void(*)(GtkWidget *)) _func;
	const char * label;
	label = gpips_gtk_menu_item_get_label(widget);
	if (label == NULL || strcmp(label, display_options_panel) == 0 || strcmp(
			label, hide_options_panel) == 0)
		return;
	func(widget);
}

void apply_on_each_options_frame_choice(GtkWidget * widget, gpointer _func) {
	void (*func)(GtkWidget *);
	func = (void(*)(GtkWidget *)) _func;
	if (GTK_IS_COMBO_BOX(widget))
		func(widget);
}

void apply_on_each_option_item(void(* function_to_apply_on_each_menu_item)(
		GtkWidget *), void(* function_to_apply_on_each_frame_choice)(
		GtkWidget *)) {
	/* Walk through items of options_menu */
	gtk_container_foreach(GTK_CONTAINER(options_menu), (GtkCallback)
			apply_on_each_options_menu_item,
			function_to_apply_on_each_menu_item);

	/* Now walk through the options frame: */
	gtk_container_foreach(GTK_CONTAINER(options_frame), (GtkCallback)
			apply_on_each_options_frame_choice,
			function_to_apply_on_each_frame_choice);
}

void disable_option_selection() {
	apply_on_each_option_item(disable_item, disable_item);
}

void enable_option_selection() {
	apply_on_each_option_item(enable_item, enable_item);
}

void update_options() {
	gint i, j;
	const char * child_resource_alias;
	const char * phase_alias_n;
	string res_true_n, phase_true_n;
	GtkWidget * menu_options, *special_prop_menu;
	GtkWidget *spm_item;

	debug_on("GPIPS_DEBUG_LEVEL");

	menu_options = smenu_options;

	/* walk through items of menu_options */
	GtkWidget * child;
	GList * menu_options_children = gtk_container_get_children(GTK_CONTAINER(
			menu_options));
	for (i = 0; i < g_list_length(menu_options_children); i++) {
		child = (GtkWidget *) g_list_nth_data(menu_options_children, i);
		if (!GTK_IS_MENU_ITEM(child))
			continue;
		if (GTK_IS_SEPARATOR(child))
			continue;

		child_resource_alias = gpips_gtk_menu_item_get_label(child);
		if (child_resource_alias == NULL || strcmp(child_resource_alias,
				display_options_panel) == 0 || strcmp(child_resource_alias,
				hide_options_panel) == 0)
			/* It must be the pin item, or the item controlling the
			 options panel. RK, 4/06/1993. */
			continue;

		if ((res_true_n = (string) hash_get(aliases, child_resource_alias))
				== HASH_UNDEFINED_VALUE)
			pips_error("update_options",
					"Hash table aliases no more consistent\n");
		// -------------

		/* find special prop menu containing item corresponding to active
		 phase*/
		special_prop_menu = gtk_menu_item_get_submenu(GTK_MENU_ITEM(child));
		//			(Menu) xv_get(child, MENU_PULLRIGHT);

		GList * spm_children = gtk_container_get_children(GTK_CONTAINER(
				special_prop_menu));
		/* find active phase which produces resource res_true_n */
		phase_true_n = rule_phase(find_rule_by_resource(res_true_n));
		if ((phase_alias_n = hash_get_key_by_value(aliases, phase_true_n))
				== HASH_UNDEFINED_VALUE) {
			/* There is no alias for currently selected phase: phase_true_n
			 We have to activate another phase
			 */
			gtk_menu_set_active(GTK_MENU(special_prop_menu), 0);
			spm_item = (GtkWidget *) g_list_nth_data(spm_children, 0);
			phase_alias_n = gpips_gtk_menu_item_get_label(spm_item);
			user_warning(
					"update_options",
					"No alias available for selected phase `%s'; selecting `%s'\n",
					phase_true_n, phase_alias_n);
			phase_true_n = hash_get(aliases, phase_alias_n);
			activate(phase_true_n);
			//			/* It does not seem to work (the doc does not talk about
			//			 xv_set for MENU_SELECTED by the way...): */
			//			xv_set(special_prop_menu_item, MENU_SELECTED, TRUE, NULL);
			//			/* Then, try harder : */
			//			xv_set(special_prop_menu, MENU_DEFAULT_ITEM,
			//					special_prop_menu_item, NULL);
			debug(2, "update_options",
					"Rule `%s' selected to produce resource `%s'\n",
					phase_alias_n, child_resource_alias);
		} else {
			/* walk through items of special_prop_m to select the activated
			 one */
			for (j = g_list_length(spm_children) - 1; j >= 0; j--) {
				const char * spm_item_label;
				spm_item = (GtkWidget *) g_list_nth_data(spm_children, j);
				if (!GTK_IS_MENU_ITEM(spm_item))
					continue;
				if (GTK_IS_SEPARATOR(spm_item))
					continue;
				spm_item_label = gpips_gtk_menu_item_get_label(spm_item);
				debug(9, "update_options", "Menu item tested:\"%s\"\n",
						spm_item_label);
				if (strcmp(spm_item_label, phase_alias_n) == 0) {
					gtk_menu_set_active(GTK_MENU(special_prop_menu), j);
					/* Update the dual options panel entry : */
					GtkWidget * combobox = hash_get(menu_item_to_combo_box,
							special_prop_menu);
					if (GTK_IS_COMBO_BOX(combobox))
						gtk_combo_box_set_active(GTK_COMBO_BOX(combobox), j);

					if (verbose_update_options)
						user_log("Options: phase %s set on.\n", phase_alias_n);
					debug(2, "update_options",
							"Rule `%s' selected to produce resource `%s'\n",
							phase_alias_n, child_resource_alias);
				}
			}
		}
		g_list_free(spm_children);
	}
	g_list_free(menu_options_children);
	display_memory_usage();

	debug_off();
}

void options_select(const char * aliased_phase) {
	string phase = hash_get(aliases, aliased_phase);

	if (phase == (string) HASH_UNDEFINED_VALUE) {
		pips_error("options_select", "aliases badly managed !!!\n");
	} else {
		if (!db_get_current_workspace_name()) {
			prompt_user("No workspace opened. Options not accounted.\n");
		} else {
			debug_on("GPIPS_DEBUG_LEVEL");
			activate(phase);
			debug_off();
			user_log("Options: phase %s (%s) set on.\n", aliased_phase, phase);
		}
	}
	/* To have the options panel consistent with the real phase set :
	 RK, 05/07/1993. */
	verbose_update_options = 0;
	update_options();
	verbose_update_options = 1;
}

void options_combo_box_change_callback(GtkWidget * item, gpointer data __attribute__((unused)))
/*	"value" is already used for a typedef... :-) RK, 11/06/1993. */
{
	char * aliased_phase = gtk_combo_box_get_active_text(GTK_COMBO_BOX(item));
	options_select(aliased_phase);
}

void options_frame_to_view_menu_gateway(GtkWidget * widget, gpointer data __attribute__((unused))) {
	const gchar * label = gtk_button_get_label(GTK_BUTTON(widget));
	gpips_execute_and_display_something_from_alias(label);
}

void options_menu_callback(GtkWidget * widget, gpointer data __attribute__((unused))) {
	const char* aliased_phase = gpips_gtk_menu_item_get_label(widget);
	options_select(aliased_phase);
}

/* The function used by qsort to compare 2 option_type structures
 according to their name string: */
static int compare_option_type_for_qsort(const void *x, const void *y) {
	return strcmp(((option_type *) x)->name, ((option_type *) y)->name);
}

static void synch_viewmenu_and_opframe_search_in_view(GtkWidget * widget,
		gpointer data) {
	LabelAndVbox * lav = (LabelAndVbox *) data;
	GtkWidget * button;
	if (GTK_IS_MENU_ITEM(widget)) {
		// if the currently processed menu_item has got the same label than the currently processed option
		if (strcmp(data, gpips_gtk_menu_item_get_label(widget)) == 0) {
			button = gtk_button_new_with_label(lav->label); // creating a "link-button" toward View
			gtk_box_pack_start(GTK_BOX(lav->vbox), button, FALSE, FALSE, 5); // packing
			gtk_box_reorder_child(GTK_BOX(lav->vbox), button, lav->position + 1); // positioning
			gtk_signal_connect(GTK_OBJECT(button), "clicked", GTK_SIGNAL_FUNC( // connecting
					options_frame_to_view_menu_gateway), NULL);
		}
	}
}

static void synch_viewmenu_and_opframe(GtkWidget * widget, gpointer data) {
	const char * label;
	LabelAndVbox * lav = (LabelAndVbox *) data;

	if (GTK_IS_LABEL(widget)) {
		label = gtk_label_get_text(GTK_LABEL(widget));
		// on cherche ds le viewmenu un menu_item avec le même label
		// si yen a un on rajoute un bouton
		lav->label = label;
		gtk_container_child_get_property(GTK_CONTAINER(lav->vbox), widget,
				"position", (GValue*)&(lav->position));
		gtk_container_foreach(GTK_CONTAINER(view_menu), (GtkCallback)
				synch_viewmenu_and_opframe_search_in_view, lav);
	}
}

/* Build the option menu by looking at the pipsmake.rc file and
 searching if there are several way to build a ressource: */
/* The option panel use the definition of the edit menu and so needs
 to be created after it: */
void build_options_menu_and_panel(GtkWidget * menu_options,
		GtkWidget * frame_vbox) {
	int i;
	option_type *all_the_options;
	makefile m = parse_makefile();
	//int max_item_width = 0;

	int number_of_options = 0;
	hash_table phase_by_made_htp = hash_table_make(hash_string, 0);
	// attention on libère jamais la mémoire de la hash_table ci dessous
	menu_item_to_combo_box = hash_table_make(hash_pointer, 0);

	smenu_options = menu_options;

	/* walking thru rules */
	MAPL(pr, {
				rule r = RULE(CAR(pr));

				/* walking thru resources made by this particular rule */
				MAPL(pvr, {
							virtual_resource vr = VIRTUAL_RESOURCE(CAR(pvr));
							string vrn = virtual_resource_name(vr);
							list p = CONS(STRING, rule_phase(r), NIL);
							list l = (list) hash_get(phase_by_made_htp, vrn);

							if ( l == (list) HASH_UNDEFINED_VALUE ) {
								hash_put(phase_by_made_htp, vrn, (char *)p);
							}
							else {
								(void) gen_nconc(l, p);
							}
						}, rule_produced(r));

			}, makefile_rules(m));

	/* walking thru phase_by_made_htp */

	/* First, try to do some measurements on the options item to be
	 able to sort and align menu items later: */
	HASH_MAP(k, v, {
				string alias1 = hash_get_key_by_value(aliases, k);
				list l = (list) v;

				if ((alias1 != HASH_UNDEFINED_VALUE) && (gen_length(l) >= 2))
				number_of_options++;
			}, phase_by_made_htp);

	all_the_options = (option_type *) malloc(number_of_options
			* sizeof(option_type));

	/* Now generate an array of the Options: */
	i = 0;
	HASH_MAP(k, v, {
				string alias1 = hash_get_key_by_value(aliases, k);
				list l = (list) v;

				if ((alias1 != HASH_UNDEFINED_VALUE) && (gen_length(l) >= 2)) {
					all_the_options[i].name = alias1;
					all_the_options[i].choices = l;
					i++;
				}
			}, phase_by_made_htp);

	/* Sort the Options: */
	qsort((char *) all_the_options, number_of_options, sizeof(option_type),
			compare_option_type_for_qsort);

	/* Create the Options in the Option menu and option panel: */
	GtkWidget * table = gtk_table_new(number_of_options, 2, TRUE);
	gtk_box_pack_start(GTK_BOX(frame_vbox), table, TRUE, FALSE, 0);
	for (i = 0; i < number_of_options; i++) {
		GtkWidget * sub_menu_option, *sub_menu_option_item;
		//GtkWidget * menu_options_item;
		GtkWidget * choices_combobox;

		/* Create the sub-menu entry of an option: */
		sub_menu_option = gtk_menu_new();
		sub_menu_option_item = gtk_menu_item_new_with_label(
				all_the_options[i].name);
		gtk_menu_item_set_submenu(GTK_MENU_ITEM(sub_menu_option_item),
				sub_menu_option);

//		g_signal_connect(GTK_OBJECT(sub_menu_option_item), "activate",
//				G_CALLBACK(options_menu_callback), NULL);
		gtk_menu_append(GTK_MENU(menu_options), sub_menu_option_item);

		/* PANEL_CLIENT_DATA is used to link the PANEL_CHOICE_STACK
		 to its dual MENU_PULLRIGHT. RK, 11/06/1993. */
		choices_combobox = gtk_combo_box_new_text();
		gtk_signal_connect(GTK_OBJECT(choices_combobox), "changed",
				GTK_SIGNAL_FUNC(options_combo_box_change_callback), NULL);
		gtk_table_attach_defaults(GTK_TABLE(table), gtk_label_new(
				all_the_options[i].name), 0, 1, i, i + 1);
		gtk_table_attach_defaults(GTK_TABLE(table), choices_combobox, 1, 2, i,
				i + 1);

		GtkWidget * subsubmenu_option_item;
		MAPL(vrn, {
					string alias2 = hash_get_key_by_value(aliases,
							STRING(CAR(vrn)));

					if (alias2 != HASH_UNDEFINED_VALUE) {
						/* Add a sub-option entry in the menu: */
						/* Attach the sub-menu to an Option menu: */
						subsubmenu_option_item = gtk_menu_item_new_with_label(alias2);
						gtk_menu_append(GTK_MENU(sub_menu_option), subsubmenu_option_item);
						g_signal_connect(GTK_OBJECT(subsubmenu_option_item), "activate",
										G_CALLBACK(options_menu_callback), NULL);

						/* ... And in the Option panel: */
						gtk_combo_box_append_text(GTK_COMBO_BOX(choices_combobox), alias2);
					}
				}, all_the_options[i].choices);

		// on indique la combobox associée à sub_menu_option menu dans une hash_table dédiée
		hash_put(menu_item_to_combo_box, sub_menu_option, choices_combobox);

		gtk_widget_show(sub_menu_option_item);
		gtk_widget_show_all(sub_menu_option);
		gtk_widget_show_all(choices_combobox);
	}
	gtk_widget_show_all(table);

	/* According to a suggestion from Guillaume Oget, it should be nice
	 to be able to select a view also from the Option panel: */

	//char * option_item_label;

	LabelAndVbox lav;
	lav.vbox = frame_vbox;
	gtk_container_foreach(GTK_CONTAINER(options_frame), (GtkCallback)
			synch_viewmenu_and_opframe, &lav);
}

/* Construct a table linking PIPS phase name to more human names, the
 aliases defined in pipsmake-rc.tex.
 */
void build_aliases() {
	static int runs = 0;
	runs++;
	char buffer[BSIZE_HERE];
	char true_name[BSIZE_HERE], alias_name[BSIZE_HERE];
	FILE *fd;

	aliases = hash_table_make(hash_string, 0);

	fd = fopen_config(WPIPS_RC, NULL,NULL);

	while (fgets(buffer, BSIZE_HERE, fd) != NULL) {
		if (buffer[0] == '-')
			continue;

		sscanf(buffer, "alias%s '%[^']", true_name, alias_name);

		string stored_name;
		if ((stored_name = hash_get(aliases, alias_name))
				!= HASH_UNDEFINED_VALUE) {
			pips_internal_error("Aliases must not be ambiguous\n"
				"\talias '%s' seems ambiguous "
				"because of previous '%s'!\n", true_name, stored_name);
		} else {
			char * upper = strdup(true_name);
			hash_put(aliases, strdup(alias_name), strupper(upper, upper));
		}
	}
	safe_fclose(fd, WPIPS_RC);
}

void display_or_hide_options_frame(GtkWidget * menu_item, gpointer data) {
	const char *message_string;

	/* Should be added : when the options panel is destroyed by the
	 window manager, toggle the menu. RK, 7/6/93. */

	message_string = gpips_gtk_menu_item_get_label(menu_item);
	if (message_string != NULL && strcmp(message_string, display_options_panel)
			== 0) {
		gtk_widget_show_all(options_window);
		gpips_gtk_menu_item_set_label(menu_item, hide_options_panel);
	} else {
		hide_window(options_window, NULL, NULL);
		gpips_gtk_menu_item_set_label(menu_item, display_options_panel);
	}
}

/* Hide and restore the menu item to reopen the option panel: */
void static options_window_done_procedure(GtkWidget * window, gpointer data) {
	hide_window(options_window, NULL, NULL);
	gpips_gtk_menu_item_set_label(options_frame_menu_item,
			display_options_panel);
}

void create_options_menu_and_window() {
	GtkWidget * vbox;
	options_frame = gtk_frame_new(NULL);
	vbox = gtk_vbox_new(FALSE, 0);
	gtk_container_add(GTK_CONTAINER(options_frame), vbox);
	gtk_container_add(GTK_CONTAINER(options_window), options_frame);

	/* Trap the FRAME_DONE event: */
	gtk_signal_connect(GTK_OBJECT(options_window), "delete-event",
			GTK_SIGNAL_FUNC(options_window_done_procedure), NULL);

	options_frame_menu_item = gtk_menu_item_new_with_label(
			display_options_panel);
	g_signal_connect(G_OBJECT(options_frame_menu_item), "activate", G_CALLBACK(
			display_or_hide_options_frame), NULL);

	options_menu = gtk_menu_new();
	gtk_menu_append(GTK_MENU(options_menu), options_frame_menu_item);

	build_aliases();
	build_options_menu_and_panel(options_menu, vbox);
	gtk_widget_show_all(options_frame);

	options_menu_item = gtk_menu_item_new_with_label("Options");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(options_menu_item), options_menu);

	gtk_menu_bar_append(GTK_MENU_BAR(main_window_menu_bar), options_menu_item);
}
