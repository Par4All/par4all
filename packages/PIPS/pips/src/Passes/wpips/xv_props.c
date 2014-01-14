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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/frame.h>
#include <xview/panel.h>

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

#include "wpips.h"

#define OPTION_PANEL_BUTTONS_ALIGNED

typedef struct 
{
   char *name;
   list choices;
}
option_type;


static Menu smenu_options;
static Menu options_menu;
Menu_item static options_panel_menu_item;
Panel options_panel;

static hash_table aliases;

static char display_options_panel[] = "Options panel...";
static char hide_options_panel[] = "Hide options panel";

	/* Flag allowing update_options also to select a option : */
int verbose_update_options = 1;


/* returns the first key which value is svp. when not found, returns NULL */
string hash_get_key_by_value(htp, svp)
hash_table htp;
string svp;
{
    HASH_MAP(kp, vp, {
	if (strcmp(vp, svp) == 0)
	    return kp;
    }, htp);

    return HASH_UNDEFINED_VALUE;
}


void
disable_panel_item(Panel_item item)
{
   xv_set(item, PANEL_INACTIVE, TRUE, 0);
}


void
enable_panel_item(Panel_item item)
{
   xv_set(item, PANEL_INACTIVE, FALSE, 0);
}


void
apply_on_each_option_item(void (* function_to_apply_on_each_menu_item)(Menu_item),
                          void (* function_to_apply_on_each_panel_item)(Panel_item))
{
   int i;
   
   /* Walk through items of menu_options */
   for (i = (int) xv_get(options_menu, MENU_NITEMS); i > 0; i--) {
      char * res_alias_n;
      
      /* find resource corresponding to the item */
      Menu_item menu_item = (Menu_item) xv_get(options_menu, MENU_NTH_ITEM, i);

      if ((bool) xv_get(menu_item, MENU_TITLE))
         /* Skip the title item: */
         break;
      
      res_alias_n = (string) xv_get(menu_item, MENU_STRING);
      if (res_alias_n == NULL
          || strcmp(res_alias_n, display_options_panel) == 0
          || strcmp(res_alias_n, hide_options_panel) == 0)
         /* It must be the pin item, or the item controlling the
            options panel. */
         continue;

      function_to_apply_on_each_menu_item(menu_item);
   }

   /* Now walk through the options panel: */
   {
      Panel_item panel_item;

      PANEL_EACH_ITEM(options_panel, panel_item)
         /* Only on the PANEL_CHOICE_STACK: */
         if ((Panel_item_type) xv_get(panel_item, PANEL_ITEM_CLASS) ==
             PANEL_CHOICE_ITEM)
            function_to_apply_on_each_panel_item(panel_item);
      PANEL_END_EACH
         }
}


void
disable_option_selection()
{
   apply_on_each_option_item(disable_menu_item, disable_panel_item);
}


void
enable_option_selection()
{
   apply_on_each_option_item(enable_menu_item, enable_panel_item);
}


void
update_options()
{
   string res_alias_n, res_true_n ,phase_alias_n ,phase_true_n;
   Menu menu_options, special_prop_m;
   Menu_item options_mi, special_prop_mi;
   int i, j;

   debug_on("WPIPS_DEBUG_LEVEL");

   menu_options = smenu_options;

   /* walk through items of menu_options */
   for (i = (int) xv_get(menu_options, MENU_NITEMS); i > 0; i--) {

      /* find resource corresponding to the item */
      options_mi = (Menu_item) xv_get(menu_options, MENU_NTH_ITEM, i);

      if ((bool) xv_get(options_mi, MENU_TITLE))
         /* Skip the title item: */
         break;
      
      res_alias_n = (string) xv_get(options_mi, MENU_STRING);
      if (res_alias_n == NULL
          || strcmp(res_alias_n, display_options_panel) == 0
          || strcmp(res_alias_n, hide_options_panel) == 0)
         /* It must be the pin item, or the item controlling the
            options panel. RK, 4/06/1993. */
         continue;
      if ((res_true_n = (string) hash_get(aliases, res_alias_n))
          == HASH_UNDEFINED_VALUE)
         pips_error("update_options", 
                    "Hash table aliases no more consistent\n");

      /* find special prop menu containing item corresponding to active 
         phase*/
      special_prop_m = (Menu) xv_get (options_mi, MENU_PULLRIGHT);

      /* find active phase which produces resource res_true_n */
      phase_true_n = rule_phase(find_rule_by_resource(res_true_n));
      if ( (phase_alias_n=hash_get_key_by_value(aliases, phase_true_n)) 
           == HASH_UNDEFINED_VALUE ) {
         /* There is no alias for currently selected phase: phase_true_n
            We have to activate another phase
            */
         special_prop_mi = xv_get(special_prop_m, MENU_NTH_ITEM, 1);
         phase_alias_n = (string) xv_get(special_prop_mi, MENU_STRING);
         user_warning("update_options",
                      "No alias available for selected phase `%s'; selecting `%s'\n",
                      phase_true_n, phase_alias_n);
         phase_true_n = hash_get(aliases, phase_alias_n);
         activate(phase_true_n);
         /* It does not seem to work (the doc does not talk about
            xv_set for MENU_SELECTED by the way...): */
         xv_set(special_prop_mi, MENU_SELECTED, TRUE, NULL);
         /* Then, try harder : */
         xv_set(special_prop_m,
                MENU_DEFAULT_ITEM, special_prop_mi,
                NULL);
         debug(2, "update_options", 
               "Rule `%s' selected to produce resource `%s'\n",
               phase_alias_n, res_alias_n);
      }
      else {
         /* walk through items of special_prop_m to select the activated 
            one */
         for (j = (int) xv_get(special_prop_m, MENU_NITEMS); j > 0; j--) {
            Panel_item panel_choice_item;

            special_prop_mi = xv_get(special_prop_m, MENU_NTH_ITEM, j);
            debug(9, "update_options", "Menu item tested:\"%s\"\n", 
                  (string) xv_get(special_prop_mi, MENU_STRING));
            if (strcmp((string)xv_get(special_prop_mi, MENU_STRING),
                       phase_alias_n ) ==0 ) {
               xv_set(special_prop_mi, MENU_SELECTED, TRUE, NULL);
               /* Update the dual options panel entry : */
               panel_choice_item = (Panel_item) xv_get(special_prop_m,
                                                       MENU_CLIENT_DATA);
               xv_set(panel_choice_item, PANEL_VALUE, j - 1, NULL);

               if (verbose_update_options)
                  user_log("Options: phase %s set on.\n", phase_alias_n);
               debug(2, "update_options", 
                     "Rule `%s' selected to produce resource `%s'\n",
                     phase_alias_n, res_alias_n);
            }
         }
      }
   }

   display_memory_usage();
   
   debug_off();
}

/* The following procedure is never used !
	I have removed it. RK, 11/06/1993. */
#if 0
void clear_options()
{
    /* string res_alias_n, res_true_n ,phase_alias_n ,phase_true_n; */
    Menu menu_options, special_prop_m;
    Menu_item options_mi, special_prop_mi;
    int i, j;

    menu_options = smenu_options;

    /* walk through items of menu_options */
    for (i=(int)xv_get(menu_options, MENU_NITEMS); i>0; i--) {

	options_mi = (Menu_item) xv_get(menu_options, MENU_NTH_ITEM, i);
	special_prop_m = (Menu) xv_get (options_mi, MENU_PULLRIGHT);

	/* walk through items of special_prop_m */
	for (j=(int)xv_get(special_prop_m, MENU_NITEMS); j>0; j--) {
	    special_prop_mi = xv_get(special_prop_m, MENU_NTH_ITEM, j);
	    xv_set(special_prop_mi, MENU_SELECTED, FALSE);
	}
    }
}
#endif

void options_select(aliased_phase)
char *aliased_phase;
{
   string phase = hash_get(aliases, aliased_phase);

   if (phase == (string) HASH_UNDEFINED_VALUE)
   {
      pips_error("options_select", "aliases badly managed !!!\n");
   }
   else {
      if ( !db_get_current_workspace_name()) {
         prompt_user("No workspace opened. Options not accounted.\n");
      }
      else {
         debug_on("WPIPS_DEBUG_LEVEL");
         activate(phase);
         debug_off();
         user_log("Options: phase %s (%s) set on.\n",
                  aliased_phase,
                  phase);
      }
   }
   /* To have the options panel consistent with the real phase set :
      RK, 05/07/1993. */
   verbose_update_options = 0;
   update_options();
   verbose_update_options = 1;
}


void
options_panel_notify(Panel_item item,
                     int valeur,
                     Event *event)
/*	"value" is already used for a typedef... :-) RK, 11/06/1993. */
{
   
   Menu menu_special_prop;
   Menu_item menu_item;
   string aliased_phase;

   menu_special_prop = (Menu) xv_get(item, PANEL_CLIENT_DATA);
   /* Look at the corresponding (dual) item in the options menu to
      get the value. RK, 11/06/1993. */
   /* Argh ! MENU_NTH_ITEM begins from 1 but
      PANEL_VALUE begins from 0...	RK, 02/07/1993. */
   menu_item = (Menu_item) xv_get(menu_special_prop,
                                  MENU_NTH_ITEM, valeur + 1);
   aliased_phase = (string) xv_get(menu_item, MENU_STRING);
   options_select(aliased_phase);
}


void
options_panel_to_view_menu_gateway(Panel_item item,
                                   Event * event)
{
   char * label = (char *) xv_get(item, PANEL_LABEL_STRING);

   wpips_execute_and_display_something_from_alias(label);
}


void
options_menu_notify(Menu menu,
                    Menu_item menu_item)
{
   string aliased_phase = (char *) xv_get(menu_item, MENU_STRING);
   options_select(aliased_phase);
}


/* The function used by qsort to compare 2 option_type structures
   according to their name string: */
static int
compare_option_type_for_qsort(const void *x,
                              const void *y)
{
   return strcmp(((option_type *) x)->name, ((option_type *) y)->name);
}



/* Build the option menu by looking at the pipsmake.rc file and
   searching if there are several way to build a ressource: */
/* The option panel use the definition of the edit menu and so needs
       to be create after it: */
void
build_options_menu_and_panel(Menu menu_options,
                             Panel options_panel)
{
   int i, j;
   option_type *all_the_options;
   makefile m = parse_makefile();
   int max_item_width = 0;
  
   int number_of_options = 0;
   hash_table phase_by_made_htp = hash_table_make(hash_string, 0);

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

   all_the_options = (option_type *)
      malloc(number_of_options*sizeof(option_type));
   
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
   qsort((char *)all_the_options,
         number_of_options,
         sizeof(option_type),
         compare_option_type_for_qsort);

   /* Create the Options in the Option menu and option panel: */
   for(i = 0; i < number_of_options; i++) {
      Menu sub_menu_option;
      Menu_item menu_options_item;
      Panel_item panel_choice_item;

      /* Create the sub-menu entry of an option: */
      sub_menu_option =
         (Menu) xv_create(NULL, MENU_CHOICE_MENU,
                          MENU_NOTIFY_PROC, options_menu_notify,
                          NULL);

      /* PANEL_CLIENT_DATA is used to link the PANEL_CHOICE_STACK
         to its dual MENU_PULLRIGHT. RK, 11/06/1993. */
      panel_choice_item = xv_create(options_panel, PANEL_CHOICE_STACK,
                                    PANEL_LAYOUT, PANEL_HORIZONTAL,
                                    PANEL_LABEL_STRING,
                                    all_the_options[i].name,
                                    PANEL_CLIENT_DATA, sub_menu_option,
                                /* Nothing selected yet: */
                                    PANEL_VALUE, 0,
                                    PANEL_NOTIFY_PROC, options_panel_notify,
                                    NULL);
      j = 0;
      MAPL(vrn, {
         string alias2 = hash_get_key_by_value(aliases, 
                                               STRING(CAR(vrn)));
         Menu_item sub_menu_option_item;

         if (alias2 != HASH_UNDEFINED_VALUE) {
            /* Add a sub-option entry in the menu: */
            sub_menu_option_item = (Menu_item) xv_create(NULL, MENUITEM,
                                                         MENU_STRING, 
                                                         alias2,
                                                         MENU_RELEASE,
                                                         NULL);
            /* Attach the sub-menu to an Option menu: */
            xv_set(sub_menu_option,
                   MENU_APPEND_ITEM, sub_menu_option_item,
                   NULL);
            /* ... And in the Option panel: */
            xv_set(panel_choice_item,
                   PANEL_CHOICE_STRING,
                   j++,
                   alias2,
                   NULL);
         }
      }, all_the_options[i].choices);
      /* Create the Option Item for this alias: */
      menu_options_item = (Menu_item) xv_create(NULL, MENUITEM,
                                                MENU_STRING,
                                                all_the_options[i].name,
                                                MENU_PULLRIGHT,
                                                sub_menu_option,
                                                MENU_RELEASE,
                                                NULL);
      /* Attach the Option Item to the Option menu: */
      xv_set(menu_options, MENU_APPEND_ITEM, menu_options_item, NULL);
      /* Add a link information from the sub_menu_option to the
         equivalent panel_choice_item for update_props(): */
      xv_set(sub_menu_option,
             MENU_CLIENT_DATA, panel_choice_item,
             NULL);
   }

   /* Now we set the width of the PANEL_CHOICE_STACK items the same to
      have a nicer vertical alignment: */
   {
      Panel_item item;

#ifdef OPTION_PANEL_BUTTONS_ALIGNED
      /* If I want something with the buttons vertically aligned: */
      PANEL_EACH_ITEM(options_panel, item)
         max_item_width = MAX(max_item_width,
                              (int) xv_get(item, PANEL_LABEL_WIDTH));
      PANEL_END_EACH

         /* It looks on my screen that this width is 8:7 larger. Apply
            a pifometric scaling factor... :-) */
         max_item_width = (max_item_width*7)/8;

      PANEL_EACH_ITEM(options_panel, item)
         xv_set(item, PANEL_LABEL_WIDTH,
                max_item_width,
                NULL);
      PANEL_END_EACH
#else
      /* If I want something a` la TeX, justified on the left and
         on the right: */
      PANEL_EACH_ITEM(options_panel, item)
         max_item_width = MAX(max_item_width,
                              (int) xv_get(item, XV_WIDTH));
      PANEL_END_EACH

      PANEL_EACH_ITEM(options_panel, item)
         xv_set(item, PANEL_LABEL_WIDTH,
                (int) xv_get(item, PANEL_LABEL_WIDTH)
                + max_item_width
                - (int) xv_get(item, XV_WIDTH),
                NULL);
      PANEL_END_EACH
#endif
   }

   /* According to a suggestion from Guillaume Oget, it should be nice
      to be able to select a view also from the Option panel: */
   {
      Panel_item option_item;
      char * option_item_label;

      PANEL_EACH_ITEM(options_panel, option_item)
         if ((Panel_item_type) xv_get(option_item, PANEL_ITEM_CLASS) ==
             PANEL_CHOICE_ITEM) {
            Menu_item view_menu_item;
            
            option_item_label = (char *) xv_get(option_item,
                                                PANEL_LABEL_STRING);

            /* Find if the View menu has an item with the same value: */
            view_menu_item =
               (Menu_item) xv_find(view_menu, MENUITEM,
                                   MENU_STRING, option_item_label,
                                /* Do not create the menu item if it
                                   does not exist: */
                                   XV_AUTO_CREATE, FALSE,
                                   NULL);
            if (view_menu_item != XV_NULL) {
               /* OK, there is also a View menu with the same name. */
               xv_create(options_panel, PANEL_BUTTON,
                         PANEL_NOTIFY_PROC, options_panel_to_view_menu_gateway,
                         PANEL_LABEL_STRING, option_item_label,
                         XV_X, xv_get(option_item, XV_X),
                         XV_Y, xv_get(option_item, XV_Y),
                         NULL);

               xv_set(option_item,
                      PANEL_LABEL_STRING, NULL,
                      PANEL_LABEL_WIDTH, 0,
                      PANEL_VALUE_X, xv_get(option_item, PANEL_VALUE_X),
                      NULL);
            }
            else {
               /* Shift a little bit the PANEL_LABEL to align with
                  other text of the menu buttons: */
               xv_set(option_item,
                      PANEL_VALUE_X, xv_get(option_item, PANEL_VALUE_X),
                      PANEL_LABEL_X, xv_get(option_item, PANEL_LABEL_X) + 8,
                      NULL);
               
            }          
         }
      PANEL_END_EACH
         }
   
#if 0
   {
      Panel_item option_item;
      char * option_item_label;
      Panel_item * the_option_panel_items;
      int number_of_options_to_link_with_the_view_menu;
      int i = 0;

      /* A conservative allocation... */
      the_option_panel_items = (Panel_item *) calloc(number_of_options,
                                                     sizeof(Panel_item *));
      
      PANEL_EACH_ITEM(options_panel, option_item) 
         {
            Menu_item view_menu_item;
            
            option_item_label = (char *) xv_get(option_item,
                                                PANEL_LABEL_STRING);

            /* Find if the View menu has an item with the same value: */
            view_menu_item =
               (Menu_item) xv_find(view_menu, MENUITEM,
                                   MENU_STRING, option_item_label,
                                /* Do not create the menu item if it
                                   does not exist: */
                                   XV_AUTO_CREATE, FALSE,
                                   NULL);
            if (view_menu_item != XV_NULL) {
               /* OK, there is also a View menu with the same name. */
               /* Put it in the array of panel items to create. We
                  cannot create it directly since it would confuse
                  PANEL_EACH_ITEM: */
               the_option_panel_items[i++] = option_item;
            }
         }
      PANEL_END_EACH
         number_of_options_to_link_with_the_view_menu = i;

      window_fit(options_panel);
      window_fit(options_frame);
   
         for(i = 0; i < number_of_options_to_link_with_the_view_menu; i++) {
               /* Overlay the label of an option item with a button
                  with the same name to select the View item: */
               xv_create(options_panel, PANEL_BUTTON,
                         PANEL_NOTIFY_PROC, options_panel_to_view_menu_gateway,
                         PANEL_LABEL_STRING, xv_get(the_option_panel_items[i],
                                                    PANEL_LABEL_STRING),
                         XV_X, xv_get(the_option_panel_items[i],
                                      XV_X),
                         XV_Y, xv_get(the_option_panel_items[i],
                                      XV_Y),
                         NULL);

               xv_set(the_option_panel_items[i],
                      PANEL_LABEL_STRING, NULL,
                      PANEL_LABEL_WIDTH, 0,
                      PANEL_VALUE_X, xv_get(the_option_panel_items[i],
                                            PANEL_VALUE_X),
                      NULL);
            }
         }
#endif
}

#define BSIZE_HERE (256)

void
build_aliases()
{
   char buffer[BSIZE_HERE];
   char true_name[BSIZE_HERE], alias_name[BSIZE_HERE];
   FILE *fd;

   aliases = hash_table_make(hash_string, 0);

   fd = fopen_config(WPIPS_RC, NULL,NULL);

   while (fgets(buffer, BSIZE_HERE, fd) != NULL) {
      if (buffer[0] == '-')
         continue;

      sscanf(buffer, "alias%s '%[^']", true_name, alias_name);

      if (hash_get(aliases, alias_name) != HASH_UNDEFINED_VALUE) {
         pips_internal_error("Aliases must not be ambiguous\n"
			     "\talias '%s' seems ambiguous!\n", true_name);
      }
      else {
         char upper[128];

         hash_put(aliases, 
                  strdup(alias_name), 
                  strdup(strupper(upper, true_name)));
      }
   }
   safe_fclose(fd, WPIPS_RC);
}

void
display_or_hide_options_panel(Menu menu,
                              Menu_item menu_item)
{
   char *message_string;


   /* Should be added : when the options panel is destroyed by the
      window manager, toggle the menu. RK, 7/6/93. */

   message_string = (char *) xv_get(menu_item, MENU_STRING);
   if (strcmp(message_string, display_options_panel) == 0)
   {
      unhide_window(options_frame);
      xv_set(menu_item, MENU_STRING, hide_options_panel, NULL);
   }
   else
   {
      hide_window(options_frame);
      xv_set(menu_item, MENU_STRING, display_options_panel, NULL);
   }
}


/* Hide and restore the menu item to reopen the option panel: */
void static
options_frame_done_procedure(Frame frame)
{
   hide_window(options_frame);
   xv_set(options_panel_menu_item, MENU_STRING, display_options_panel, NULL);  
}


void
create_options_menu_and_window()
{
   options_panel = (Panel) xv_create(options_frame, PANEL,
                                     PANEL_LAYOUT, PANEL_VERTICAL,
                                     /* RK, 11/06/1993. There is no room : */
                                     PANEL_ITEM_Y_GAP, 0,
                                     PANEL_ITEM_X_GAP, 0,
                                     NULL);

   /* Trap the FRAME_DONE event: */
   xv_set(options_frame, FRAME_DONE_PROC, options_frame_done_procedure, NULL);

   options_panel_menu_item =
      (Menu_item) xv_create(NULL, MENUITEM,
                            MENU_STRING, display_options_panel,
                            MENU_NOTIFY_PROC, display_or_hide_options_panel,
                            NULL);
   options_menu = (Menu) xv_create(XV_NULL, MENU_COMMAND_MENU,
                                   MENU_TITLE_ITEM, "Selecting PIPS Options ",
                                   MENU_GEN_PIN_WINDOW, main_frame, "Options Menu",
                                   MENU_APPEND_ITEM, options_panel_menu_item,
                                   NULL);

   build_aliases();
   build_options_menu_and_panel(options_menu, options_panel);
   
   (void) xv_create(main_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Options",
                    PANEL_ITEM_MENU, options_menu,
                    NULL);

   window_fit(options_panel);
   window_fit(options_frame);
}
