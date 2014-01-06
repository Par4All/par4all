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
/* Single choice handling */

/* Difference with previous release: 
   1/ schoose_close() must be called in order to close the schoose window.
   2/ schoose() has one more argument, because cancel button is created.
   bb 04.06.91
 */

#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>

#include "genC.h"
#include "misc.h"

#include "wpips.h"

enum
{
   ABBREV_MENU_WITH_TEXT_AFTER_SELECTION = 9841,
      ABBREV_MENU_WITH_TEXT_GENERATE_MENU = 1431       
};

static Panel_item choice, choices, ok, help, cancel;

static void (* apply_on_choice)(char *);
static void (* apply_on_cancel)(void);

void static
schoose_help_notify(Panel_item item,
                    Event * event)
{
   display_help("SingleChoice");
}

/* Cette routine est appelé d'une part lorsqu'on a cliqué sur OK pour
   valider un nom tapé textuellement, d'autre part lorsqu'on clique sur
   un choix. */
void static
schoose_ok_notify(Panel_item item,
                  Event * event)
{
   /* Suppose qu'item et event sont folklo car on peut être appelé par
      schoose_choice_notify */
   char *curchoice;
   int i, nchoices;
   int item_is_in_the_list;

   curchoice = strdup((char *) xv_get(choice, PANEL_VALUE, 0));
   if (strlen(curchoice)==0)
      prompt_user("Choose one item or cancel");
   else {
      /* Modified to verify that an correct item is selected.
         RK, 21/05/1993. */
      nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);
      item_is_in_the_list = FALSE;
      for(i = 0; i < nchoices; i++)
         if (strcmp((char *)xv_get(choices, PANEL_LIST_STRING, i),
                    curchoice) == 0) {
            item_is_in_the_list = TRUE;
            break;
         }
      if (item_is_in_the_list == FALSE)
         prompt_user("You have to choose one item of the list!");
      else {
				/* Normal case : */
         (*apply_on_choice)(curchoice);
         schoose_close();
      }
   }

   free(curchoice);
}

/* schoose_close() can be called even when schoose window is already closed.
 */
void
schoose_close()
{
   int nchoices;

   hide_window(schoose_frame);

   nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, NULL);

   /* Delete all the rows, ie nchoices rows from row 0: */
   xv_set(choices,
          PANEL_LIST_DELETE_ROWS, 0, nchoices,
          NULL);

   xv_set(choice, PANEL_VALUE, "", NULL);
}


void
schoose_cancel_notify(Panel_item item,
                      Event * event)
{
   schoose_close();

   (*apply_on_cancel)();
}

/* Function used to update the text panel according to the list panel: */
int static
schoose_choice_notify(Panel_item item,
                      char * item_string,
                      Xv_opaque client_data,
                      Panel_list_op op,
                      Event * event,
                      int row)
{
   switch (op) {    
     case PANEL_LIST_OP_SELECT:
       xv_set(choice,
              PANEL_VALUE, item_string,
              NULL);
       break;
     
     /* Avoid deletion and insertion with the edit menu of button 3: */
     case PANEL_LIST_OP_DELETE:
     case PANEL_LIST_OP_VALIDATE:
        return XV_ERROR;
     
     case PANEL_LIST_OP_DESELECT:
       break;
       
     default:
       pips_assert("schoose_choice_notify: unknown operation !", 0);
     }

   /* Accept the operation by default: */
   return XV_OK;
}


/* Avoid the schoose_frame destruction and act as cancel: */
void static
schoose_frame_done_proc(Frame frame)
{
   (*apply_on_cancel)();
   hide_window(frame);
}


void
schoose(char * title,
	gen_array_t array,
        char * initial_choice,
        void (*function_for_ok)(char *),
        void (*function_for_cancel)(void))
{
   int i;
   int nchoices;
   int argc = gen_array_nitems(array);

   apply_on_choice = function_for_ok;
   apply_on_cancel = function_for_cancel;

   xv_set(schoose_frame, FRAME_LABEL, title, NULL);

   /* reset the choice set to empty */
   nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);

   /* Delete all the rows, ie nchoices rows from row 0: */
   xv_set(choices,
          PANEL_LIST_DELETE_ROWS, 0, nchoices,
          NULL);

   for (i = 0; i < argc; i++) {
       string name = gen_array_item(array, i);
       xv_set(choices, PANEL_LIST_STRING, i, name, NULL);
   }

   /* Initialise au choix initial ou à défaut le premier : */
   xv_set(choice, PANEL_VALUE, gen_array_item(array, 0), NULL);
   if (initial_choice != NULL)
   {
       for (i = 0; i < argc; i++) 
       {
	   string name = gen_array_item(array, i);
	   if (strcmp(initial_choice, name) == 0) {
	       xv_set(choice, PANEL_VALUE, name, NULL);
	       xv_set(choices, PANEL_LIST_SELECT, i, TRUE, NULL);
	       break;
	   }
       }
   }

   unhide_window(schoose_frame);
   /* move the pointer to the center of the query window */
   pointer_in_center_of_frame(schoose_frame);
}


/* Accept only typed text in the menu list: */
static void
schoose_abbrev_menu_with_text_text_notify(Panel_item text_item,
                                /* int Value, ? */
                                          Event * event)
{
   void (* real_user_notify_function)(char *);
   
   char * text = (char *) xv_get(text_item, PANEL_VALUE);

   debug_on("WPIPS_DEBUG_LEVEL");
   debug(9, "schoose_abbrev_menu_with_text_text_notify", "Entering ...\n");
   
   real_user_notify_function =
      (void (*)(char *)) xv_get(text_item, XV_KEY_DATA,
                                ABBREV_MENU_WITH_TEXT_AFTER_SELECTION);
   real_user_notify_function(text);
   
   debug_off();
}


/* The function that calls the real user notifying function: */
static void
abbrev_menu_with_text_menu_notify(Menu menu, Menu_item menu_item)
{
   void (* real_user_notify_function)(char *);
   char * menu_choice = (char *) xv_get(menu_item, MENU_STRING);

   debug_on("WPIPS_DEBUG_LEVEL");
   debug(9, "abbrev_menu_with_text_menu_notify", "Entering...\n");
   
   real_user_notify_function =
      (void (*)(char *)) xv_get(menu, MENU_CLIENT_DATA);
   
   real_user_notify_function(menu_choice);

   debug_off();
}


static Notify_value
abbrev_menu_event_filter_proc(Panel panel,
                              Event * event,
                              Notify_arg arg,
                              Notify_event_type type)
{
   Panel_item item;
   Rect * rect;
   Menu (* generate_menu)(void);
   
   debug_on("WPIPS_DEBUG_LEVEL");
   debug(9, "abbrev_menu_event_filter_proc", "Entering ...\n");
   
   /* See example p. 675 in the XView Programming Manual: */
   if (event_is_down(event)) {
      /* Find the Panel_item */
      PANEL_EACH_ITEM(panel, item) 
         {
            rect = (Rect *) xv_get(item, XV_RECT);
            if (rect_includespoint(rect,
                                   event->ie_locx,
                                   event->ie_locy)) {
               generate_menu =
                  (Menu (* )()) xv_get(item,
				       XV_KEY_DATA,
				       ABBREV_MENU_WITH_TEXT_GENERATE_MENU);

               if (generate_menu != NULL) {
		   /* OK, we clicked on a abbrev_menu_with_text menu: */
		   Menu new_menu;
		   void (* a_menu_notify_procedure)(Menu, Menu_item);
		   /* If there is an old menu, remove it: */
                  Menu old_menu =  (Menu) xv_get(item, PANEL_ITEM_MENU);

		  debug(9, "abbrev_menu_event_filter_proc",
			"OK, we clicked on a abbrev_menu_with_text menu.\n");
		  
                  if (old_menu != NULL)
                     xv_destroy(old_menu);

                  /* Create the new menu: */
                  new_menu = generate_menu();

                  a_menu_notify_procedure = (void (*)(Menu, Menu_item)) xv_get(new_menu, MENU_NOTIFY_PROC);
                  /* menu_return_value() seems to be the default
                     MENU_NOTIFY_PROC in XView... Hum, internal
                     details... */
		   /* Quite strange: with gcc without -static on
                      SunOS4.1.4, this test is never true... :-( Well,
                      remove this micro-optimization and always
                      reinstall the MENU_NOTIFY_PROC: */
		   /*
		      if (a_menu_notify_procedure == menu_return_value) {
		      */
                     /* The new_menu has not attached a notify
                        procedure. Get the one given at creation time
                        of the panel: */
                     xv_set(new_menu, MENU_NOTIFY_PROC,
                            abbrev_menu_with_text_menu_notify,
                            NULL);
		     debug(9, "abbrev_menu_event_filter_proc",
			   "Attaching abbrev_menu_with_text_menu_notify...\n");
		   /*
		      }
		      */
		  

                  {
                     /* Associate the real notify function to the menu too: */
                     void (* after_selection)(char *);
                     
                     after_selection = (void (*)(char *))
                        xv_get(item,
                               XV_KEY_DATA,
                               ABBREV_MENU_WITH_TEXT_AFTER_SELECTION);
                     
                     xv_set(new_menu,
                            MENU_CLIENT_DATA,
                            after_selection,
                            NULL);
                  }
                  xv_set(item, PANEL_ITEM_MENU, new_menu);
               }
            }
         }
      PANEL_END_EACH
         }
   debug_off();
   
   /* Now call the normal event procedure: */
   return notify_next_event_func(panel, (Notify_event) event, arg, type);
}


/* Create an abbreviation menu attached with a text item.
 after_selection() is called when a selection is done or a text has been
 entered. It can be seen as a new widget. */
Panel_item
schoose_create_abbrev_menu_with_text(Panel main_panel,
                                     char * label_string,
                                     int value_display_length,
                                     int x,
                                     int y,
                                     Menu (* generate_menu)(void),
                                     void (* after_selection)(char *))
{
   Panel_item item, text;

   item = xv_create(main_panel, PANEL_ABBREV_MENU_BUTTON,
                    PANEL_LABEL_STRING, label_string,
                    PANEL_VALUE_DISPLAY_LENGTH, value_display_length,
                    PANEL_LAYOUT, PANEL_HORIZONTAL,
                    PANEL_VALUE_X, x,
                    PANEL_VALUE_Y, y,
                    /* No real menu yet: */
                    PANEL_ITEM_MENU, xv_create(NULL, MENU,
                                               MENU_STRINGS, "* none *", NULL,
                                               NULL),
                    XV_KEY_DATA, ABBREV_MENU_WITH_TEXT_GENERATE_MENU, generate_menu,
                    XV_KEY_DATA, ABBREV_MENU_WITH_TEXT_AFTER_SELECTION, after_selection,
                    
                    NULL);

   notify_interpose_event_func(main_panel,
                               abbrev_menu_event_filter_proc,
                               NOTIFY_SAFE);

   text = xv_create(main_panel, PANEL_TEXT,
                    PANEL_VALUE_DISPLAY_LENGTH, value_display_length,
                    PANEL_VALUE_STORED_LENGTH, 128,
                    PANEL_READ_ONLY, FALSE,
                    PANEL_NOTIFY_PROC, schoose_abbrev_menu_with_text_text_notify,
                    XV_KEY_DATA, ABBREV_MENU_WITH_TEXT_AFTER_SELECTION,
                    after_selection,
                    /* Strange, 21 does a
                       Program received signal SIGSEGV, Segmentation fault.
                       0x1cf988 in attr_check_use_custom () */
                    PANEL_VALUE_X, x + 25,
                    PANEL_VALUE_Y, y,
                    /* PANEL_ITEM_X_GAP, 22,*/
                    NULL);
   
   return text;
}


void
create_schoose_window()
{
   schoose_frame = xv_create(main_frame, FRAME,
                             XV_SHOW, FALSE,
                             FRAME_DONE_PROC, hide_window,
                             NULL);

   schoose_panel = xv_create(schoose_frame, PANEL, NULL);

   choice = xv_create(schoose_panel, PANEL_TEXT,
                      PANEL_LABEL_STRING, "Current choice",
                      PANEL_VALUE_DISPLAY_LENGTH, 30,
                      PANEL_NOTIFY_PROC, schoose_ok_notify,
                      XV_X, xv_col(schoose_panel, 0),
                      NULL);

   choices = xv_create(schoose_panel, PANEL_LIST,
                       PANEL_LABEL_STRING, "Available choices",
                       PANEL_LIST_DISPLAY_ROWS, 5,
                       PANEL_NOTIFY_PROC, schoose_choice_notify,
                       PANEL_CHOOSE_ONE, TRUE,
                       XV_X, xv_col(schoose_panel, 0),
                       XV_Y, xv_rows(schoose_panel, 1),
                       NULL);

   ok = xv_create(schoose_panel, PANEL_BUTTON,
                  PANEL_LABEL_STRING, "OK",
                  PANEL_NOTIFY_PROC, schoose_ok_notify,
                  XV_X, xv_col(schoose_panel, 5),
                  XV_Y, xv_rows(schoose_panel, 5),
                  NULL);

   cancel = xv_create(schoose_panel, PANEL_BUTTON,
                      PANEL_LABEL_STRING, "Cancel",
                      PANEL_NOTIFY_PROC, schoose_cancel_notify,
                      NULL);

   help = xv_create(schoose_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Help",
                    PANEL_NOTIFY_PROC, schoose_help_notify,
                    NULL);

   (void) xv_set(schoose_panel, PANEL_DEFAULT_ITEM, ok, NULL);

   window_fit(schoose_panel);
   window_fit(schoose_frame);
   
   /* Avoid the schoose_frame destruction: */
   xv_set(schoose_frame,
          FRAME_DONE_PROC, schoose_frame_done_proc,
          NULL);
}
