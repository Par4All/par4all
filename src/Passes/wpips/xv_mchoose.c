/* 	%A% ($Date: 1995/09/28 10:25:35 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_xv_mchoose[] = "%A% ($Date: 1995/09/28 10:25:35 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */

/* Multiple choices handling */

#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/notice.h>
#include <xview/notify.h>
#include <types.h>

#include "genC.h"
#include "misc.h"

#include "wpips.h"

static Panel_item mchoices, choices, ok, cancel, help;
static char mchoices_notify_buffer[SMALL_BUFFER_LENGTH];

static void (*apply_on_mchoices)(int *, char **) = NULL;
static void (*cancel_on_mchoices)(void) = NULL;

static void mchoose_help_notify(item, event)
Panel_item item;
Event *event;
{
    display_help("MultipleChoice");
}


void static
mchoose_ok_notify(Panel_item item,
                  Event * event)
{
   char *mchoices_args[ARGS_LENGTH];
   char buffer[SMALL_BUFFER_LENGTH];
   int mchoices_length = 0;
   int i, nchoices, len;
   int item_is_in_the_list = FALSE;
   char *p;

   hide_window(mchoose_frame);

   /* The OK button becomes inactive through RETURN: */
   xv_set(mchoose_panel, PANEL_DEFAULT_ITEM, NULL, NULL);
   xv_set(mchoices, PANEL_NOTIFY_PROC, NULL);

   nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, NULL);
   mchoices_length = 0;

   /*
     for (i = 0; i < nchoices; i++) {
     if ((int) xv_get(choices, PANEL_LIST_SELECTED, i) == TRUE) {
     char *s = strdup(xv_get(choices, PANEL_LIST_STRING, i));
     args_add(&mchoices_length, mchoices_args, s);		     
     }
     }
     Read the text line instead, 1 word separated by ' '.
     RK, 19/05/1993.
     */
   strcpy(mchoices_notify_buffer, (char *) xv_get(mchoices, PANEL_VALUE));
   p = mchoices_notify_buffer;
   while(sscanf(p, "%s%n", buffer, &len) == 1) {
      args_add(&mchoices_length, mchoices_args, strdup(buffer));
      item_is_in_the_list = FALSE;
      for(i = 0; i < nchoices; i++)
         if (strcmp((char *)xv_get(choices, PANEL_LIST_STRING, i), buffer) == 0) {
            item_is_in_the_list = TRUE;
            break;
         }
      if (item_is_in_the_list == FALSE)
         break;
      p += len;
   }

   /*	At least on item selected, and in the list.
        RK, 21/05/1993.
	*/
   if (mchoices_length == 0 || item_is_in_the_list == FALSE) {
      char *s;
      s = mchoices_length == 0 ? "You have to select at least 1 item!" :
         "You have selected an item not in the choice list!";
      args_free(&mchoices_length, mchoices_args);
      prompt_user(s);
      return;
   }

   (*apply_on_mchoices)(&mchoices_length, mchoices_args);

   args_free(&mchoices_length, mchoices_args);

   /* Delay the graphics transformations. RK, 21/05/1993. */

   /* Delete all the rows, ie nchoices rows from row 0: */
   xv_set(choices,
          PANEL_LIST_DELETE_ROWS, 0, nchoices,
          NULL);

   xv_set(mchoices, PANEL_VALUE, "", NULL);
}


void static
mchoose_cancel_notify(Panel_item item,
                      Event * event)
{
   hide_window(mchoose_frame);
   
   /* The OK button becomes inactive through RETURN: */
   xv_set(mchoose_panel, PANEL_DEFAULT_ITEM, NULL, NULL);
   xv_set(mchoices, PANEL_NOTIFY_PROC, NULL);
   
   (*cancel_on_mchoices)();
   
   /* Delete all the rows, ie nchoices rows from row 0: */
   xv_set(choices,
          PANEL_LIST_DELETE_ROWS, 0, (int) xv_get(choices,
                                                  PANEL_LIST_NROWS,
                                                  NULL),
          NULL);

   xv_set(mchoices, PANEL_VALUE, "", NULL);
}


/* Avoid the mchoose_frame destruction and act as cancel: */
void static
mchoose_frame_done_proc(Frame frame)
{
   mchoose_cancel_notify(NULL, NULL);
}


/*
Notify_value
try_to_avoid_mchoose_destruction(Notify_client client,
                                 Destroy_status status)
{
   fprintf(stderr, "%x, %x\n", client, status);
   
   if (status == DESTROY_CHECKING) {
      notify_veto_destroy(client);
      prompt_user("You are not allowed to destroy the Mchoose frame !");
   }
   else if (status == DESTROY_CLEANUP)
      return notify_next_destroy_func(client, status);

   return NOTIFY_DONE;
}
*/


/* Function used to update the text panel according to the list panel: */
int static
mchoose_notify(Panel_item item,
               char * item_string,
               Xv_opaque client_data,
               Panel_list_op op,
               Event * event,
               int row)
{
   switch (op) {    
     case PANEL_LIST_OP_SELECT:
     case PANEL_LIST_OP_DESELECT:
     {
        int nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);

        mchoices_notify_buffer[0] = '\0';

	/* Now it is mchoices_notify_buffer which is used for the selection.
           No size verification implemented yet... :-)
           RK, 19/05/1993. */

        while (nchoices--) {
           if ((int) xv_get(choices, PANEL_LIST_SELECTED, nchoices) == TRUE) {
              strcat(mchoices_notify_buffer,
                     (char *) xv_get(choices, PANEL_LIST_STRING, nchoices));
              strcat(mchoices_notify_buffer, " ");
           }
        }

        xv_set(mchoices, PANEL_VALUE, mchoices_notify_buffer, NULL);
        break;
     }
     
     /* Avoid deletion and insertion with the edit menu of button 3: */
     case PANEL_LIST_OP_DELETE:
     case PANEL_LIST_OP_VALIDATE:
       return XV_ERROR;
     
     default:
       pips_assert("schoose_choice_notify: unknown operation !", 0);
   }

   /* Accept the operation by default: */
   return XV_OK;
}


void
mchoose(char * title,
        int argc,
        char * argv[],
        void (*function_ok)(int *, char **),
        void (*function_cancel)(void))
{
   int i;
   int nchoices;

   apply_on_mchoices = function_ok;
   cancel_on_mchoices = function_cancel;
   mchoices_notify_buffer[0] = '\0';

   xv_set(mchoose_frame, FRAME_LABEL, title, NULL);

   /* reset the choice set to empty */
   nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);

   /* Delete all the rows, ie nchoices rows from row 0: */
   xv_set(choices,
          PANEL_LIST_DELETE_ROWS, 0, nchoices,
          NULL);

   for (i = 0; i < argc; i++) {
      xv_set(choices, PANEL_LIST_STRING, i, argv[i], 0);
   }

   unhide_window(mchoose_frame);

   /* move the pointer to the center of the query window */
   pointer_in_center_of_frame(mchoose_frame);

   /* The OK button becomes active through RETURN: */
   xv_set(mchoose_panel, PANEL_DEFAULT_ITEM, ok, NULL);
   xv_set(mchoices, PANEL_NOTIFY_PROC, mchoose_ok_notify);
}


void
create_mchoose_window()
{
   mchoose_panel = xv_create(mchoose_frame, PANEL, 0);

/* Carriage return enable for "OK". RK, 19/05/1993. */

   mchoices = xv_create(mchoose_panel, PANEL_TEXT,
                        PANEL_LABEL_STRING, "Current choices",
                        PANEL_VALUE_DISPLAY_LENGTH, 30,
                        PANEL_VALUE_STORED_LENGTH, 128,
                        /* PANEL_NOTIFY_PROC, mchoose_ok_notify, */
                        XV_X, xv_col(mchoose_panel, 0),
                        NULL);

   choices = xv_create(mchoose_panel, PANEL_LIST,
                       PANEL_LABEL_STRING, "Available choices",
                       PANEL_LIST_DISPLAY_ROWS, 5,
                       PANEL_NOTIFY_PROC, mchoose_notify,
                       PANEL_CHOOSE_ONE, FALSE,
                       XV_X, xv_col(mchoose_panel, 0),
                       XV_Y, xv_rows(mchoose_panel, 1),
                       0);

   ok = xv_create(mchoose_panel, PANEL_BUTTON,
                  PANEL_LABEL_STRING, "OK",
                  PANEL_NOTIFY_PROC, mchoose_ok_notify,
                  XV_X, xv_col(mchoose_panel, 5),
                  XV_Y, xv_rows(mchoose_panel, 5),
                  NULL);
	
   cancel = xv_create(mchoose_panel, PANEL_BUTTON,
                      PANEL_LABEL_STRING, "Cancel",
                      PANEL_NOTIFY_PROC, mchoose_cancel_notify,
                      NULL);

   help = xv_create(mchoose_panel, PANEL_BUTTON,
                    PANEL_LABEL_STRING, "Help",
                    PANEL_NOTIFY_PROC, mchoose_help_notify,
                    NULL);

   window_fit(mchoose_panel);
   window_fit(mchoose_frame);

   /*
     Just to veto the xv_destroy(): 
   (void) notify_interpose_destroy_func(mchoose_frame,
                                        try_to_avoid_mchoose_destruction);
   xv_set((Window) xv_get(mchoose_frame, XV_XID)
                                        */
   /* Avoid the mchoose_frame destruction: */
   xv_set(mchoose_frame,
          FRAME_DONE_PROC, mchoose_frame_done_proc,
          NULL);
}
