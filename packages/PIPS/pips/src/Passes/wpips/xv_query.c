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
#include <stdarg.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>

#include "xv_sizes.h"

#include "genC.h"
#include "misc.h"

#include "wpips.h"

static Panel_item query_pad;
static Panel_item query_cancel_button;

static char *query_help_topic;
static success (*apply_on_query)(char *);

void
start_query(char * window_title,
            char * query_title,
            char * help_topic,
            success (* ok_func)(char *),
            void (* cancel_func)(Panel_item, Event *))
{
   xv_set(query_frame, FRAME_LABEL, window_title, NULL);
   /*	     PANEL_NOTIFY_PROC, cancel_query_notify, */

   if (cancel_func == NULL)
      /* No cancel button requested: */
      xv_set(query_cancel_button, 
             PANEL_INACTIVE, TRUE,
             NULL);
   else
      xv_set(query_cancel_button, PANEL_NOTIFY_PROC, cancel_func,
             PANEL_INACTIVE, FALSE,
             NULL);

   xv_set(query_pad, PANEL_LABEL_STRING, query_title, NULL);

   xv_set(query_pad, PANEL_VALUE, "", NULL);

   query_help_topic = help_topic;

   apply_on_query = ok_func;

   unhide_window(query_frame);

   /* move the pointer to the center of the query window */
   pointer_in_center_of_frame(query_frame);

   /* Valide le focus sur l'entrée : */
   /* J'ai l'impression que le code pre'ce'dent est pluto^t obsole`te... RK. */
   /* Mais pourquoi cela ne marche pas ??? */
   win_set_kbd_focus(query_frame, xv_get(query_panel, XV_XID));
   /* xv_set(query_xwindow , WIN_SET_FOCUS, NULL); */
   xv_set(query_panel, PANEL_CARET_ITEM, query_pad, NULL);
}

void query_canvas_event_proc(window, event)
Xv_Window window;
Event *event;
{
  debug_on("WPIPS_EVENT_DEBUG_LEVEL");
  debug(2,"query_canvas_event_proc",
	"Event_id %d, event_action %d\n",
	event_id(event), event_action(event));
  debug_off();
    switch(event_id(event)) {
        case LOC_WINENTER :
	  /* enter_window(window); */
	  break;
	case '\r' :
	  /* ie. return key pressed */
	  if (event_is_up(event)) 
	      /* ie. key is released. It is necessary to use this event
		 because notice_prompt() (in prompt_user() (in 
		 end_query_notify() )) also returns on up RETURN.
		 This can cause the notice to return immediately when it is 
		 called on down RETURN.
		 There schould be another possibility: put a mask to ignore
		 key release events on the window which owns notice_prompt().
		 This was done in create_main_window() but seems without 
		 effect.
		*/
	      end_query_notify(NULL, event);
	  break;
	default : ;
      }
}

/* Pour debug seulement : */
void end_query_pad_notify(Panel_item item, Event *event)
{
  debug_on("WPIPS_EVENT_DEBUG_LEVEL");
  debug(2,"find_dead_code",
	"end_query_pad_notify: Event_id %d, event_action %d\n",
	event_id(event), event_action(event));
  debug_off();
}


void end_query_notify(item, event)
Panel_item item;
Event *event;
{
    
    char *s = (char *) xv_get(query_pad, PANEL_VALUE);

    /* Dans le cas ou` on vient d'un retour charriot dans le texte : */
/* Cela ne peut bien entendu pas marcher... :-( */
/*
    xv_set(xv_get(query_panel, PANEL_DEFAULT_ITEM, NULL),
	   PANEL_BUSY, FALSE,
	   NULL);
*/    
    if (apply_on_query(s)) {
	hide_window(query_frame);
	/* Remet le bouton OK a l'e'tat normal : */
/*	xv_set(xv_get(query_panel, PANEL_DEFAULT_ITEM, NULL),
	       PANEL_BUSY, FALSE,
	       NULL);
*/
    }
}

void help_query_notify(item, event)
Panel_item item;
Event *event;
{
    display_help(query_help_topic);
}

/* Ne fait rien d'autre que de fermer la fene^tre... */
void
cancel_query_notify(Panel_item item,
                    Event * event)
{
   hide_window(query_frame);
}


/* Cancel clear the string value and return: */
void
cancel_user_request_notify(Panel_item item,
                           Event * event)
{
   xv_set(query_pad, PANEL_VALUE, "", NULL);
   hide_window(query_frame);
   /* Just return the "": */
   xv_window_return((Xv_opaque) xv_get(query_pad, PANEL_VALUE));
}


success
end_user_request_notify(char * the_answer)
{
   hide_window(query_frame);
   xv_window_return((Xv_opaque) the_answer);
   /* In fact, the function above never return... */
   return TRUE;
}


string
wpips_user_request(char * a_printf_format,
                   va_list args)
{
   char * the_answer;
   
   static char message_buffer[SMALL_BUFFER_LENGTH];

   (void) vsprintf(message_buffer, a_printf_format, args);

   start_query("User Query",
               message_buffer, 
               "UserQuery",
               end_user_request_notify,
               cancel_user_request_notify);

   user_log("User Request...\n");

   /* Loop on the Query window: */
   the_answer = (string) xv_window_loop(query_frame);

   /* Log the answer for possible rerun through tpips: */
   user_log("%s\n\"%s\"\nEnd User Request\n",
            message_buffer, the_answer);
   
   /* returns a newly allocated string...
    */
   if (the_answer && !string_undefined_p(the_answer))
       the_answer = strdup(the_answer);

   return the_answer;
}


void create_query_window()
{
    query_panel = xv_create(query_frame, PANEL, NULL);

/* Semble ne servir a` rien. RK, 9/11/93. */
    xv_set(canvas_paint_window(query_panel), 
	   WIN_CONSUME_EVENT, LOC_WINENTER, NULL, 
/*	   WIN_IGNORE_X_EVENT_MASK, KeyReleaseMask, */
	   WIN_EVENT_PROC, query_canvas_event_proc, 
	   NULL);


    query_pad = xv_create(query_panel, PANEL_TEXT, 
			  PANEL_VALUE_DISPLAY_LENGTH, 20,
			  PANEL_VALUE_STORED_LENGTH, 128,
			  PANEL_NOTIFY_PROC, end_query_notify,
			  NULL);

    xv_set(query_panel, PANEL_DEFAULT_ITEM,
		xv_create(query_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, " OK ",
		     XV_Y, xv_row(query_panel, 2),
		     XV_X, QUERY_WIDTH/2-100,
		     PANEL_NOTIFY_PROC, end_query_notify,
		     NULL),NULL);

    (void) xv_create(query_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Help",
		     PANEL_NOTIFY_PROC, help_query_notify,
		     NULL);

    query_cancel_button = xv_create(query_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Cancel",
		     NULL);
}
