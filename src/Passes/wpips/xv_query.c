#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/text.h>

#include <types.h>

#include "xv_sizes.h"

#include "genC.h"
#include "misc.h"

#include "wpips.h"

static Panel_item query_pad;

static char *query_help_topic;
static success (*apply_on_query)();

void start_query(window_title, query_title, help_topic, func)
char *window_title, *query_title, *help_topic;
success (*func)();
{
    Display *dpy;
    Window query_xwindow;

    xv_set(query_frame, FRAME_LABEL, window_title, NULL);

    xv_set(query_pad, PANEL_LABEL_STRING, query_title, NULL);

    xv_set(query_pad, PANEL_VALUE, "", NULL);

    query_help_topic = help_topic;

    apply_on_query = func;

    unhide_window(query_frame);

    /* move the pointer to the center of the query window */
    dpy = (Display *)xv_get(main_frame, XV_DISPLAY);
    query_xwindow = (Window) xv_get(query_frame, XV_XID);
    XWarpPointer(dpy, None, query_xwindow, None, None, None, None, 
		 QUERY_WIDTH/2, QUERY_HEIGHT/2);
}

void query_canvas_event_proc(window, event)
Xv_Window window;
Event *event;
{
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

Xv_opaque end_query_notify(item, event)
Panel_item item;
Event *event;
{
    
    char *s = (char *) xv_get(query_pad, PANEL_VALUE);

    if (apply_on_query(s)) {
	hide_window(query_frame);
    }
}

Xv_opaque help_query_notify(item, event)
Panel_item item;
Event *event;
{
    display_help(query_help_topic);
}

Xv_opaque cancel_query_notify(item, event)
Panel_item item;
Event *event;
{
    hide_window(query_frame);
}

void create_query_window()
{
    query_panel = xv_create(query_frame, PANEL, NULL);

    xv_set(canvas_paint_window(query_panel), 
	   WIN_CONSUME_EVENT, LOC_WINENTER, 
/*	   WIN_IGNORE_X_EVENT_MASK, KeyReleaseMask, */
	   WIN_EVENT_PROC, query_canvas_event_proc, 
	   NULL);


    query_pad = xv_create(query_panel, PANEL_TEXT, 
			  PANEL_VALUE_DISPLAY_LENGTH, 20,
			  PANEL_VALUE_STORED_LENGTH, 128,
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

    (void) xv_create(query_panel, PANEL_BUTTON,
		     PANEL_LABEL_STRING, "Cancel",
		     PANEL_NOTIFY_PROC, cancel_query_notify,
		     NULL);
}
