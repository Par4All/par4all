 /* Multiple choices handling */
#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <types.h>

#include "genC.h"
#include "misc.h"

#include "wpips.h"

static Panel_item mchoices, choices, ok, help;
static void (*apply_on_mchoices)();

static void mchoose_help_notify(item, event)
Panel_item item;
Event *event;
{
    display_help("MultipleChoice");
}

static void ok_notify(item, event)
Panel_item item;
Event *event;
{
    char *mchoices_args[ARGS_LENGTH];
    int mchoices_length = 0;

    int i, nchoices;

    hide_window(mchoose_frame);

    nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);
    mchoices_length = 0;

    for (i = 0; i < nchoices; i++) {
	if ((int) xv_get(choices, PANEL_LIST_SELECTED, i) == TRUE) {
	    char *s = strdup(xv_get(choices, PANEL_LIST_STRING, i));
	    args_add(&mchoices_length, mchoices_args, s);		     
	}
    }

    for (i = 0; i < nchoices; i++) {
	xv_set(choices, PANEL_LIST_DELETE, 0, 0);
    }

    xv_set(mchoices, PANEL_VALUE, "", 0);

    (*apply_on_mchoices)(&mchoices_length, mchoices_args);

    args_free(&mchoices_length, mchoices_args);
}



static void mchoices_notify(item, op, event)
Panel_item item;
Panel_list_op op;
Event *event;
{
    char buffer[SMALL_BUFFER_LENGTH];

    int nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);

    buffer[0] = '\0';

    while (nchoices--) {
	if ((int) xv_get(choices, PANEL_LIST_SELECTED, nchoices) == TRUE) {
	    strcat(buffer, xv_get(choices, PANEL_LIST_STRING, nchoices));
	    strcat(buffer, " ");
	}
    }

    xv_set(mchoices, PANEL_VALUE, buffer, 0);
}



void mchoose(title, argc, argv, f)
char *title;
int argc;
char *argv[];
void (*f)();
{
    int i;
    int nchoices;

    apply_on_mchoices = f;

    xv_set(mchoose_frame, FRAME_LABEL, title, 0);

    /* reset the choice set to empty */
    nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);

    for (i = 0; i < nchoices; i++) {
	xv_set(choices, PANEL_LIST_DELETE, 0, 0);
    }

    for (i = 0; i < argc; i++) {
	xv_set(choices, PANEL_LIST_STRING, i, argv[i], 0);
    }

    unhide_window(mchoose_frame);
}


void create_mchoose_window()
{
    mchoose_panel = xv_create(mchoose_frame, PANEL, 0);

    mchoices = xv_create(mchoose_panel, PANEL_TEXT,
			 PANEL_LABEL_STRING, "Current choices",
			 PANEL_VALUE_DISPLAY_LENGTH, 30,
			 PANEL_VALUE_STORED_LENGTH, 128,
			 XV_X, 10,
			 XV_Y, 10,
			 0);

    choices = xv_create(mchoose_panel, PANEL_LIST,
			PANEL_LABEL_STRING, "Available choices",
			PANEL_LIST_DISPLAY_ROWS, 5,
			PANEL_NOTIFY_PROC, mchoices_notify,
			PANEL_CHOOSE_ONE, FALSE,
			XV_X, 10,
			XV_Y, 40,
			0);

    ok = xv_create(mchoose_panel, PANEL_BUTTON,
		   PANEL_LABEL_STRING, "OK",
		   PANEL_NOTIFY_PROC, ok_notify,
		   XV_X, 320,
		   XV_Y, 40,
		   0);

    help = xv_create(mchoose_panel, PANEL_BUTTON,
		   PANEL_LABEL_STRING, "HELP",
		   PANEL_NOTIFY_PROC, mchoose_help_notify,
		   XV_X, 320,
		   XV_Y, 70,
		   0);

    window_fit(mchoose_panel);
    window_fit(mchoose_frame);
}
