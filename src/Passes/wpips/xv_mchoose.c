 /* Multiple choices handling */
#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/notice.h>
#include <types.h>

#include "genC.h"
#include "misc.h"

#include "wpips.h"

static Panel_item mchoices, choices, ok, help;
static char mchoices_notify_buffer[SMALL_BUFFER_LENGTH];

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
	char buffer[SMALL_BUFFER_LENGTH];
    int mchoices_length = 0;
    int i, nchoices, len;
	int item_is_in_the_list;
	char *p;

    nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);
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
	strcpy(mchoices_notify_buffer, (char *)xv_get(mchoices,PANEL_VALUE));
	p = mchoices_notify_buffer;
	while(sscanf(p,"%s%n",&buffer,&len) == 1) {
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
		int result;
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

	for (i = 0; i < nchoices; i++) {
	xv_set(choices, PANEL_LIST_DELETE, 0, 0);
	}

	xv_set(mchoices, PANEL_VALUE, "", 0);

    hide_window(mchoose_frame);
}



static void mchoices_notify(item, op, event)
Panel_item item;
Panel_list_op op;
Event *event;
{
    int nchoices = (int) xv_get(choices, PANEL_LIST_NROWS, 0);

    mchoices_notify_buffer[0] = '\0';

	/* Now it is mchoices_notify_buffer which is used for the selection.
		No size verification implemented yet... :-)
		RK, 19/05/1993. */

    while (nchoices--) {
	if ((int) xv_get(choices, PANEL_LIST_SELECTED, nchoices) == TRUE) {
	    strcat(mchoices_notify_buffer,
			xv_get(choices, PANEL_LIST_STRING, nchoices));
	    strcat(mchoices_notify_buffer, " ");
	}
    }

    xv_set(mchoices, PANEL_VALUE, mchoices_notify_buffer, 0);
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
    mchoices_notify_buffer[0] = '\0';

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

/* Carriage return enable for "OK". RK, 19/05/1993. */

    mchoices = xv_create(mchoose_panel, PANEL_TEXT,
			 PANEL_LABEL_STRING, "Current choices",
			 PANEL_VALUE_DISPLAY_LENGTH, 30,
			 PANEL_VALUE_STORED_LENGTH, 128,
			 PANEL_NOTIFY_LEVEL,PANEL_SPECIFIED,
			 PANEL_NOTIFY_STRING,"\n",
			PANEL_NOTIFY_PROC, ok_notify,
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
