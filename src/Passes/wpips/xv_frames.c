#include <stdio.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>
#include <types.h>

#include "genC.h"
#include "wpips.h"
#include "xv_sizes.h"

#define BOUND(x,lb,ub) \
    ((x)<(lb)) ? (lb) :\
    ((x)>(ub)) ? (ub) : (x)
/*#define MAX(a, b) ((a)>(b) ? (a) : (b))*/

static int display_width, display_height;

void place_frame(frame, l, t)
Frame frame;
int l, t;
{
    Rect rect;

    frame_get_rect(frame, &rect);

		/* We need to estimate the size of the decor added by the widow
			manager, Y_WM_DECOR_SIZE & X_WM_DECOR_SIZE. RK, 9/10/1993. */
    rect.r_top = BOUND(t, 0, MAX(0,display_height-rect.r_height-Y_WM_DECOR_SIZE));
    rect.r_left = BOUND(l, 0, MAX(0,display_width-rect.r_width-X_WM_DECOR_SIZE));

    frame_set_rect(frame, &rect);
}

void create_frames()
{
	int i;
    Display *dpy;
    Xv_Screen screen;
    int screen_no,display_width,display_height;

    main_frame = xv_create(NULL, FRAME, 
			   FRAME_LABEL, "XView Pips", 
/*			   XV_WIDTH, WPIPS_WIDTH, 
			   XV_HEIGHT, WPIPS_HEIGHT,
	*/		   NULL);


    /* get the display dimensions */
    dpy = (Display *)xv_get(main_frame, XV_DISPLAY);
    screen = (Xv_Screen)xv_get(main_frame, XV_SCREEN);
    screen_no = (int)xv_get(screen, SCREEN_NUMBER);

    display_width = DisplayWidth(dpy, screen_no);
    display_height = DisplayHeight(dpy, screen_no);

    log_frame = xv_create(main_frame, FRAME, 
			       XV_SHOW, FALSE,
			       FRAME_DONE_PROC, close_log_subwindow,
			       XV_WIDTH, DIALOG_WIDTH, 
			       XV_HEIGHT, DIALOG_HEIGHT, 
			       NULL);

    xv_set(log_frame, FRAME_LABEL, "Pips Log Window", NULL);


		/* Footers added to edit window.
			RK, 21/05/1993. */
	for (i = 0; i < 2; i++)
    	edit_frame[i] = xv_create(main_frame, FRAME, 
			   XV_SHOW, FALSE,
			   FRAME_DONE_PROC, hide_window,
			   XV_WIDTH, EDIT_WIDTH, 
			   XV_HEIGHT, EDIT_HEIGHT, 
			   FRAME_SHOW_FOOTER, TRUE,
			   FRAME_LEFT_FOOTER, "<",
			   FRAME_RIGHT_FOOTER, ">",
			   NULL);


    help_frame = xv_create(main_frame, FRAME, 
			   FRAME_LABEL, "Pips On Line Help Facility",
			   XV_SHOW, FALSE,
			   FRAME_DONE_PROC, hide_window,
			   XV_WIDTH, HELP_WIDTH, 
			   XV_HEIGHT, HELP_HEIGHT, 
			   NULL);

    mchoose_frame = xv_create(main_frame, FRAME,
			      XV_SHOW, FALSE,
			      FRAME_DONE_PROC, hide_window,
			      NULL);

    schoose_frame = xv_create(main_frame, FRAME,
			      XV_SHOW, FALSE,
			      FRAME_DONE_PROC, hide_window,
			      NULL);

    query_frame = xv_create(main_frame, FRAME,
			    XV_SHOW, FALSE,
			    FRAME_DONE_PROC, hide_window,
			    XV_WIDTH, QUERY_WIDTH, 
			    XV_HEIGHT, QUERY_HEIGHT, 
			    NULL);

    properties_frame = xv_create(main_frame, FRAME,
				FRAME_LABEL, "Properties panel",
			      XV_SHOW, FALSE,
			      XV_WIDTH, display_width - EDIT_WIDTH -2*X_WM_DECOR_SIZE, 
			      FRAME_DONE_PROC, hide_window,
			      NULL);
}

void place_frames()
{
    Rect rect;
    int main_l, main_t, main_w, main_h;
    int main_center_l, main_center_t;
   
    Frame full_frame;
    Xv_Screen screen;
    Display *dpy;
    int screen_no;

    /* get the display dimensions */
    full_frame = (Frame) xv_create(XV_NULL, FRAME, NULL);
    dpy = (Display *)xv_get(full_frame, XV_DISPLAY);
    screen = (Xv_Screen)xv_get(full_frame, XV_SCREEN);
    screen_no = (int)xv_get(screen, SCREEN_NUMBER);
	xv_destroy(full_frame);

    display_width = DisplayWidth(dpy, screen_no);
    display_height = DisplayHeight(dpy, screen_no);

    /* warning: some window managers do NOT place the top frame (main_frame) 
       themselves. In this case add this fonction call and modify the call 
       to place_frames().
     
       place_frame(main_frame, 
		(display_width-WPIPS_WIDTH)/2, 
		(display_height-WPIPS_HEIGHT)/2);
     */
    place_frame(main_frame, 
		0, 
		(display_height-WPIPS_HEIGHT)/2);

    frame_get_rect(main_frame, &rect);

    main_t = rect.r_top;
    main_w = rect.r_width;
    main_h = rect.r_height;
    main_l = rect.r_left;

    main_center_t = main_t+main_h/2;
    main_center_l = main_l+main_w/2;

    /* in the bottom left : */
    place_frame(log_frame, 0, display_height);

    /* in the upper right corner : */
    place_frame(edit_frame[0], display_width, 0);

    /* in the bottom right corner : */
    place_frame(edit_frame[1], display_width, display_height);

    /* in the upper */
    place_frame(help_frame, 
		main_l + (main_w - HELP_WIDTH)/2,
		main_t - HELP_HEIGHT);

    /* in the upper left corner : */
    place_frame(mchoose_frame, 0, 0);

    /* in the upper left corner : */
    place_frame(schoose_frame, 0, 0);

    /* in the upper left corner */
    place_frame(query_frame, 
		main_l - QUERY_WIDTH + MIN(main_w, QUERY_WIDTH)/3, 
		main_t - QUERY_HEIGHT - Y_WM_DECOR_SIZE);
}
