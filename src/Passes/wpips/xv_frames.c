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

    rect.r_top = BOUND(t, 0, MAX(0,display_height-rect.r_height));
    rect.r_left = BOUND(l, 0, MAX(0,display_width-rect.r_width));

    frame_set_rect(frame, &rect);
}

void create_frames()
{
    main_frame = xv_create(NULL, FRAME, 
			   FRAME_LABEL, "XView Pips", 
/*			   XV_WIDTH, WPIPS_WIDTH, 
			   XV_HEIGHT, WPIPS_HEIGHT,
	*/		   NULL);

    log_frame = xv_create(main_frame, FRAME, 
			       XV_SHOW, FALSE,
			       FRAME_DONE_PROC, close_log_subwindow,
			       XV_WIDTH, DIALOG_WIDTH, 
			       XV_HEIGHT, DIALOG_HEIGHT, 
			       0);

    xv_set(log_frame, FRAME_LABEL, "Pips Log Window", NULL);


    edit_frame[0] = xv_create(main_frame, FRAME, 
			   XV_SHOW, FALSE,
			   FRAME_DONE_PROC, hide_window,
			   XV_WIDTH, EDIT_WIDTH, 
			   XV_HEIGHT, EDIT_HEIGHT, 
			   0);
    edit_frame[1] = xv_create(main_frame, FRAME, 
			   XV_SHOW, FALSE,
			   FRAME_DONE_PROC, hide_window,
			   XV_WIDTH, EDIT_WIDTH, 
			   XV_HEIGHT, EDIT_HEIGHT, 
			   0);


    help_frame = xv_create(main_frame, FRAME, 
			   FRAME_LABEL, "Pips On Line Help Facility",
			   XV_SHOW, FALSE,
			   FRAME_DONE_PROC, hide_window,
			   XV_WIDTH, HELP_WIDTH, 
			   XV_HEIGHT, HELP_HEIGHT, 
			   0);

    mchoose_frame = xv_create(main_frame, FRAME,
			      XV_SHOW, FALSE,
			      FRAME_DONE_PROC, hide_window,
			      0);

    schoose_frame = xv_create(main_frame, FRAME,
			      XV_SHOW, FALSE,
			      FRAME_DONE_PROC, hide_window,
			      0);

    query_frame = xv_create(main_frame, FRAME,
			    XV_SHOW, FALSE,
			    FRAME_DONE_PROC, hide_window,
			    XV_WIDTH, QUERY_WIDTH, 
			    XV_HEIGHT, QUERY_HEIGHT, 
			    0);
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

    display_width = DisplayWidth(dpy, screen_no);
    display_height = DisplayHeight(dpy, screen_no);

    /* warning: some window managers do NOT place the top frame (main_frame) 
       themselves. In this case add this fonction call and modify the call 
       to place_frames().
     
       place_frame(main_frame, 
		(display_width-WPIPS_WIDTH)/2, 
		(display_height-WPIPS_HEIGHT)/2);
     */

    frame_get_rect(main_frame, &rect);

    main_t = rect.r_top;
    main_w = rect.r_width;
    main_h = rect.r_height;
    main_l = rect.r_left;

    main_center_t = main_t+main_h/2;
    main_center_l = main_l+main_w/2;

    /* in the bottom */
    place_frame(log_frame, 
		main_l,
		main_t + main_h);

    /* in the bottom left corner */
    /*place_frame(edit_frame, 
		main_l - EDIT_WIDTH + MIN(main_w, EDIT_WIDTH)/3, 
		main_t + main_h - MIN(main_h, EDIT_HEIGHT)/3);
		*/
    /* in the upper right corner */
    place_frame(edit_frame[0], 
		main_l + main_w - MIN(main_w, EDIT_WIDTH)/3,
		main_t - EDIT_HEIGHT + MIN(main_h, EDIT_HEIGHT)/3);

    /* in the bottom right corner */
    place_frame(edit_frame[1], 
		main_l + main_w - MIN(main_w, EDIT_WIDTH)/3, 
		main_t + main_h - MIN(main_h, EDIT_HEIGHT)/3);

    /* in the upper right corner */
    /*place_frame(help_frame, 
		main_l + main_w - MIN(main_w, HELP_WIDTH)/3,
		main_t - HELP_HEIGHT + MIN(main_h, HELP_HEIGHT)/3);
		*/
    /* in the upper */
    place_frame(help_frame, 
		main_l + (main_w - HELP_WIDTH)/2,
		main_t - HELP_HEIGHT);

    /* in the upper left corner */
    place_frame(mchoose_frame, 
		main_l - MULTIPLE_CHOICE_WIDTH + 
		  MIN(main_w, MULTIPLE_CHOICE_WIDTH)/3, 
		main_t - MULTIPLE_CHOICE_HEIGHT + 
		  MIN(main_h, MULTIPLE_CHOICE_HEIGHT)/3);

    /* in the upper left corner */
    place_frame(schoose_frame, 
		main_l - SINGLE_CHOICE_WIDTH + 
		  MIN(main_w, SINGLE_CHOICE_WIDTH)/3, 
		main_t - SINGLE_CHOICE_HEIGHT + 
		  MIN(main_h, SINGLE_CHOICE_HEIGHT)/3);

    /* in the upper left corner */
    place_frame(query_frame, 
		main_l - QUERY_WIDTH + MIN(main_w, QUERY_WIDTH)/3, 
		main_t - QUERY_HEIGHT + MIN(main_h, QUERY_HEIGHT)/3);
}
