#include <stdio.h>
#include <sys/types.h>
#include <sys/param.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>
#include <xview/panel.h>

#include "genC.h"
#include "misc.h"

#include "wpips.h"

extern char *getwd();



/* unmap a frame from the screen */
void hide_window(frame)
Frame frame;
{
    xv_set(frame, XV_SHOW, FALSE, NULL);
}



/* map a frame on the screen */
void unhide_window(frame)
Frame frame;
{
    xv_set(frame, FRAME_CLOSED, FALSE, NULL);
    xv_set(frame, XV_SHOW, TRUE, NULL);
}


/* Centre la souris sur une fene^tre : */
void pointer_in_center_of_frame(Frame frame)
{
  Display *dpy;
  Window xwindow;
  Rect rect;
    
  dpy = (Display *)xv_get(main_frame, XV_DISPLAY);
  xwindow = (Window) xv_get(frame, XV_XID);
  frame_get_rect(frame, &rect);
  XWarpPointer(dpy, None, xwindow, None, None, None, None, 
	       rect.r_width/2, rect.r_height/2);
}
