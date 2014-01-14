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
