#include <stdio.h>
#include <sys/types.h>
#include <sys/param.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <xview/xview.h>

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
    xv_set(frame, XV_SHOW, TRUE, NULL);
}
