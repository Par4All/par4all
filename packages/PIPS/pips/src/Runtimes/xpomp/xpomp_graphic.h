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
/* Define the display handler type: */
typedef int XPOMP_display;

/* To open a new display: */
extern XPOMP_display
XPOMP_open_display(int X_window_size,
		   int Y_window_size);

/* The user function to close a display window: */
extern void
XPOMP_close_display(XPOMP_display display);

/* Get the current default XPOMP display: */
extern int
XPOMP_get_current_default_display();

/* Set the current default XPOMP display and return the old one: */
extern int
XPOMP_set_current_default_display(XPOMP_display screen);

/* Get the pixel bit width of a display window: */
extern int
XPOMP_get_depth(XPOMP_display screen);

/* Select the color map: */
extern int
XPOMP_set_color_map(XPOMP_display screen,
		    int pal,
		    int cycle,
		    int start,
		    int clip);

/* Load a user defined colormap and select it: */
extern int 
XPOMP_set_user_color_map(XPOMP_display screen,
			 unsigned char * red,
			 unsigned char * green,
			 unsigned char * blue);

/* Wait for a mouse click: */
extern int
XPOMP_wait_mouse(XPOMP_display screen,
		 int * X,
		 int * Y,
		 int * state);

/* Just test for a mouse click: */
extern int
XPOMP_is_mouse(XPOMP_display screen,
	       int * X,
	       int * Y,
	       int * state);

/* The user interface to show something uncooked. Return -1 if it
   fails, 0 if not: */
extern int
XPOMP_flash(XPOMP_display screen,
	    unsigned char * data_array,
	    int X_data_array_size,
	    int Y_data_array_size,
	    int X_offset,
	    int Y_offset,
	    int X_zoom_ratio,
	    int Y_zoom_ratio);


/* Show a float array and interpolate color between min_value to
   max_value. If min_value = max_value = 0, automatically scale the
   colors: */
extern int
XPOMP_show_float(XPOMP_display screen,
		 const float * data_array,
		 const int X_data_array_size,
		 const int Y_data_array_size,
		 const int X_offset,
		 const int Y_offset,
		 const int X_zoom_ratio,
		 const int Y_zoom_ratio,
		 double min_value,
		 double max_value);

/* Show a double array and interpolate color between min_value to
   max_value. If min_value = max_value = 0, automatically scale the
   colors: */
extern int
XPOMP_show_double(XPOMP_display screen,
		  const double * data_array,
		  const int X_data_array_size,
		  const int Y_data_array_size,
		  const int X_offset,
		  const int Y_offset,
		  const int X_zoom_ratio,
		  const int Y_zoom_ratio,
		  double min_value,
		  double max_value);

/* Scroll a window: */
int
XPOMP_scroll(XPOMP_display screen,
	     int delta_Y);

/* Draw a frame from corner (X0,Y0) to corner (X1,Y1) and add a title: */
int
XPOMP_draw_frame(XPOMP_display screen,
		 char * title,
		 int title_color,
		 int background_color,
		 int X0,
		 int Y0,
		 int X1,
		 int Y1,
		 int color);

/* Print out a small help about keyboard usage in xPOMP: */
extern void
XPOMP_show_usage(void);
