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

/* Print out a small help about keyboard usage in xPOMP: */
extern void
XPOMP_show_usage(void);
