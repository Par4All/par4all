/* Define the display handler type: */
typedef int HPFC_display;

/* To open a new display: */
extern HPFC_display
HPFC_open_display(int X_window_size,
		  int Y_window_size);

/* The user function to close a display window: */
extern void
HPFC_close_display(HPFC_display display);

/* Get the current default HPFC display: */
extern int
HPFC_get_current_default_display();

/* Set the current default HPFC display and return the old one: */
extern int
HPFC_set_current_default_display(HPFC_display screen);

/* Get the pixel bit width of a display window: */
extern int
HPFC_get_depth(HPFC_display screen);

/* Select the color map: */
extern int
HPFC_set_color_map(HPFC_display screen,
		   int pal,
		   int cycle,
		   int start,
		   int clip);

/* Load a user defined colormap and select it: */
extern int 
HPFC_set_user_color_map(HPFC_display screen,
			unsigned char * red,
			unsigned char * green,
			unsigned char * blue);

/* Wait for a mouse click: */
extern int
HPFC_wait_mouse(HPFC_display screen,
		int * X,
		int * Y);

/* Just test for a mouse click: */
extern int
HPFC_is_mouse(HPFC_display screen,
	      int * X,
	      int * Y);

/* The user interface to show something uncooked. Return -1 if it
   fails, 0 if not: */
extern int
HPFC_flash(HPFC_display screen,
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
HPFC_show_float(HPFC_display screen,
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
HPFC_show_double(HPFC_display screen,
		 const double * data_array,
		 const int X_data_array_size,
		 const int Y_data_array_size,
		 const int X_offset,
		 const int Y_offset,
		 const int X_zoom_ratio,
		 const int Y_zoom_ratio,
		 double min_value,
		 double max_value);

/* Print out a small help about keyboard usage in xHPFC: */
extern void
HPFC_show_usage(void);
