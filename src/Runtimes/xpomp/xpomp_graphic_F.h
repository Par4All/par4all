c       The fortran headers for the HPFC graphical library.

c       Have a look to hpfc_graphic.h to have 
c       a more precise description.

c       Ronan.Keryell@cri.ensmp.fr

c       To open a new display:
	integer hpfc_open_display
	external hpfc_open_display

c       The user function to close a display window:
c	hpfc_close_display

c       Get the current default HPFC display:
	integer hpfc_get_current_default_display
	external hpfc_get_current_default_display

c       Set the current default HPFC display and return the old one:
	integer hpfc_set_current_default_display
	external hpfc_set_current_default_display

c       Get the pixel bit width of a display window:
	integer hpfc_get_depth
	external hpfc_get_depth

c       Select the color map
	integer hpfc_set_color_map
	external hpfc_set_color_map

c       Load a user defined colormap and select it:
	integer hpfc_set_user_color_map
	external hpfc_set_user_color_map

c       Wait for a mouse click:
	integer hpfc_wait_mouse
	external hpfc_wait_mouse

c       Just test for a mouse click:
	integer hpfc_is_mouse
	external hpfc_is_mouse

c       The user interface to show something uncooked. 
c	Return -1 if it fails, 0 if not:
	integer hpfc_flash
	external hpfc_flash

c       Show a float array and interpolate color between min_value
c       tomax_value. If min_value = max_value = 0, 
c       automatically scale the	colors:
	integer hpfc_show_float
	external hpfc_show_float

c       Show a double array and interpolate color between 
c       min_value to max_value. If min_value = max_value = 0, 
c       automatically scale the colors:
	integer hpfc_show_double
	external hpfc_show_double

c	Print out a small help about keyboard usage in xHPFC:
c       hpfc_show_usage
