! $RCSfile: xpomp_graphic_F.h,v $ (version $Revision$)
! $Date: 1996/08/30 19:55:58 $, 
!
!  The fortran headers for the HPFC graphical library.

c     Have a look to xpomp_graphic.h to have 
c     a more precise description.
c
c     Ronan.Keryell@cri.ensmp.fr
c
c     To open a new display:
!fcd$ io xpomp_open_display
      integer xpomp_open_display
      external xpomp_open_display

c     The user function to close a display window:
c     xpomp_close_display
!fcd$ io xpomp_close_display

c     Get the current default HPFC display:
!fcd$ io xpomp_get_current_default_display
      integer xpomp_get_current_default_display
      external xpomp_get_current_default_display

c     Set the current default HPFC display and return the old one:
!fcd$ io xpomp_set_current_default_display
      integer xpomp_set_current_default_display
      external xpomp_set_current_default_display

c     Get the pixel bit width of a display window:
!fcd$ io xpomp_get_depth
      integer xpomp_get_depth
      external xpomp_get_depth

c     Select the color map
!fcd$ io xpomp_set_color_map
      integer xpomp_set_color_map
      external xpomp_set_color_map

c     Load a user defined colormap and select it:
!fcd$ io xpomp_set_user_color_map
      integer xpomp_set_user_color_map
      external xpomp_set_user_color_map

c     Wait for a mouse click:
!fcd$ io xpomp_wait_mouse
      integer xpomp_wait_mouse
      external xpomp_wait_mouse

c     Just test for a mouse click:
!fcd$ io xpomp_is_mouse
      integer xpomp_is_mouse
      external xpomp_is_mouse

c     The user interface to show something uncooked. 
c     Return -1 if it fails, 0 if not:
!fcd$ io xpomp_flash
      integer xpomp_flash
      external xpomp_flash

c     Show a float array and interpolate color between min_value
c     tomax_value. If min_value = max_value = 0, 
c     automatically scale the colors:
!fcd$ io xpomp_show_float
      integer xpomp_show_float
      external xpomp_show_float

c     Show a double array and interpolate color between 
c     min_value to max_value. If min_value = max_value = 0, 
c     automatically scale the colors:
!fcd$ io xpomp_show_double
      integer xpomp_show_double
      external xpomp_show_double

c      Print out a small help about keyboard usage in xHPFC:
c     xpomp_show_usage
!fcd$ io xpomp_show_usage

!
! that is all for $RCSfile: xpomp_graphic_F.h,v $
!
