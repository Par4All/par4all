! $RCSfile: xpomp_graphic_F.h,v $ (version $Revision$)
! $Date: 1996/08/31 13:16:18 $, 
!
!  The fortran headers for the XPOMP graphical library.
!
!  Have a look to show_fxpomp_graphic.h for a more precise description.
!  The FC IO directives tells HPFC that the directive is an IO...
!
! (c) Ronan.Keryell@cri.ensmp.fr 1996
!     Centre de Recherche en Informatique,
!     École des mines de Paris.
!
! To open a new display:
!fcd$ io xpomp_open_display

! The user function to close a display window:
!fcd$ io xpomp_close_display

! Get the current default HPFC display:
!fcd$ io xpomp_get_current_default_display

! Set the current default HPFC display and return the old one:
!fcd$ io xpomp_set_current_default_display

! Get the pixel bit width of a display window:
!fcd$ io xpomp_get_depth

! Select the color map
!fcd$ io xpomp_set_color_map

! Load a user defined colormap and select it:
!fcd$ io xpomp_set_user_color_map

! Wait for a mouse click:
!fcd$ io xpomp_wait_mouse

! Just test for a mouse click:
!fcd$ io xpomp_is_mouse

! The user interface to show something uncooked. 
! Return -1 if it fails, 0 if not:
!fcd$ io xpomp_flash

! Show a float/double array and interpolate color between min_value 
! and max_value. 
! If min_value = max_value = 0, automatically scale the colors.
!fcd$ io xpomp_show_real4
!fcd$ io xpomp_show_real8

! Print out a small help about keyboard usage in xPOMP:
!fcd$ io xpomp_show_usage

!
! that is all for $RCSfile: xpomp_graphic_F.h,v $
!
