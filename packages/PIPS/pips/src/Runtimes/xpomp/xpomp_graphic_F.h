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
! 
! $Id$
!
!  The fortran headers for the XPOMP graphical library.
!
!  Have a look to xpomp_graphic.h for a more precise description.
!  The FC IO directives tells HPFC that the directive is an IO...
!
! (c) Ronan.Keryell@cri.ensmp.fr 1996
!     Centre de Recherche en Informatique,
!     École des mines de Paris.
!

! Tells HPFC where to link (Link eDitor I/O):
!fcd$ ld io -L$XPOMP_RUNTIME/$PIPS_ARCH -L$PIPS_ROOT/Runtime/xpomp/$PIPS_ARCH -lxpomp

! To open a new display:
!fcd$ io xpompf open display

! The user function to close a display window:
!fcd$ io xpompf close display

! Get the current default HPFC display:
!fcd$ io xpompf get current default display

! Set the current default HPFC display and return the old one:
!fcd$ io xpompf set current default display

! Get the pixel bit width of a display window:
!fcd$ io xpompf get depth

! Select the color map
!fcd$ io xpompf set color map

! Load a user defined colormap and select it:
!fcd$ io xpompf set user color map

! Wait for a mouse click:
!fcd$ io xpompf wait mouse

! Just test for a mouse click:
!fcd$ io xpompf is mouse

! The user interface to show something uncooked. 
! Return -1 if it fails, 0 if not:
!fcd$ io xpompf flash

! Show a float/double array and interpolate color between min value 
! and max value. 
! If min value = max value = 0, automatically scale the colors.
!fcd$ io xpompf show real4
!fcd$ io xpompf show real8

! Scroll a window:
!fcd$ io xpompf scroll

! Draw a frame from corner (X0,Y0) to corner (X1,Y1) and add a title:
!fcd$ io xpompf draw frame

! Print out a small help about keyboard usage in xPOMP:
!fcd$ io xpompf show usage

!
! that is all
!
