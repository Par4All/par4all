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
! size parameters for a simple wave propagation
!

      implicit none

! Computed area
      integer maxiter
      parameter(maxiter = 1024)

      integer x_size, y_size
      parameter(x_size = 128)
      parameter(y_size = 128)

! various physical (?) constants
      real*8 g, c
      parameter(c = 1.0/300.0)
      parameter(g = 9.81)

      real*8 friction
      parameter(friction = 0.98)

      real*8 alpha
      parameter(alpha = c*g)

! DROP description
      integer radius
      parameter(radius = 5)

      real*8 height
      parameter(height = 5.0)

! XPOMP settings
      integer x_display_zoom, y_display_zoom
      parameter(x_display_zoom = 2)
      parameter(y_display_zoom = 2)

      integer x_display_size, y_display_size
      parameter(x_display_size = x_size*x_display_zoom)     
      parameter(y_display_size = y_size*y_display_zoom)

      integer zero_color
      parameter(zero_color = 128)

! HPF objects
!hpf$ template t(x_size,y_size)
!hpf$ processors p(2,2)
!hpf$ distribute t(block,block) onto p

! end of it
