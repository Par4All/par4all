! $RCSfile: wave_parameters.h,v $ (version $Revision$)
! $Date: 1996/09/03 23:19:57 $, 
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

! end of $RCSfile: wave_parameters.h,v $
