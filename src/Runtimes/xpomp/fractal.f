      program fractal

      implicit none

      include 'xpomp_graphic_F.h'
      
!     Algorithm parameters:
      integer n_iteration
      parameter (n_iteration = 256)
      real*8 big_value
      parameter (big_value = 100)
!     distorsion = 2 to have normal Mandelbrot:
      real*8 distorsion
      parameter (distorsion = 2)

!     Size of the iteration space:
      integer x_size, y_size
      parameter(x_size = 200)     
      parameter(y_size = 200)

!     The zooming ratio to display this iteration space:      
      integer x_display_zoom, y_display_zoom
      parameter(x_display_zoom = 2)     
      parameter(y_display_zoom = 2)

!     The zooming factor used when zooming with the button:
      real*8 zooming_factor
      parameter (zooming_factor = 4)

      integer x_display_size, y_display_size
      parameter(x_display_size = x_size*x_display_zoom)     
      parameter(y_display_size = y_size*y_display_zoom)
      
      integer k, x, y
      integer display
      character image(0:x_size - 1, 0:y_size - 1)
      
      real*8 zoom, xcenter, ycenter
      real*8 zr, zrp, zi, cr, ci, d
      integer status, button
      
! Some HPF distributions:
! cyclic would make more sense as far as load balancing is concerned
! However HPFC would not be very good at it...
!hpf$ processor pe(2, 2)
!hpf$ template space(0:x_size - 1, 0:y_size - 1)
!hpf$ distribute space(block, block) onto pe
!hpf$ align image with space
      
      display = xpomp_open_display(x_display_size, y_display_size)
      status = xpomp_set_color_map(display, 1, 1, 0, 0)
      
      xcenter = 0
      ycenter = 0
      zoom = 5

      call xpomp_show_usage

      print *, 'Use mouse button 1 to zoom in, button 2 to recenter'
      print *, '    and button 3 to zoom out'

!     Main loop:
 10   continue

!     Compute a fractal image:
chpf$ independent, new(cr,ci,zr,zi,d,zrp)
      do x = 0, x_size - 1
         cr = xcenter + (x - x_size/2)*zoom/x_size
chpf$ independent
         do y = 0, y_size - 1      
            ci = ycenter + (y - y_size/2)*zoom/y_size
            zr = cr
            zi = ci
           do k = 0, n_iteration - 1
               d = zr*zr + zi*zi
               zrp = zr*zr - zi*zi + cr
               zi = distorsion*zr*zi + ci
               zr = zrp
               if (d .gt. big_value) goto 300
            enddo
 300        continue
            image(x, y) = CHAR(k)
         enddo
      enddo

c     Display the image:
      status = xpomp_flash(display, image, x_size, y_size, 
     &     0, 0, x_display_zoom, y_display_zoom)

c     Wait for user interaction:
      button = xpomp_wait_mouse(display, x, y)
      xcenter = xcenter + (x/x_display_zoom - x_size/2)*zoom/x_size
      ycenter = ycenter + (y/y_display_zoom - y_size/2)*zoom/y_size
      
      if (button.eq.1) then
       zoom = zoom/zooming_factor
      else if (button.eq.3) then
       zoom = zoom*zooming_factor
      endif

      print *, 'Position (', xcenter, ',', ycenter, '), zoom =', zoom
      goto 10

      end
