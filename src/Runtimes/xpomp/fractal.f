! testing Xpomp display facilities in fortran
!
! (c) Ronan.Keryell@cri.ensmp.fr 1996
!
! $RCSfile: fractal.f,v $ (version $Revision$)
! $Date: 1996/09/01 21:51:06 $, 
!
      program fractal

      implicit none

! XPOMP library interface:
      include 'xpomp_graphic_F.h'

! Algorithm parameters:
      integer n_iteration
      parameter (n_iteration = 256)
      real*8 big_value
      parameter (big_value = 100)
! distorsion = 2 to have normal Mandelbrot:
      real*8 distorsion
      parameter (distorsion = 2)

! Size of the iteration space:
      integer x_size, y_size
      parameter(x_size = 400)     
      Parameter(y_size = 400)

! The zooming ratio to display this iteration space:      
      integer x_display_zoom, y_display_zoom
      parameter(x_display_zoom = 2)     
      parameter(y_display_zoom = 2)

! The zooming factor used when zooming with the button:
      real*8 zooming_factor
      parameter (zooming_factor = 4)

      integer x_display_size, y_display_size
      parameter(x_display_size = x_size*x_display_zoom)     
      parameter(y_display_size = y_size*y_display_zoom)
      
      integer counter
      integer k, x, y
      integer display
      character image(0:x_size-1, 0:y_size-1)
      
      real*8 zoom, xcenter, ycenter
      real*8 zr, zrp, zi, cr, ci, d
      integer status, button
      
! Some HPF distributions:
! cyclic would make more sense as far as load balancing is concerned
! However HPFC would not be very good at it...
      integer nproc
      parameter (nproc=5)
!hpf$ processors pe(nproc)
!hpf$ template space(0:x_size - 1, 0:y_size - 1)
!hpf$ distribute space(*,block) onto pe
!hpf$ align image with space
      

! Initialize XPOMP
      call xpomp_open_display(x_display_size, y_display_size, display)
      call xpomp_set_color_map(display, 1, 1, 0, 0, status)

      call xpomp_show_usage

      print *, 'Mouse button 1 to zoom in, 2 to recenter, 3 to zoom out'

! Initial position
      xcenter = 0
      ycenter = 0
      zoom = 5

! Main loop:
      counter = 0
 10   continue

!!!fcd$ time
! Compute a fractal image:
!hpf$ independent, new(cr,ci,zr,zi,d,zrp)
      do y = 0, y_size - 1      
         ci = ycenter + (y - y_size/2)*zoom/y_size
!hpf$    independent
         do x = 0, x_size - 1
            cr = xcenter + (x - x_size/2)*zoom/x_size
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
!!!fcd$ end time ('Computation of one image')

! Display the image:
      call xpomp_flash(display, image, x_size, y_size, 
     &     0, 0, x_display_zoom, y_display_zoom, status)

! Wait for user interaction:
      call xpomp_wait_mouse(display, x, y, button)
      xcenter = xcenter + (x/x_display_zoom - x_size/2)*zoom/x_size
      ycenter = ycenter + (y/y_display_zoom - y_size/2)*zoom/y_size
      
      if (button.eq.1) then
         zoom = zoom/zooming_factor
      else if (button.eq.3) then
         zoom = zoom*zooming_factor
      endif

      print *, 'Position (', xcenter, ',', ycenter, '), zoom =', zoom

! hpfc does not like infinite loop without an exit point...
! some bug related to the management of unstructured to be investigated
! I added this counter and exit as a temporary fix - FC.
      counter = counter+1
      if (counter.gt.100000) goto 20
      goto 10
 20   continue
      end

! end of $RCSfile: fractal.f,v $

