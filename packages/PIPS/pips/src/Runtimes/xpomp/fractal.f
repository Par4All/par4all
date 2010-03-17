! $Id$
!
! testing Xpomp display facilities in Fortran 77 / HPF
!
! (c) Ronan.Keryell@cri.ensmp.fr 1996
!
      program fractal

      implicit none

! XPOMP library interface:
      include 'xpomp_graphic_F.h'

! also tells that a file of stubs is needed:
!fcd$ stubs xpomp_stubs.f

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
      parameter(x_size = 800)     
      Parameter(y_size = 800)

! The zooming ratio to display this iteration space:      
      integer x_display_zoom, y_display_zoom
      parameter(x_display_zoom = 1)     
      parameter(y_display_zoom = 1)

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
      integer status, button, state
      integer X0, Y0, X1, Y1
      
! Some HPF distributions:
! cyclic would make more sense as far as load balancing is concerned
! However HPFC would not be very good at it...
! No communication -> block distribution on the second dimension
!   for best cache effects 
!
!hpf$ processors pe(number_of_processors())
!hpf$ distribute image(*,block) onto pe
      

! Initialize XPOMP
      call xpompf open display(x_display_size, y_display_size, display)
      call xpompf set color map(display, 1, 1, 0, 0, status)

      call xpompf show usage

      print *, 'Mouse button 1 to zoom in, 2 to recenter, 3 to zoom out'
      print *, 'A mouse + Shift: restart.'

! Initial position
      xcenter = 0
      ycenter = 0
      zoom = 5

! Main loop:
      counter = 0
 10   continue

!fcd$ time
! Compute a fractal image:
!hpf$ independent, new(cr,ci,zr,zi,d,zrp,k)
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

! Display the image:
      call xpompf flash(display, image, x_size, y_size, 
     &     0, 0, x_display_zoom, y_display_zoom, status)

!fcd$ end time ('Computation of one image')

! Wait for user interaction:
      call xpompf wait mouse(display, x, y, state, button)

! Recenter
      xcenter = xcenter + (x/x_display_zoom - x_size/2)*zoom/x_size
      ycenter = ycenter + (y/y_display_zoom - y_size/2)*zoom/y_size
      
      if (state .eq. 1) then

! A button with shift -> restart from the begining:
       xcenter = 0
       ycenter = 0
       zoom = 5
      else if (button .eq. 1) then

! Zoom in
         zoom = zoom/zooming_factor
! Display the frame that will be zoomed in:
         X0 = x - x_display_size/zooming_factor/2
         Y0 = y - y_display_size/zooming_factor/2
         X1 = x + x_display_size/zooming_factor/2
         Y1 = y + y_display_size/zooming_factor/2
         call xpompf draw frame(display, ' ', 255, -1,
     &        X0, Y0, X1, Y1, -181, status)
      else if (button .eq .3) then

! Zoom out
         zoom = zoom*zooming_factor
      endif

      print *, 'Position (', xcenter, ',', ycenter, '), zoom =', 
     &     zoom, ' button =', button, ', state =', state

! hpfc does not like infinite loop without an exit point...
! some bug related to the management of unstructured to be investigated
! I added this counter and exit as a temporary fix - FC.
      counter = counter+1
      if (counter.gt.100000) goto 20
      goto 10
 20   continue
      end

! end of it

