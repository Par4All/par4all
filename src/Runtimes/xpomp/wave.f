! Wave propagation in HPF with xpomp
!
! Ronan KERYELL (POMP C version) and 
! Fabien COELHO (HPF version)
!
! $RCSfile: wave.f,v $ (version $Revision$)
! $Date: 1996/09/03 23:19:56 $, 
!

!
! Initialize areas
!
      subroutine init_area(depth, area)

      include 'wave_parameters.h'

      real*8 
     $     depth(x_size,y_size),
     $     area(x_size,y_size)

!hpf$ align with t:: depth, area

      integer i, j

! area  = c      and 0.0 on [1,:],[:,1] 
! depth = 2600.0 and 800.0 on [:,1:32]

!hpf$ independent
      do i=1, x_size
         area(i, j) = 0.0
      end do

!hpf$ independent
      do j=2, y_size
!hpf$    independent
         do i=2, x_size
            area(i,j) = c
         end do
      end do

!hpf$ independent
      do j=2, y_size
         area(1, j) = 0.0
      end do

!hpf$ independent 
      do i=1, x_size/4
         area(i,y_size/2) = 0.0
      enddo

!hpf$ independent 
      do i=65, 95
         area(i,y_size/2) = 0.0
      enddo

!hpf$ independent
      do j=(x_size/4)+5, x_size/2
!hpf$    independent
         do i=1, x_size
            depth(i,j) = 2600.0
         end do
      end do

!hpf$ independent
      do j=(y_size/4)+1, y_size
!hpf$    independent
         do i=1, x_size
            depth(i,j) = 800.0
         end do
      end do

      end subroutine



!
! Impact of a drop at X, Y on WAVE
!
      subroutine drop(wave, x, y)
      
      include 'wave_parameters.h'

      real*8 wave(x_size, y_size)
      integer x, y
!hpf$ align with t:: wave
      
      integer i, j, d, dl
      parameter(dl = radius*radius)

      print *, 'drop at ', x, ' ', y

!hpf$ independent, new(d)
      do j=max(1,y-radius), min(y_size,y+radius)
!hpf$    independent
         do i=max(1,x-radius), min(y_size,x+radius)
            d = (i-x)*(i-x) + (j-y)*(j-y)
            if (d.le.dl) wave(i,j) = wave(i,j) + height*(25-d)
         end do
      end do

      end subroutine



!
! Iterate once
!
      subroutine iterate(speed, wave, depth, area)

      include 'wave_parameters.h'

      real*8 
     $     speed(x_size,y_size,2), 
     $     wave(x_size,y_size),
     $     depth(x_size,y_size),
     $     area(x_size,y_size)

      real*8 div
      
      integer i, j

!hpf$ align with t:: speed, wave, depth, area

!      print *, 'iterating'

! update speed:
!hpf$ independent
      do j=1, y_size-1
!hpf$    independent
         do i=1, x_size-1
            speed(i,j,1) = speed(i,j,1) +
     $           alpha * (wave(i+1,j)-wave(i,j))
            speed(i,j,2) = speed(i,j,2) +
     $           alpha * (wave(i,j+1)-wave(i,j))
         end do
      end do

! update wave:
!hpf$ independent, new(div)
      do j=2, y_size
!hpf$    independent
         do i=2, x_size
            div = speed(i,j,1) - speed(i-1,j,1) + 
     $           speed(i,j,2) - speed(i,j-1,2)
            wave(i,j) =  friction * 
     $           (wave(i,j) + div * area(i,j) * (depth(i,j)+wave(i,j)))
         end do
      end do

      end subroutine



!
! Image computation
!
      subroutine compute_image(image, wave)
      
      include 'wave_parameters.h'

      character image(x_size,y_size)
      real*8 wave(x_size,y_size)

      integer i, j, k

!hpf$ align with t:: image, wave
      
!      integer kmin, kmax
!      kmin=255
!      kmax=0
! , reduction(kmin,kmax)

!hpf$ independent, new(k)
      do j=1, y_size
!hpf$    independent
         do i=1, x_size
            k = wave(i,j)+zero_color
!            kmin = min(kmin, k)
!            kmax = max(kmax, k)
            image(i,j) = CHAR(k)
         end do
      end do

!      print *, 'max = ', kmax, ' min = ', kmin

      end subroutine



!
! MAIN
!
      program wave_propagation

      include 'wave_parameters.h'

      integer i, j, iter, x, y

      real*8 
     $     speed(x_size,y_size,2), 
     $     wave(x_size,y_size),
     $     depth(x_size,y_size),
     $     area(x_size,y_size)

! XPOMP library interface:
      include 'xpomp_graphic_F.h'

      character image(x_size,y_size)

      integer display, status, button, state, x_display, y_display

! HPF mappings:
!hpf$ align with t:: speed, wave, depth, area

! Initialize XPOMP
      call xpomp_open_display(x_display_size, y_display_size, display)
      call xpomp_set_color_map(display, 0, 1, 128, -1, status)
      call xpomp_show_usage
      
!      print *, 'click somewhere in the area'
!      call xpomp_wait_mouse(display, x, y, button, state) 
      
      call init_area(depth, area)

! Initialize speed and wave
! speed = 0.0
! wave  = 0.0

!hpf$ independent
      do j=1, y_size
!hpf$    independent
         do i=1, x_size
            speed(i, j, 1) = 0.0
            speed(i, j, 2) = 0.0
            wave(i, j) = 0.0
         enddo
      enddo

      do iter=1, maxiter

! Checks for a drop
         call xpomp_is_mouse
     $        (display, x_display, y_display, state, button)
         
         if (button.eq.1) then
            x = (x_display + x_display_zoom-1)/x_display_zoom
            y = (y_display + y_display_zoom-1)/y_display_zoom

            call drop(wave, x, y)
         end if

! Iterate wave propagation once

         call iterate(speed, wave, depth, area)

! Compute the image and show it

         call compute_image(image, wave)
         call xpomp_flash(display, image, x_size, y_size,
     $        0, 0, x_display_zoom, y_display_zoom, status)

      end do

      end program

! end of $RCSfile: wave.f,v $

