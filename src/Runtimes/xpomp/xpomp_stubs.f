! $Id$
!
! List of fake functions to have PIPS happy with 
! the same « effects » as the xPOMP graphical library.
! !fcd$ io directive is used by the HPFC compiler to consider
! these functions as IO routines.
! !fcd$ fake directives tells not to compile these functions,
! since they will be provided somewhere else.
!
!     Ronan.Keryell@cri.ensmp.fr
!
      subroutine xpompf open display(x, y, d)
      integer x, y, d
!fcd$ io
!fcd$ fake
      print *, x, y
      read *, d
      end
      
      subroutine xpompf close display(d)
      integer d
!fcd$ io
!fcd$ fake
      print *, d
      end
      
      subroutine xpompf get current default display(d)
      integer d
!fcd$ io
!fcd$ fake
      read *, d
      end
      
      subroutine xpompf set current default display(d, r)
      integer d, r
!fcd$ io
!fcd$ fake
      print *, d
      read *, r
      end
      
      subroutine xpompf get depth(d)
      integer d
!fcd$ io
!fcd$ fake
      read *, d
      end
      
      subroutine xpompf set color map(screen,
     &     pal, cycle, start, clip, r)
      integer screen, pal, cycle, start, clip, r
!fcd$ io
!fcd$ fake
      print *, screen, pal, cycle, start, clip
      read *, r
      end
      
      subroutine xpompf set user color map(screen,
     &     red, green, blue, r)
      integer screen
      character red(256), green(256), blue(256)
      integer r, i
!fcd$ io
!fcd$ fake
      do i = 1, 256
         print *, screen, red(i), green(i), blue(i)
      enddo
      read *, r
      end

      subroutine xpompf wait mouse(screen, X, Y, state, r)
      integer screen, X, Y, state, r
!fcd$ io
!fcd$ fake
      print *, screen
      read *, X, Y, state, r
      end

      subroutine xpompf is mouse(screen, X, Y, state, r)
      integer screen, X, Y, state, r
!fcd$ io
!fcd$ fake
      print *, screen
      read *, X, Y, state, r
      end
      
      subroutine xpompf flash(window,
     &     image,
     &     X data array size, Y data array size,
     &     X offset, Y offset,
     &     X zoom ratio, Y zoom ratio,
     &     status)
      integer window
      integer X data array size, Y data array size
      character image(X data array size, Y data array size)
      integer X offset, Y offset
      integer X zoom ratio, Y zoom ratio
      integer status, x, y
!fcd$ io
!fcd$ fake
      print *, window, X data array size, Y data array size,
     &     X offset, Y offset,
     &     X zoom ratio, Y zoom ratio
      do x = 1, X data array size
         do y = 1, Y data array size
            print *, image(x, y)
         enddo
      enddo
      read *, status
      end

      subroutine xpompf show real4(screen, image,
     &     X data array size, Y data array size,
     &     X offset, Y offset,
     &     X zoom ratio, Y zoom ratio,
     &     min value, max value,
     &     status)
      integer screen
      integer X data array size, Y data array size
      real*4 image(X data array size, Y data array size)
      integer X offset, Y offset
      integer X zoom ratio, Y zoom ratio
      real*8 min value, max value
      integer status, x, y
!fcd$ io
!fcd$ fake
      print *, screen, X data array size, Y data array size,
     &     X offset, Y offset,
     &     X zoom ratio, Y zoom ratio
      do x = 1, X data array size
         do y = 1, Y data array size
            print *, image(x, y)
         enddo
      enddo
      read *, status
      end

      subroutine xpompf show real8(screen, image,
     &     X data array size, Y data array size,
     &     X offset, Y offset,
     &     X zoom ratio, Y zoom ratio,
     &     min value, max value,
     &     status)
      integer screen
      integer X data array size, Y data array size
      real*8 image(X data array size, Y data array size)
      integer X offset, Y offset
      integer X zoom ratio, Y zoom ratio
      real*8 min value, max value
      integer status, x, y
!fcd$ io
!fcd$ fake
      print *, screen, X data array size, Y data array size,
     &     X offset, Y offset,
     &     X zoom ratio, Y zoom ratio
      do x = 1, X data array size
         do y = 1, Y data array size
            print *, image(x, y)
         enddo
      enddo
      read *, status
      end
      
      subroutine xpompf scroll(window, y, result)
      integer window, y, result
!fcd$ io
!fcd$ fake
      print *, window, y
      read *, result
      end
      
      subroutine xpompf draw frame(window,
     &     title,
     &     title color, background color,
     &     X0, Y0, X1, Y1,
     &     color,
     &     status)
      integer window
      character*(*) title
      integer title color, background color
      integer X0, Y0, X1, Y1
      integer color
      integer status
!fcd$ io
!fcd$ fake
      print *, window, title, title color, background color,
     &     X data array size, Y data array size,
     &     X0, Y0, X1, Y1, color
      read *, status
      end
      
      subroutine xpompf show usage()
!fcd$ io
!fcd$ fake
      print *, 'Some help...'
      end
