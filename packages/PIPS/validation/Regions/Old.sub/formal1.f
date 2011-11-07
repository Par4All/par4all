!
      program reg
      real a(10)
      call s1(a)
      end
! here, the dimension declaration
! must not be included in the region computation...
      subroutine s1(x)
      real x(1)
      call s2(x)
      end
!
      subroutine s2(y)
      real y(1)
      integer i
      do i=1, 10
         y(i) = 1.0
      enddo
      end
