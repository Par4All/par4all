      program test12

!================================================
! Test spaghettifier
!================================================

      implicit none

      integer a, b, c, n, m, k
      real r      

      b=5
      a=2
      c=3

      do j=1,10,1
         b = 5
         a = 8
         do i=1,10,1
            n = a*i + b
            m = b*i + j
            k = k + n * m
         enddo
         a = b + k
      end do

      if (a.LT.b) then
       b = 5
       a = 17
      else 
       b = 7
       a = 89
      end if

      do while (a .LT. b)
         r = a*r
         a = a+9
      end do

      stop
      end
      
