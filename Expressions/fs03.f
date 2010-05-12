!
! forward substitution example 
!
      program fs03

      integer n, i
      parameter (n=100)

      real a(n), x

      do i=1, n
         x = a(i)*a(i)
         a(i) = x + x
      enddo

      x = a(4)
      a(i-3) = x + x
      a(i-2) = x - x

      print *, a(5)

      end
