!
! forward substitution example 
!
      program fs02

      integer n, i, j
      parameter (n=100)

      real a(n,n), b(n)
      real x1, x2, x3

      do i=1, n
         x1 = b(i)
         x2 = x1*x1 + x1
         do j=1, n
            x3 = x1 + x2
            a(i,j) = a(i,j)*x3+x2-1
         enddo
      enddo

      print *, a(5,20)

      end
