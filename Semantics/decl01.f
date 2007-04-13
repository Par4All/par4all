      subroutine decl01(x,n)

C     Check that declaration information is used in preconditions.
C     If it is used, i == n+1 && n >= 1 before the print.
C     If not, i >= 1 && i >=n+1.

      real x(n)

      do i = 1, n
         x(i) = 0.
      enddo

      print *, i

      end
