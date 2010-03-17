      subroutine decl02(x, y, n, m)

C     Check that declaration information is used in preconditions.
C     If it is used, i == n+1 && n >= 1 before the print.
C     If not, i >= 1 && i >=n+1.

      real x(n, m), y(m, n)

      do i = 1, n
         x(i,i) = 0.
      enddo

      do i = 1, n
         do j = 1, m
            y(j,i) = 1.
         enddo
      enddo

      print *, i, j

      end
