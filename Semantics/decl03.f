      subroutine decl03(x, y, n, m)

C     Check that continuation condition is present in subroutine
C     transformer

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
