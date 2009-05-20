      subroutine scalarization(x, y, n)
      real x(n), y(n)
      
C     Expected result: nothing to scalarize
      do i=1,n
         x(i) = y(i)
      enddo

      end
