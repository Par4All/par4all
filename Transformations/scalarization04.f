      subroutine scalarization(x, y, n)
      real x(n,n), y(n,n)
      real t(100)
      
C     Expected result: nothing to scalarize
      do i=1,n
         t(i+1) = x(i,i)
         x(i,i) = y(i,i)
         y(i,i) = t(i)
      enddo

      end
