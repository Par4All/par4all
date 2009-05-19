      subroutine scalarization(x, y, n)
      real x(n,n), y(n,n)
      real tt
      
C     Expected result: nothing to scalarize
      do i=1,n
         tt = x(i,i)
         x(i,i) = y(i,i)
         y(i,i) = tt
      enddo

      end
