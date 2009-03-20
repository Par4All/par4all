      subroutine scalarization02(x, y, n)
      real x(n,n), y(n,n)
      real t(100), t2(100), tt
      
C     Expected result: t(i) should be scalarized
      do j = 1,n
         do i=1,n
            t(i) = x(i,j)
            x(i,j) = y(i,j)
            y(i,j) = t(i)
         enddo
      enddo

C     Expected result: nothing to scalarize
      do i=1,n
         tt = x(i,i)
         x(i,i) = y(i,i)
         y(i,i) = tt
      enddo

C     Expected result: nothing to scalarize
      do i=1,n
         t2(i+1) = x(i,i)
         x(i,i) = y(i,i)
         y(i,i) = t2(i)
      enddo

      end
