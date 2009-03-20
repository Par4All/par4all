      subroutine scalarization01(x, y, n)
      real x(n,n), y(n,n)
      real t(100), t2(100)
      
C     Expected result : t(i) scalarized
      do j = 1,n
         do i=1,n
            t(i) = x(i,j)
            x(i,j) = y(i,j)
            y(i,j) = t(i)
         enddo
      enddo

C     Expected result : t2(i) NOT scalarized (b/c copied out)
      do j = 1,n
         do i=1,n
            t2(i)  = x(i,j)
            x(i,j) = y(i,j)
            y(i,j) = t2(i)
         enddo
      enddo

      print *, t2(n)

      end
