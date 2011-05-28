C     Expected result: t(i) scalarized (although it is copied out)

      subroutine scalarization(x,y,n)
      real x(n,n), y(n,n)
      real t(100)      
      do i = 1,n
         do j = 1,n
            t(i)   = x(i,j)
            x(i,j) = y(i,j)
            y(i,j) = t(i)
         enddo
      enddo
      print *, t(n)
      end

      program scalarization02
      parameter(n=100)
      real x(n,n), y(n,n)     
      read *,x,y
      call scalarization(x,y,n)
      print *,x,y
      end

