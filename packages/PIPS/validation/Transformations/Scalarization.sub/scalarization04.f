C     Expected result: nothing to scalarize

      subroutine scalarization(x,y,n)
      real x(n,n), y(n,n)
      real t(100)      
      do i=1,n
         t(i+1) = x(i,i)
         x(i,i) = y(i,i)
         y(i,i) = t(i)
      enddo
      end

      program scalarization04
      parameter(n=100)
      real x(n,n), y(n,n)      
      read *,x,y
      call scalarization(x,y,n)
      print *,x,y
      end
