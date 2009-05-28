C     Expected result: nothing to scalarize

      subroutine scalarization(x,y,n)
      real x(n,n), y(n,n)
      real tt
      do i=1,n
         tt = x(i,i)
         x(i,i) = y(i,i)
         y(i,i) = tt
      enddo
      end

      program scalarization03
      parameter(n=100)
      real x(n,n), y(n,n)      
      read *,x,y
      call scalarization(x,y,n)
      print *,x,y
      end

