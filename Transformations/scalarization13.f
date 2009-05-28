C     Expected result: nothing needs to be scalarized
C
C     NOTE: scalarization takes place all the same

      subroutine scalarization(x,y,n)
      real x(n), y(n)      
      do i=1,n
         x(i) = y(i)
      enddo
      end

      program scalarization01
      parameter(n=100)
      real x(n,n), y(n,n)      
      read *,y
      call scalarization(x,y,n)
      print *,x,y
      end
