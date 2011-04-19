C     Expected result: t(i) scalarized

C     Same as scalarization38.f, but with a name conflict if the prefix
C     specified in scalarization39.tpips is forced to uppercase

      subroutine scalarization(x,y,n)
      real x(n,n), y(n,n)
      real t(100), s_0
      s_0 = 0.
      do i = 1,n
         do j = 1,n
            t(i)   = x(i,j)
            x(i,j) = y(i,j)
            y(i,j) = t(i)
         enddo
      enddo
      print *, s_0
      end

      program scalarization39
      parameter(n=100)
      real x(n,n), y(n,n)      
      read *,x,y
      call scalarization(x,y,n)
      print *,x,y
      end

