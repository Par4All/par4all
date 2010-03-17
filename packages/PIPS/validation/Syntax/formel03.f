c     passage de procedure formelle
C     abort "no source file for x"
      program formel03
      external q

C     Forbidden in Fortran 77:
c      p = q

      call foo(q)
      end
      subroutine foo(x)
      external x
      real y(100)
      
      do 100 j = 1, 100
         call x(i)
         y(i) = 0.
 100  continue
      return
      end
      subroutine q(i)
      i = i + 1
      return
      end
