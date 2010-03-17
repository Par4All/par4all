c     passage de procedure formelle
C     abort "no source file for x"
      program formel
      external q

C     Forbidden in Fortran 77:
c      p = q

      call foo(q)
      end
      subroutine foo(x)
      real y(100)
      
      do 100 j = 1, 100
         y(i) = x(i)
 100  continue
      return
      end
      function q(i)
      q = float(i)
      return
      end
