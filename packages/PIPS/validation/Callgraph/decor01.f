      program decor01

C     Check that call graphs can be decorated by all possible decoration
c     See decor01.tpips

      n = 0

      call B(x, y, z, n)

      print *, z, n

      end

      subroutine B(x, y, z, n)

      print *, x+y
      z = 2
      n = n + 1

      end
