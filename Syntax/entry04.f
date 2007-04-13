      subroutine entry04(x, y, n)

C     Goal: make sure that offset for formal parameters are respected

      complex x(n)
      character*1 y(n)

      print *, x(1)

      return

      entry increment(y, x, n)

      print *, y(2)

      m = m + 1

      end
