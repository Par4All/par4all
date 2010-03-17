      subroutine entry06(x, n)

C     Goal: make sure that more than one entry can be processed

      complex x(n)
      character*1 y(m)

      print *, x(1)

      return

      entry increment(y, m)

      print *, y(2)

      m = m + 1

      return

      entry decrement(y, m)

      print *, y(3)

      m = m - 1

      end
