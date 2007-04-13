      subroutine entry07(x, n)

C     Goal: make sure that the control flows between entries, from
C     increment() into decrement()

      complex x(n)
      character*1 y(m)

      print *, x(1)

      return

      entry increment(y, m)

      print *, y(2)

      m = m + 1

      entry decrement(y, m)

      print *, y(3)

      m = m - 1

      end
