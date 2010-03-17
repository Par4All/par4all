      subroutine entry08(x, n)

C     Goal: make sure that static variables are properly processed

      complex x(n)
      character*1 y(m)

      save l

      print *, x(1), w

      return

      entry increment(y, m)

      print *, y(2), u

      m = m + 1

      end
