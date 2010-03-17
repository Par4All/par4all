      subroutine entry02(x, n)

C     Goal: make sure that varying size formal parameters such as y
C     are properly processed.

      complex x(n)
      character*1 y(m)

      print *, x(1)

      return

      entry increment(y, m)

      print *, y(2)

      m = m + 1

      end
