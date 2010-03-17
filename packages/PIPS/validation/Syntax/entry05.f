      program entry05

C     Goal to check that initial.f really can be parsed

      real x(10), y(20)

      call foo(x, 10)

      call bar(y, 20)

      end

      subroutine foo(x, n)

C     Goal: make sure that varying size formal parameters such as y
C     are properly processed.

      complex x(n)
      character*1 y(m)

      print *, x(1)

      return

      entry bar(y, m)

      print *, y(2)

      m = m + 1

      end
