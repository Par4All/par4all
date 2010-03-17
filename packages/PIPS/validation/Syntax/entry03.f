      subroutine entry03(x, n)

C     Goal: make sure callees are fixed.

      complex x(n)
      character*1 y(m)

      call foo
      print *, x

      return

      entry increment(y, m)

      call bar
      print *, y

      m = m + 1

      end

      subroutine foo
      print *, 'foo'
      end

      subroutine bar
      print *, 'bar'
      end
