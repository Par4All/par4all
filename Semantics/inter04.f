      program inter03

C     Check that aliases do not cause core dumps

      common /foo/ j

      call incr(j)

      print *, j

      end

      subroutine incr(i)
      common /foo/ x
      x = 2.
      i = i + 1
      end

