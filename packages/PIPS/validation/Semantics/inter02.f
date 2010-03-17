      program inter02

C     Check that type mismatches are detected and do not cause core dumps

      x = 0.

      call incr(x)

      print *, x

      end

      subroutine incr(i)
      i = i + 1
      end

