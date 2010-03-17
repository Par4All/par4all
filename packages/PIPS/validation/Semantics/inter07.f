      program inter07

C     Check that type mismatches are detected and do not cause core dumps:
C     this case is not detected but the value of i is lost as it should be.
C     Floating point scalar variables are not analyzed

      i = 0

      call incr(i)

      print *, i

      end

      subroutine incr(x)
      x = x + 1.
      end
