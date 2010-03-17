      program inter06

C     Check that type mismatches are detected and do not cause core
C     dumps: the type mismatch is detected and a warning is issued.
C     Floating point scalar variables are not analyzed, this is the
C     difference with inter02

      call incr(x)

      print *, x

      end

      subroutine incr(i)
      i = i + 1
      end

