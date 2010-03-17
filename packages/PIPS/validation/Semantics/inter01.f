      program inter01

C     Check that insufficient calls do not cause core dump in
C     transformers and preconditions: it is now trapped by the
C     effects analysis

      call foo(i)

      print *, i

      end

      subroutine foo(i, j)
      i = 1
      j = 2
      end
