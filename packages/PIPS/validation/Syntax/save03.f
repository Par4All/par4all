      program save03

C     Bug: external should imply rom storage and an error message for the SAVE
C     In fact, PIPS parser allocates a local variable SAVE03:FOO. f77 core dumps.
C     g77 generates an error.

      external foo

      save foo

      call foo(i)

      print *, i

      end

      subroutine foo(i)
      i = i + 1
      end

