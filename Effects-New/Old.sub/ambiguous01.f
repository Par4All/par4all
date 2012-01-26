      program ambiguous01

C     Purpose: to show that effects on I are not translated in a unique way.
C     Its name depends on the parsing order.

      call foo

      call bar

      end

      subroutine foo
      common /toto/i
      i = 1
      end

      subroutine bar
      common /toto/i
      i = 1
      end
