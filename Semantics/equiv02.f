      program equiv02

C     Check alias detection

      common i

      i = 0

      call foo

      print *, i

      end

      subroutine foo
      common x

      x = 3.

      end
