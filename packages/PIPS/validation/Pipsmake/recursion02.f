      program recursion02

C     Goal: Make sure that pipsmake remove required resources

      call foo01(x)
      call foo04(x)

      end

      subroutine foo01(y)

      call foo02(x)
      call foo03(x)

      end

      subroutine foo02(y)

      call foo03(x)
      call foo01(x)

      end

      subroutine foo03(y)

      call foo01(x)
      call foo02(x)


      end

      subroutine foo04(y)

      call foo05(x)

      end

      subroutine foo05(y)

      call foo04(x)

      end
