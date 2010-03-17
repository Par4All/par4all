      program recursion01

C     Goal: Make sure that pipsmake detects recursion before looping forever

      call foo01(x)

      end

      subroutine foo01(y)

      call foo02(x)

      end

      subroutine foo02(y)

      call foo03(x)

      end

      subroutine foo03(y)

      call foo01(x)

      end
