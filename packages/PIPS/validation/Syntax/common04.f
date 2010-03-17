      program common04

      common /foofoo/ x

      call foo()

      print *, x

      end

      subroutine foo

      common /foofoo/ y

      call bar()

      y = y + 1.

      end

      subroutine bar

      common /foofoo/ z

      z= 3.

      end
