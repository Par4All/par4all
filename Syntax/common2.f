      program common2
      common /toto/ x, y
      call foo
      end
      subroutine foo
      common /toto/ x, y, z
      end
