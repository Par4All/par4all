      program common2
      common /bar/ x
      call foo
      end
      subroutine foo
      common /bar/ i
      i=1
      end
