      program common3
      common /bar/ x,y
      call foo
      end
      subroutine foo
      common /bar/ i,y
      i=1
      end
