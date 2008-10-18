      program wtype3
      real x(10,10)
      call foo(x)
      end
      subroutine foo(y)
      real y(100)
      y(3)=0
      end
