      program wtype5
      real x(10,10)
      i=3
      call foo(x(1,i))
      end
      subroutine foo(y)
      real y(20)
      y(18)=1
      end
