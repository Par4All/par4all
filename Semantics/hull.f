      program hull
      i = 1
      call foo(i)
      i = 2
      call foo(i)
      end
      subroutine foo(j)
      j = j +1
      end
