      program capply
      integer i
      i = 2
      print *, i
      call foo(i)
      end

      subroutine foo(i)
      integer i, j
      j = 3
      print *, i, j
      call bla(i, j)
      end

      subroutine bla(i, j)
      integer i, j, k
      k = 5 
      print *, i, j, k + 3
      end

      
