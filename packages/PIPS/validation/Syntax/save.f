      program saveglobal

      call foo(i)

      call foo(i+1)

      print *, i

      end

      subroutine foo(k)
      integer j
      save

      i = k
      j = k+1

      end
