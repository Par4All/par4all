      program funcside1
      integer foo1
      external foo1

      i1 = 10

      i1 = foo1(i1)

      print *, i1

      end

      integer function foo1(j)
      j = j + 1
      foo1 = 2
      end
