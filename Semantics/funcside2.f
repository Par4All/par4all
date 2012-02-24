      program funcside2

      external foo2
      integer foo2

      i2 = 20

      i2 = foo2(i2)

      print *, i2

      end

      integer function foo2(j)
      j = j + 1
      foo2 = foo2 + 2
      end
