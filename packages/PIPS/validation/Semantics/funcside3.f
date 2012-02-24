      program funcside3

      external foo3
      integer foo3

      i3 = 30

      i3 = foo3(i1, i2)

      print *, i3

      end

      integer function foo3(j, k)
      foo3 = j + k
      j = j + 1
      k = k + 2
      end
