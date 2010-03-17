      integer function entry10(x, n)

C     Goal: make sure that overlapping entries are correctly processed

      complex x(n)
      character*1 y(m)

      print *, x(1)

      entry foo

      return

      entry increment(y, m)

      print *, y(2)

      m = m + 1

      increment = 4

      end
