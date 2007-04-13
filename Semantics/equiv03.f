      program equiv03

C     Check alias detection

      common i, z

      i = 0

      y = 2. + xincr(y)

      print *, i

      end

      real function xincr(y)
      common x, z

      y = y + 1.
      x = 3.
      z = y

      xincr = 4.

      end
