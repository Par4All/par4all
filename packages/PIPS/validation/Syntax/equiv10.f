      program equiv10

C     Check that an array reference may occur *before* an array declaration

      real y(100)
      equivalence (x, z), (u(3), v)
      real u(100)

      do i = 2, 10
         y(i) = 0.
      enddo

      print *, i, j

      end
