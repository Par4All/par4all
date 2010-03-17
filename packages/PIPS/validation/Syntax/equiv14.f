      program equiv14

C     Check offsets

      real x(100), y(100)
      common /foo/x
      equivalence (x(2), y), (y(50), z)
      common /foo/u

      do i = 2, 10
         y(i) = 0.
      enddo

      print *, i, j

      end
