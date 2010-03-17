      program equiv13

C     Check that chains of undeclared arrays are correctly processed: in fact PIPS
C     does not handle this

      real x(100), y(100), z(100)
      equivalence (x(50), y), (y(50), z)
c      real x(100), y(100), z(100)

      do i = 2, 10
         y(i) = 0.
      enddo

      print *, i, j

      end
