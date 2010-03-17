      program equiv11

C     Check that an external cannot appear in an EQUIVALENCE: PIPS does not

      real y(100)
      equivalence (x, z), (u, v)
      real u(100)
      external v

      do i = 2, 10
         y(i) = 0.
      enddo

      print *, i, j

      end
