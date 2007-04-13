      program equiv17

C     Check that offsets in the dynamic area are correct regardless of 
C     equivalences. Variables z1 and z2 are not equivalenced on purpose
C     to check their allocation offsets.

      real x(50), w(100)

      equivalence (x(25),y)
      equivalence (u, w(50))

      x(1) = y + w(10) + z1 + z2

      print *, u, y

      end
