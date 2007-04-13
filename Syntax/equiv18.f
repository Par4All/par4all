      program equiv18

C     Check that offsets in a common area are correct regardless of 
C     equivalences. Offsets:
C     x=0
C     y=96
C     u=200
C     w=4

      real x(50), w(100)

      equivalence (x(25),y)
      equivalence (u, w(50))

      common /foo/ x, u

      x(1) = y + w(10)

      print *, u, y

      end
