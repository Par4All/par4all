      program save06

C     Check that offsets in the static area are correct regardless of 
C     equivalences. Variable v is not equivalenced on purpose: it should
C     be allocated *after* the equivalenced variables in the static area.

C     Size of static area: 604
C     offsets for u=396 (from w(50))
C     offsets for v=600 (added at the end of area, after equivalenced variables)
C     offsets for w=200 (located after the (x,y) chain)
C     offsets for x=0
C     offsets for y=96 (from x(25))
      save u
      save v
      save w
      save x
      save y

      real x(50), w(100)

      equivalence (x(25),y)
      equivalence (u, w(50))

      x(1) = y + w(10)

      print *, u, y

      end
