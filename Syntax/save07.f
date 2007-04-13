      program save07

C     Check that offsets in the static area are correct regardless of 
C     equivalences. Variable v is not equivalenced on purpose: it should
C     be allocated *after* the equivalenced variables in the static area.

C     Size of static area: 904, 300 for the (x,y) chain and 600 for the (u,w) chain, 4 for v
C     offsets for u=300 (after the (x,y) chain)
C     offsets for v=900 (added at the end of area, after equivalenced variables)
C     offsets for w=500 (located from u(50))
C     offsets for x=100
C     offsets for y=0
      save u
      save v
      save w
      save x
      save y

      real x(50), y(50), u(100), w(100)

      equivalence (x(25),y(50))
      equivalence (u(100), w(50))

      x(1) = y + w(10)

      print *, u, y

      end
