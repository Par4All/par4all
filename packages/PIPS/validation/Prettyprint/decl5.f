! clean declarations must drop unused equivalenced...
      program decl5
! common FOO
      common /foo/ a, b, c
      real d(3)
      equivalence (d(1),a)
! common BLA
      common /bla/ a1, a2, a3
      real a1, a2, a3, b1, b2, b3
      equivalence (a1, b1)
      equivalence (a2, b2)
      equivalence (a3, b3)
! locals
      real c1(10), c2(5), c3(2)
      equivalence (c2(1), c1(5))
      equivalence (c3(1), c1(4))
! use
      print *, b, b3, c2(2), c3(2)
      end
