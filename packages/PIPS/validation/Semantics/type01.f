      program type01

C     Goal: check the extension of the semantics analysis to non-integer
C     scalar variables

      logical l1
      character*4 s1
      real r1
      double precision d1
      complex c1

      l1 = .TRUE.
      l1 = .FALSE.
      l1 = .NOT. l1
      s1 = "TOTO"
      r1 = 1.57
      d1 = 1.23456789
      c1 = cmplx(1.,1.)

      print *, l1, s1, r1, d1, c1

      end

