      program equiv06

C     This seems to be compatible with the Fortran 77 standard

      real t(3)
      common /foo/ a, b, c
      equivalence (t(1),a), (t(3),c)

      end
