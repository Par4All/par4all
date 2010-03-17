      program equiv05

C     The equivalence chain is not consistent

      real t(3)
      common /foo/ a, b, c
      equivalence (t(1),a), (t(2),c)

      end
