      program equiv22(n)

C     Bug: equivalence with formal parameter

      common /nm/m
      equivalence (n,m)

      print *, n, m

      end
