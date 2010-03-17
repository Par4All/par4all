      program equiv21

C     Bug EDF: DATA sur une variable en equivalence avec une autre
C     declaree dans un common

      data n/3/
      common /nm/m
      equivalence (n,m)

      common /nmp/mp
      data np/4/
      equivalence (np,mp)

      print *, m, mp

      end


