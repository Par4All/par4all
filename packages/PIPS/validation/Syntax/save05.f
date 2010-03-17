      program save05

C     Check that offsets in the static area are correct regardless of 
C     equivalences

      real w

      save u
      save x
      save y

      equivalence (x,y)
      equivalence (w1, w2)

      x = y + w

      print *, u, w1, w2

      end
