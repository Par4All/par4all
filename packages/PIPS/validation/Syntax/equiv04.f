      program equiv04

C     Detection of impossible equivalences

      common /foo/ x, y
      common /bar/ u, v

      equivalence (x,u), (x,y)

      print *, u, v, x, y

      end
