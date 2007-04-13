      program equiv16

C     Check equivalences and commons: an error must be detected

      common /foo/ x, y
      equivalence (x,y)

      print *, x, y

      end
