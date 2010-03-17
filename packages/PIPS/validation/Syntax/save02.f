      program save02

C     Bug to detect: double incompatible storage declaration for variable i

      common /foo/ i

      save i

      i = 1

      print *, i

      end
