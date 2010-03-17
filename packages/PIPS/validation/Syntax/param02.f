      PROGRAM PARAM02

C     A parameter cannot be put in a common

      parameter (numtst=1)

      common / foo / numtst

      print *, numtst

      end
