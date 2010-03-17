      PROGRAM PARAM01

C     A variable cannot be declared a parameter

      common / foo / numtst

      parameter (numtst=1)

      print *, numtst

      end
