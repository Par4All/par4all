      program pointer01

C     Check the syntactic analysis of pointers

      real x(10)

      pointer (i, x)

      i = malloc(10)

      print *, x(1)

      end
