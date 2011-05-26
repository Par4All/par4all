C bug seen in Transformations/eval.c: modulo evaluatiaon for negative arguments

      program main

      i = mod(3, 2)
      print *, i, "must be 1"

      i = mod(-3, 2)
      print *, i, "must be -1"

      i = mod(3, -2)
      print *, i, "must be 1"

      i = mod(-3, -2)
      print *, i, "must be -1"

      end
