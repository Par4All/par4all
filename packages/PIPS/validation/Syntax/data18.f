      program data18

C     Use non-integer variables in DATA

      character*10 s
      logical b

      data i, x, s, b /1, 2., "3", .TRUE./

      print *, i, x, s, b

      end
