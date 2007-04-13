      program equiv15

C     Check equivalences and substrings

      character*132 x
      character*4 y

      equivalence (x(13:16), y(2:3))

      print *, x, y

      end
