C     Check the piece of code added by Ronan to detect uses of
C     uninitialized variables

C     Results with real codes such as oa118.f and extr.f are garbage,
C     but it work for toy examples

C     Note that information gathered for the last print is garbage

C     Implicit variables such as __IO_EFFECTS:LUNS are a special case
C     that is not dealt with. And effects on LUNS are not well defined

      program initialization01

      j = i
      i = 1
      j = i
      print *, j, k, l
      read *, i, j, k, l
      print *, j, k, l
      end
