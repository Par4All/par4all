      program data10

C     Check that DATA statement in the midst of executable statements
C     are taken into account

      real y(100)

      i = 1

      data x /2.13/

      data (y(i), i = 1, 30) /1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
     &     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
     &     29, 30/

      print *, x, y(1)

      end
