      program data11
      integer x(10), y(10), z(10)
      data i, j, k /3*11/
      data (x(i), y(i), z(i), i = 1, 2) /1, 2, 3, 4, 5 ,6/

      print *, (x(i), i = 1, 2),
     & (y(i), i = 1, 2),
     & (z(i), i = 1, 2)

      end
