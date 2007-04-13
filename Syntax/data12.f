      program data12
      integer x(10)
      complex z(10)

C     Complex value is coerced to integer and this should not be
C     interpreted as eight values: 1 1 1 2 1 1 1 2!

c      data (x(i), i = 1, 2) /2*(3*1,2)/

c      data (z(i), i = 1, 2) /2*(3*1,2)/
      data (x(i), i = 1, 2) /2*(3,2)/

      data (z(i), i = 1, 2) /2*(3,2)/

      print *, (x(i), i = 1, 8)

      print *, (z(i), i = 1, 8)

      end
