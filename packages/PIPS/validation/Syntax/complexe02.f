      program complexe02
      complex z

c     This is a complex constant
      print *, (5, 6)

c     This is diagnosed as an invalid implicit DO
c     print *, (5, 6, 7)

      print *, (5, 6, I = 1, 2)

      end
