      program equiv20

C     Check that equivalenced variables declared in common are properly allocated

      integer x(2), y(2)
      integer x1, y2
      common /foo/ x, y2
      equivalence (x1,x(1)), (y(2),y2)

      x(2) = 3

      print *, y(1)

      end
