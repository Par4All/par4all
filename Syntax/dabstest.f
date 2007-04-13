      program dabstest

c     problem with intrinsic declaration because intrinsic are predeclared in PIPS

      double precision dabs
      double precision x,y
      read *, y
      x = dabs(y)
      print *, x
      end
