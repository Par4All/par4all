      program cmplx01

      complex z

      read *, x, y

c     This is a Fortran complex constant:

      z = (1., 2.)

c     This is not a Fortran syntax: only numerical constants are allowed

      z = (x+y, x*y)

      print *, z

      end

      subroutine foo

      parameter (ur=3.)
      parameter (ui=4.)

c     This is not another Fortran complex constant:

      z = (ur, ui)

      print *, z

      end
