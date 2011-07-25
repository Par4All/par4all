!
! Check simplification of float constants
!
! Like float01.f, but check behavior for negative values
!

      program float03
      real x, y, z

      x = -1.
      y = -2.
      z = x+y

      read *, x, y

      print *, z

      end
