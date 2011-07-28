!
! Check simplification of float constants
!
! Like float01.f, but check behavior for "0."
! Do we want to handle "0." as a value different from 0?
!

      program float02
      real x, y, z

      x = 1.
      y = 0.
      z = x+y

      read *, x, y

      print *, z

      end
