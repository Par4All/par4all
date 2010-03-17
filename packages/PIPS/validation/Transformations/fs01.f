!
! forward substitution example 01
!
      program fs01

      real x, y, z, w, u, v

      x = 0.2226473
      y = x + x*x
      z = y + x
      w = z*z*z + y*x 
      
      print *, 'w = ', w

      u = 0.3236433
      v = u + 2*u
      
      end
