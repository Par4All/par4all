C     test de la (non) capture des intrinsics, e.g. REAL used as a 
C     statement function.

      program capture02
      real(x) = x**2
      if(real(1.).lt.0.) print *, real(2)
      end
