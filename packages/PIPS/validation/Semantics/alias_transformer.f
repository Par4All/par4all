C This program is likely not to confomr to the Fortran standard because
C of aliasing between a formal parameter and a global variable.

C The result found by tpips is wrong.

C The aliasing should be checked before the analysis

      PROGRAM ALIAS_TRANSFORMER
      COMMON I
      I = 0
      CALL FOO(I)
      PRINT *, I
      END
      SUBROUTINE FOO(J)
      COMMON I
      J = J + 1
      I = I + 2
      END
