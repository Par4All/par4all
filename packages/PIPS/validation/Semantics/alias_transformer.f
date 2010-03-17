      PROGRAM ALIAS_TRANSFORMER
      COMMON I
      I = 0
      CALL FOO(I)
      END
      SUBROUTINE FOO(J)
      COMMON I
      J = J + 1
      I = I + 2
      END
