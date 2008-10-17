      PROGRAM ALIAS
      I = 1
      CALL FOO(I,I)
      PRINT *,I
      END
      SUBROUTINE FOO(J,K)
      J = J+K
      K = J*K
      PRINT *,J,K
      END
