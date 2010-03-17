C     Semantics analysis and dead code

C     Partial evaluation may generate dead code and trivial tests

      PROGRAM DEAD2

      I = 1
      J = 1
      IF (I.EQ.J) THEN
         K = 2
      ELSE
         K = 4
      ENDIF
      IF (1.EQ.1) THEN
         K = 2
      ELSE
         K = 4
      ENDIF
      PRINT *,I,J,K
      END
