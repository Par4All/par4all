      PROGRAM DO
      REAL A(100)
c
      K = 24
      X = 1.1
      Y = 82.2
      DO I = 1,20
         IF ( SIN(X) .GT. 1. ) THEN
         X = X*X
         X = X+X
         ENDIF
         CALL P(X)
         IF ( X .GT. Y ) THEN
            X = X - 0.5
            Y = Y -0.1
         ELSE
            X = X + 0.5
            CALL P(X)
         ENDIF
         A(I) = 1.
         PRINT '(2F10.5)', X, Y
      ENDDO
      END
C
      SUBROUTINE P(Q)
      Q = Q + 1.
c      W = W - 1.
      END
