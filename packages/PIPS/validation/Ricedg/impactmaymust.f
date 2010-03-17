      PROGRAM IMPACT
      COMMON /COM/ T
      INTEGER A
      CALL FOO(A, A)
      END

      SUBROUTINE FOO(X,Y)
      COMMON /COM/ T
      X = 0
      IF (T.GT.0) THEN
         X = 1
         DO WHILE (T.GT.10)
            X = X + 1
            T = T - 1
         ENDDO
      ELSE
         Y = 2
      END IF
      Y = 10
      CALL FOO1(X)
      Y = 3
      END

      SUBROUTINE FOO1(X)
      IF (T.GT.0) THEN
         X = 10
C      ELSE
C         X = 9   
      ENDIF
      END





