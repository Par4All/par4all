      PROGRAM IMPACT
      COMMON /COM/ T
      INTEGER A, B, C(5)
      A = 0
      B = 1
      CALL FOO(A, B)
      CALL FOO(A, A)
C      CALL FOO(T,A)
C      CALL FOO(A,T)
C      CALL FOO1(C(1), C(2))
      END

      SUBROUTINE FOO(X,Y)
      COMMON /COM/ T
      X = 0
      IF (T.GT.0) THEN
         X = 1
      ELSE
         Y = 2
      END IF
      Y = 3
      END

      SUBROUTINE FOO1(X, Y)
      INTEGER X(4), Y(3)
      X(1) = 1
      X(2) = X(2) + T
      Y(1) = Y(1) + T
      END

