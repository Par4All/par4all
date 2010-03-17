      PROGRAM DECL06

C     Check dimensionning by real expressions in FOO

      INTEGER N
      PARAMETER (N=10)
      REAL A(N), B(N), C(N), D(N)
      CALL FOO(A, B, C, D, N-4, 1)
      END

      SUBROUTINE FOO(X, Y, Z, W, N, M)
      INTEGER N, M
      REAL X(N+1)
      REAL Y(N+1.0)
      REAL Z(N*1.5)
      REAL W(N*M)
      INTEGER I
      DO 10 I=1, N+1
         X(I) = 1
 10   CONTINUE
      END
