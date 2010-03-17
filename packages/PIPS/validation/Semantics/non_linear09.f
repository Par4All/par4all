      SUBROUTINE NON_LINEAR09 (IS, L, M, U, X, Y)

      IMPLICIT REAL*8 (A-H, O-Z)
      DIMENSION U(1), X(1), Y(1)
C
C   Set initial parameters.
C
      N = 2 ** M
      K = U(1)
      NU = K / 64
      N1 = N / 2
      LK = 2 ** (L - 1)
      LI = 2 ** (M - L)
      LJ = 2 * LI
      KU = LI + 1
      KN = KU + NU
C
      DO 100 K = 0, LK - 1
        I11 = K * LJ + 1
        I12 = I11 + LI
        I21 = K * LI + 1
        I22 = I21 + N1
C
C   This loop is vectorizable.
C
        DO 100 I = 0, LI - 1
          U1 = U(KU+I)
          U2 = IS * U(KN+I)
          X11 = X(I11+I)
          X12 = X(I11+I+N)
          X21 = X(I12+I)
          X22 = X(I12+I+N)
          T1 = X11 - X21
          T2 = X12 - X22
          Y(I21+I) = X11 + X21
          Y(I21+I+N) = X12 + X22
          Y(I22+I) = U1 * T1 - U2 * T2
          Y(I22+I+N) = U1 * T2 + U2 * T1
 100  CONTINUE
C
      RETURN
      END
