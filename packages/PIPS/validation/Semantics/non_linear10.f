      SUBROUTINE CFFT (IS, M, U, X, Y, NQ)

      IMPLICIT REAL*8 (A-H, O-Z)
      PARAMETER (PI = 3.141592653589793238D0)
      COMMON /COUNT/ KTTRANS(256)
      DIMENSION U(1), X(1), Y(1)

C     Synthesize preconditions
      if(is.lt.-1.or.is.gt.1
     & .or.m.lt.5.or.m.gt.6
     & .or.nq.lt.1.or.nq.gt.64) stop

C
      IF (IS .EQ. 0)  THEN
C
C     Initialize the U array with sines and cosines in a manner that
C     permits stride one access at each FFT iteration.
C
        N = 2 ** M
        NU = N
        U(1) = 64 * N + M
        KU = 2
        KN = KU + NU
        LN = 1
C
        DO 110 J = 1, M
          T = PI / LN
C
C   This loop is vectorizable.
C
          DO 100 I = 0, LN - 1
            TI = I * T
            U(I+KU) = COS (TI)
            U(I+KN) = SIN (TI)
 100      CONTINUE
C
          KU = KU + LN
          KN = KU + NU
          LN = 2 * LN
 110    CONTINUE
C
        RETURN
      ENDIF
C
C   Check if input parameters are invalid.
C
      K = U(1)
      MX = MOD (K, 64)
      IF ((IS .NE. 1 .AND. IS .NE. -1) .OR. M .LT. 1 .OR. M .GT. MX)    
     $  THEN
C       WRITE (6, 1)  IS, M, MX
 1      FORMAT ('CFFTZ: EITHER U HAS NOT BEEN INITIALIZED, OR ELSE'/    
     $    'ONE OF THE INPUT PARAMETERS IS INVALID', 3I5)
C       STOP
      ENDIF
C>> A normal call to CFFTZ starts here.  M1 is the number of the first
C     variant radix-2 Stockham iterations to be performed.  The second
C     variant is faster on most computers after the first few
C     iterations, since in the second variant it is not necessary to
C     access roots of unity in the inner DO loop.  Thus it is most
C     efficient to limit M1 to some value.  For many vector computers,
C     the optimum limit of M1 is 6.  For scalar systems, M1 should
C     probably be limited to 2.
C
      N = 2 ** M
C      M1 = MIN (M / 2, 6)
      M1 = MIN (M / 2, 2)
      M2 = M - M1
      N2 = 2 ** M1
      N1 = 2 ** M2
C
C   Perform one variant of the Stockham FFT.
C
      DO 120 L = 1, M1, 2
        CALL FFTZ1 (IS, L, M, U, X, Y)
        IF (L .EQ. M1) GOTO 140
        CALL FFTZ1 (IS, L + 1, M, U, Y, X)
 120  CONTINUE
C
C   Perform a transposition of X treated as a N2 x N1 x 2 matrix.
C
      CALL TRANS (N1, N2, X, Y)
      KTTRANS(NQ) = KTTRANS(NQ) + 1
C
C   Perform second variant of the Stockham FFT from Y to X and X to Y.
C
      DO 130 L = M1 + 1, M, 2
        CALL FFTZ2 (IS, L, M, U, Y, X)
        IF (L .EQ. M) GOTO 180
        CALL FFTZ2 (IS, L + 1, M, U, X, Y)
 130  CONTINUE
C
      GOTO 160
C
C   Perform a transposition of Y treated as a N2 x N1 x 2 matrix.
C
 140  CALL TRANS (N1, N2, Y, X)
      KTTRANS(NQ) = KTTRANS(NQ) + 1
C
C   Perform second variant of the Stockham FFT from X to Y and Y to X.
C
      DO 150 L = M1 + 1, M, 2
        CALL FFTZ2 (IS, L, M, U, X, Y)
        IF (L .EQ. M) GOTO 160
        CALL FFTZ2 (IS, L + 1, M, U, Y, X)
 150  CONTINUE
C
      GOTO 180
C
C   Copy Y to X.
C
 160  CONTINUE
      DO 170 I = 1, 2 * N
        X(I) = Y(I)
 170  CONTINUE
C
 180  CONTINUE
      RETURN
      END
      SUBROUTINE FFTZ2 (IS, L, M, U, X, Y)

      IMPLICIT REAL*8 (A-H, O-Z)
      DIMENSION U(1), X(1), Y(1)
C
      N = 2 ** M
      K = U(1)
      NU = K / 64
      N1 = N / 2
      LK = 2 ** (L - 1)
      LI = 2 ** (M - L)
      LJ = 2 * LK
      KU = LI + 1
C
      DO 100 I = 0, LI - 1
        I11 = I * LK + 1
        I12 = I11 + N1
        I21 = I * LJ + 1
        I22 = I21 + LK
        U1 = U(KU+I)
        U2 = IS * U(KU+I+NU)
C
C   This loop is vectorizable.
C
        DO 100 K = 0, LK - 1
          X11 = X(I11+K)
          X12 = X(I11+K+N)
          X21 = X(I12+K)
          X22 = X(I12+K+N)
          T1 = X11 - X21
          T2 = X12 - X22
          Y(I21+K) = X11 + X21
          Y(I21+K+N) = X12 + X22
          Y(I22+K) = U1 * T1 - U2 * T2
          Y(I22+K+N) = U1 * T2 + U2 * T1
 100  CONTINUE
C
      RETURN
      END
C
      SUBROUTINE TRANS (N1, N2, X, Y)
C
      IMPLICIT REAL*8 (A-H, O-Z)
      DIMENSION X(1), Y(1)
C
      N = N1 * N2

C      IF (N1 .LT. 32 .OR. N2 .LT. 32) THEN
      IF (N1 .GE. N2) THEN
        GOTO 100
      ELSE
        GOTO 120
      ENDIF
C      ELSE
C        GOTO 140
C      ENDIF
C
 100  DO 110 J = 0, N2 - 1
C
C   This loop is vectorizable.
C
        DO 110 I = 0, N1 - 1
          Y(I*N2+J+1) = X(I+J*N1+1)
          Y(I*N2+J+1+N) = X(I+J*N1+1+N)
 110  CONTINUE
C
      GOTO 180
C

 120  DO 130 I = 0, N1 - 1
C
C   This loop is vectorizable.
C
        DO 130 J = 0, N2 - 1
          Y(J+I*N2+1) = X(J*N1+I+1)
          Y(J+I*N2+1+N) = X(J*N1+I+1+N)
 130  CONTINUE
C
      GOTO 180

 140  N11 = N1 + 1
      N21 = N2 + 1
      IF (N1 .GE. N2) THEN
        K1 = N1
        K2 = N2
        I11 = N1
        I12 = 1
        I21 = 1
        I22 = N2
      ELSE
        K1 = N2
        K2 = N1
        I11 = 1
        I12 = N2
        I21 = N1
        I22 = 1
      ENDIF
C
      DO 150 J = 0, K2 - 1
        J1 = J * I11 + 1
        J2 = J * I12 + 1
C
C   This loop is vectorizable.
C
        DO 150 I = 0, K2 - 1 - J
          Y(N21*I+J2) = X(N11*I+J1)
          Y(N21*I+J2+N) = X(N11*I+J1+N)
 150  CONTINUE
C
      DO 160 J = 1, K1 - K2 - 1
        J1 = J * I21 + 1
        J2 = J * I22 + 1
C
C   This loop is vectorizable.
C
        DO 160 I = 0, K2 - 1
          Y(N21*I+J2) = X(N11*I+J1)
          Y(N21*I+J2+N) = X(N11*I+J1+N)
 160  CONTINUE
C
      DO 170 J = K1 - K2, K1 - 1
        J1 = J * I21 + 1
        J2 = J * I22 + 1
C
C   This loop is vectorizable.
C
        DO 170 I = 0, K1 - 1 - J
          Y(N21*I+J2) = X(N11*I+J1)
          Y(N21*I+J2+N) = X(N11*I+J1+N)
 170  CONTINUE
C
 180  RETURN
      END
      SUBROUTINE FFTZ1 (IS, L, M, U, X, Y)

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
