C
C
      PROGRAM unroll
C
      PARAMETER (N=100)
      REAL T(N)
C
      DO 10 I = 1, 100
         T(I) = -2*I
 10   CONTINUE
C
      END
