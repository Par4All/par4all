C
C
      PROGRAM UNROLL09
C
      PARAMETER (N=100)
      REAL T(N)
C
      DO 10 I = 1, 4
         IF(I.EQ.M) GO TO 10
         T(I+N) = -2*I
 10   CONTINUE
C
      END
