      PROGRAM unroll07
C
      PARAMETER (N=4)
      REAL T(N,N)
Cxxx
      DO 10 I = 1, N
Cxxx
      DO 10 J = 1, N
         T(I,J) = -N*I+J
 10   CONTINUE
C
      END
