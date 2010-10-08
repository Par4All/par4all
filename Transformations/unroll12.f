C     To try unrolling with epilogue
C
      PROGRAM unroll12
C
      PARAMETER (N=100)
      REAL T(N+1)
C
 100  DO 10 I = 1, N
         T(I) = -2*I
 10   CONTINUE
      DO 20 I = 1, N+1
         T(I) = -2*I
 20   CONTINUE
C
      print *, i
      END
