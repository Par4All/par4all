      PROGRAM unroll08
C
      PARAMETER (N=4)
      REAL T(N,N)

C     Outer unrolling with different labels

Cxxx
      DO 10 I = 1, N
C
         DO 20 J = 1, N
            T(I,J) = -N*I+J
 20      CONTINUE
 10   CONTINUE

C     Inner unrolling with different labels

      DO 30 I = 1, N
Cxxx
         DO 40 J = 1, N
            T(I,J) = -N*I+J
 40      CONTINUE
 30   CONTINUE

C     Outer unrolling with same labels

Cxxx
      DO 11 I = 1, N
C
         DO 11 J = 1, N
            T(I,J) = -N*I+J
 11   CONTINUE

C     Inner unrolling with same labels

      DO 31 I = 1, N
Cxxx
         DO 31 J = 1, N
            T(I,J) = -N*I+J
 31   CONTINUE
C
      END
