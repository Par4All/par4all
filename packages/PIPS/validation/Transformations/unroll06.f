      PROGRAM unroll06
C
      PARAMETER (N=4)
      REAL T(N)
C     First: pragma inside a comment; Second: just the pragma; third:
C     nor pragma at all
C
Cxxx
      DO 10 I = 1, N
         T(I) = -2*I
 10   CONTINUE
Cxxx
      DO 20 I = 1, N
         T(I) = -2*I
 20   CONTINUE
C
      DO 30 I = 1, N
         T(I) = -4*I
 30   CONTINUE
C
      END
