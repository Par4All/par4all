      PROGRAM CA
      DIMENSION A(10)
C
      DO 101 N = 10, 20
C
         DO 100 I = 15, N
C
            PRINT *,A(I)
C
 100        CONTINUE
 101     CONTINUE
C
      END
