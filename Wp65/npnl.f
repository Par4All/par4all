      PROGRAM NPNL
      INTEGER SIZE
      PARAMETER (SIZE=100)
      REAL C(SIZE,SIZE),A(SIZE,SIZE),B(SIZE,SIZE),T(SIZE,SIZE)

       DO 100 I = 1, SIZE
         DO 200 J = 1, SIZE
            T(I,J) = A(I,J)
             DO 300 K = 1, SIZE
              A(I,J) = A(I+1,J+1)                                    
 300       CONTINUE
 200    CONTINUE
 100  CONTINUE

      DO 99994 I = 1, SIZE
         DO 99997 J = 1, SIZE
            DO 99998 K = 1, SIZE
               B(I,J) = C(I+1,J+1)                                 
99998       CONTINUE
            DO 99996 K = 1, SIZE
               C(I,J) = B(I+1,J+1)                                  
99996       CONTINUE
99997    CONTINUE
99994 CONTINUE
      END
