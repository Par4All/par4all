      PROGRAM TEST2
      INTEGER SIZE,I,J,K,X
      PARAMETER (SIZE=10)
      INTEGER C(SIZE,SIZE),A(SIZE,SIZE),B(SIZE,SIZE)
         
      DO 100 I = 1, SIZE
         DO 200 J = 1, SIZE
            A(I,J)=1
 200     CONTINUE
 100  CONTINUE

      DO 300 I = 1, SIZE-1
         DO 400 J = 1, SIZE-1
            X=  A(I,J)
            C(I,J) = B(X,J)                                    
 400        CONTINUE
 300     CONTINUE
      END
