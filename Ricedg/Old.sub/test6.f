      PROGRAM TEST6
      REAL MAT(10,10)

      DO 10 I=1,10
         MAT(I,I) = 0.0
         DO 20 J=1,I-1
            MAT(I,J)=1.0
            MAT(J,I) = -1.0
 20      CONTINUE
 10   CONTINUE
         END
