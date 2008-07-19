      PROGRAM TEST5
      REAL V1(10,10),V2(10,10)

      DO 10 I=1,10
         DO 10 J=1,10
            V1(I,J)=0.0
            V2(I,J)=V2(I,J) + V1(I+1,J)
 10      CONTINUE
      END
