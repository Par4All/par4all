      SUBROUTINE test2(Y,N)
      INTEGER N
      REAL*8 Y(2:2*N)

      DO 10 I=1,N
         DO 10 J=1,N
            Y(I+J) = Y(I)+Y(I+J)
 10         CONTINUE
            END
