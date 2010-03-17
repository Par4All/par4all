      PROGRAM SAMAN
      INTEGER I,J, N, WORK(100), A(200)
      
      N = 50

      DO J =1, 100
         WORK(J) = 0
      ENDDO

      DO J = 1,200
         DO I = 1,N
            WORK(I) = I+J
         ENDDO
         DO I = 1,2*N
            A(J) = WORK(I)
         ENDDO
      ENDDO
      END
      
