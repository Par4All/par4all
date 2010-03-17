C CODE FROM THE PAPER "ARRAY PRIVATIZATION FOR PARALLEL
C EXECUTION OF LOOPS" FROM Z. LI, ICS'92.

      SUBROUTINE LI921(A,N)
      INTEGER N, A(N), M

      A(1) = 0
      DO I = 1,N
         DO J = 1,N
            DO K = 2,N
               A(K) = 0
               M = A(K-1)
            ENDDO
         ENDDO
      ENDDO

      PRINT *, I, J, K, M

      END
