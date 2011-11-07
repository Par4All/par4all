C
      PROGRAM ESSBOP
C
      INTEGER AA(25)
C      
      
      CALL ESSBOU(AA)
C
      END
C
C
      SUBROUTINE ESSBOU(A)
C
      INTEGER A(25)
C
      K = 0

      DO I=1,5
            A(K) = 0
            K = K * K
      ENDDO
C
      END
