C PROJNE : stands for PROJecton Not Exact. 
C It is the same program as renpar6.f, but
C for the function that modifies K.
C In this case, the transformer contains only
C an enequality, and the projection cannot preserve
C the must approximation for the array WORK.

      SUBROUTINE PROJNE(A,N,K,M)
      INTEGER N,K,M,A(N)
      DIMENSION WORK(100,100)
         K = M * M
      IF (K.GT.0) THEN
         DO I = 1,N
            DO J = 1,N
               WORK(J,K) = J + K
            ENDDO
            
            CALL TOTO(K)
            
            DO J = 1,N
               WORK(J,K) = J * J - K * K
               A(I) = A(I) + WORK(J,K) + WORK(J,K-1)
            ENDDO
         ENDDO
      ENDIF
      END

      
      SUBROUTINE TOTO(I)
      IF (I.LT.10) THEN
         I = I + 10
      ENDIF
      END


