C     Matrix multiplicaiton, version with call to MATMUL

      PROGRAM MAIN   
      PARAMETER (L=200, M=200, N=200)
      REAL Z(L,N), X(L,M), Y(M,N)

C     Initialization

      DO I = 1,L
         DO J = 1,M
            X(I,J) = I
         ENDDO
      ENDDO

      DO J = 1,M
         DO K = 1,N
            Y(J,K) = J
         ENDDO
      ENDDO
     
      CALL MATMUL(Z,X,Y,L,M,N)
   
      END

      SUBROUTINE MATMUL(Z,X,Y,L,M,N)
      REAL Z(L,N), X(L,M), Y(M,N)
      
    
      DO I=1,L
         DO K=1,N
            Z(I,K) = 0.
            DO J=1,M
               Z(I,K) = Z(I,K) + X(I,J)*Y(J,K)
            ENDDO
         ENDDO
      ENDDO
      END



















