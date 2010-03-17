C
C     GAUSS METHOD
C
C     DEPENDENCE COMPUTAION WITH AND WITHOUT EXECUTION CONTEXT
C     PRIVATIZATION
C
      SUBROUTINE GAUSS(A, N)
C     
      INTEGER N
      REAL A(N,N)
C     
      DO L = 1,N-1
         XAUX = A(L,L)
         DO J = L+1, N
            A(L,J) = A(L,J) / XAUX
         ENDDO
         DO I = L+1, N
            XAUX = A(I,L)
            DO K = L+1,N
               A(I,K) = A(I,K) - XAUX*A(L,K)
            ENDDO
         ENDDO
      ENDDO         
C     
      RETURN
      END
