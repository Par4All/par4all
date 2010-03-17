C
C     CHOLESKI METHOD - VERSION 1
C
C     PRIVATIZATION
C     DEPENDENCE COMPUTATION WITH AND WITHOUT EXECUTION CONTEXT
C
      SUBROUTINE CHOLES(A, P, N)
      REAL X, A(N,N), P(N)
C
      DO I = 1, N
         X = A(I,I)
         DO K = 1, I-1
            X = X - A(I,K)*A(I,K)
         ENDDO
         P(I) = 1.0 / SQRT(X)
         DO J = I+1, N
            X = A(I,J)
            DO KK = 1, I-1
               X = X -A(I,J) * A(I,KK)
            ENDDO
            A(J,I) = X * P(I)
         ENDDO
      ENDDO
C
      RETURN
      END
