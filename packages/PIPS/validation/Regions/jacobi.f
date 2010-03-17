C
C     JACOBI METHOD
C
C     LOOP DISTRIBUTION
C     INTERNAL SEQUENTIAL LOOP AND EXTERNAL PARALLEL LOOP
C
      SUBROUTINE JACOBI(N, A, B, X, Y)
      INTEGER N
      REAL A(N,N), B(N), X(N), Y(N)
C
      DO I = 1,N
         Y(I) = B(I)
         DO J = 1,N
            Y(I) = Y(I) - X(J)*A(I,J)
         ENDDO
      ENDDO
C
      DO I = 1,N
         X(I) = X(I) +Y(I)/A(I,I)
      ENDDO
C
      RETURN
      END
