      SUBROUTINE EXACT(A,B,N,M,J)

      INTEGER N, A(N,N), B(N,N), M, J

      DO I = 1, N
         J = J + I
         A(I,J) = 0
         J = J + 1
         B(I,J) = 0
      ENDDO
      END
