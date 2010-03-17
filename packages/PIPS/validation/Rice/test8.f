      SUBROUTINE TEST8(L,N,M,A,B,C)
      INTEGER L,N,M
      REAL A(L,M), B(M,N), C(L,M)

      DO 10 I = 1, L 
         DO 20 J = 1, N
            C(I,J) = 0.0
            DO 30 K = 1, M 
               C(I,J) = C(I,J) + A(I,K) * B(K,J)
 30         CONTINUE
 20      CONTINUE
 10   CONTINUE
      END
