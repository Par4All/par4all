      SUBROUTINE TEST7(N,A,B)
      INTEGER N
      REAL A(N,N),B(N,N)

      DO 10 L = 1, N
         DO 20 J = L+1, N
            A(L,J) = A(L,J)/A(L,L)
 20      CONTINUE
         DO 30 I = L+1, N
            DO 40 K = L+1, N
               B(I,K) = A(I,L) * A(L,K)
               A(I,K) = A(I,K) - B(I,K)
 40         CONTINUE
 30      CONTINUE
 10   CONTINUE
      END
