C     The two array are overlapped ?

      PROGRAM ALIAS
      REAL A(10000)
      M = 50
      CALL FOO(A,M)
      END

      SUBROUTINE FOO(X,N)
      REAL X(N,N)
      L = 20
      DO I=1,L
         DO J =1,L
            CALL TOTO(X(I,J),X(2*I,2*J),L)
         ENDDO
      ENDDO
      END
      
      SUBROUTINE TOTO(V1,V2,L)
      REAL V1(L,L),V2(L,L)
  
      DO I=1,L
         DO J =1,L
            V1(2*I-J,J+I)=I+J
         ENDDO
      ENDDO
      END








