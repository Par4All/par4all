C     The two arrays are not overlapped

      PROGRAM ALIAS
      REAL A(10000)
      M = 50
      CALL FOO(A,M)
      END

      SUBROUTINE FOO(X,N)
      REAL X(100,100)
      L = 20
      DO I=1,L
         DO J =1,L
            CALL TOTO(X(I,J),X(2*I,2*J),I,J,L)
         ENDDO
      ENDDO
      END
      
      SUBROUTINE TOTO(V1,V2,II,JJ,L)
      REAL V1(L),V2(L)  
      DO I=1,L
         V1(I)=I
      ENDDO
      END








