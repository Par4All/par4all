      PROGRAM ALIAS 
      REAL A(100), B(100)
      M =10
      N =10
      CALL P(A(1),B(11),M,N)
      END

      SUBROUTINE P(V1,V2,M1,N1)
      REAL V1(M1),V2(N1)
      DO I =1,M1
         V1(I)=I
      ENDDO
      DO I =1,N1
         V2(I)=I
      ENDDO
      
      END
      






