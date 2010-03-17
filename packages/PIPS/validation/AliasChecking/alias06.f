      PROGRAM MAIN  
      REAL A(10000)  
      M = 50  
      DO I=1,M  
         CALL SUB1(A(I),M)  
      ENDDO   
      END  

      SUBROUTINE SUB1(X,N)  
      REAL X(N,N)  
      L = 20   
      DO I = 1,L  
         DO J =1,L   
            CALL SUB2(X(I,J),X(2*I,2*J),L)   
         ENDDO  
      ENDDO  
      END   

      SUBROUTINE SUB2(V1,V2,L)  
      REAL V1(L),V2(L)  
      DO I = 1,L  
         V1(I)=I  
      ENDDO  
      END 
