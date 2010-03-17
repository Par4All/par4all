* This example has array bound violations !!!

      SUBROUTINE TEST(X,N)
      INTEGER X(N)
     
      PRINT *, 'THIS IS A TEST'
     
    
      DO 100 I=1,N
         
            X(I+1)=0
  
 100  CONTINUE
 
      P = X(-10) +2
      
      END
