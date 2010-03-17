* Example with indirection's  array reference 

      PROGRAM INDIRECTION
      REAL X(500),Y(500)
     
      PRINT *, 'THIS IS A TEST'
      N = 10
      S = 0.0
      DO 100 I=1,N-1
         S = S + X(Y(I))*Y(X(N-I))
 100  CONTINUE   
      WRITE (*,'("b",F15.5)')S
      END

