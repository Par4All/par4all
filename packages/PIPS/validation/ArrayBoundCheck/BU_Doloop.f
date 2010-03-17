* Example with array reference in do loop
* After apply PARTIAL_EVAL, we do not need SUPPRESS_DEAD_CODE any more !

      SUBROUTINE DOLOOP(X,N)
      REAL X(N,N)
     
      PRINT *, 'THIS IS A TEST'
     
      S = 0.0
      DO 200 I=1,N
         DO 100 J= 1,N/2
            S = S + X(I,2*J-1)
 100     CONTINUE
 200  CONTINUE
      
      END
