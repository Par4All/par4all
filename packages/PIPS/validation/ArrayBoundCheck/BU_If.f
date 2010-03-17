* Example with array reference in the condition of test
      PROGRAM TEST
      PARAMETER (N=10)
      INTEGER T(N),X
     
      PRINT *, 'THIS IS A TEST'
      
      IF (T(5).LE.10) THEN 
         T(X) = 10
      ELSE
         T(5) = 10
      ENDIF

      PRINT *,T
      END

