* Example with simple unstructured

      PROGRAM TEST
      INTEGER X(50)    
     
    
      DO 100 I=1,10
         X(I) = 0
         IF ( I.GT.5) GOTO 200        
 100  CONTINUE   
    
 200  PRINT *, 'THIS IS A TEST'
      END

