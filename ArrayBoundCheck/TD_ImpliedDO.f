* Example with Implied-DO 

      PROGRAM IMPLIEDDO
      REAL X(20,20)
      
      WRITE (6,*), ((X(I,J),J=1,10),I=1,10)
      END

