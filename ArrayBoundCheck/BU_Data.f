* Example with the DATA declaration
* Array bound checking in PIPS do not add test to this declaration

      PROGRAM TEST
      REAL X(50)
      DATA (X(I),I=-5,10)/3*0.0/
      
      N = 20
      
      DO 100 I=1,N
         X(I)=0.
 100  CONTINUE         
      END

