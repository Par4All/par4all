* Example with array references in the condition of while loop
    
      PROGRAM WHILELOOP
      PARAMETER (NUM=10)
      REAL TOSORT(NUM),TEMP
      INTEGER I,J,NUM

      DO 100 I=1,NUM-1
         DO 100 J=I,1,-1
            DO WHILE (TOSORT(J).GT.TOSORT(J+1))
               TEMP =TOSORT(J)
               TOSORT(J)=TOSORT(J+1)
               TOSORT(J+1)=TEMP
            ENDDO

 100  CONTINUE
      END

