      SUBROUTINE CREATE


      IF       (I .EQ. 1)  THEN
	  GOTO 9999
      ELSE IF  (I .EQ. 2)  THEN
          I  = 3
      ENDIF
8888  RETURN
9999  i = 4
      END
