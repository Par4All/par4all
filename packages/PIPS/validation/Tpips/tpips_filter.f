C
C
      PROGRAM stf
C
      INTEGER I
      INTEGER J
C
      IF (I .EQ. 1) GOTO 100
      J = 2
      GOTO 200

 100  CONTINUE
      J = 3

 200  CONTINUE
      END
