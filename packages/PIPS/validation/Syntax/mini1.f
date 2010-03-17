      PROGRAM MINI1
      INTEGER  NIRR
      REWIND NIRR
      IF (INIRR.NE.1) THEN
    1    READ(NIRR,1000,END=2)
         GOTO 1
      ENDIF
      INIRR=0
2     CONTINUE   
      END
