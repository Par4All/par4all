
*DECK                                            ERRORS
      SUBROUTINE ERRORS (MESAGE)
      CHARACTER*80 MESAGE
C**********************************************************************
C ERROR HANDLING ROUTINE.  PRINT MESSAGE, ABORT JOB.
C**********************************************************************
      WRITE(6,10) MESAGE
  10  FORMAT(1X,A)
      STOP ' ERRORS'
      END
