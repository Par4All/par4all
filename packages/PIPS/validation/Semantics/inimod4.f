      PROGRAM INIMOD4

C     Simplified version of inimod4

      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 )
      PARAMETER  ( XLZ =       30.0D0 )
      PARAMETER  ( XCZ =       15.0D0 )

      MPCON =  NINT( MP * XCZ / XLZ )
      PRINT *, MPCON
      END
