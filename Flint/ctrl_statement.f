C     This program is written for testing module TYPING:
C     * Verification the syntax of control statement *
C
C     Written by PHAM DINH Son - trainee 03/00-09/00
C     May, 30, 2000
C
      PROGRAM CTRL_STATEMENT
      INTEGER I

C     Without arguments
C     =================
      CONTINUE

C     At most one argument: 
C        * Integer constant with not more than 5 digits
C        * Character constant
C     =======================
      STOP
      STOP 12345
      STOP "STRING"

      STOP 1234567890
      STOP I

      END
