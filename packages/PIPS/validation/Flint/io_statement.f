C     This program is written for testing module TYPING:
C     * Verification the syntax of IO statements *
C
C     Written by PHAM DINH Son - trainee 03/00-09/00.
C     May, 30, 2000
C
      PROGRAM IO_STATEMENT
      INTEGER I
      CHARACTER CH
      LOGICAL L

 1234 PRINT *, "1234 IS A LABEL STATEMENT"

C     UNIT specifier : Integer or character expression
C     ================================================
      READ(UNIT=12345, FMT=0)
      READ(UNIT="STRING", FMT=0)
      READ(UNIT=I, FMT=0)
      READ(UNIT=CH, FMT=0)

      READ(UNIT=.TRUE., FMT=0)
      READ(UNIT=L, FMT=0)

C     Specifier must be a label statement
C     Like: ERR, END
C     ===================================
      READ(UNIT=12345, FMT=0, ERR=1234, END=1234)

      READ(UNIT=12345, FMT=0, ERR=L, END=I)
      READ(UNIT=12345, FMT=0, ERR=1234, END=CH)

C     Specifier must be a label statement, integer or character expression
C     Like: FMT, REC
C     ====================================================================
      READ(UNIT=12345, FMT=1234, REC=0)
      READ(UNIT=12345, FMT=I, REC=CH)
      READ(UNIT=12345, FMT="STRING", REC=12345678)

      READ(UNIT=12345, FMT=L, REC=.TRUE.)

C     Specifier must be a character expression
C     Like: FILE, STATUS, ACCESS, FORM, BLANK
C     ========================================
      OPEN(UNIT=123, FILE="STRING", STATUS="OLD", ACCESS="SEQUENTIAL")

      OPEN(UNIT=123, FILE=CH, STATUS=I, ACCESS=1234, FORM=L)

C     Specifier must be an integer expression
C     Like: RECL
C     ========================================
      OPEN(UNIT=123, FILE="STRING", RECL=123)
      OPEN(UNIT=123, FILE="STRING", RECL=I)

      OPEN(UNIT=123, FILE="STRING", RECL=CH)


C     Specifier must be a LOGICAL variable or array element
C     Like: EXIST, OPENED, NAMED
C     =====================================================
      INQUIRE(IOSTAT=I, OPENED=L, NAMED=L, EXIST=L)

      INQUIRE(IOSTAT=I, OPENED=.TRUE., NAMED=I, EXIST=CH)

C     Specifier must be an INTEGER variable or array element
C     Like: IOSTAT, NUMBER, NEXTREC
C     =====================================================
      INQUIRE(IOSTAT=I, NUMBER=I, NEXTREC=I)

      INQUIRE(IOSTAT=12, NUMBER=12, NEXTREC=1234)
      INQUIRE(IOSTAT=12, NUMBER=L, NEXTREC=CH)

C     Specifier must be a CHARACTER variable or array element
C     Like: NAME, SEQUENTIAL, DIRECT, FORMATTED, UNFORMATTED
C     ======================================================
      INQUIRE(IOSTAT=I, NAME=CH, SEQUENTIAL=CH, DIRECT=CH)

      INQUIRE(IOSTAT=I, NAME="STRING", SEQUENTIAL=12, DIRECT=I)

      END



