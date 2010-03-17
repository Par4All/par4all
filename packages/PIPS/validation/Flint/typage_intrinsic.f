C     This program is written for testing module TYPING
C
C
      PROGRAM TYPAGE_INTRINSIC
      LOGICAL L
      INTEGER I
      REAL*4 R
      REAL*8 D
      COMPLEX*8 C
      COMPLEX*16 E
      CHARACTER CH

C     Conversion intrinsics
C     =====================
      I = INT(R)
      I = INT(D)
      I = INT(C)
      I = INT(CH)               ! ERROR
      I = INT(L)                ! ERROR

      R = REAL(I)
      R = REAL(D)
      R = REAL(C)
      R = REAL(CH)              ! ERROR
      R = REAL(L)               ! ERROR

      D = DBLE(I)
      D = DBLE(R)
      D = DBLE(C)
      D = DBLE(CH)              ! ERROR
      D = DBLE(L)               ! ERROR

      C = CMPLX(I)
      C = CMPLX(R)
      C = CMPLX(D)
      C = CMPLX(CH)             ! ERROR
      C = CMPLX(L)              ! ERROR

C     Argument & Result: REAL or DOUBLE
C     ========================
      I = TAN(I)                ! ERROR
      R = TAN(R)
      D = TAN(D)
      I = TAN(C)                ! ERROR
      I = TAN(L)                ! ERROR

C     Argument & Result: REAL, DOUBLE or COMPLEX
C     =================================
      I = EXP(I)                ! ERROR
      R = EXP(R)
      D = EXP(D)
      C = EXP(C)
      I = EXP(L)                ! ERROR

      I = LOG(I)                ! ERROR
      R = LOG(R)
      D = LOG(D)
      C = LOG(C)
      I = LOG(L)                ! ERROR

C     Argument & Result: INT, REAL or DOUBLE
C     =============================
      I = MIN(I, 15, I)
      R = MIN(I, 15, R)
      D = MIN(I, 15, D)
      C = MIN(I, 15, C)         ! ERROR
      I = MIN(I, 15, L)         ! ERROR

      I = MOD(I, I)
      R = MOD(R, R)
      D = MOD(D, D)
      I = MOD(R, D)
      C = MOD(C, C)             ! ERROR

C     Intrinsic ABS: IRDC -> IRDR
C     ===========================
      I = ABS(I)
      R = ABS(R)
      D = ABS(D)
      R = ABS(C)
      C = ABS(C)
      I = ABS(L)                ! ERROR

C     Type_argument --> Type_result
C     ===================================
C     INT  ->  INT
      I = IABS(I)
      I = IABS(R)               ! ERROR
      I = IABS(D)               ! ERROR
      I = IABS(C)               ! ERROR
      
C     REAL -> REAL
      R = ALOG(I)               ! ERROR
      R = ALOG(R)
      R = ALOG(D)               ! ERROR
      R = ALOG(C)               ! ERROR
      
C     DOUBLE -> DOUBLE
      D = DLOG(I)               ! ERROR
      D = DLOG(R)               ! ERROR
      D = DLOG(D)
      D = DLOG(C)               ! ERROR
      
C     COMPLEX -> COMPLEX 
      C = CLOG(I)               ! ERROR
      C = CLOG(R)               ! ERROR
      C = CLOG(D)               ! ERROR
      C = CLOG(C)

C     CHAR -> INT
      I = ICHAR(CH)
      D = ICHAR(R)              ! ERROR
      
      I = LEN(CH)
      I = LEN(R)                ! ERROR
      
      I = INDEX(CH)
      I = INDEX(R)              ! ERROR
      
C     INT      ->   CHAR
      CH = CHAR(I)
      CH = CHAR(D)              ! ERROR

C     REAL     ->   INT
      I = MIN1(I, I)            ! ERROR
      I = MIN1(R, R)
      I = MIN1(I, R)            ! ERROR
      I = MIN1(I, R, D)         ! ERROR
      I = MIN1(I, R, D, C)      ! ERROR
      
C     COMPLEX  ->   REAL
      R = AIMAG(I)              ! ERROR
      R = AIMAG(R)              ! ERROR
      R = AIMAG(D)              ! ERROR
      R = AIMAG(C)
      
C     REAL     ->   DOUBLE
      D = DPROD(I)              ! ERROR
      D = DPROD(R)
      D = DPROD(D)              ! ERROR
      D = DPROD(C)              ! ERROR
      
C     INT      ->   REAL
      R = AMIN0(I, I)
      R = AMIN0(I, R)           ! ERROR
      R = AMIN0(R, D)           ! ERROR
      R = AMIN0(D, C)           ! ERROR
      
C     CHAR     ->   LOGICAL
      L = LGE(I, CH)            ! ERROR
      L = LGE(R, CH)            ! ERROR
      L = LGE(CH, CH)
      
      END






