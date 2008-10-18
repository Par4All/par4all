*DECK FMTGEN
      SUBROUTINE FMTGEN(F,T,M,ICK)
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION F(M)
      COMMON/IO/IN,IOUT,IPUNCH
      COMMON/FMCONS/FOUR,ONE,HALF,TWO,ZERO,TEN,TENM9,F20,F42,F500
      COMMON/FM/GA(17),RPITWO,FMZERO(17),TOL,CUT0S,CUTSM,CUTML
      EQUIVALENCE(APPROX,OLDSUM)
 2001 FORMAT(41H0FAILURE IN FMGEN FOR SMALL T   IX > 50, /
     $ 6H IX = ,I3,7H,  T = ,D20.14)
C
C     ICK IS AN ERROR INDICATOR.
C     ON RETURN, ICK=0 IMPLIES THAT ALL IS WELL.
C     IF ON RETURN, ICK IS NON-ZERO, THE ASSYMPTOTIC EXPANSION
C     HAS FAILED.
      ICK=0
C     TEST FOR TYPE OF ALGORITHM.
      IF( ABS(T).GT.CUT0S)GO TO 3
C*********************************************************************
C        FILL F(M) FOR ARGUMENT OF ZERO.
C*********************************************************************
      DO 2 I=1,M
    2 F(I)=FMZERO(I)
      RETURN
C
C     TEST FOR EVALUATION OF THE EXP.
    3 TEXP=ZERO
      IF( ABS(T).GE.CUTML)GO TO 150
      TEXP= EXP(-T)
      IF( ABS(T).GE.CUTSM)GO TO 80
C*********************************************************************
C        0 .LT. T .LT. 10
C*********************************************************************
      A= FLOAT(M-1)+HALF
      TERM=ONE/A
      SUM=TERM
      DO 20 IX=2,400
      A=A+ONE
      TERM=TERM*T/A
      SUM=SUM+TERM
      IF( ABS(TERM/SUM)-TOL)30,20,20
   20 CONTINUE
      WRITE(IOUT,2001)IX,T
C***      CALL LNK1E
      STOP
   30 F(M)=HALF*SUM*TEXP
      GO TO 160
C
C*********************************************************************
C        10 .LE. T .LT. 42
C*********************************************************************
   80 A= FLOAT(M-1)
      B=A+HALF
      A=A-HALF
      TX=ONE/T
      MM1=M-1
      APPROX=RPITWO* SQRT(TX)*(TX**MM1)
      IF(MM1)90,110,90
   90 DO 100 IX=1,MM1
      B=B-ONE
  100 APPROX=APPROX*B
  110 FIMULT=HALF*TEXP*TX
      SUM=ZERO
      IF(FIMULT)120,140,120
  120 FIPROP=FIMULT/APPROX
      TERM=ONE
      SUM =ONE
      NOTRMS=  INT(T)+MM1
      DO 130 IX=2,NOTRMS
      TERM=TERM*A*TX
      SUM=SUM+TERM
      IF( ABS(TERM*FIPROP/SUM)-TOL)140,140,130
  130 A=A-ONE
      RATIO= ABS((TERM*FIPROP/SUM))
C     WRITE(IOUT,2002)T,RATIO
      ICK=1
      RETURN
  140 F(M)=APPROX-FIMULT*SUM
      GO TO 160
C*********************************************************************
C        T .GE. 42
C*********************************************************************
  150 TX= FLOAT(M)-HALF
      F(M)=HALF*GA(M)/(T**TX)
C*********************************************************************
C        RECUR DOWNWARDS TO F(1)
C*********************************************************************
  160 TX=T+T
      SUM= FLOAT(M+M-3)
      MM1=M-1
      IF(MM1)170,190,170
  170 DO 180 IX=1,MM1
      F(M-IX)=(TX*F(M-IX+1)+TEXP)/SUM
  180 SUM=SUM-TWO
  190 RETURN
      END
