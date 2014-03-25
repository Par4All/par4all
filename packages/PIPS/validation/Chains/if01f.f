C     Fortran version
C     use-def chains with if/else
C     no dependence between if and else case have to be done
      PROGRAM IF01F
      
      INTEGER R, A
      
      R = -1
      
      IF (RAND(0) .LE. 1) THEN
        R = 1
      ELSE
        R = 0
      END IF

      A = R
      RETURN 
      END
      
C***********************************************************************
      FUNCTION RAND(N)
C***********************************************************************
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      COMMON /RANDNO/ R3(127),R1,I2
      SAVE S, T, RMC
      SAVE IW
      DATA S, T, RMC /0.0D0, 1.0D0, 1.0D0/
      DATA IW /-1/
C
      IF(R1.LT.1.0D0.AND.N.EQ.0) GO TO 60
      IF(IW.GT.0) GO TO 30
   10 IW=IW+1
      T=T*0.5D0
      R1=S
      S=S+T
      IF(S.GT.R1.AND.S.LT.1.0D0) GO TO 10
      IKT=(IW-1)/12
      IC=IW-12*IKT
      ID=2**(13-IC)
      DO 20 I=1,IC
   20 RMC=0.5D0*RMC
      RM=0.015625D0*0.015625D0
   30 I2=127
      IR=MOD(ABS(N),8190)+1
   40 R1=0.0D0
      DO 50 I=1,IKT
      IR=MOD(17*IR,8191)
   50 R1=(R1+REAL(IR/2))*RM
      IR=MOD(17*IR,8191)
      R1=(R1+REAL(IR/ID))*RMC
      R3(I2)=R1
      I2=I2-1
      IF(I2.GT.0) GO TO 40
   60 IF(I2.EQ.0) I2=127
      T=R1+R3(I2)
      IF(T.GE.1.0D0) T=(R1-0.5D0)+(R3(I2)-0.5D0)
      R1=T
      R3(I2)=R1
      I2=I2-1
      RAND=R1
      RETURN
      END
      