C     Bug: an intrinsic is overriden
C     SG: no longer a bug, PIPS supports this :p

      FUNCTION SIGN(X,Y)
C
      X1=0.47619
      Y1=X1
      X2=0.9523
      Y2=X2
      IREG=3
      IF(X.GT.X2.AND.Y.GT.Y1) IREG=1
      IF(X.GT.X1.AND.X.LE.X2.AND.Y.GT.Y2) IREG=1
      IF(X.GT.X1.AND.X.LE.X2.AND.Y.GT.Y1.AND.Y.LE.Y2) IREG=2
      GO TO (6,7,8),IREG
C
    6 CONTINUE
      SIGN=0.02*(2.1**2)
      RETURN
C
    7 CONTINUE
      SIGN=0.03*(2.1**2)
      RETURN
C
    8 CONTINUE
      SIGN=0.05*(2.1**2)
      RETURN
C
      END 
