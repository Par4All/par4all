CDECK RATNUC
      SUBROUTINE  RATNUC
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      SAVE
C     ------------------------------------------------------------------
      COMMON  /VECTA/  VA(401),SA(401),PA(401),PEA(401),TA(401)
      COMMON  /VECTA/  RA(401),TAUA(401),FRA(401),FCA(401)
      COMMON  /VECTA/  QNUC(401),RATX(401)
      COMMON  /VECTA/  CPA(401),DADA(401)
      COMMON  /VECTA/  CC(401),UA(401),QT(401),EA(401)
      COMMON  /VECTA/  FCT(401),TAUC(401)
      COMMON  /VECTA/  PVS(401)
      COMMON  /VECTA/  OPACA(401),DOPDVA(401),DOPDTA(401)
      COMMON  /VECTAI/ KONV(401)
C     ------------------------------------------------------------------
      COMMON  /COMPOS/  XXX,YYY,ZZZ
      COMMON  /COMPOS/  XXA(401),YYA(401),ZZA(401)
      COMMON  /COMPOS/  EFF(401),DRODEF(401)
      COMMON  /COMPOSI/ JPOINT
C     ------------------------------------------------------------------
      COMMON  /COMPROG/  FACMIX,FLUM,PEXTER,TAUEX,PSTART
      COMMON  /COMPROI/  NP,NPP1
C     ------------------------------------------------------------------
      DO 1 I=2,NPP1
         QNUC(I)=0.0D0
         RATX(I)=0.0D0
         X=XXA(I)
         Y=YYA(I)
         ZCN=ZZA(I)
         EPP=0.D0
         ECN=0.D0
         T=TA(I)/1.D+6
         T13=T**(1.D0/3.D0)
         RO=1.D0/VA(I)
         IF(T.GE.1.0D0) THEN
            IF(X.GE.1.D-20) THEN
               SQTR=SQRT(RO/T**3)
               W=1.22D16*(X/(1.D0+X))*(1.D0+SQTR)/SQRT(T13)
     1              *EXP(-102.6D0/T13)
               ALPHA=5.48D17*(Y*0.25D0/X)**2*EXP(-100.D0/T13)
C     changed to avoid data dependencies
C               IF(ALPHA.LE.1.D5) GAMA=SQRT(ALPHA*(2.D0+ALPHA))-ALPHA
C               IF(ALPHA.GT.1.D5) GAMA=1.D0-0.5D0/ALPHA
C     1              *(1.D0-1.D0/ALPHA)
               IF(ALPHA.LE.1.D5) THEN
                  GAMA=SQRT(ALPHA*(2.D0+ALPHA))-ALPHA
                  ELSE
                     GAMA=1.D0-0.5D0/ALPHA *(1.D0-1.D0/ALPHA)
               ENDIF
               PSI=1.D0+GAMA*(0.959D0+0.47D0*W)/(1.D0+W)
               G11=1.D0+0.0012D0*T13+0.0078D0*T13**2+0.0006D0*T
               F11=1.D0+0.25D0*SQTR
               EPP=2.206D6*RO*X**2/T13**2*EXP(-33.81D0/T13)
     1              *F11*G11*PSI
               IF(T.GE.6.D0) THEN
                  IF(ZCN.GE.1.D-10) THEN
                     F14=1.D0+1.75D0*SQTR
                     G14=1.D0+0.0027D0*T13-0.0037D0*T13**2-0.0007D0*T
                     ECN=7.94D27*RO*X*ZCN/T13**2*EXP(-152.313D0/T13)
     1                    *F14*G14
                  ENDIF
               ENDIF
 2             CONTINUE
               E=EPP+ECN
               RAT=EPP/6256.D0+ECN/6035.D0
               RAT=RAT/1.D15
               QNUC(I)=E
               RATX(I)=RAT
            ENDIF
         ENDIF
    1 CONTINUE
      QNUC(1)=0.0D0
      RATX(1)=0.0D0
      RETURN
      END
