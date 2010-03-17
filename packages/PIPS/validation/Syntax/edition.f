C     Bug: implicit DO in IO list not recognized
C
      SUBROUTINE EDITION(NFICH,KPRINT)
C*********************************************************
C     EDITION DU FICHIER  NFICH
C     PAR PLANS  K=CSTE
C     LES TABLEAUX IL,JL,KL SERVENT A PRECISER
C     LES INDICES DES NOEUDS QUI SERONT REPRESENTES
C*********************************************************
      DIMENSION IL(71),JL(36),KL(22),JD(2),JF(2),TIT(2,7)
      COMMON/IO/LEC,IMP,KIMP,NXYZ,NGEO,NDIST
      COMMON/CTI/TITXYZ(8),TITRE0(8),IT0,TITRE(8),IT

c     COMMON/CT/TI(52,21,5),TK(57,52)
      DIMENSION T(52,21,60)
      COMMON/CT/T
      DIMENSION TI(52,21,5),TK(57,52)      
      EQUIVALENCE (T(1,1,1),TI(1,1,1)), (T(1,1,6),TK(1,1))

      COMMON/CI/I1,I2,IMAX,I1P1,I1P2,I2M1,I2M2,IBF
      COMMON/CJ/J1,J2,JMAX,J1P1,J1P2,J2M1,J2M2,JA,JB,JAM1,JBP1
      COMMON/CK/K1,K2,KMAX,K1P1,K1P2,K2M1,K2M2
      COMMON/GAMMA/GAM,GAM1,GAM2,GAM3,GAM4,GAM5,GAM6,GAM7,GAM8,GAM9
      DATA TIT/'        MA','CH        ',
     1         '      ENTR','OPIE      ',
     2         '  PRESSION',' STATIQUE ',
     3         '         R','O         ',
     4         '    ANGLE ','(OX,OY)   ',
     5         '    ANGLE ','(OX,OZ)   ',
     6         ' PAS DE TE','MPS X 1000'/
      DATA NT,JPM,JPP,JPMAX,KPMAX/7,18,19,36,5/
      DATA JD/1,19/,JF/18,36/
C
      RADDEG=45./ATAN(1.)
      PA0=GAM1*(GAM7*0.5)**GAM5
C
C     DEFINITION DES TABLEAUX DE TRI
C
      DO 10 I=I1,I2
  10  IL(I)=I
      DO 20 JP=1,3
  20  JL(JP)=JP
      JL(4)=6
      JL(5)=9
      JL(6)=12
      JL(7)=14
      DO 21 JP=8,JPM
   21 JL(JP)=7+JP
      DO 25 JP=1,JPM
  25  JL(JP+JPM)=JMAX-1-JL(JP)
      DO 30 KP=1,4
  30  KL(KP)=KP
      KL(5 )=KMAX-2
C
c     BUFFER IN (NFICH,0) (TITRE(1),IT)
      IF(UNIT(NFICH)) 40,9999,9999
   40 CONTINUE
      WRITE(IMP,1000) IT
      REWIND NFICH
C
      DO 100 N=1,NT
      ICHIF=KPRINT/10**(NT-N)-10*(KPRINT/10**(NT-N+1))
      IF(ICHIF.EQ.0) GO TO 100
      DO 200 KP=1,KPMAX
c     BUFFER IN (NFICH,0) (TITRE(1),IT)
      IF(UNIT(NFICH)) 41,9999,9999
   41 CONTINUE
      DO 210 II=I1,I2
c     BUFFER IN (NFICH,0) (TI(1,1,1),TI(JMAX,KMAX,5))
      IF(UNIT(NFICH)) 42,9999,9999
   42 CONTINUE
      I=I1+I2-II
      IP=IL(I)
      IF(IP.EQ.0) GO TO 210
      K=KL(KP)
      DO 220 JP=1,JPMAX
      J=JL(JP)
      RO =TI(J+1,K+1,1)
      ROU=TI(J+1,K+1,2)
      ROV=TI(J+1,K+1,3)
      ROW=TI(J+1,K+1,4)
      IF((ROU**2+ROV**2).LT.1.E-12) ROV=1.E-6
      IF((ROU**2+ROW**2).LT.1.E-12) ROW=1.E-6
      QQ=(ROU*ROU+ROV*ROV+ROW*ROW)/(RO*RO)
      P=RO*(GAM1-GAM2*QQ)
      IF(N.EQ.1) TTT=SQRT(RO*QQ/(GAM*P))
      IF(N.EQ.2) TTT=((RO**GAM)/(GAM*P))
      IF(N.EQ.3) TTT=P/PA0
      IF(N.EQ.4) TTT=RO
      IF(N.EQ.5) TTT=ATAN2(ROV,ROU)*RADDEG
      IF(N.EQ.6) TTT=ATAN2(ROW,ROU)*RADDEG
      IF(N.EQ.7) TTT=TI(J+1,K+1,5)*1000.
      TK(IP,JP)=TTT
  220 CONTINUE
  210 CONTINUE
C
      DO 230 KAS=1,2
      JDEB=JD(KAS)
      JFIN=JF(KAS)
      WRITE(IMP,2000) K,(TIT(L,N),L=1,2)
      WRITE(IMP,2500) (JL(JP),JP=JDEB,JFIN)
      WRITE(IMP,2501)
C
      DO 230 I=I1,I2
      IP=IL(I)
      IF(IP.EQ.0) GO TO 230
      IF(N.GE.5) GO TO 231
      WRITE(IMP,2001) I,(TK(IP,JP),JP=JDEB,JFIN)
      GO TO 230
  231 CONTINUE
      WRITE(IMP,2002) I,(TK(IP,JP),JP=JDEB,JFIN)
  230 CONTINUE
      REWIND NFICH
  200 CONTINUE
  100 CONTINUE
C
C*******************************************************************
 1000 FORMAT(///,' NOMBRE D ITERATIONS EFFECTUEES',I5/
     1          ,' -----------------------------------')
 2500 FORMAT(2X,'J',18(5X,I2))
 2501 FORMAT(1X,130('-'))
 2001 FORMAT(I3,2X,18F7.4)
 2002 FORMAT(I3,2X,18F7.2)
 2000 FORMAT(1H1,44X,'   K=',I2,2A10,/
     1          ,45X,'------------------------------',/)
C
      RETURN
 9999 CONTINUE
      STOP7
      END
      FUNCTION UNIT(I)
      UNIT = I
      END
