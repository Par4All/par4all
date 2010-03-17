C
      PROGRAM OA118
C
C     ******************************************************************
C     *
C     *
C     *    ONERA--DIRECTION DE L'AERODYNAMIQUE
C     *
C     *
C     *               DIVISION OAT2
C     *
C     *
C     *
C     *    DEVELOPPED BY  T.H. LE    (PT.24-30)
C     *
C     *
C     *    VECTORISED BY  H. BOILLOT    (PT.25-38)
C     * non //
C     *
C     ******************************************************************
C
C     Modifications:
C      - in OAMAT1, common CFN is shorter than in OAVITEL; since PIPS
C     requires uniquely defined commons, it had to be extended with
C     the array AIRE(NFAC); Francois Irigoin, December 1990
C      - in OAMAT1 and OAVITEL common OO is shorter than in OA118; as
C     mentionned above, it had to be extended; Francois Irigoin, December 1990
C      - in SOLV, IF()GOTO replaced by IF THEN ENDIF; Francois Irigoin, 
C     March 1991
C      - in SOLV3, IF THEN GOTO 1 ENDIF replaced by ELSE; Francois Irigoin,
C     April 1991
C      - in PHWPAN, IF GOTO 6 replaced by IF THEN ENDIF; Francois Irigoin,
C     April 1991
C      - in PHWAK, IF GOTO 6 replaced by IF THEN ENDIF; Francois Irigoin,
C     April 1991
C      - in CLOCK, call to TIME commented out; Francois Irigoin, March 1993
C
      PARAMETER(NFAC=49,NKJ=3,NDOM=1,ALWAK=+2.,NSAU=1,
     1          ORI=+0.0,CLO=1.000,ALPHA=02.0,DERI=+0.0,SREF=1./3.,
     2            NDOW=1,DWAK=100.)
      COMMON/OASET/PIO2
      COMMON/EE/E11,E22,E33
C
      COMMON/GEF/XF(NFAC,0:4),YF(NFAC,0:4),ZF(NFAC,0:4)
      COMMON/OO/O(NFAC),VS(NFAC),PS(NFAC)
      COMMON/FX/FXX(NFAC),FXY(NFAC),FXZ(NFAC)
      COMMON/FE/FEX(NFAC),FEY(NFAC),FEZ(NFAC)
      COMMON/CFN/FNX(NFAC),FNY(NFAC),FNZ(NFAC),AIRE(NFAC)
      COMMON/PCP/XO(NFAC),YO(NFAC),ZO(NFAC)
      COMMON/OACOF/AA(NFAC,NFAC)
      COMMON/GRA/GRD(NFAC,3)
      COMMON/OT/SWX(4,NKJ),SWY(4,NKJ),SWZ(4,NKJ),OTXE(NKJ)
      COMMON/OAWAK/PHW(NFAC)
      DIMENSION IY(2),CPC(NFAC)
      DIMENSION XLE(NKJ),CD(NKJ),JEX(NKJ),JIX(NKJ)
      DIMENSION INE1(NDOW),INE2(NDOW),INI1(NDOW),INI2(NDOW)
      DIMENSION IMD(NDOM),JMD(NDOM),IFA(NDOM)
      DIMENSION R(NKJ),RL(NKJ),PSV(NKJ,NKJ)
      DIMENSION CC0(NKJ),CC(NKJ,NKJ)
      INTEGER*4 CLOCK,T1,T2,T3,T4,T

  102 FORMAT(1H1,53X,20HDONNEES GEOMETRIQUES//22X,1HI,7X,1HX,7X,1HY,9X,
     11HZ,8X,2HXO,7X,2HYO,7X,2HZO,8X,2HNX,6X,2HNY,6X,2HNZ//)
  104 FORMAT(2X,I5,3(1X,F12.6),1X,3(1X,F12.6),1X,3(1X,F12.6))
  106 FORMAT(7X,3(1X,F12.6))
  600 FORMAT(E12.5,5(1X,E12.5))
  601 FORMAT(62X,'*INCIDENCE:ALPHA=*',F7.4,//)
  606 FORMAT(1H1,30X,'RESULTATS COMPLETS DU CALCUL',//,5X,1HI,7X,1HX,10X
     1,1HY,10X,1HZ,12X,1HU,10X,1HV,10X,1HW,8X,2HKP/)
  607 FORMAT(40X,I3,5X,4(1X,I3),5X,4(1X,I3))
  608 FORMAT(1X,I5,1X,3(F12.6,1X),2X,3(F12.6,1X),2X,F12.6)
  609 FORMAT(/,40X,'ORIGINE DU REPERE:X=',F10.4,8X,'*LONGUEUR DE REFERENC
     *:C=',F10.4,//63X,'*NOMBRE DE FACETTES:*',I5)
  610 FORMAT(1X,'*CONTROLE*',2X,I5,5(E12.5,6X))
  613 FORMAT(//////64X,'*PARAMETRES DE CALCUL*',///)
  700 FORMAT(2I6,8(2X,E12.5))
  731 FORMAT(2I6,8E12.5)
  732 FORMAT(3I5,8E12.5)
  733 FORMAT(1X,10(2X,I5))
       OPEN(2,FILE='oares')
C
C     ******************************************************************
C     ***PARAMETRES DE CALCUL*******************************************
C     ******************************************************************
C
      T1 = CLOCK()
      PI=4.*ATAN(1.)
      PIO2=2.*PI
      PIO4=4.*PI
      RPIO4=1./PIO4
      IY(1)=1
      IY(2)=-1
C
      ALFA=ALPHA*PI/180.
      BETTA=DERI*PI/180.
      AMWAK=ALWAK*PI/180.
      UINFI=COS(ALFA)*COS(BETTA)
      VINFI=SIN(BETTA)*COS(ALFA)
      WINFI=SIN(ALFA)
C
      INE1(1)=43
      INE2(1)=45
      INI1(1)=47
      INI2(1)=49
C
      IN=0
      DO 171 N=1,NDOW
      IN1=INE1(N)
      IN2=INE2(N)
      DO 172 I=1,IN2-IN1+1
      IN=IN+1
      JEX(IN)=IN1+I-1
  172 CONTINUE
  171 CONTINUE
C
      IN=0
      DO 173 N=1,NDOW
      IN1=INI1(N)
      IN2=INI2(N)
      DO 174 I=1,IN2-IN1+1
      IN=IN+1
      JIX(IN)=IN2-I+1
  174 CONTINUE
  173 CONTINUE
C#       WRITE(100,733)(JEX(N),N=1,NKJ)
C#       WRITE(100,733)(JIX(N),N=1,NKJ)
C
C
C     ******************************************************************
C     ***DISCRETISATION DE L OBSTACLE EN FACETTES***********************
C     ******************************************************************
C
      T1=CLOCK()
      CALL OAMET
      T2=CLOCK()
      T=T2-T1
       WRITE(100,*) ' OAMET :',T
C
C#      READ(2,*) (IMD(I),I=1,NDOM)
C#      READ(2,*) (JMD(I),I=1,NDOM)
      IFA(1)=0
      DO 221 N=2,NDOM
      IFA(N)=IFA(N-1)+IMD(N-1)*JMD(N-1)
  221 CONTINUE
C
      PRINT 102
      DO 206 I=1,NFAC
      WRITE(100,104)I,XF(I,1),YF(I,1),ZF(I,1),XO(I),YO(I),ZO(I),
     1          FNX(I),FNY(I),FNZ(I)
      DO 204 K=2,4
      WRITE(100,106)XF(I,K),YF(I,K),ZF(I,K)
  204 CONTINUE
  206 CONTINUE
C
      WRITE(100,613)
      WRITE(100,601)ALPHA
      WRITE(100,609)ORI,CLO,NFAC
C
      DO 209 I=1,NFAC
      VS(I)=FNX(I)*UINFI+FNY(I)*VINFI+FNZ(I)*WINFI
  209 CONTINUE
C
C
C
C DEFINITION DES NAPPES
C
      E11=COS(AMWAK)
      E22=0.
      E33=SIN(AMWAK)
C
      DO 175 N=1,NKJ
      IP=JEX(N)
      IQ=JIX(N)
      SWX(1,N)=XF(IP,2)
      SWX(2,N)=XF(IP,2)+DWAK*E11
      SWX(3,N)=XF(IP,3)+DWAK*E11
      SWX(4,N)=XF(IP,3)
      SWY(1,N)=YF(IP,2)
      SWY(2,N)=YF(IP,2)+DWAK*E22
      SWY(3,N)=YF(IP,3)+DWAK*E22
      SWY(4,N)=YF(IP,3)
      SWZ(1,N)=ZF(IP,2)
      SWZ(2,N)=ZF(IP,2)+DWAK*E33
      SWZ(3,N)=ZF(IP,3)+DWAK*E33
      SWZ(4,N)=ZF(IP,3)
      XLE(N)=0.5*(XF(N,1)+XF(N,4))
      CD(N)=0.5*(XF(IP,2)+XF(IP,3))-XLE(N)
      WRITE(100,732)N,IP,IQ,SWX(1,N),SWX(2,N),SWY(1,N),SWY(2,N),
     1                  SWZ(1,N),SWZ(2,N)
  175 CONTINUE
C
C     ******************************************************************
C     ***COEFFICIENTS DE LA MATRICE****************************
C     ******************************************************************
C
       T2=CLOCK()
C
      CALL OAMAT1
C
       T3=CLOCK()
      T=T3-T2
      WRITE(100,*)' TEMPS OAMAT1 :',T
      DO 999 I=1,NFAC
      WRITE(100,*)I
C#      WRITE(100,*)(AA(I,J),J=1,NFAC)
999   CONTINUE
      
C#      WRITE(100,*)(O(I),I=1,NFAC)
C
      DO 250 I=1,NFAC
      PS(I)=O(I)
  250 CONTINUE
C
C
C     ******************************************************************
C     ***CALCUL DE MU0**************************
C
C     ************** RESOLUTION PAR M.G. ********
      T3=CLOCK()
      CALL GRAD1(PS,O)
       T4=CLOCK()
      T=T4-T3
      WRITE(100,*)' TEMPS RESOLUTION MU0 PAR M.G.= ',T
C
      DO 114 I=1,NKJ
      JE=JEX(I)
      JI=JIX(I)
      CC0(I)=O(JE)-O(JI)
  114 CONTINUE
C
C*******CALCUL DES NKJ CHAMPS CIRCULATOIRES *********************
      DO 998 J=1,NKJ
C
      CALL PHWAK(J)
C
C
C     ******************************************************************
C     ***CALCUL DE MUJ**************************
C
C     ************** RESOLUTION PAR M.G. ********
C
       T3=CLOCK()
      CALL GRAD1(PHW,O)
       T4=CLOCK()
      T=T4-T3
      WRITE(100,*)' TEMPS RESOLUTION MUW',J,' PAR M.G.= ',T
C
      DO 116 I=1,NKJ
      JE=JEX(I)
      JI=JIX(I)
      CC(J,I)=O(JE)-O(JI)
  116 CONTINUE
  998 CONTINUE
C
C  ****CALCUL MUW, K.J. CONTINUITE DES MU********
C
C
      DO 98 J=1,NKJ
      DO 96 JJ=1,NKJ
      PSV(J,JJ)=CC(JJ,J)
   96 CONTINUE
   98 CONTINUE
      DO 1098 J=1,NKJ
      PSV(J,J)=PSV(J,J)-1
1098  CONTINUE
C
      DO 69 J=1,NKJ
   69 RL(J)=-CC0(J)
C
      CALL SOLV(NKJ,PSV,RL)
C
      DO 90 J=1,NKJ
      OTXE(J)=RL(J)
   90 CONTINUE
C
      CALL PHWPAN
C
      DO 222 I=1,NFAC
  222 PS(I)=PS(I)+PHW(I)
C
C     ******************************************************************
C     ***CALCUL DE MU***************************
C
C     ************** RESOLUTION PAR M.G. ********
C
      T3=CLOCK()
      CALL GRAD1(PS,O)
      T4=CLOCK()
      T=T4-T3
       WRITE(100,*)' TEMPS RESOLUTION MU PAR M.G.= ',T
C*
      DO 349 I=1,NKJ
      JE=JEX(I)
      JI=JIX(I)
      OOO=OTXE(I)-O(JE)+O(JI)
      WRITE(100,610)I,OTXE(I),O(JE),O(JI),OOO
  349 CONTINUE
C
C     ******************************************************************
C     ***CALCUL DU GRADIENT DE MU******************************
C     ******************************************************************
C
      DO 886 N=1,NDOM
      IM=IMD(N)
      JM=JMD(N)
      IFAC0=IFA(N)
       T3=CLOCK()
       CALL OAVITEL(IM,JM)
       T4=CLOCK()
       T=T4-T3
       WRITE(100,*)' TEMPS OAVITEL.= ',T

  886 CONTINUE
C
C     ******************************************************************
C     ***VITESSE ET PRESSION AUX POINTS DE CONTROLE*********************
C     ******************************************************************
C
      DO 366 I=1,NFAC
      GRD(I,1)=-GRD(I,1)-FNX(I)*VS(I)+UINFI
      GRD(I,2)=-GRD(I,2)-FNY(I)*VS(I)+VINFI
      GRD(I,3)=-GRD(I,3)-FNZ(I)*VS(I)+WINFI
C     GRD(I,1)=-GRD(I,1)
C     GRD(I,2)=-GRD(I,2)
C     GRD(I,3)=-GRD(I,3)
      V2=GRD(I,1)*GRD(I,1)+GRD(I,2)*GRD(I,2)+GRD(I,3)*GRD(I,3)
      CPC(I)=1.-V2
C     CPC(I)=SQRT(V2)
  366 CONTINUE
C
C
      CX=0.
      CY=0.
      CZ=0.
      DO 417 I=1,NFAC
C     CX=CX+CPC(I)*FNX(I)*AIRE(I)
      CX=CX+AIRE(I)
C     CY=CY+CPC(I)*FNY(I)*AIRE(I)
      CZ=CZ+CPC(I)*FNZ(I)*AIRE(I)
  417 CONTINUE
      CY=CZ/SREF
       WRITE(100,610)IDT,CX,CY,CZ
       WRITE(100,606)
      REWIND 11
      AMACH=CY
      WRITE(11,730)AMACH,ALPHA
      DO 509 I=1,NFAC
      XR=XO(I)-ORI
      YR=YO(I)
      ZR=ZO(I)
      OR=O(I)
      WRITE(11,730)XR,YR,ZR,GRD(I,1),GRD(I,2),GRD(I,3),CPC(I),OR
      WRITE(100,608)I,XO(I),YO(I),ZO(I),GRD(I,1),
     1GRD(I,2),GRD(I,3),CPC(I)
  509 CONTINUE
      REWIND 11
  730 FORMAT(8F10.5)
      REWIND 12
      JF=JMD(1)
      NS=(JF-NSAU)/2
      NPE=NFAC/JF
      NPI=NPE
      NPOL=1
      AMACH=CY
      RE=999.
      WRITE(12,731)NS,NPOL,AMACH,ALPHA,RE,RE
C
      DO 709 J=1,NS
      YR=YO(J)
      CLN=0.
      CLA=0.
      WRITE(12,731)NPE,NPI,YR
      DO 707 I=1,NPE
      IR=JF*(I-1)+J
      XR=XO(IR)/CD(J)-XLE(J)/CD(J)
      ZR=ZO(IR)/CD(J)
      CLNI=-CPC(IR)*FNZ(IR)*AIRE(IR)
      CLAI=CPC(IR)*FNX(IR)*AIRE(IR)
      CLN=CLN+CLNI
      CLA=CLA+CLAI
      WRITE(12,731)J,I,ZR,XR,CPC(IR),O(IR)
  707 CONTINUE
      DO 708 I=1,NPE
      IR=JF*I-J+1
      XR=XO(IR)/CD(J)-XLE(J)/CD(J)
      ZR=ZO(IR)/CD(J)
      CLNI=-CPC(IR)*FNZ(IR)*AIRE(IR)
      CLAI=CPC(IR)*FNX(IR)*AIRE(IR)
      CLN=CLN+CLNI
      CLA=CLA+CLAI
      WRITE(12,731)J,I,ZR,XR,CPC(IR),O(IR)
  708 CONTINUE
      CLN=CLN/(YF(J,4)-YF(J,1))/CD(J)
      CLA=CLA/(YF(J,4)-YF(J,1))/CD(J)
      CLL=CLN*UINFI-CLA*WINFI
      PRINT 104,J,YR,CLN,CLA,CLL
  709 CONTINUE
      REWIND 12
C
      STOP
      END
      SUBROUTINE OAMET
C
      PARAMETER(NFAC=49,NTR1=1,NTR2=1)
      COMMON/GEF/XF(NFAC,0:4),YF(NFAC,0:4),ZF(NFAC,0:4)
      COMMON/FX/FXX(NFAC),FXY(NFAC),FXZ(NFAC)
      COMMON/FE/FEX(NFAC),FEY(NFAC),FEZ(NFAC)
      COMMON/CFN/FNX(NFAC),FNY(NFAC),FNZ(NFAC),AIRE(NFAC)
      COMMON/PCP/XO(NFAC),YO(NFAC),ZO(NFAC)
      DIMENSION ITR1(NTR1),ITR2(NTR2)
C
  121 FORMAT(I8,3E12.6)
  122 FORMAT(1X,10(2X,I5))
C
C     ******************************************************************
C     *** MAILLAGE D'UNE DEMI-VOILURE *************
C     ******************************************************************
C
C
      REWIND 2
      CD=1.008930411365
      DO 29 IJ=1,NFAC
      DO 19 K=1,4
      READ(2,121) NBID,XF(IJ,K),YF(IJ,K),ZF(IJ,K)
      XF(IJ,K)=XF(IJ,K)/(3.*CD)
      YF(IJ,K)=YF(IJ,K)/3.
      ZF(IJ,K)=ZF(IJ,K)/(3.*CD)
   19 CONTINUE
   29 CONTINUE
C
      READ(2,*) NBID,NBID
C#      READ(2,*)(ITR1(I),I=1,NTR1)
C#      READ(2,*)(ITR2(I),I=1,NTR2)
C#      PRINT 122,(ITR1(I),I=1,NTR1)
C#      PRINT 122,(ITR2(I),I=1,NTR2)
      DO 39 IJ=1,NFAC
      XF(IJ,0)=XF(IJ,4)
      YF(IJ,0)=YF(IJ,4)
      ZF(IJ,0)=ZF(IJ,4)
   39 CONTINUE
C
C
C     ***REPERE LOCAL,POINT DE CONTROLE*****************
C
      DO 205 I=1,NFAC
C
      FL1X=XF(I,3)-XF(I,1)
      FL1Y=YF(I,3)-YF(I,1)
      FL1Z=ZF(I,3)-ZF(I,1)
      FL2X=XF(I,4)-XF(I,2)
      FL2Y=YF(I,4)-YF(I,2)
      FL2Z=ZF(I,4)-ZF(I,2)
      C1= FL1Y*FL2Z-FL1Z*FL2Y
      C2= FL1Z*FL2X-FL1X*FL2Z
      C3= FL1X*FL2Y-FL1Y*FL2X
      AN=SQRT(C1*C1+C2*C2+C3*C3)
      AIRE(I)=0.5*AN
      AN=1./AN
      FNX(I)=C1*AN
      FNY(I)=C2*AN
      FNZ(I)=C3*AN
      FXX(I)=XF(I,3)+XF(I,2)-XF(I,1)-XF(I,4)
      FXY(I)=YF(I,3)+YF(I,2)-YF(I,1)-YF(I,4)
      FXZ(I)=ZF(I,3)+ZF(I,2)-ZF(I,1)-ZF(I,4)
      PP=SQRT(FXX(I)**2+FXY(I)**2+FXZ(I)**2)
      PP=1./PP
      FXX(I)=FXX(I)*PP
      FXY(I)=FXY(I)*PP
      FXZ(I)=FXZ(I)*PP
      FEX(I)=FNY(I)*FXZ(I)-FNZ(I)*FXY(I)
      FEY(I)=FNZ(I)*FXX(I)-FNX(I)*FXZ(I)
      FEZ(I)=FNX(I)*FXY(I)-FNY(I)*FXX(I)
C
      XO(I)=.25*(XF(I,1)+XF(I,2)+XF(I,3)+XF(I,4))
      YO(I)=.25*(YF(I,1)+YF(I,2)+YF(I,3)+YF(I,4))
      ZO(I)=.25*(ZF(I,1)+ZF(I,2)+ZF(I,3)+ZF(I,4))
  205 CONTINUE
C
      DO 206 JJ=1,NTR1
      J=ITR1(JJ)
      XO(J)=(XF(J,1)+XF(J,2)+XF(J,3))/3.
      YO(J)=(YF(J,1)+YF(J,2)+YF(J,3))/3.
      ZO(J)=(ZF(J,1)+ZF(J,2)+ZF(J,3))/3.
  206 CONTINUE
C
      DO 207 JJ=1,NTR2
      J=ITR2(JJ)
      XO(J)=(XF(J,1)+XF(J,3)+XF(J,4))/3.
      YO(J)=(YF(J,1)+YF(J,3)+YF(J,4))/3.
      ZO(J)=(ZF(J,1)+ZF(J,3)+ZF(J,4))/3.
  207 CONTINUE
C
C
      RETURN
      END
      SUBROUTINE PHWPAN
C
      PARAMETER(NFAC=49,NKJ=3,REFD=1.E-06,PREM=1.E-20)
      COMMON/OASET/PIO2
      COMMON/GEF/XF(NFAC,0:4),YF(NFAC,0:4),ZF(NFAC,0:4)
      COMMON/EE/E11,E22,E33
      COMMON/OT/SWX(4,NKJ),SWY(4,NKJ),SWZ(4,NKJ),OTXE(NKJ)
      COMMON/OAWAK/PHW(NFAC)
      COMMON/PCP/XO(NFAC),YO(NFAC),ZO(NFAC)
      COMMON/CFN/FNX(NFAC),FNY(NFAC),FNZ(NFAC),AIRE(NFAC)
      DIMENSION IY(2)
      DIMENSION FN1(NKJ),FN2(NKJ),FN3(NKJ)
      DIMENSION FX(NKJ),FY(NKJ),FZ(NKJ)
  678 FORMAT(3I5,8(2X,E12.5))
C
      IY(1)=1.
      IY(2)=-1.
      PIO4=PIO2*2.
C
C
CDIR$ BLOCK
      DO 10 J=1,NKJ
      FN1(J)=E22*(SWZ(4,J)-SWZ(1,J))-E33*(SWY(4,J)-SWY(1,J))
      FN2(J)=E33*(SWX(4,J)-SWX(1,J))-E11*(SWZ(4,J)-SWZ(1,J))
      FN3(J)=E11*(SWY(4,J)-SWY(1,J))-E22*(SWX(4,J)-SWX(1,J))
      FN=SQRT(FN1(J)**2+FN2(J)**2+FN3(J)**2)
      FN1(J)=FN1(J)/FN
      FN2(J)=FN2(J)/FN
      FN3(J)=FN3(J)/FN
      FX(J)=FN2(J)*E33-FN3(J)*E22
      FY(J)=FN3(J)*E11-FN1(J)*E33
      FZ(J)=FN1(J)*E22-FN2(J)*E11
   10 CONTINUE
C
C
C
      DO 2 I=1,NFAC
   2  PHW(I)=0.
C
      DO 5 J=1,NKJ
      DO 6 K=1,4
      KK=K+1-(K/4)*4
      DX=SWX(KK,J)-SWX(K,J)
      DY=SWY(KK,J)-SWY(K,J)
      DZ=SWZ(KK,J)-SWZ(K,J)
      D=SQRT(DX**2+DY**2+DZ**2)
      D1LX=DX*E11+DY*E22+DZ*E33
      IF(ABS(D1LX).GE.REFD) THEN
      D1LY=DX*FX(J)+DY*FY(J)+DZ*FZ(J)
      PT=D1LY/D1LX
      DO 4 KS=1,2
      DO 1 I=1,NFAC
      R1X=XO(I)-SWX(K,J)
      R1Y=IY(KS)*YO(I)-SWY(K,J)
      R1Z=ZO(I)-SWZ(K,J)
      R1=SQRT(R1X**2+R1Y**2+R1Z**2)
      R2X=XO(I)-SWX(KK,J)
      R2Y=IY(KS)*YO(I)-SWY(KK,J)
      R2Z=ZO(I)-SWZ(KK,J)
      R2=SQRT(R2X**2+R2Y**2+R2Z**2)
      R1LX=R1X*E11+R1Y*E22+R1Z*E33
      R1LY=R1X*FX(J)+R1Y*FY(J)+R1Z*FZ(J)
      R2LX=R2X*E11+R2Y*E22+R2Z*E33
      R2LY=R2X*FX(J)+R2Y*FY(J)+R2Z*FZ(J)
      Z=R1X*FN1(J)+R1Y*FN2(J)+R1Z*FN3(J)+PREM

C      Z=CVMGP(Z,-ABS(Z),ABS(Z)-REFD)
       IF (ABS(Z)-REFD.LT.0) THEN
       Z=-ABS(Z)
       ENDIF
      E1=Z**2+R1LX**2
      E2=Z**2+R2LX**2
      H1=R1LX*R1LY
      H2=R2LX*R2LY
      EH1=PT*E1-H1
      EH2=PT*E2-H2
      AT=ATAN2((Z*(R1*EH2-R2*EH1)),(Z*Z*R1*R2+EH1*EH2))
      PHW(I)=PHW(I)+AT*OTXE(J)
    1 CONTINUE
    4 CONTINUE
      ENDIF
    6 CONTINUE
   5  CONTINUE
C
      RETURN
      END
      SUBROUTINE PHWAK(N)
C
      PARAMETER(NFAC=49,NKJ=3,REFD=1.E-06)
      COMMON/OASET/PIO2
      COMMON/GEF/XF(NFAC,0:4),YF(NFAC,0:4),ZF(NFAC,0:4)
      COMMON/EE/E11,E22,E33
      COMMON/OT/SWX(4,NKJ),SWY(4,NKJ),SWZ(4,NKJ),OTXE(NKJ)
      COMMON/OAWAK/PHW(NFAC)
      COMMON/PCP/XO(NFAC),YO(NFAC),ZO(NFAC)
      COMMON/CFN/FNX(NFAC),FNY(NFAC),FNZ(NFAC),AIRE(NFAC)
      DIMENSION FN1(NKJ),FN2(NKJ),FN3(NKJ)
      DIMENSION FX(NKJ),FY(NKJ),FZ(NKJ)
      DIMENSION IY(2)
  678 FORMAT(3I5,8(2X,E12.5))
C
      PIO4=PIO2*2.
C
      IY(1)=1
      IY(2)=-1
C
CDIR$ BLOCK
      J=N
      FN1(J)=E22*(SWZ(4,J)-SWZ(1,J))-E33*(SWY(4,J)-SWY(1,J))
      FN2(J)=E33*(SWX(4,J)-SWX(1,J))-E11*(SWZ(4,J)-SWZ(1,J))
      FN3(J)=E11*(SWY(4,J)-SWY(1,J))-E22*(SWX(4,J)-SWX(1,J))
      FN=SQRT(FN1(J)**2+FN2(J)**2+FN3(J)**2)
      FN1(J)=FN1(J)/FN
      FN2(J)=FN2(J)/FN
      FN3(J)=FN3(J)/FN
      FX(J)=FN2(J)*E33-FN3(J)*E22
      FY(J)=FN3(J)*E11-FN1(J)*E33
      FZ(J)=FN1(J)*E22-FN2(J)*E11
C
C
C
      DO 2 I=1,NFAC
   2  PHW(I)=0.
C
      J=N
      DO 6 K=1,4
      KK=K+1-(K/4)*4
      DX=SWX(KK,J)-SWX(K,J)
      DY=SWY(KK,J)-SWY(K,J)
      DZ=SWZ(KK,J)-SWZ(K,J)
      D=SQRT(DX**2+DY**2+DZ**2)
      D1LX=DX*E11+DY*E22+DZ*E33
C     IF(ABS(D1LX).LT.REFD) GOTO 6
      IF(ABS(D1LX).GE.REFD) THEN
      D1LY=DX*FX(J)+DY*FY(J)+DZ*FZ(J)
      PT=D1LY/D1LX
      DO 4 KS=1,2
      DO 1 I=1,NFAC
      R1X=XO(I)-SWX(K,J)
      R1Y=IY(KS)*YO(I)-SWY(K,J)
      R1Z=ZO(I)-SWZ(K,J)
      R1=SQRT(R1X**2+R1Y**2+R1Z**2)
      R2X=XO(I)-SWX(KK,J)
      R2Y=IY(KS)*YO(I)-SWY(KK,J)
      R2Z=ZO(I)-SWZ(KK,J)
      R2=SQRT(R2X**2+R2Y**2+R2Z**2)
      R1LX=R1X*E11+R1Y*E22+R1Z*E33
      R1LY=R1X*FX(J)+R1Y*FY(J)+R1Z*FZ(J)
      R2LX=R2X*E11+R2Y*E22+R2Z*E33
      R2LY=R2X*FX(J)+R2Y*FY(J)+R2Z*FZ(J)
      Z=R1X*FN1(J)+R1Y*FN2(J)+R1Z*FN3(J)
C      Z=CVMGP(Z,-ABS(Z),ABS(Z)-REFD)

       IF (ABS(Z)-REFD.LT.0) THEN
       Z=-ABS(Z)
       ENDIF



      E1=Z**2+R1LX**2
      E2=Z**2+R2LX**2
      H1=R1LX*R1LY
      H2=R2LX*R2LY
      EH1=PT*E1-H1
      EH2=PT*E2-H2
      AT=ATAN2((Z*(R1*EH2-R2*EH1)),(Z*Z*R1*R2+EH1*EH2))
      PHW(I)=PHW(I)+AT
    1 CONTINUE
    4 CONTINUE
      ENDIF
    6 CONTINUE
C
      RETURN
      END
       SUBROUTINE SOLV(N,A,B)
      PARAMETER(NKJ=3)
       DIMENSION A(NKJ,NKJ),B(1)
       DO 1 I=1,N
       T=1./A(I,I)
       DO 2 J=I,N
    2  A(I,J)=A(I,J)*T
       B(I)=B(I)*T
       IF(I.NE.N) THEN
       IP=I+1
       DO 3 K=IP,N
       T=A(K,I)
       DO 4 J=IP,N
    4  A(K,J)=A(K,J)-T*A(I,J)
       B(K)=B(K)-T*B(I)
    3  CONTINUE
       ENDIF
    1  CONTINUE
       DO 5 II=2,N
       I=N+1-II
       IP=I+1
       DO 6 J=IP,N
    6  B(I)=B(I)-A(I,J)*B(J)
    5  CONTINUE
       RETURN
       END
      SUBROUTINE OAVITEL(IFM,JFM)
C
      PARAMETER(NFAC=49)
      COMMON/OO/O(NFAC),VS(NFAC),ps(nfac)
      COMMON/PCP/XO(NFAC),YO(NFAC),ZO(NFAC)
      COMMON/GRA/GRD(NFAC,3)
      COMMON/CFN/FNX(NFAC),FNY(NFAC),FNZ(NFAC),AIRE(NFAC)
C
C
      IMFM1=IFM-1
      JMFM1=JFM-1
      JM1O2=JMFM1/2
      JM1O2M1=JM1O2-1
      JM1O2P3=JM1O2+3
C
      IFAC=1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=XO(IFAC+1)-XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=YO(IFAC+1)+YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=ZO(IFAC+1)-ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC))
      DM2=O(IFAC+1)-O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 19 J=2,JM1O2M1
      IFAC=IFAC+1
C
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=XO(IFAC+1)-XO(IFAC-1)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=YO(IFAC+1)-YO(IFAC-1)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=ZO(IFAC+1)-ZO(IFAC-1)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=O(IFAC+1)-O(IFAC-1)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   19 CONTINUE
C
      IFAC=IFAC+1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=-4.*XO(IFAC-1)+XO(IFAC-2)+3.*XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=-4.*YO(IFAC-1)+YO(IFAC-2)+3.*YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=-4.*ZO(IFAC-1)+ZO(IFAC-2)+3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=-4.*O(IFAC-1)+O(IFAC-2)+3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      IFAC=IFAC+1
C
      IFAC=IFAC+1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=4.*XO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=4.*YO(IFAC+1)-YO(IFAC+2)-3.*YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=4.*ZO(IFAC+1)-ZO(IFAC+2)-3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC))
      DM2=4.*O(IFAC+1)-O(IFAC+2)-3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 18 J=JM1O2P3,JMFM1
      IFAC=IFAC+1
C
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=XO(IFAC+1)-XO(IFAC-1)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=YO(IFAC+1)-YO(IFAC-1)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=ZO(IFAC+1)-ZO(IFAC-1)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=O(IFAC+1)-O(IFAC-1)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   18 CONTINUE
C
      IFAC=IFAC+1
      R1=4.*XO(IFAC+JFM)-XO(IFAC+2*JFM)-3.*XO(IFAC)
      R2=-XO(IFAC-1)+XO(IFAC)
      R4=4.*YO(IFAC+JFM)-YO(IFAC+2*JFM)-3.*YO(IFAC)
      R5=-YO(IFAC-1)-YO(IFAC)
      R7=4.*ZO(IFAC+JFM)-ZO(IFAC+2*JFM)-3.*ZO(IFAC)
      R8=-ZO(IFAC-1)+ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=4.*O(IFAC+JFM)-O(IFAC+2*JFM)-3.*O(IFAC)
      DM2=-O(IFAC-1)+O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 39 I=2,IMFM1
      IFAC=IFAC+1
C
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=XO(IFAC+1)-XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=YO(IFAC+1)+YO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=ZO(IFAC+1)-ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=O(IFAC+1)-O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
C POINT COURANT
C
      DO 29 J=2,JM1O2M1
      IFAC=IFAC+1
C
      IP1=IFAC+JFM
      IM1=IFAC-JFM
      JP1=IFAC+1
      JM1=IFAC-1
      R1=(XO(IP1)-XO(IM1))
      R2=(XO(JP1)-XO(JM1))
      R4=(YO(IP1)-YO(IM1))
      R5=(YO(JP1)-YO(JM1))
      R7=(ZO(IP1)-ZO(IM1))
      R8=(ZO(JP1)-ZO(JM1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IP1)-O(IM1))
      DM2=(O(JP1)-O(JM1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   29 CONTINUE
C
      IFAC=IFAC+1
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=-4.*XO(IFAC-1)+XO(IFAC-2)+3.*XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=-4.*YO(IFAC-1)+YO(IFAC-2)+3.*YO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=-4.*ZO(IFAC-1)+ZO(IFAC-2)+3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=-4.*O(IFAC-1)+O(IFAC-2)+3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      IFAC=IFAC+1
C
      IFAC=IFAC+1
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=4.*XO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=4.*YO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=4.*ZO(IFAC+1)-ZO(IFAC+2)-3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=4.*O(IFAC+1)-O(IFAC+2)-3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 28 J=JM1O2P3,JMFM1
      IFAC=IFAC+1
C
      IP1=IFAC+JFM
      IM1=IFAC-JFM
      JP1=IFAC+1
      JM1=IFAC-1
      R1=(XO(IP1)-XO(IM1))
      R2=(XO(JP1)-XO(JM1))
      R4=(YO(IP1)-YO(IM1))
      R5=(YO(JP1)-YO(JM1))
      R7=(ZO(IP1)-ZO(IM1))
      R8=(ZO(JP1)-ZO(JM1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IP1)-O(IM1))
      DM2=(O(JP1)-O(JM1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   28 CONTINUE
C
      IFAC=IFAC+1
C
      R1=(XO(IFAC+JFM)-XO(IFAC-JFM))
      R2=-XO(IFAC-1)+XO(IFAC)
      R4=(YO(IFAC+JFM)-YO(IFAC-JFM))
      R5=-YO(IFAC-1)-YO(IFAC)
      R7=(ZO(IFAC+JFM)-ZO(IFAC-JFM))
      R8=-ZO(IFAC-1)+ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(O(IFAC+JFM)-O(IFAC-JFM))
      DM2=-O(IFAC-1)+O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   39 CONTINUE
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=XO(IFAC+1)-XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=YO(IFAC+1)+YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=ZO(IFAC+1)-ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=O(IFAC+1)-O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 49 J=2,JM1O2M1
      IFAC=IFAC+1
C
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=(XO(IFAC+1)-XO(IFAC-1))
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=(YO(IFAC+1)-YO(IFAC-1))
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=(ZO(IFAC+1)-ZO(IFAC-1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=(O(IFAC+1)-O(IFAC-1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   49 CONTINUE
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=-4.*XO(IFAC-1)+XO(IFAC-2)+3.*XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=-4.*YO(IFAC-1)+YO(IFAC-2)+3.*YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=-4.*ZO(IFAC-1)+ZO(IFAC-2)+3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=-4.*O(IFAC-1)+O(IFAC-2)+3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      IFAC=IFAC+1
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=4.*XO(IFAC+1)-XO(IFAC+2)-3.*XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=4.*YO(IFAC+1)-YO(IFAC+2)-3.*YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=4.*ZO(IFAC+1)-ZO(IFAC+2)-3.*ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=4.*O(IFAC+1)-O(IFAC+2)-3.*O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      DO 48 J=JM1O2P3,JMFM1
      IFAC=IFAC+1
C
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=(XO(IFAC+1)-XO(IFAC-1))
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=(YO(IFAC+1)-YO(IFAC-1))
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=(ZO(IFAC+1)-ZO(IFAC-1))
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=(O(IFAC+1)-O(IFAC-1))
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
   48 CONTINUE
C
      IFAC=IFAC+1
      R1=(-4.*XO(IFAC-JFM)+XO(IFAC-2*JFM)+3.*XO(IFAC))
      R2=-XO(IFAC-1)+XO(IFAC)
      R4=(-4.*YO(IFAC-JFM)+YO(IFAC-2*JFM)+3.*YO(IFAC))
      R5=-YO(IFAC-1)-YO(IFAC)
      R7=(-4.*ZO(IFAC-JFM)+ZO(IFAC-2*JFM)+3.*ZO(IFAC))
      R8=-ZO(IFAC-1)+ZO(IFAC)
C
      D=R1*R5*FNZ(IFAC)+R4*R8*FNX(IFAC)-R7*R5*FNX(IFAC)-R8*R1*FNY(IFAC)
     1  +R2*R7*FNY(IFAC)-R2*R4*FNZ(IFAC)
      A1=(R5*FNZ(IFAC)-R8*FNY(IFAC))/D
      A2=(R8*FNX(IFAC)-R2*FNY(IFAC))/D
      A3=(R2*FNY(IFAC)-R5*FNX(IFAC))/D
      A4=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
      A5=(R1*FNZ(IFAC)-R7*FNX(IFAC))/D
      A6=(R7*FNY(IFAC)-R4*FNZ(IFAC))/D
C
      DM1=(-4.*O(IFAC-JFM)+O(IFAC-2*JFM)+3.*O(IFAC))
      DM2=-O(IFAC-1)+O(IFAC)
C
      GRD(IFAC,1)=DM1*A1+DM2*A4
      GRD(IFAC,2)=DM1*A2+DM2*A5
      GRD(IFAC,3)=DM1*A3+DM2*A6
C
      RETURN
      END
      SUBROUTINE OAMAT1
C
      PARAMETER(NFAC=49,REFD=1.E-06,PREM=1.E-37)
      COMMON/OASET/PIO2
      COMMON/OACOF/AA(NFAC,NFAC)
      COMMON/OO/O(NFAC),VS(NFAC),ps(nfac)
      COMMON/GEF/XF(NFAC,0:4),YF(NFAC,0:4),ZF(NFAC,0:4)
      COMMON/PCP/XO(NFAC),YO(NFAC),ZO(NFAC)
      COMMON/GRA/GRD(NFAC,3)
      COMMON/FX/FXX(NFAC),FXY(NFAC),FXZ(NFAC)
      COMMON/FE/FEX(NFAC),FEY(NFAC),FEZ(NFAC)
      COMMON/CFN/FNX(NFAC),FNY(NFAC),FNZ(NFAC),aire(nfac)
      DIMENSION IY(2)
C
      IY(1)=1
      IY(2)=-1
      ICAS1=0
      ICAS2=0
C
      DO 1 I=1,NFAC
      DO 1 J=1,NFAC
         AA(I,J)=0.
 1    CONTINUE
      DO 2 I=1,NFAC
         O(I)=0.
 2    CONTINUE
C
      DO 5 J=1,NFAC
      B11=FXX(J)
      B12=FXY(J)
      B13=FXZ(J)
      B21=FEX(J)
      B22=FEY(J)
      B23=FEZ(J)
      B31=FNX(J)
      B32=FNY(J)
      B33=FNZ(J)
C
      BX=B11*XO(J)+B13*ZO(J)
      BY=B21*XO(J)+B23*ZO(J)
      COEFZ=B31*XO(J)+B33*ZO(J)
C
      DO 7 K=1,4
      KK=K-1
      XN=XF(J,KK)-XO(J)
      YN=YF(J,KK)-YO(J)
      ZN=ZF(J,KK)-ZO(J)
      XIN=B11*XN+B12*YN+B13*ZN
      ETAN=B21*XN+B22*YN+B23*ZN
C
      XM=XF(J,K)-XO(J)
      YM=YF(J,K)-YO(J)
      ZM=ZF(J,K)-ZO(J)
      XIM=B11*XM+B12*YM+B13*ZM
      ETAM=B21*XM+B22*YM+B23*ZM
C
      XIMN=XIM-XIN
      ETAMN=ETAM-ETAN
      COEFX=XIN+BX
      COEFY=ETAN+BY
C
      DD=SQRT(XIMN**2+ETAMN**2)
      IF (ABS(XIMN).GT.REFD) THEN
          ICAS1=ICAS1+1
          AAA=ETAMN/XIMN
          VSJDD=VS(J)/DD
          DO 10 KS=1,2
             DO 15 I=1,NFAC
                A=IY(KS)*YO(I)-YO(J)
                XXIN=B11*XO(I)+(B13*ZO(I)+(B12*A-COEFX))
                YTAN=B21*XO(I)+(B23*ZO(I)+(B22*A-COEFY))
                Z   =B31*XO(I)+(B33*ZO(I)+(B32*A-COEFZ))+PREM
C               Z   =CVMGP(Z,-ABS(Z),ABS(Z)-REFD)
                IF (ABS(Z)-REFD.LT.0) THEN
                   Z=-ABS(Z)
                ENDIF

C
                RN=(Z**2)+XXIN**2
                A =AAA*RN-YTAN*XXIN
                RM=(Z**2)+(XXIN-XIMN)**2
                B =AAA*RM-(YTAN-ETAMN)*(XXIN-XIMN)
                XXIN=VSJDD*(XXIN*ETAMN-YTAN*XIMN)
                RM=SQRT(RM+(YTAN-ETAMN)**2)
                RN=SQRT(RN+YTAN**2)
C
                YTAN=((RN+RM)+DD)/((RN+RM)-DD)
                B=ATAN2(A*(Z*RM)-B*(Z*RN),(Z*RN)*(Z*RM)+A*B)
                AA(I,J)=AA(I,J)+B
                O(I)=ALOG(YTAN)*XXIN+(O(I)-B*VS(J)*Z)
 15          CONTINUE
 10       CONTINUE
C
       ELSEIF (DD.GT.REFD) THEN
          ICAS2=ICAS2+1
          VSJDD=VS(J)/DD
          DO 20 KS=1,2
CDIR$ BLOCK
             DO 25 I=1,NFAC
                A=IY(KS)*YO(I)-YO(J)
                XXIN=B11*XO(I)+(B13*ZO(I)+(B12*A-COEFX))
                YTAN=B21*XO(I)+(B23*ZO(I)+(B22*A-COEFY))
                Z   =B31*XO(I)+(B33*ZO(I)+(B32*A-COEFZ))
C
                A=SQRT((Z**2)+XXIN**2+YTAN**2)
     +            +SQRT((Z**2)+(XXIN-XIMN)**2+(YTAN-ETAMN)**2)
                O(I)=O(I)+ALOG((A+DD)/(A-DD))
     +                    *(VSJDD*(XXIN*ETAMN-YTAN*XIMN))
 25          CONTINUE
 20       CONTINUE
C
       ENDIF
 7    CONTINUE
 5    CONTINUE
C
      PRINT *,' ***** OAMAT: ICAS1,ICAS2 = ',ICAS1,ICAS2
C
      RETURN
      END
      SUBROUTINE PMAT(V,W)
C
      PARAMETER(NFAC=49)
      COMMON/OACOF/AA(NFAC,NFAC)
      DIMENSION V(1),W(1),C(NFAC)
C
      DO 10,I=1,NFAC
      DO 20,J=1,NFAC
   20 C(J)=AA(I,J)
C
      W(I)=SDOT(NFAC,C,1,V,1)
   10 CONTINUE
C
C
      RETURN
      END

       SUBROUTINE GRAD1(B,X)
C multi gradient  sans fenetrage
C 
       PARAMETER(NFAC=49,IRMAX=50)
      COMMON/OASET/PIO2
      COMMON/OACOF/AA(NFAC,NFAC)
       DIMENSION A(IRMAX,IRMAX),B(NFAC),X(NFAC)
       DIMENSION ALFA(IRMAX),R(NFAC,IRMAX),S(NFAC,IRMAX)
       DIMENSION Y(NFAC),Z(NFAC),RS(IRMAX),D(IRMAX)
       RRM=1.E-20
      DO 111 I=1,NFAC
C 111 X(I)=B(I)/AA(I,I)
      Y(I)=0.
  111 X(I)=0.
C      CALL PMAT(X,Y)
       DO 1 I=1,NFAC
    1  R(I,1)=B(I)-Y(I)
       IPM=50
       IPP=1
       DO 2 IP=1,IPM
       IPP0=IPP
       IPP=IPP+1
       IPQ=IP
       DO 21 I=1,NFAC
   21  Y(I)=R(I,IPP0)
       CALL PMAT(Y,Z)
       DO 22 I=1,NFAC
   22  S(I,IPP0)=Z(I)
       DO 3 II=1,IPQ
       DO 4 I=1,NFAC
    4  Y(I)=S(I,II)
       DO 5 JJ=II ,IPQ
       DO 6 I=1,NFAC
    6  Z(I)=S(I,JJ)
       A(II,JJ)=SDOT(NFAC,Y,1,Z,1)
    5  A(JJ,II)=A(II,JJ)
       DO 7 I=1,NFAC
    7  Z(I)=R(I,IPP0)
    3  RS(II)=SDOT(NFAC,Y,1,Z,1)
      CALL SOLV3(IPQ,A,RS,D)
       DO 41 I=1,NFAC
   41  R(I,IPP)=R(I,IPP0)
       DO 42 II=1,IPQ
       DO 42 I=1,NFAC
   42  X(I)=X(I)+RS(II)*R(I,II)
       DO 43 II=1,IPQ
       DO 43 I=1,NFAC
   43  R(I,IPP)=R(I,IPP)-RS(II)*S(I,II)
       DO 10 I=1,NFAC
   10  Y(I)=R(I,IPP)
       RR=SDOT(NFAC,Y,1,Y,1)
       WRITE(6,*)IP,RR
       WRITE(100,*)IP,RR

       IF(RR.LT.RRM)GO TO 11
    2  CONTINUE
   11  CONTINUE
       RETURN
  100  FORMAT(I5,E12.5)
  101  FORMAT(10E12.5)
       END


       FUNCTION SDOT(N,X,INCX,Y,INCY)
       REAL*4 X(1),Y(1),SDOT
       SDOT=0.0
       IX=1
       IY=1
       DO 10 I=1,N
          SDOT=SDOT+X(IX)*Y(IY)
          IX=IX+INCX
          IY=IY+INCY
  10   CONTINUE
       RETURN
       END
       
       
       SUBROUTINE SOLV3(NP1,A,B,D)
C     
C     RESOLUTION DU SYSTEME AX=B AVEC RETOUR DE LA SOLUTION DANS B
C     ON DEFINI CETTE RESOLUTION DE MANIERE RECURENTE SUR LA DIMENSION
C     ON SUPPOSERA QUE LA N MATRICE A DEJA ETE TRAITEE
C     ON SUPPOSERA TOUS LES RESIDUS CONSERVES
C     ON RESOUT ICI LE SYSTEME DE DIMENSION NP1=N + 1
C     LA PARTIE INF DE A CONTIENT L
C     
C     on utilise ici une decomposition L D Lt
       
       DIMENSION A(50,50),B(50),D(50),R(50),Z(50)
C     
C     PARTIE INITIALISATION CAS N=1
C     A(1,1) RESTE A SA VALEUR
C     D(1)=A(1,1)
       
       DO 1 K=1,NP1
          
          DO 2 IP=1,K-1
             R(IP)=D(IP)*A(K,IP)
 2        CONTINUE
          
          IF(K.EQ.NP1) THEN
             D(K)=A(K,K)
             DO 3 IP=1,K-1
                D(K)=D(K)-A(K,IP)*R(IP)
 3           CONTINUE
          ELSE
             I=NP1
             A(I,K)=A(K,I) 
             DO 4 IP=1,K-1
                A(I,K)=A(I,K)-A(I,IP)*R(IP)
 4           CONTINUE
             A(I,K)=A(I,K)/D(K)
          ENDIF
 1     CONTINUE 
C     
C     RESOLUTION DES SYSTEMES LZ=B
C     DY=Z LtX=Y
C     DESCENTE
       Z(1)=B(1)
       DO 100 I=2,NP1
          S=0.
          DO 101,J=1,I-1
             S=S+A(I,J)*Z(J)
 101      CONTINUE
          Z(I)=B(I)-S
 100   CONTINUE
C     
C     DIAGONALE
C     
       DO 200 I=1,NP1
          Z(I)=Z(I)/D(I)
 200   CONTINUE
C     
C     REMONTEE
C     
       B(NP1)=Z(NP1)
       IND=NP1-1
       DO 300 I=2,NP1
          S=0
          DO 301 J=IND+1,NP1
             S=S+A(J,IND)*B(J)
 301      CONTINUE
          B(IND)=Z(IND)-S
          IND=IND-1
 300   CONTINUE
C     
       RETURN
       END

       INTEGER*4 FUNCTION CLOCK()
C      CLOCK = TIME()
       CLOCK = 1
       RETURN
       END
