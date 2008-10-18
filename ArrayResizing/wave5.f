c
c
c modifications for spec
c
c
c initialized some uninitialized (assumed to be zero) values
c disabled some code paths not used in the benchmark
c scaled down benchmark (5 iterations instead of 20)
c 11/5/90 mrg changed view of array BUFFER
c 11/5/90 mrg changed STOP 'NSTOP' since it caused verification difficulties
c 11/5/90 mrg eliminated use of unitialized varable CPTIM
c  4/11/91 J.Lo changed CDIR$ to CCDIR$ 
C 8/6/91  dzn Use legal ANSI FORTRAN-77 array bounds declarations; completely
C             remove `CDIR$ IVDEP' vectorizer directives; suppress
C             uninitialized formal warnings in SETB.
C 10/20 change to double precision
C           
c
C **********************************************************************
C *                                                                    *
C *  Copyright, 1986, The Regents of the University of California.     *
C *  This software was produced under a U. S. Government contract      *
C *  (W-7405-ENG-36) by the Los Alamos National Laboratory, which is   *
C *  operated by the University of California for the U. S. Department *
C *  of Energy. The U. S. Government is licensed to use, reproduce,    *
C *  and distribute this software. Permission is granted to the public *
C *  to copy and use this software without charge, provided that this  *
C *  Notice and any statement of authorship are reproduced on all      *
C *  copies. Neither the Government nor the University makes any       *
C *  warranty, express or implied, or assumes any liability            *
C *  responsibility for the use of this software.                      *
C *                                                                    *
C **********************************************************************
C
      PROGRAM WAVE
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
C     WAVE   CRAY VERSION: 2.1, REVISED 85/2/21
C     APERIODIC IN Y, INJECTION, RESTRUCTURED
C
      PARAMETER ( NBB=512)
C THE NEXT NUMBER MUST BE GREATER THAN NX2 * NY2:
      PARAMETER( NC1 = 78885)
      PARAMETER ( N1D=5000 )
      PARAMETER ( NYY=514 )
      PARAMETER ( NNS=25 )
      PARAMETER ( NDIM=503 )
      PARAMETER ( NNS1 = NNS + 1, NNS2 = NNS * 2 )
      PARAMETER ( NCL = 3 * NYY)
C
      COMMON /HTIM/ PSHCPT, PSHELT, FIOCPT, FIOELT
      COMMON  PBUF(NBB,5),Q(NC1),TEMP(2*N1D),TBUF(5,NBB)
      COMMON /POTEN1/ SS(NC1),W(NC1),AX(NC1)
      COMMON /POTEN2/ AX1(NC1),AY(NC1),AY1(NC1)
      COMMON /POTEN3/ AZ(NC1),AZ1(NC1)
      COMMON /CURR/   CX(NC1),CY(NC1),CZ(NC1)
      COMMON /EFIELD/ EX(NC1),EY(NC1),EZ(NC1),BX(NC1),BY(NC1),BZ(NC1)
      COMMON /QBK/    QBK(NC1)
      DIMENSION X(NBB),Y(NBB),VX(NBB),VY(NBB),VZ(NBB)
      EQUIVALENCE (X(1),PBUF(1,1)),(Y(1),PBUF(1,2)),(VX(1),PBUF(1,3)),
     *  (VY(1),PBUF(1,4)),(VZ(1),PBUF(1,5))
C
      COMMON/PARCM/NX,NY,L3,L4,KBND,NX2,NX1,NY1,NY2,IBCDL,IBCDR,
     *   IBCDT,IBCDB,KBR,LSTL,LSTR,NINJL,NINJR,KBNDY,INOP,
     *   XMAX,YMAX,DT,QMULT,WMULT,XLEFT,XRIGHT,
     *   HX,HY,HXI,HYI,VMPX1,VMPY1,
     *   VMPZ1,ELOST,QLOSTL,QLOSTR,QLOSTT,QLOSTB,
     *   YTOP,YBOT,DRINJ,DLINJ,VRINJ,VLINJ,
     *   TRPAR,TLPAR,DTLPAR,DTRPAR,QINJL,QINJR,EINJ
C
      COMMON/FKICK/XKICK,DXKICK, YK1,YK2,FNKICK,VMPXH,VMPYH,EHEAT
     1 ,VOSC,DELE,DLASER,TWIDTH,TPEAK,XLASER
      COMMON/PARAM/IT,NC,NB,NP,IMPSW,NRAND,KJSMTH,KQSMTH,
     *    KEI,KEE,KSP,NPTIME,
     *    NSP,NPX(NNS),NPY(NNS),NSPEC(NNS),IDNX(NNS1),IDNY(NNS1),
     *    IBCNDL(NNS),IBCNDR(NNS),IBCNDT(NNS),IBCNDB(NNS),
     *    LOADED(NNS),LOSTL(NNS),LOSTR(NNS),INJL(NNS),INJR(NNS)
      COMMON/PARAMR/ PI,T,AA,SINA,SINP,TDLAZ,AMPH,PHAS,VEA,FNK,
     *    TAU,TAU1,TAU2,EXC,EYC,EZC,BXC,BYC,BZC,WSPEC(NNS),QSPEC(NNS),
     *    AYL(NCL),AYR(NCL),AYLD(NCL),AYRD(NCL),
     *    AZL(NCL),AZR(NCL),AZLD(NCL),AZRD(NCL),
     *    VMPX(NNS),VMPY(NNS),VMPZ(NNS),
     *    XDRIFT(NNS),YDRIFT(NNS),ZDRIFT(NNS),
     *    XN(NNS2),DENX(NNS2),YN(NNS2),DENY(NNS2),
     *    XLJ(NNS),XRJ(NNS),XTJ(NNS),XBJ(NNS),FEREM,AFRACR,
     *    DINJR(NNS),DINJL(NNS),VINJR(NNS),VINJL(NNS),
     *    TINJR(NNS),TINJL(NNS),IBATCH(NNS)
      COMMON /PARAM1/ EKX(NNS),EKY(NNS),EKZ(NNS),EKJ(NNS),
     *    VLIMA,VLIMB,VXAVGA,VXAVGB,VYAVGA,
     *    VYAVGB,VZAVGA,VZAVGB,TFIN,EFIN,
     *    TDEC(NDIM),EDEC(NDIM,6),ELOSTB,AVPOYN,APOYNT(N1D),
     *    EPOUTR(50),EPOUTL(50),EPOUTB(50),EPOUTT(50),
     *    EPINR(50),EPINL(50),EPINB(50),EPINT(50),
     *    DEP,TEP,EPMAX,KSPLIT,NPTSA,NPTSB,NPLOT,NENRG,NHIST,NFT1,NFT2,
     *    NENRG2,KDIAG,LDEC,NEP
      COMMON /NSTEPS/ NTIMES
C
      CHARACTER*72 TITLE
      DATA TITLE /' SAMPLE WEIBEL INSTABILITY           '/
C THE NEXT NUMBER IS THE NUMBER OF CYCLES TO BE RUN:
c orig version was 20 steps - for spec, we have selected 5 steps
C	PLV
      DATA TLMIN /1800.D0/
c specmod
	DATA KSP0 / 0 /
        DATA CPTIM/ 0.0D0 /

CHW   CALL SAMPLE(5HTFILE,4)
CHW   CALL SAMPON
c        call fpi
      NC=NC1
      NB=NBB
      TRUN = 620.D0
C
      OPEN(9, FILE = 'WAVE.OUT', STATUS='UNKNOWN')
C
C          BEGINING OF DATA READ CYCLE
C
      CALL GENPRB
      CALL JOBTIM(TIM1)
C
      IF ((NX+2)*(NY+2).GT.NC) STOP 'DIMEN'
      IF (SINA.NE.0.D0) THEN
C
C          ADJUST YMAX TO FIT OBLIQUELY INCIDENT WAVE
C
           INT=MAX0(1,IDINT(SINA*YMAX*.5D0/PI+.5D0))
           YMAX=2.D0*PI*INT/SINA
           WRITE (9,140) YMAX
      ENDIF
      WRITE (9,150) TITLE,NX,NY,XMAX,YMAX,DT
      DT2DX2=DT**2*((NX/XMAX)**2+(NY/YMAX)**2)
      IF (DT2DX2.GT.1.D0) WRITE (9,160) DT2DX2
      IF (DT2DX2.GT.1.D0.AND.IMPSW.EQ.0) CALL ENDRUN('COURANT')

C                   ** INITIALIZE PROBLEM **
      T=0.D0
      CALL JOBTIM(TMI1)
      CALL INIT
      CALL JOBTIM(TMI2)
      TMICP = TMI2 - TMI1
      WRITE (9,170)
      WRITE (9,180) NTIMES
10    IF (NTIMES.EQ.0) THEN
           LDEC = -LDEC
20      CONTINUE
CHW        CALL GETB(PBUF,L4,KSP0,TBUF)
           L3 = 1
           IF (KSP0.NE.0) THEN
                 KSP=IABS(KSP0)
                 XLEFT=XLJ(KSP)
                 XRIGHT=XRJ(KSP)
                 QMULT=QSPEC(KSP)
                 WMULT=WSPEC(KSP)
CHW              CALL PDIAG
                 GO TO 20
           ENDIF
           L3 = -1
CHW        CALL DIAGNS
           LDEC = IABS(LDEC)
           NTIMES = -1
           GO TO 10
      ENDIF

  30  IF (NTIMES.LE.IT) THEN
C   ** RUN COMPLETED! **
CHW        CALL SAMPOFF
CHW        CALL SAMPTRM
           CALL JOBTIM(TIM2)
           WRITE (9,100) IT
           CPUTIM = TIM2 - TIM1
C          WRITE(9,120) CPUTIM
C          WRITE(9,250) TMICP
c           call printfpi_counts
c ** spec mod
c some systems print out the nstop differently and this caused a problem
c with verification.  To resolve this issue, we modified the statement
c below.
c old code
c           STOP 'NSTOP'
c new code
            STOP
      ELSE
C MAIN LOOP.....
           IT = IT + 1
           T = T + DT
           CALL TRANS
           CALL FIELD
           L3 = -1
           CALL DIAGNS
           IF (MOD(IT,5).EQ.0) THEN
                WRITE (9,190) IT
           ENDIF
C  CHECK TIME:
           IF ( CPTIM.GT.TRUN .AND. TRUN.LT.TLMIN ) THEN
                WRITE(9,510) CPTIM
                WRITE(9,520) IT,T
c                STOP 'CPTIME'
		 STOP
           ENDIF
           GO TO 30
      ENDIF

100   FORMAT ('  NSTOP STOP ', I6, ' CYCLES COMPLETED')
120   FORMAT (' OVERALL       ELAPSED TIME IS ', F10.4)
140   FORMAT ('  NEW YMAX=',E12.4)
150   FORMAT (1X,A72/'0  NX=',I4,'  NY=',I4,'  XMAX=',E12.4,'  YMAX='
     1 ,E12.4,'  DT=',E12.4)
160   FORMAT (' WARNING**BAD COURANT CONDITION, DT2DX2 = ',E10.3)
170   FORMAT (' INITIALIZATION COMPLETED!' )
180   FORMAT (' NUM TIMESTEPS TO BE RUN = ',I6)
190   FORMAT (' TIME STEP',I6,' COMPLETED ')
250   FORMAT (' INITIALIZING  ELAPSED TIME IS ', F10.4)
510   FORMAT ('  CPTIME STOP   CPTIME=',F7.0,12X)
520   FORMAT (6X,' IT=',I6,3X,' PROBLEM TIME=',F9.3)
      END
