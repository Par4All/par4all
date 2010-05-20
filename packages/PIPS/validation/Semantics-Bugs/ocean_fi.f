C
C     1993 version - Modified by Beatrice Creusillet for PIPS -
C
C     MODIFICATIONS:
C
C     AUGUST 1995:
C     ~~~~~~~~~~~~
C     - CXxxxxx and CYyyyyy commentaries removed for clarity.
C     - CPUTIM and ELAPSE subroutines uncommented.
C     - BLOCK DATA moved in main procedure (Francois Irigoin) .
C     - Macro definition LSTMOD(MODEHI) in module DEALSE replaced where
C       it is used.
C     - (1.,0.) replaced by CMPLX(1.,0.) wherever needed.
C     - In RUNINF, computed GOTO's are replaced by block if's.
C     - Passed through stf; one empty if block removed afterwards.
C     - Add common /TIMING/ in DATAST, TEMPFO, TEMPHY, PSIFOU, PSIPHY,
C       ZETAFO, ZETAPH, VVELFO, VVELPH, UVELFO, UVELPH, DEALSE, in order
C       to help the interprocedural propagation.
C     - Common /NMBRS/ declared in RCS, CSR, ACAC, SCSC to
C       propagate major predicates (like N2P==129) downwards to FTRVMT 
C       (fi, June 1993).
C     - Add common /FT/ in TEMPHY
C     - Add common /MEMORY/ in OCEAN
C     - variable NUMBER in common /DEBUG/ renamed into NUM1 to avoid clashes
C       with variable NUMBER in common /OPTONS/
C     - use full declaration of /FT/ in UVELFO (Francois Irigoin)
C     - Syntactic construct IF(.TRUE.) THEN ... ENDIF added around the
C       time loop to compute accuracte preconditions and transformers.
C       This should be useless with unspaghettify...
C       (Francois Irigoin)
C     - useless STOP in OCEAN commented out to avoid problems with
C       Ronan's dead code elimination (Francois Irigoin)
C     - management of NPTS in SCSC modified to restore
C       exactly NPTS; preconditions fail to find this because they use
C       a transformer computed with no knowledge of NPTS parity;
C       Same transform in ACAC.
C       Note that NPTS is directly saved and restored in RCS.
C       (Francois Irigoin)
C     - artificial IF(.TRUE.)THEN removed; do not forget to unspaghettify
C       the code before processing (Francois Irigoin)
C     - combination of arithmetic IF and RETURNs restructured manually
C       in CPRINT
C     - forwarding of N1's assignment in subsequent WRITE in CPRINT;
C       I guess the programmer did not realize N1 was directly available
C       in COMMON /OPTONS/; he did not realize the potential impact
C       of N1 reassignement in the middle of the time loop; common
C       /OPTONS/ might be useless!
C
C******NOTE: USE OF MACHINE-DEPENDENT CONSTANT: ZERO IN SUBROUTINE RUNIN
C
C            ZERO = 1.0E-200 (MAY HAVE TO REPLACE WITH YOUR MACHINES
C                             RELATIVE PRECISION FOR REAL DATA TYPE)
C

C***********************************************************************
C                        OCEAN FORTRAN PROGRAMS
C                        CREATION DATE: 3/22/81
C***********************************************************************
      PROGRAM OCEAN
C
C  D(Z)/DT + (U,V).GRAD(Z) = RNU*DELSQ(Z) + ALPHA*D(TEMP)/DX
C
C  D(TEMP)/DT + (U,V).GRAD(TEMP) = RKAPA*DELSQ(TEMP) + BETA*V
C
C
C
C THIS PROGRAM SIMULATES TWO-DIMENSIONAL FLUID FLOW ON A   (N2+1) * N1
C GRID. IT IS POSSIBLE TO SPECIFY NON-LINEAR DISTRIBUTION OF DATA POINTS
C THE Y-DIRECTION (THE FIRST INDEX CORRESPONDS TO Y DIRECTION, SECOND TO
C X DIRECTION). N1 AND N2 HAVE TO BE POWERS OF TWO GREATER OR EQUAL TO 8
C
C THE PROGRAM USES LARGE CORE MEMORY. IT REQUIRES 8*(N2+1)*N1+(N2-1)*(N2
C WORDS OF IT. IF NOTEMP = 1 IT IS POSSIBLE TO SAVE 2*(N2+1)*N1 WORDS,
C IF LMAPP = 1 YOU SAVE ADDITIONAL (N2-1)*(N2+1) WORDS.
C
C IF  LMAPP = 2 , LCM REQUIREMENTS ARE AT LEAST  3 * (N2-1) * (N2-1)
C MORE MEMORY COULD BE NEEDED IN THIS CASE BECAUSE OF EIGSET AND LCMRQR.
C
C
C FOR CDC7600 RUNS: DELETE ROUTINES LCMREQ, IN AND OUT FROM THE END OF
C                   THIS PACKAGE. ON CDC7600 THE LCM IS USED THROUGH
C                   THE ASSEMBLY ROUTINES.
C
C                   LIBRARY.
C FOR CRAY-1 RUNS : DELETE ROUTINES SECOND, DATE, TIME, EXSET AND
C                   LOG2 FROM THE END OF THIS PACKAGE. THEY ARE
C                   OBTAINED AT LOAD TIME FROM CRAY LIBRARIES AND
C                   FFTLIB.
C                   ALSO, ADJUST THE SIZE OF PARAMETER LCMSIZ IN
C                   THE ROUTINE LCMREQ SO THAT THE COMMON IS LARGE
C                   ENOUGH TO EMULATE CDC7600 LCM REQUIRED FOR THE RUN.
C
C THE FOLLOWING PARAMETERS DEFINE THE SIZE OF THE RUN:
C
      PARAMETER (NX=128,NY=128)
      PARAMETER (NA1=2* (NY+1),NA2=NX/2,NE=NY-1,NW1=NY+1)
      PARAMETER (NW2=128,NEX=256)
      PARAMETER (NI=2*NY-1,NEQ=3*NY-2,NCW2=2*NW1+1,NCW3=4*NW1+1)
C
C PLEASE REFER TO THE DIMENSIONING AND EQUIVALENCING INSTRUCTIONS IN
C THE CASE YOUR COMPILER CANNOT HANDLE PARAMETER STATEMENTS...
C COMMON BLOCKS:
C
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
C
C THE DIMENSION OF  GSOLV  HAS TO BE AT LEAST (N2-1)
C
      COMMON /SOLV/LVECT,LSKIP,NVECT,NVSKIP,GSOLV(127)
C
C THE DIMENSIONS OF  GMAPP  AND  G  HAVE TO BE AT LEAST (N2+1)
C
      COMMON /MAPING/GMAPP(129),G(129)
      COMMON /MFLOW/U(129)
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      COMMON /PI/PI
      COMMON /NMBRS/N1H,N1Q,N1E,N2M,N2P,N2DP,N2DM,N2PD
      COMMON /EXSIZE/NMAX2
      COMMON /LADDR/LZN,LTN,LPN,LZO,LTO,LPY,LZX,LZY,LEIG,LE,LH,LEV
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      PARAMETER (LCMSIZ=165117)
      COMMON /MEMORY/MEMSIZ,RM(LCMSIZ)
C
C
C COMPLEX ARRAYS HAVE TO BE DIMENSIONED EXACTLY AS THE FORMULAS SAY:
C            CA        (N2+1) X N1/2
C            CWORK     N2+1
C            CWORK2    N2+1
C            CWORK3    N2+1
C            EX        MAX(2*N2, N1)
C
C EX ARRAY MUST NOT BE THE FIRST ARRAY IN THE MEMORY.
C
      COMPLEX CA(NW1,NA2),CWORK(NW1),CWORK2(NW1)
      COMPLEX CWORK3(NW1),EX(NEX),CF
C
C REAL ARRAYS HAVE TO BE DIMENSIONED EXACTLY AS THE FORMULAS SAY:
C            A      2*(N2+1) X N1/2
C            E      (N2-1)   X (N2-1)
C            WORK   N2+1     X MAX(6, 2*MIN(64, N1/2))   (AT LEAST)
C            EVAL   N2-1
C            IR     N2-1
C            IC     N2-1
C
      REAL A(NA1,NA2),E(NE,NE)
      REAL WORK(NW1*NW2)
      REAL EVAL(NE),IR(NE),IC(NE)
      INTEGER VALID
C
C EQUIVALENCE STATEMENTS ARE SAVING MEMORY SPACE. THE FOLLOWING PATTERN
C HAS TO BE FOLLOWED EXACTLY:
C
C     A-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C     CA+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C     EVAL+-+IR-+-+-IC+-+-+E+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C
C
C     WORK+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C     CWORK-+-+-+CWORK2-+-+-CWORK3+-+-+
C
C      EQUIVALENCE (A,CA,EVAL), (A(NY),IR), (A(NI),IC), A(NEQ),E) - ORGI
C      EQUIVALENCE (A,CA,EVAL), (EVAL(NY),IR), (EVAL(NI),IC),
C     *             (EVAL(NEQ),E) - MACFARLANE'S FIX. BAD.
C POINTER'S FIX. ANSI REQUIRES THAT THE NUMBER OF SUBSCRIPTS MATCH
C THE NUMBER OF DIMENSIONS OF THE ARRAY. THAT WAS THE PROBLEM WITH
C THE ORGINAL STATEMENT. THE PROBLEM WITH MACFARLANE'S FIX IS THAT
C NY,NI,NEQ>NE (THE DIMENSION OF EVAL). HERE 3*NE+1-NA1 IS THE 3*NE+1TH
C ELEMENT OF A AND 3*NE+1 = NEQ.
      EQUIVALENCE (A,CA,EVAL), (A(NY,1),IR), (A(NI,1),IC),
     +            (A(3*NE+1-NA1,2),E)
      EQUIVALENCE (WORK,CWORK), (WORK(NCW2),CWORK2), (WORK(NCW3),CWORK3)
      COMMON /VERIFY/VDAT(15),CDAT(15)
      COMMON /DEBUG/NUM1,TLAST
C
C     former BLOCK DATA
C
C
      DATA TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT/0.,0.,0.,0.,
     +     0.,0.,0./
      DATA TXSHUF,TUSHUF,TXPRNT,TXLOG2/0.,0.,0.,0./
      DATA EXXRCS,EXXCSR,EXSCSC,EXACAC,EFTVMT/0.,0.,0.,0.,0./
      DATA NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT/0,0,0,0,0,0,
     +     0/
      DATA NXSHUF,NUSHUF,NXPRNT,NXLOG2/0,0,0,0/
      DATA VDAT/0.10000E+03,-0.10000E+03,0.10000E-04,-0.10000E+03,
     +     -0.15708E+01,0.99983E+02,-0.18147E+01,-0.20427E+01,
     +     -0.22452E+01,-0.24181E+01,0.95976E+02,-0.95977E+02,
     +     0.10000E-04,-0.95978E+02,-0.25589E+01/
C
C PI IS A USEFULL CONSTANT
C
      DATA PI/3.141592653589793/
      DATA NUM1/1/,TLAST/0.0/
C
C
C
C NOW WE CAN START WITH THE PROGRAM...
C SUBROUTINE START TELLS OCEAN WHAT TO DO:
C
      CALL CPUTIM(CPU1)
      CALL ELAPSE(TCLCK1)
      OPEN (1,FILE='OCV')
      OPEN (6,FILE='OCO6')
      WRITE (1,FMT=9000)

      CALL START(NX,NY)
C
C IF  LMAPP = 1 , H  (THE SIZE OF THE BOX IN Y DIRECTION)  SHOULD BE
C SPECIFIED IN THE MAIN PROGRAM.
C
      H = PI
C
C DEFINING CONSTANTS THAT WILL BE NEEDED:
C
      N1H = N1/2
      N1Q = N1/4
      N1E = N1/8
      N2M = N2 - 1
      N2P = N2 + 1
      N2DP = 2*N2 + 1
      N2DM = 2*N2 - 1
      N2PD = 2*N2P
      NW = N2PD*N1H
      NWH = NW/2
      NWQ = NW/4
      NWEIG = N2M
      IF (LMAPP.NE.1) NWEIG = (N2M+3)*N2M
      NMAX2 = MAX(2*N2,N1)
      K1Q = N1E + 1
      K2Q = K1Q + N1E
      K3Q = K2Q + N1E
      K1QM = K1Q - 1
      K2QM = K2Q - 1
      K3QM = K3Q - 1
C
C LCM ALLOCATION AND COMPUTATION OF ADDRESSES:
C
      LARGE = 8*NW
      IF (NOTEMP.EQ.1) LARGE = 6*NW
      IF (LMAPP.NE.2) THEN
C
          LARGE = LARGE + NWEIG

      ELSE
C
          IF (KPRESS.GT.1) LARGE = MAX(LARGE,3*N2M*N2M) + NWEIG + NWEIG
          IF (KPRESS.LE.1) LARGE = MAX(LARGE+NWEIG,3*N2M*N2M)
      END IF
C
      LZN = LCMREQ(LARGE)
      LTN = LZN + NW
      IF (NOTEMP.EQ.1) LTN = LZN
      LPN = LTN + NW
      LZO = LPN + NW
      LTO = LZO + NW
      IF (NOTEMP.EQ.1) LTO = LZO
      LPY = LTO + NW
      LZX = LPY + NW
      LZY = LZX + NW
      LPRESS = LZN + LARGE - NWEIG
      LEIG = LPRESS
      IF ((KPRESS.GT.1) .AND. (LMAPP.NE.1)) LEIG = LPRESS - NWEIG
C
C THESE LCM ADDRESSES ARE USED BY EIGSET AND LCMRQR
C
      LE = LZN
      LH = LE + N2M*N2M
      LEV = LH + N2M*N2M
C
      CALL EXSET(EX,NMAX2)
C
C SET ALL  U(I) TO 0. TO AVOID TROUBLE IF NO MEAN FLOW IS DESIRED
C
      DO 10 I = 1,N2P
          U(I) = 0.
   10 CONTINUE
C
C SET UP ARRAYS ASSOCIATED WITH THE MAPPING FUNCTION:
C
      IF (LMAPP.EQ.1) THEN
C
C LINEAR MAPPING - EIGENVALUES KNOWN, EIGENFUNCTIONS ARE SIMPLY SINES
C AND COSINES, SO THAT NO MATRICES ARE NEEDED:
C
          F = H/REAL(N2)
          FZ = PI/H
          FZSQ = FZ*FZ
          GMAPP(1) = 0.
          G(1) = FZ
          DO 20 I = 1,N2M
              GMAPP(I+1) = F*REAL(I)
              G(I+1) = FZ
              EVAL(I) = - (REAL(I)**2)*FZSQ
   20     CONTINUE
          GMAPP(N2P) = H
          G(N2P) = FZ
          CALL OUT(EVAL,LEIG,NWEIG)
C
      END IF
C
C READ THE DATA:
C
      IF ((KDATA.EQ.1) .OR. (KDATA.EQ.2)) CALL DATAST(A,N2P,N1,GMAPP)
      FX = 2.0*PI/BOXX
      FX2 = 2.*FX
C
C PREPARE THE INPUT FOR PROCESSING:
C
      IF (KDATA.EQ.1) THEN
C
C THE INPUT WAS STREAMFUNCTION IN PHYSICAL SPACE - SHUFFLE THE DATA AND
C GO TO FOURIER SPACE:
C
          CALL IN(A,LPN,NW)
          CALL SHUF(A,N2P,N1,WORK)
          NPTS = N1
          NSKIP = N2P
          MTRN = N2P
          MSKIP = 1
          ISIGN = -1
          CALL XLOG2(NPTS,LOG)
          IEXSHF = NMAX2/NPTS
          CALL RCS(A,EX)
          NPTS = 2*N2
          NSKIP = 1
          MTRN = N1H
          MSKIP = N2P
          ISIGN = -1
          CALL XLOG2(NPTS,LOG)
          IEXSHF = NMAX2/NPTS
          CALL ACAC(A,EX,WORK)
C           F = 0.125/REAL(N1*N2)
          F = 0.0625/REAL(N1*N2)
          DO 30 J = 1,N1H
              DO 40 I = 1,N2PD
                  A(I,J) = A(I,J)*F
   40         CONTINUE
   30     CONTINUE
          CALL OUT(A,LPN,NW)
          CALL OUT(A,LZN,NW)
      END IF
C
CC*****C H E C K   F O U R I E R   C O E F F I C I E N T S**************
C
C         WRITE (6,7010) A(1,1),A(1,2),A(1,3)
C         WRITE (6,7020) A(2,1),A(2,2),A(2,3)
C         WRITE (6,7030) A(3,1),A(3,2),A(3,3)
C 7010    FORMAT(/2X,26HCHECK FOURIER COEFFICIENTS/4X,8HA(1,1) =,1PE13.6
C     1   2X,8HA(1,2) =,1PE13.6,2X,8HA(1,3) =,1PE13.6)
C 7020    FORMAT(4X,8HA(2,1) =,1PE13.6,2X,8HA(2,2) =,1PE13.6,2X,
C     1   8HA(2,3) =,1PE13.6)
C 7030    FORMAT(4X,8HA(3,1) =,1PE13.6,2X,8HA(3,2) =,1PE13.6,2X,
C     1   8HA(3,3) =,1PE13.6/)
C
C
      IF (KDATA.NE.3) THEN
C                                                     2
C STREAMFUNCTION KNOWN - FIND THE VORTICITY:  Z =  DEL  (PSI):
C
          CALL IN(EVAL,LEIG,NWEIG)
          KZN = LZN
          LVECT = N2M
          LSKIP = 2
          NVECT = 2
          NVSKIP = 1
          DO 50 J = 1,N1H
              CALL IN(CWORK,KZN,N2PD)
              F = REAL((J-1)* (J-1))*FX*FX
              DO 60 I = 2,N2
                  CWORK(I) = (EVAL(I-1)-F)*CWORK(I)
   60         CONTINUE
              CWORK(1) = -F*CWORK(1)
              CWORK(N2P) = 0.0
              CALL OUT(CWORK,KZN,N2PD)
              KZN = KZN + N2PD
   50     CONTINUE
          NPTS = 2*N2
          NSKIP = 1
          MTRN = N1H
          MSKIP = N2P
          ISIGN = 1
          CALL XLOG2(NPTS,LOG)
          IEXSHF = NMAX2/NPTS
          CALL IN(A,LZN,NW)
          CALL ACAC(A,EX,WORK)
          DO 70 J = 1,N1H
              DO 80 I = 1,N2PD
                  A(I,J) = 0.5*A(I,J)
   80         CONTINUE
   70     CONTINUE
          CALL OUT(A,LZO,NW)
          IF (NOTEMP.NE.1) THEN
              IF (KDATA.NE.1) THEN
                  CALL IN(A,LTN,NW)
C
C NEW TEMPERATURE KNOWN - SET UP T(OLD):
C
                  NPTS = 2*N2
                  NSKIP = 1
                  MTRN = N1H
                  MSKIP = N2P
                  ISIGN = 1
                  CALL XLOG2(NPTS,LOG)
                  IEXSHF = NMAX2/NPTS
                  CALL ACAC(A,EX,WORK)
                  DO 90 J = 1,N1H
                      DO 100 I = 1,N2PD
                          A(I,J) = A(I,J)*0.5
  100                 CONTINUE
   90             CONTINUE
                  CALL OUT(A,LTO,NW)

              ELSE
C
C TEMPERATURE IS SUPPLIED IN PHYSICAL SPACE - SHUFFLE THE DATA AND
C GO TO FOURIER REPRESENTATION:
C
                  CALL IN(A,LTN,NW)
                  CALL SHUF(A,N2P,N1,WORK)
                  NPTS = N1
                  NSKIP = N2P
                  MTRN = N2P
                  MSKIP = 1
                  ISIGN = -1
                  CALL XLOG2(NPTS,LOG)
                  IEXSHF = NMAX2/NPTS
                  CALL RCS(A,EX)
                  F = 0.5/REAL(N1)
                  DO 110 J = 1,N1H
                      DO 120 I = 1,N2PD
                          A(I,J) = F*A(I,J)
  120                 CONTINUE
  110             CONTINUE
                  CALL OUT(A,LTO,NW)
                  NPTS = 2*N2
                  NSKIP = 1
                  MTRN = N1H
                  MSKIP = N2P
                  ISIGN = -1
                  CALL XLOG2(NPTS,LOG)
                  IEXSHF = NMAX2/NPTS
                  CALL ACAC(A,EX,WORK)
C           F = 0.25/REAL(N2)
                  F = 0.125/REAL(N2)
                  DO 130 J = 1,N1H
                      DO 140 I = 1,N2PD
                          A(I,J) = F*A(I,J)
  140                 CONTINUE
  130             CONTINUE
                  CALL OUT(A,LTN,NW)
              END IF

          END IF

          T = 0.
          DTT = DT
          NSTEPS = 1
      END IF

      CALL RUNINF
      CALL BUG('AFT INITAL')
      
C#
c      IF (.TRUE.) THEN    
  150 CONTINUE
C
C THIS IS THE BEGINNING OF THE TIME STEP.
C
      IF ((MOD(NSTEPS,KTF).EQ.1) .AND. (NOTEMP.NE.1)) CALL TEMPFO(CA,
     +    N2P,N1H,CWORK)
      IF (MOD(NSTEPS,KPF).EQ.1) CALL PSIFOU(CA,N2P,N1H,CWORK)
      IF (MOD(NSTEPS,KZF).EQ.1) CALL ZETAFO(CA,N2P,N1H,CWORK)
      IF (MOD(NSTEPS,KVF).EQ.1) CALL VVELFO(CA,N2P,N1H,FX,CWORK)
      IF (MOD(NSTEPS+1,KTA).EQ.1) THEN
C
C TIME AVERAGE VORTICITY:
C
          DTT = DT
          T = T - 0.5*DT
          CALL IN(A,LZO,NW)
          NPTS = 2*N2
          NSKIP = 1
          MTRN = N1H
          MSKIP = N2P
          ISIGN = -1
          IEXSHF = NMAX2/NPTS
          CALL XLOG2(NPTS,LOG)
          CALL ACAC(A,EX,WORK)
          F = 0.125/REAL(N2)
          DO 160 J = 1,N1H
              DO 170 I = 1,N2PD
                  A(I,J) = A(I,J)*F
  170         CONTINUE
  160     CONTINUE
          CALL OUT(A,LZO,NW)
          ISIGN = 1
          MTRN = N1Q
          KZN = LZN
          KZO = LZO
          DO 180 K = 1,2
              CALL IN(A,KZO,NWH)
              CALL IN(A(1,K2Q),KZN,NWH)
              DO 190 J = 1,N1Q
                  K2QJ = J + K2QM
                  DO 200 I = 1,N2PD
                      A(I,J) = A(I,J) + 0.5*A(I,K2QJ)
  200             CONTINUE
  190         CONTINUE
              CALL OUT(A,KZN,NWH)
              CALL ACAC(A,EX,WORK)
              DO 210 J = 1,N1Q
                  DO 220 I = 1,N2PD
                      A(I,J) = 0.5*A(I,J)
  220             CONTINUE
  210         CONTINUE
              CALL OUT(A,KZO,NWH)
              KZO = KZO + NWH
              KZN = KZN + NWH
  180     CONTINUE
          IF (NOTEMP.NE.1) THEN
C
C TIME AVERAGE TEMPERATURE:
C
              ISIGN = -1
              MTRN = N1H
              CALL IN(A,LTO,NW)
              CALL ACAC(A,EX,WORK)
              F = 0.125/REAL(N2)
              DO 230 J = 1,N1H
                  DO 240 I = 1,N2PD
                      A(I,J) = F*A(I,J)
  240             CONTINUE
  230         CONTINUE
              CALL OUT(A,LTO,NW)
              ISIGN = 1
              MTRN = N1Q
              KTO = LTO
              KTN = LTN
              DO 250 K = 1,2
                  CALL IN(A,KTO,NWH)
                  CALL IN(A(1,K2Q),KTN,NWH)
                  DO 260 J = 1,N1Q
                      K2QJ = J + K2QM
                      DO 270 I = 1,N2PD
                          A(I,J) = A(I,J) + 0.5*A(I,K2QJ)
  270                 CONTINUE
  260             CONTINUE
                  CALL OUT(A,KTN,NWH)
                  CALL ACAC(A,EX,WORK)
                  DO 280 J = 1,N1Q
                      DO 290 I = 1,N2PD
                          A(I,J) = 0.5*A(I,J)
  290                 CONTINUE
  280             CONTINUE
                  CALL OUT(A,KTO,NWH)
                  KTO = KTO + NWH
                  KTN = KTN + NWH
  250         CONTINUE
          END IF

      END IF
C
C THE BEGINING OF A NORMAL TIME STEP
C (U IS X VELOCITY AND V IS Y VELOCITY)
C FIND (-U) = D(PSI)/DY
C
C
      IF (NSTEPS.LT.3) THEN
          CALL IN(CA,LPN,NW)
          CALL PSIPHY(CA,N2P,N1H,EX,G,U,WORK,CWORK3,+1)

      ELSE
          CALL IN(CA,LPN,NW)
          IF (MOD(NSTEPS,KPP).EQ.1) CALL PSIPHY(CA,N2P,N1H,EX,G,U,WORK,
     +        CWORK3,-1)
      END IF
C
      NPTS = 2*N2
      NSKIP = 1
      MTRN = N1H
      MSKIP = N2P
      ISIGN = 1
      CALL XLOG2(NPTS,LOG)
      IEXSHF = NMAX2/NPTS
      CALL IN(CA,LPN,NW)
      F = 0.
      DO 300 I = 1,N2
          DO 310 J = 1,N1H
              CA(I,J) = CMPLX(-F*AIMAG(CA(I,J)),F*REAL(CA(I,J)))
  310     CONTINUE
          F = F + 1.
  300 CONTINUE
      DO 320 J = 1,N1H
          CA(N2P,J) = 0.
  320 CONTINUE
      CALL SCSC(CA,EX,WORK)
      CALL OUT(CA,LPY,NW)
      IF (KSM.EQ.1) CALL MEAN(CA,U,G)
      IF (MOD(NSTEPS,KUF).EQ.1) CALL UVELFO(CA,N2P,N1H,G,EX,CWORK)
      CALL IN(CA,LPN,NW)
C
C FIND  V = D(PSI)/DX
C
      F = 0.
      DO 330 J = 1,N1H
          DO 340 I = 1,N2P
              CA(I,J) = CMPLX(-F*AIMAG(CA(I,J)),F*REAL(CA(I,J)))
  340     CONTINUE
          F = F + FX
  330 CONTINUE
      CALL ACAC(CA,EX,WORK)
      CALL OUT(CA,LPN,NW)
      IF ((MOD(NSTEPS,KPP).EQ.1) .AND. (NOTEMP.NE.1)) CALL IN(CA,LPN,NW)
      NPTS = 2*N2
      NSKIP = 1
      MTRN = N1H
      MSKIP = N2P
      ISIGN = 1
      CALL XLOG2(NPTS,LOG)
      IEXSHF = NMAX2/NPTS
      IF (NOTEMP.NE.1) THEN
C
C T(TOTAL) = T-BETA*Y
C WHERE BETA < 0 MEANS   STABLE STRATIFICATION AND
C       BETA > 0 MEANS UNSTABLE STRATIFICATION
C
          KTO = LTO
          DTB = 0.5*BETA*DTT
          DO 350 J = 1,N1H
              CALL IN(CWORK,KTO,N2PD)
              F = -REAL(J-1)*FX*DTT*0.25
              DO 360 I = 1,N2P
                  CWORK(I) = CWORK(I)*CMPLX(1.,U(I)*F) + DTB*CA(I,J)
  360         CONTINUE
              CALL OUT(CWORK,KTO,N2PD)
              KTO = KTO + N2PD
  350     CONTINUE
       END IF

      CALL IN(CA,LZN,NW)
C
C FIND DZ/DY:
C
      F = 0.
      DO 370 I = 1,N2
          DO 380 J = 1,N1H
              CA(I,J) = CMPLX(-F*AIMAG(CA(I,J)),F*REAL(CA(I,J)))
  380     CONTINUE
          F = F + 1.
  370 CONTINUE
      DO 390 J = 1,N1H
          CA(N2P,J) = 0.
  390 CONTINUE
      CALL SCSC(CA,EX,WORK)
      CALL OUT(CA,LZY,NW)
      CALL IN(CA,LZN,NW)
      CALL ACAC(CA,EX,WORK)
      DO 400 J = 1,N1H
          DO 410 I = 1,N2P
              CA(I,J) = CA(I,J)*0.5
  410     CONTINUE
  400 CONTINUE
      CALL OUT(CA,LZN,NW)
      NPTS = N1
      NSKIP = N2P
      MTRN = N2P
      MSKIP = 1
      ISIGN = 1
      CALL XLOG2(NPTS,LOG)
      IEXSHF = NMAX2/NPTS
      IF (MOD(NSTEPS,KZP).EQ.1) CALL ZETAPH(CA,N2PD,N1H,EX,WORK,-1)
      IF (MOD(NSTEPS,KZP).EQ.1) CALL IN(CA,LZN,NW)
C
C FIND DZ/DX:
C
      F = 0.
      DO 420 J = 1,N1H
          DO 430 I = 1,N2P
              CA(I,J) = CMPLX(-F*AIMAG(CA(I,J)),F*REAL(CA(I,J)))
  430     CONTINUE
          F = F + FX2
  420 CONTINUE
      CALL CSR(CA,EX)
      CALL OUT(CA,LZX,NW)
      CALL IN(CA,LZY,NW)
      CALL CSR(CA,EX)
      CALL OUT(CA,LZY,NW)
      CALL IN(CA,LPN,NW)
      CALL CSR(CA,EX)
      CALL OUT(CA,LPN,NW)
      CALL IN(CA,LPY,NW)
      CALL CSR(CA,EX)
      CALL OUT(CA,LPY,NW)
C
C WE ARE IN PHYSICAL SPACE NOW - FIND  (U,V).GRAD (Z):
C
      IF (MOD(NSTEPS,KUP).EQ.1) CALL UVELPH(A,N2PD,N1H,WORK,-1)
      IF (MOD(NSTEPS,KVP).EQ.1) CALL VVELPH(A,N2PD,N1H,WORK,-1)
      KPY = LPY
      KPN = LPN
      KZY = LZY
      KZX = LZX
      UKSM1F = REAL(2-KSM)
      DO 440 K = 1,4
          CALL IN(A,KPY,NWQ)
          CALL IN(A(1,K1Q),KPN,NWQ)
          CALL IN(A(1,K2Q),KZY,NWQ)
          CALL IN(A(1,K3Q),KZX,NWQ)
          DO 450 I = 1,N2PD
              IPH = (I+1)/2
              GG = G(IPH)
              UU = U(IPH)*UKSM1F
              DO 460 J = 1,N1E
                  A(I,J) = A(I,J)*GG + UU
                  A(I,K2QM+J) = A(I,J)*A(I,K3QM+J) -
     +                          A(I,K1QM+J)*A(I,K2QM+J)*GG
  460         CONTINUE
  450     CONTINUE
          CALL OUT(A,KPY,NWQ)
          CALL OUT(A(1,K2Q),KZY,NWQ)
          KPN = KPN + NWQ
          KPY = KPY + NWQ
          KZX = KZX + NWQ
          KZY = KZY + NWQ
  440 CONTINUE
C
C [Z(OLD) + (MEAN -U) * (-DZ(OLD)/DX)] + (U,V).GRAD(Z)
C
      ISIGN = -1
      CALL IN(A,LZY,NW)
      CALL RCS(A,EX)
      CALL OUT(A,LZY,NW)
      KZO = LZO
      KZN = LZN
      KZY = LZY
      DTFAC = 0.125*DTT/REAL(N1)
      F1 = DTT*FX*0.25
      F = F1
      DO 470 K = 1,2
          CALL IN(CA,KZO,NWH)
          CALL IN(CA(1,K2Q),KZN,NWH)
          CALL OUT(CA(1,K2Q),KZO,NWH)
          CALL IN(CA(1,K2Q),KZY,NWH)
          DO 480 J = 1,N1Q
              K2QJ = J + K2QM
              F = F - F1
              DO 490 I = 1,N2P
                  CA(I,J) = CA(I,J)*CMPLX(1.,F*U(I)) + DTFAC*CA(I,K2QJ)
  490         CONTINUE
  480     CONTINUE
          CALL OUT(CA,KZN,NWH)
          KZN = KZN + NWH
          KZO = KZO + NWH
          KZY = KZY + NWH
  470 CONTINUE
      IF (NOTEMP.NE.1) THEN
C
C FIND DT/DY:
C
          NPTS = 2*N2
          NSKIP = 1
          MTRN = N1H
          MSKIP = N2P
          ISIGN = 1
          CALL XLOG2(NPTS,LOG)
          IEXSHF = NMAX2/NPTS
          CALL IN(CA,LTN,NW)
          F = 0.
          DO 500 I = 1,N2
              DO 510 J = 1,N1H
                  CA(I,J) = CMPLX(-F*AIMAG(CA(I,J)),F*REAL(CA(I,J)))
  510         CONTINUE
              F = F + 1.
  500     CONTINUE
          DO 520 J = 1,N1H
              CA(N2P,J) = 0.
  520     CONTINUE
          CALL SCSC(CA,EX,WORK)
          CALL OUT(CA,LZY,NW)
          CALL IN(CA,LTN,NW)
          CALL ACAC(CA,EX,WORK)
          CALL OUT(CA,LTN,NW)
          NPTS = N1
          NSKIP = N2P
          MTRN = N2P
          MSKIP = 1
          CALL XLOG2(NPTS,LOG)
          IEXSHF = NMAX2/NPTS
          IF (NSTEPS.EQ.1) THEN
              CALL TEMPHY(A,N2PD,N1H,EX,WORK,0)

          ELSE IF ((MOD(NSTEPS,KTP).EQ.1) .OR.
     +             (MOD(NSTEPS,KPP).EQ.1)) THEN
              CALL TEMPHY(A,N2PD,N1H,EX,WORK,-1)
          END IF
C
C TEMPERATURE INFLUENCE:
C
          F = -FX
          KZN = LZN
          KZX = LZX
          KTN = LTN
          DTA = 0.5*ALPHA*DTT
          DO 530 K = 1,2
              CALL IN(CA,KZN,NWH)
              CALL IN(CA(1,K2Q),KTN,NWH)
              DO 540 J = 1,N1Q
                  K2QJ = J + K2QM
                  F = F + FX
                  DO 550 I = 1,N2P
                      CA(I,K2QJ) = CMPLX(-F*AIMAG(CA(I,K2QJ)),
     +                             F*REAL(CA(I,K2QJ)))
                      CA(I,J) = CA(I,J) + DTA*CA(I,K2QJ)
  550             CONTINUE
  540         CONTINUE
              CALL OUT(CA,KZN,NWH)
              CALL OUT(CA(1,K2Q),KZX,NWH)
              KZX = KZX + NWH
              KTN = KTN + NWH
              KZN = KZN + NWH
  530     CONTINUE
          CALL IN(A,LZX,NW)
          CALL CSR(A,EX)
          CALL OUT(A,LZX,NW)
          CALL IN(A,LZY,NW)
          CALL CSR(A,EX)
          CALL OUT(A,LZY,NW)
          KPN = LPN
          KPY = LPY
          KZX = LZX
          KZY = LZY
C
C IN PHYSICAL SPACE - FIND  (U,V).GRAD(T):
C
          DO 560 K = 1,4
              CALL IN(A,KPY,NWQ)
              CALL IN(A(1,K1Q),KPN,NWQ)
              CALL IN(A(1,K2Q),KZY,NWQ)
              CALL IN(A(1,K3Q),KZX,NWQ)
              DO 570 I = 1,N2PD
                  IPH = (I+1)/2
                  GG = G(IPH)
                  DO 580 J = 1,N1E
                      A(I,J) = A(I,J)*A(I,K3QM+J) -
     +                         A(I,K1QM+J)*A(I,K2QM+J)*GG
  580             CONTINUE
  570         CONTINUE
              CALL OUT(A,KZY,NWQ)
              KPN = KPN + NWQ
              KPY = KPY + NWQ
              KZX = KZX + NWQ
              KZY = KZY + NWQ
  560     CONTINUE
          ISIGN = -1
          CALL IN(A,LZY,NW)
          CALL RCS(A,EX)
          CALL OUT(A,LZY,NW)
          KTN = LTN
          KTO = LTO
          KZY = LZY
          DO 590 K = 1,2
              CALL IN(CA,KTO,NWH)
              CALL IN(A(1,K2Q),KTN,NWH)
              DO 600 J = K2Q,N1H
                  DO 610 I = 1,N2PD
                      A(I,J) = 0.5*A(I,J)
  610             CONTINUE
  600         CONTINUE
              CALL OUT(A(1,K2Q),KTO,NWH)
              CALL IN(CA(1,K2Q),KZY,NWH)
              DO 620 J = 1,N1Q
                  K2QJ = J + K2QM
                  DO 630 I = 1,N2P
                      CA(I,J) = CA(I,J) + DTFAC*CA(I,K2QJ)
  630             CONTINUE
  620         CONTINUE
              CALL OUT(CA,KTN,NWH)
              KTN = KTN + NWH
              KTO = KTO + NWH
              KZY = KZY + NWH
  590     CONTINUE
      END IF

      DTF = 0.25*DTT*FX
      F = -DTF
      KZN = LZN
      KTN = LTN
      DO 640 K = 1,2
          IF (NOTEMP.NE.1) CALL IN(CA,KTN,NWH)
          CALL IN(CA(1,K2Q),KZN,NWH)
          DO 650 J = 1,N1Q
              K2QJ = J + K2QM
              F = F + DTF
              IF (NOTEMP.NE.1) THEN
                  DO 660 I = 1,N2P
                      WORK(I) = F*U(I)
                      CA(I,J) = CMPLX(1.0,-WORK(I))*CA(I,J)/
     +                          (1.0+WORK(I)*WORK(I))
  660             CONTINUE
              END IF

              DO 670 I = 1,N2P
                  WORK(I) = F*U(I)
                  CA(I,K2QJ) = CMPLX(1.0,-WORK(I))*CA(I,K2QJ)/
     +                         (1.0+WORK(I)*WORK(I))
  670         CONTINUE
  650     CONTINUE
          IF (NOTEMP.NE.1) CALL OUT(CA,KTN,NWH)
          CALL OUT(CA(1,K2Q),KZN,NWH)
          KTN = KTN + NWH
          KZN = KZN + NWH
  640 CONTINUE
      NPTS = 2*N2
      NSKIP = 1
      MTRN = N1H
      MSKIP = N2P
      ISIGN = -1
      IEXSHF = NMAX2/NPTS
      CALL XLOG2(NPTS,LOG)
      CALL IN(A,LZN,NW)
      CALL ACAC(A,EX,WORK)
      CALL OUT(A,LZN,NW)
      IF (NOTEMP.NE.1) THEN
          CALL IN(A,LTN,NW)
          CALL ACAC(A,EX,WORK)
          CALL OUT(A,LTN,NW)
      END IF

      KZN = LZN
      KPN = LPN
      F2 = 1./ (4.*REAL(N2))
      F1 = RNU*DTT
C
C FIND PRESSURE NOW OR NEVER (IF REQUIRED...)
C
      CALL IN(EVAL,LEIG,NWEIG)
      LVECT = N2M
      LSKIP = 2
      NVECT = 2
      NVSKIP = 1
      FXSQ = FX*FX
C
C UPDATE VORTICITY AND FIND THE NEW STREAMFUNCTION:
C
      DO 680 J = 1,N1H
          CALL IN(CWORK,KZN,N2PD)
          F = REAL((J-1)* (J-1))*FXSQ
          DO 690 I = 2,N2
              CWORK(I) = CWORK(I)*F2/ (1.0-F1* (EVAL(I-1)-F))
              CWORK2(I) = CWORK(I)/ (EVAL(I-1)-F)
  690     CONTINUE
          CWORK(1) = CWORK(1)*F2/ (1.0+F1*F)
          CWORK2(1) = 0.0
          IF (F.NE.0.0) CWORK2(1) = -CWORK(1)/F
          CWORK(N2P) = 0.0
          CWORK2(N2P) = 0.0
          CALL OUT(CWORK,KZN,N2PD)
          CALL OUT(CWORK2,KPN,N2PD)
          KPN = KPN + N2PD
          KZN = KZN + N2PD
  680 CONTINUE
C
C DE-ALIASE UPDATED FIELDS IF REQUESTED:
C
      IF (NOALS.EQ.1) CALL DEALSE(CA,N2P,N1H,LZN)
      IF (NOALS.EQ.1) CALL DEALSE(CA,N2P,N1H,LPN)
      IF (NOTEMP.NE.1) THEN
C
C UPDATE TEMPERATURE:
C
          IF (NOALS.EQ.1) CALL IN(EVAL,LEIG,NWEIG)
          KTN = LTN
          F1 = RKAPA*DTT
          DO 700 J = 1,N1H
              CALL IN(CWORK,KTN,N2PD)
              F = REAL((J-1)* (J-1))*FXSQ
              DO 710 I = 2,N2
                  CWORK(I) = CWORK(I)*F2/ (1.0-F1* (EVAL(I-1)-F))
  710         CONTINUE
              CWORK(1) = CWORK(1)*F2/ (1.0+F1*F)
              CWORK(N2P) = 0.0
              CALL OUT(CWORK,KTN,N2PD)
              KTN = KTN + N2PD
  700     CONTINUE
C
C DE-ALIASE IF REQUESTED:
C
          IF (NOALS.EQ.1) CALL DEALSE(CA,N2P,N1H,LTN)
      END IF
C
C AT LAST, THE END OF THE TIME STEP:
C
      DTT = DT + DT
      T = T + DT
      NSTEPS = NSTEPS + 1
      IF (NUMBER.GE.NSTEPS) GO TO 150
C#
c      END IF
C
      WRITE (6,FMT=9010) TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,
     +  TFTVMT,TXSHUF,TUSHUF,TXPRNT,TXLOG2
      WRITE (6,FMT=9020) EXXRCS,EXXCSR,EXSCSC,EXACAC,EFTVMT
      WRITE (6,FMT=9030) NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,
     +  NFTVMT,NXSHUF,NUSHUF,NXPRNT,NXLOG2
C
      CALL BUG('AFTER  ALL')
C
      CALL CPUTIM(CPU2)
      CALL ELAPSE(TCLCK2)
C
      TWALL = MOD(TCLCK2-TCLCK1+1000000.0,1000000.0)
      WRITE (6,FMT=9040) TWALL
      WRITE (1,FMT=*)
      WRITE (1,FMT=*) 'ELAPSED CPU TIME IN SECONDS: ',CPU2 - CPU1
C MFLOPS RATE = MFLOP 1 CPU CRAY X-MP/CPU TIME
      WRITE (1,FMT=*) 'MFLOPS RATE: ',1657.863/ (CPU2-CPU1)
      WRITE (1,FMT=*) 'ELAPSED WALLCLOCK TIME IN SECONDS: ',TWALL
      WRITE (1,FMT=9050)
      WRITE (1,FMT=9080) (CDAT(I),I=1,15)
      VALID = 1
      DO 720 I = 1,15
          IF (I.EQ.3 .OR. I.EQ.13) THEN
              IF (ABS(CDAT(I)).GT.ABS(VDAT(I))) VALID = 0

          ELSE IF (ABS(VDAT(I)-CDAT(I)).GT.ABS(.01*VDAT(I))) THEN
              VALID = 0
          END IF

  720 CONTINUE
C
      IF (VALID.EQ.1) THEN
          WRITE (1,FMT=9060)

      ELSE
          WRITE (1,FMT=9070)
      END IF

c      STOP

 9000 FORMAT (1X,
     +   'OUTPUT FOR PERFECT CLUB BENCHMARK: OCEAN INSTRUMENTED VERSION'
     +       ,/,1X,'$Revision: 1.A.1 $ $Author: psharma $',/)
C
 9010 FORMAT (/,2X,'SUBROUTINE    IN, EX-TIME FOR FIRST CALL =',1P,
     +       E13.6,' SECONDS',/,2X,
     +       'SUBROUTINE   OUT, EX-TIME FOR FIRST CALL =',1P,E13.6,/,
     +       2X,'SUBROUTINE   RCS, EX-TIME FOR FIRST CALL =',1P,E13.6,
     +       /,2X,'SUBROUTINE   CSR, EX-TIME FOR FIRST CALL =',1P,
     +       E13.6,/,2X,'SUBROUTINE  SCSC, EX-TIME FOR FIRST CALL =',
     +       1P,E13.6,/,2X,'SUBROUTINE  ACAC, EX-TIME FOR FIRST CALL ='
     +       ,1P,E13.6,/,2X,
     +       'SUBROUTINE FTRVMT EX-TIME FOR FIRST CALL =',1P,E13.6,/,
     +       2X,'SUBROUTINE  SHUF, EX-TIME FOR FIRST CALL =',1P,E13.6,
     +       /,2X,'SUBROUTINE UNSHUF EX-TIME FOR FIRST CALL =',1P,
     +       E13.6,/,2X,'SUBROUTINE RPRINT EX-TIME FOR FIRST CALL =',
     +       1P,E13.6,/,2X,'SUBROUTINE XLOG2, EX-TIME FOR FIRST CALL ='
     +       ,1P,E13.6)
 9020 FORMAT (/,2X,'SUBROUTINE   RCS, EX-TIME LAST TIMESTEP  =',1P,
     +       E13.6,' SECONDS',/,2X,
     +       'SUBROUTINE   CSR, EX-TIME LAST TIMESTEP  =',1P,E13.6,/,
     +       2X,'SUBROUTINE  SCSC, EX-TIME LAST TIMESTEP  =',1P,E13.6,
     +       /,2X,'SUBROUTINE  ACAC, EX-TIME LAST TIMESTEP  =',1P,
     +       E13.6,/,2X,'SUBROUTINE FTRVMT EX-TIME LAST TIMESTEP  =',
     +       1P,E13.6)
 9030 FORMAT (/,2X,'SUBROUTINE    IN, CUMULATIVE CALLS TO    =',I12,/,
     +       2X,'SUBROUTINE   OUT, CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE   RCS, CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE   CSR, CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE  SCSC, CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE  ACAC, CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE FTRVMT CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE  SHUF, CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE UNSHUF CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE RPRINT CUMULATIVE CALLS TO    =',I12,/,2X,
     +       'SUBROUTINE XLOG2, CUMULATIVE CALLS TO    =',I12)
 9040 FORMAT (/,2X,'MEASURE OF WALL-CLOCK TIME (ENVIRONMENT SENSITIVE)'
     +       ,/,2X,'FOR EXECUTION ONLY =',1P,E13.6,1X,'SECONDS')
C
 9050 FORMAT (/,1X,'VALIDATION PARAMETERS:',/)
 9060 FORMAT (/,/,1X,'RESULTS FOR THIS RUN ARE:   VALID')
 9070 FORMAT (/,/,1X,'RESULTS FOR THIS RUN ARE: INVALID')
 9080 FORMAT (5E15.5)

      END
C***********************************************************************
      SUBROUTINE DEALSE(CA,N2P,N1H,LOC)
      COMPLEX CA(N2P,1)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
C
C THIS FUNCTION GIVES THE INDEX OF THE HIGHEST MODE TO BE RETAINED
C IN TERMS OF THE NUMBER OF MODES WHICH CAN BE RESOLVED:
C
C#      LSTMOD (MODEHI) = INT(0.666666666666666*REAL(MODEHI-1)+0.999)
C
      NW = 2*N2P*N1H
      CALL IN(CA,LOC,NW)
      N1CUT = INT(0.666666666666666*REAL(N1H-1)+0.999) + 1
      N2CUT = INT(0.666666666666666*REAL(N2P-1)+0.999) + 1
C
C THIS DEALIASING PROCEDURE HAS TO BE IMPROVED IF NONLINEAR MAPPING
C IN Y-DIRECTION IS USED:
C
      DO 10 J = N1CUT,N1H
          DO 20 I = 1,N2P
              CA(I,J) = 0.0
   20     CONTINUE
   10 CONTINUE
      DO 30 J = 1,N1H
          DO 40 I = N2CUT,N2P
              CA(I,J) = 0.0
   40     CONTINUE
   30 CONTINUE
      CALL OUT(CA,LOC,NW)
C
      END
C***********************************************************************
C     FGPS  2-MAY-99 SET KPP = KZP = KUP = KVP = 249 AND NUMBER = 250
C     FGPS 18-APR-88 SET KPP = KZP = KUP = KVP = 499 AND NUMBER = 500
C
      SUBROUTINE START(NX,NY)
C
C THIS SUBROUTINE IS USED TO SPECIFY OPTIONS FOR PROGRAM OCEAN.
C
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
C
C N1 AND N2 ARE THE DIMENSIONS OF THE MATRIX. THEY HAVE TO BE POWERS
C OF  2  AND GREATER THAN OR EQUAL TO  8
C
      N1 = NX
      N2 = NY
C
C KDATA SPECIFIES THE WAY DATA IS SUPPLIED TO THE PROGRAM.
C
C            KDATA = 1     MEANS THAT IT WILL BE IN PHYSICAL SPACE
C                  = 2     MEANS THAT IT WILL BE IN FOURIER SPACE
C                  = 3     MEANS THAT THIS IS A RESTART RUN (->READ TAPE
C
      KDATA = 1
C
C LMAPP SPECIFIES THE MAPPING IN THE Y DIRECTION.
C
C            LMAPP = 1     MEANS LINEAR MAPPING (-> EIGSET NOT CALLED)
C                  = 2     MEANS NON-LINEAR MAPPING
C                  = 3     MEANS RESTART RUN, NON-LINEAR MAPPING
C
      LMAPP = 1
C
C NOTEMP EXCLUDES TEMPERATURE FIELD FROM CALCULATIONS WHEN SET TO 1.
C
      NOTEMP = 0
C
C NOALS TURNS ON DEALIASING AT EACH TIME STEP WHEN SET TO 1:
C (TO BE USED ONLY WITH LINEAR MAPPING)
C
      NOALS = 1
C
C TIME AVERAGING IS PERFORMED EACH KTA TIME STEPS:
C
      KTA = 50
C
C THE FOLLOWING PARAMETERS SPECIFY WHEN TO CALL CORRESPONDING OUTPUT
C SUBROUTINES. IF YOU WANT, FOR EXAMPLE, PSIPHY TO BE CALLED EVERY
C 50 TIME STEPS, YOU SHOULD SET KPP TO 50.
C
C WARNING: SETTING A PARAMETER TO 1 TURNS OFF THE SUBROUTINE. IT DOES
C          NOT CAUSE IT TO BE CALLED EVERY TIME STEP.
C          YOU SHOULD NEVER SET ANY OF THEM TO 0, SINCE THEY ARE USED
C          WITH THE FUNCTION MOD(NSTEPS,KXYZ).
C
      KPP = 249
      KZP = 249
      KTP = 50
      KUP = 249
      KVP = 249
      KPF = 1
      KZF = 1
      KTF = 1
      KUF = 1
      KVF = 1
      KPRESS = 1
C
C KTAPE SPECIFIES HOW WILL HISTORY TAPE BE WRITTEN:
C
C          KTAPE = 1  WRITE ONLY EIG-STUFF ON TAPE
C                = 2  WRITE ONLY DATA ON TAPE
C                = 3  WRITE BOTH EIG-STUFF AND DATA ON TAPE
C
C ANY OTHER VALUE WILL PREVENT THE HISTORY TAPE FROM BEING WRITTEN.
C
      KTAPE = 0
C
C NUMBER SPECIFIES THE NUMBER OF TIME STEPS IN A SINGLE RUN:
C
      NUMBER = 250
C
C KSM GOVERNS THE USE OF THE MEAN FLOW OPTION.
C
C                   KSM = 0       MEANS THAT U(MEAN) IS ZERO
C                         1       MEANS THAT IT SHOULD BE COMPUTED
C                         2       MEANS THAT IT IS USER-SPECIFIED
C
      KSM = 1
C
C THAT WAS ALL. RETURN TO MOTHER OCEAN.
C
      END
C***********************************************************************
      SUBROUTINE GSET(GMAPP)
      DIMENSION GMAPP(1)
      COMMON /PI/PI
      COMMON /NMBRS/NSKIP(4),N2P,NDUMY1(3)

      F = PI/REAL(N2P-1)
C
C GSET DEFINES THE MAPPING FUNCTION.
C
      DO 10 I = 1,N2P
          GMAPP(I) = F*REAL(I-1)
   10 CONTINUE
      END
C***********************************************************************
C     FGPS 12-MAY-88 RESET "RLAMDA = 1.00"
C     FGPS 18-APR-88 RESET "DT" TO DT = 0.0001, OTHERWISE THE INCREASED
C                    FUNCTION RESULTS IN NUMERICAL INSTABILITY
C     FGPS 15-APR-88 INCREASED THE INITIAL AMPLITUDE PRESCRIBED FOR THE
C                    FUNCTION BY *50
C     FGPS 16-FEB-88 RESET "DT" TO DT = 0.001
C
      SUBROUTINE DATAST(A,N2P,N1,GMAPP)
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /PI/PI
      COMMON /LADDR/LZN,LTN,LPN,LZO,LTO,LPY,LZX,LZY,LEIG,LDUMY1(3)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      DIMENSION A(N2P,1),GMAPP(1)
C
C PROGRAM TO INITIALIZE THE RUN (EITHER IN PHYSICAL OR FOURIER SPACE)
C PHYSICAL SPACE: KDATA=1    --> A (I,J) ... POINT (I-1, J-1)
C FOURIER SPACE : KDATA=2    --> CA(I,J) ... MODE  (I-1, J-1)
C
C ALSO, IF YOU WANT TO SPECIFY THE MEAN FLOW (KSM=2), DO IT HERE.
C
      COMMON /MFLOW/U(129)
C
      ALPHA = 10.0
C
C UNSTABLE CASE:
      BETA = 1.0
C      RLAMDA = 1.35
C     WHEN CODE IS USED AS A BENCHMARK, SET RLAMDA = 1.00
      RLAMDA = 1.00
      BOXX = 2.0*PI*RLAMDA
      FX = 2.0*PI/BOXX
C
C ASSUMING THAT GSET IS CALLED. IF LMAPP = 1, H SHOULD BE SET TO
C A DEFINED POSITIVE NUMBER.
C
      H = GMAPP(N2P)
C
C COMPUTE VISCOSITY AND HEAT CONDUCTIVITY FROM PRANDTL NUMBER
C AND THE RATIO OF RAYLEIGH NUMBER TO ITS CRITICAL VALUE:
C
      PRANDT = 10.0
      RRATIO = 20.0
      W = ALPHA*BETA*PRANDT*4.0/ (27.0*RRATIO)
      RNU = SQRT(ABS(W))
      RKAPA = RNU/PRANDT
C
C DETERMINE DIFFUSION LIMIT FOR THE TIME STEP AND COMPARE IT TO
C THE CHOSEN VALUE:
C
      DT = 0.0001
      X = FX*REAL(N1/2-1)
      Y = REAL(N2P-1)
      DTMAX = 1.0/ ((X*X+Y*Y)*MAX(RNU,RKAPA))
      WRITE (6,FMT=9000) DTMAX,DT
      NW = N2P*N1
C
C
      EPS = 100.
      DO 10 J = 1,N1
          X = REAL(J-1)*2.0*PI/REAL(N1)
          DO 20 I = 1,N2P
              Y = REAL(I-1)*PI/REAL(N2P-1)
              A(I,J) = EPS*SIN(X)*SIN(Y)
   20     CONTINUE
   10 CONTINUE
C
      WRITE (6,FMT=9010)
      CALL OUT(A,LPN,NW)
      DO 30 J = 1,N1
          DO 40 I = 1,N2P
              A(I,J) = 0.0
   40     CONTINUE
   30 CONTINUE
      WRITE (6,FMT=9020)
      CALL OUT(A,LTN,NW)

 9000 FORMAT (' DIFFUSION CONSTRAINT ON DT: DTMAX = ',F10.6,/,
     +       ' ACTUAL DT CHOSEN FOR THIS RUN: DT = ',F10.6)
 9010 FORMAT (/,/,/,' INITIAL STREAMFUNCTION IS   A SIN(X) SIN(Y),',/,
     +       ' WHERE  0 <= X <= 2*PI  AND  0 <= Y <= PI.',/,
     +       ' THIS IMPLIES TWO CONVECTION CELLS INITIALLY.',/,
     +       ' NOTE THAT THE PRESCRIBED INITIAL STREAM-',/,
     +       ' FUNCTION VS X, FOR Y=YMAX/2, IS PRINTED OUT',/,
     +       ' BELOW FOR NSTEPS=1 (TIME ZERO) AND ALSO FOR',/,
     +       ' THE NEXT STEP, NSTEPS=2',/,/)
 9020 FORMAT (' INITIAL TEMPERATURE PERTURBATION IS ZERO.',/,
     +       ' NOTE THAT THE INITIAL TEMPERATURE DROP VS',/,
     +       ' Y IS PRINTED OUT BELOW, WHERE INITIALLY',/,
     +       ' THIS IS INDEPENDENT OF X,  I.E., 1 <= IX',/,
     +       ' <= 64, AND WHERE THE Y REPRESENTATION IN',/,
     +       ' PHYSICAL SPACE SPANS 1 <= JY <= 129.   A',/,
     +       ' KEY VARIABLE, THE EVOLUTION OF WHICH CAN',/,
     +       ' MONITOR CODE PERFORMANCE, IS  "TEMP PHY(',/,
     +       ' JY=65,IX=33)"; I.E., A MEASURE OF TEMPER-',/,
     +       ' ATURE IN PHYSICAL SPACE AT THE CENTERPOINT',/,
     +       ' OF THE BOUNDARY SEPARATING THE TWO CONVEC-',/,
     +       ' TION CELLS.')

      END
C***********************************************************************
      SUBROUTINE RUNINF
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
C
C THIS SUBROUTINE TELLS US WHAT  OCEAN  WILL BE DOING:
C
C*****MACHINE-DEPENDENT CONSTANT (REPLACE FOR YOUR MACHINE)
C
C*********************
      ZERO = 1.0E-200
C*********************
C
      WRITE (6,FMT=9000)
C
      CALL CPUTIM(CPUTM)
C      CALL DATE   (TODAY)
C      CALL TIME   (NOW)
      WRITE (6,FMT=9010) CPUTM
C
      MOREST = NUMBER - NSTEPS + 1
      WRITE (6,FMT=9020) N1,N2,MOREST,KTA
      IF (KSM.EQ.1) WRITE (6,FMT=9030)
      IF (KSM.EQ.2) WRITE (6,FMT=9040)
C
C#      GO TO (1,2,3), KDATA
C#      WRITE(6,400)
C#      GO TO 10
C#1     WRITE(6,401)
C#      GO TO 10
C#2     WRITE(6,402)
C#      GO TO 10
C#3     WRITE(6,403)
C
      IF (KDATA.EQ.1) THEN
          WRITE (6,FMT=9060)

      ELSE IF (KDATA.EQ.2) THEN
          WRITE (6,FMT=9070)

      ELSE IF (KDATA.EQ.3) THEN
          WRITE (6,FMT=9080)

      ELSE
          WRITE (6,FMT=9050)
      END IF
C
C#10    GO TO (11,123,123), LMAPP
C#      WRITE(6,500)
C#      GO TO 20
C#11    WRITE(6,501)
C#      GO TO 20
C#123   WRITE(6,523)
C
      IF (LMAPP.EQ.1) THEN
          WRITE (6,FMT=9100)

      ELSE IF (LMAPP.EQ.2) THEN
          WRITE (6,FMT=9110)

      ELSE IF (LMAPP.EQ.3) THEN
          WRITE (6,FMT=9110)

      ELSE
          WRITE (6,FMT=9090)
      END IF
C
C#20    GO TO (21), NOTEMP
C#      WRITE(6,600)
C#      GO TO 30
C#21    WRITE(6,601)
C
      IF (NOTEMP.EQ.1) THEN
          WRITE (6,FMT=9130)

      ELSE
          WRITE (6,FMT=9120)
      END IF
C
C#30    GO TO (31,32,33), KTAPE
C#      WRITE(6,700)
C#      GO TO 40
C#31    WRITE(6,701)
C#      GO TO 40
C#32    WRITE(6,702)
C#      GO TO 40
C#33    WRITE(6,703)
C
      IF (KTAPE.EQ.1) THEN
          WRITE (6,FMT=9150)

      ELSE IF (KTAPE.EQ.2) THEN
          WRITE (6,FMT=9160)

      ELSE IF (KTAPE.EQ.3) THEN
          WRITE (6,FMT=9170)

      ELSE
          WRITE (6,FMT=9140)
      END IF
C
      IF (KPP.NE.1) WRITE (6,FMT=9180) KPP
      IF (KZP.NE.1) WRITE (6,FMT=9190) KZP
      IF (KTP.NE.1) WRITE (6,FMT=9200) KTP
      IF (KUP.NE.1) WRITE (6,FMT=9210) KUP
      IF (KVP.NE.1) WRITE (6,FMT=9220) KVP
      IF (KPF.NE.1) WRITE (6,FMT=9230) KPF
      IF (KZF.NE.1) WRITE (6,FMT=9240) KZF
      IF (KTF.NE.1) WRITE (6,FMT=9250) KTF
      IF (KUF.NE.1) WRITE (6,FMT=9260) KUF
      IF (KVF.NE.1) WRITE (6,FMT=9270) KVF
      IF (KPRESS.NE.1) WRITE (6,FMT=9280) KPRESS
C
      IF (NOALS.EQ.1) WRITE (6,FMT=9290)
C
      REY = 1.0/ (RNU+ZERO)
      RH = H*H*H*H*ABS(ALPHA*BETA)/ (RNU*RKAPA+ZERO)
C
C CRITICAL RAYLEIGH NUMBER IS   (PI**4) * 27/4:
C
      RHCRIT = 657.5113648
      RRATIO = RH/RHCRIT
      PR = RNU/ (RKAPA+ZERO)
      WRITE (6,FMT=9300) BOXX,H,RNU,RKAPA,ALPHA,BETA,REY,RH,RRATIO,PR,
     +  DT,T
C
C
C FORMAT STATEMENTS:
C
 9000 FORMAT (/,12X,' OOOOOO     CCCCCC    EEEEEEE    AAAAAA    N     N'
     +       ,/,12X,'O      O   C      C   E         A      A   NN    N'
     +       ,/,12X,'O      O   C          E         A      A   N N   N'
     +       ,/,12X,'O      O   C          EEEE      AAAAAAAA   N  N  N'
     +       ,/,12X,'O      O   C          E         A      A   N   N N'
     +       ,/,12X,'O      O   C      C   E         A      A   N    NN'
     +       ,/,12X,' OOOOOO     CCCCCC    EEEEEEE   A      A   N     N'
     +       )
C
C200   FORMAT (////' THIS RUN BEGINS ON DATE ',A8,' AT TIME ',A8,/,
C     1            ' CPU TIME USED SO FAR IS',F9.6,' SECONDS.')
 9010 FORMAT (/,/,/,/,' CPU TIME USED SO FAR IS',F9.6,' SECONDS.',/,/,
     +       'W A R N I N G ******** POSSIBLE UNDERFLOW ********',/,
     +       'THE STATEMENT "ZERO = 1.0E-200" APPEARS IN RUNINF.')
C
 9020 FORMAT (/,/,' THIS RUN USES ',I3,' BY ',I3,' GRID, AND DOES',I5,
     +       ' TIME STEPS',/,' TIME AVERAGING IS DONE EVERY',I3,
     +       ' STEPS')
 9030 FORMAT (' MEAN FLOW IS INCLUDED IN CALCULATIONS')
 9040 FORMAT (' MEAN FLOW IS SPECIFIED BY USER')
C
 9050 FORMAT (' KDATA HAS ILLEGAL VALUE')
 9060 FORMAT (' INITIAL DATA IN PHYSICAL SPACE')
 9070 FORMAT (' INITIAL DATA IN FOURIER SPACE')
 9080 FORMAT (/,' THIS IS A RESTART RUN',/)
C
 9090 FORMAT (' LMAPP HAS AN ILLEGAL VALUE')
 9100 FORMAT (' LINEAR MAPPING IN BOTH DIRECTIONS')
 9110 FORMAT (' NON-LINEAR MAPPING IN Y DIRECTION')
C
 9120 FORMAT (' TEMPERATURE FIELD INCLUDED IN CALCULATIONS')
 9130 FORMAT (' CALCULATIONS DONE WITHOUT TEMPERATURE FIELD')
C
 9140 FORMAT (/,' NO HISTORY TAPE WILL BE WRITTEN',/)
 9150 FORMAT (/,' HISTORY TAPE WILL INCLUDE OPERATOR MATRICES ONLY',/)
 9160 FORMAT (/,' HISTORY TAPE WILL NOT INCLUDE OPERATOR MATRICES',/)
 9170 FORMAT (/,' HISTORY TAPE WILL INCLUDE ALL IMPORTANT ARRAYS',/)
C
 9180 FORMAT (' OUTPUT PHYSICAL STREAMFUNCTION EVERY',I4,' TIME STEPS')
 9190 FORMAT (' OUTPUT PHYSICAL VORTICITY      EVERY',I4,' TIME STEPS')
 9200 FORMAT (' OUTPUT PHYSICAL TEMPERATURE    EVERY',I4,' TIME STEPS')
 9210 FORMAT (' OUTPUT PHYSICAL X VELOCITY     EVERY',I4,' TIME STEPS')
 9220 FORMAT (' OUTPUT PHYSICAL Y VELOCITY     EVERY',I4,' TIME STEPS')
 9230 FORMAT (' OUTPUT FOURIER  STREAMFUNCTION EVERY',I4,' TIME STEPS')
 9240 FORMAT (' OUTPUT FOURIER  VORTICITY      EVERY',I4,' TIME STEPS')
 9250 FORMAT (' OUTPUT FOURIER  TEMPERATURE    EVERY',I4,' TIME STEPS')
 9260 FORMAT (' OUTPUT FOURIER  X VELOCITY     EVERY',I4,' TIME STEPS')
 9270 FORMAT (' OUTPUT FOURIER  V VELOCITY     EVERY',I4,' TIME STEPS')
 9280 FORMAT (' OUTPUT PHYSICAL PRESSURE       EVERY',I4,' TIME STEPS')
C
 9290 FORMAT (' DE-ALIASING TURNED ON.')
C
 9300 FORMAT (/,/,/,/,' PHYSICAL PARAMETERS FOR THIS RUN ARE:',/,
     +       ' BOX DIMENSIONS          : ',F10.6,' BY ',F10.6,/,
     +       ' VISCOSITY (NU)          : ',E24.14,/,
     +       ' HEAT CONDUCTIVITY (KAPA): ',E24.14,/,
     +       ' ALPHA AND BETA          : ',F10.6,' AND',F10.6,/,
     +       ' REYNOLDS NUMBER         : ',E24.14,/,
     +       ' RAYLEIGH NUMBER         : ',E24.14,/,
     +       ' RATIO RAY/RAY(CRITICAL) : ',E24.14,/,
     +       ' PRANDTL  NUMBER         : ',E24.14,/,/,
     +       ' THE TIME STEP IS',F10.6,/,' WE BEGIN AT  T =',F10.6,/,/,
     +       /,1H ,60 (2H+-),/,/)

      END
C***********************************************************************
      SUBROUTINE SHUF(A,N2P,N1,WORK)
      DIMENSION A(N2P,1),WORK(1)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      LOGICAL LFIRST
      DATA LFIRST/.TRUE./
      SAVE LFIRST
C
      IF (LFIRST) CALL CPUTIM(T1)
C
      DO 10 J = 1,N1,2
          DO 20 I = 1,N2P
              II = I + I
              WORK(II-1) = A(I,J)
              WORK(II) = A(I,J+1)
   20     CONTINUE
          DO 30 I = 1,N2P
              A(I,J) = WORK(I)
              A(I,J+1) = WORK(I+N2P)
   30     CONTINUE
   10 CONTINUE
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TXSHUF = T2 - T1
      END IF
C
      NXSHUF = NXSHUF + 1
C
      END
C***********************************************************************
      SUBROUTINE UNSHUF(A,N2P,N1,WORK)
      DIMENSION A(N2P,1),WORK(1)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      LOGICAL LFIRST
      DATA LFIRST/.TRUE./
      SAVE LFIRST
C
      IF (LFIRST) CALL CPUTIM(T1)
C
      DO 10 J = 1,N1,2
          DO 20 I = 1,N2P
              WORK(I) = A(I,J)
              WORK(I+N2P) = A(I,J+1)
   20     CONTINUE
          DO 30 I = 1,N2P
              II = I + I
              A(I,J) = WORK(II-1)
              A(I,J+1) = WORK(II)
   30     CONTINUE
   10 CONTINUE
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TUSHUF = T2 - T1
      END IF
C
      NUSHUF = NUSHUF + 1
C
      END
C***********************************************************************
      SUBROUTINE MEAN(CA,U,G)
      DIMENSION U(1),G(1)
      COMPLEX CA(1)
      COMMON /NMBRS/NSKIP(4),N2P,NDUMY1(3)

      DO 10 I = 1,N2P
          U(I) = -G(I)*REAL(CA(I))
   10 CONTINUE
      END
C***********************************************************************
C
C     FGPS  8-APR-88 SET UP RPRNT2 AS A MODIFICATION OF RPRINT.  THE IN
C                    TENT IS TO PRINT THE VALUE OF A QUANTITY AT THE CEN
C                    OF THE FIRST VORTEX CELL IN ORDER TO MONITOR EXECUT
C
      SUBROUTINE RPRNT2(R,NAME,N1H,N2PD,IMODE)
C
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMMON /VERIFY/VDAT(15),CDAT(15)
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      REAL R(N2PD,1)
      CHARACTER*8 NAME
C
      LOGICAL LFIRST
      DATA LFIRST/.TRUE./
      SAVE LFIRST
C
      IF (LFIRST) CALL CPUTIM(T1)
C
      WRITE (6,FMT=9000) NSTEPS
C
C
      IX1 = 1 + N1H/4
      IX2 = 1 + N1H/2
      JY1 = 1 + N2PD/4
      WRITE (6,FMT=9010) NAME,JY1,IX1,R(JY1,IX1)
      WRITE (6,FMT=9020) NAME,JY1,IX2,R(JY1,IX2)
      IF (NSTEPS.EQ.51) THEN
          CDAT(7) = R(JY1,IX2)

      ELSE IF (NSTEPS.EQ.101) THEN
          CDAT(8) = R(JY1,IX2)

      ELSE IF (NSTEPS.EQ.151) THEN
          CDAT(9) = R(JY1,IX2)

      ELSE IF (NSTEPS.EQ.201) THEN
          CDAT(10) = R(JY1,IX2)

      ELSE IF (NSTEPS.EQ.NUMBER) THEN
          CDAT(15) = R(JY1,IX2)
      END IF
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TXPRNT = T2 - T1
      END IF
C
      NXPRNT = NXPRNT + 1
C
C
 9000 FORMAT (/,2X,8HNSTEPS =,I7)
 9010 FORMAT (2X,17HTEMPERATURE DROP:,/,2X,
     +       41HAT THE CENTER OF FIRST CONVECTION CELL = ,A8,4H(JY=,I3,
     +       4H,IX=,I2,3H) =,1P,E13.6)
 9020 FORMAT (2X,41HAT CENTER OF CONVECTION-CELL BOUNDARY  = ,A8,4H(JY=,
     +       I3,4H,IX=,I2,3H) =,1P,E13.6,/)

      END
C
C***********************************************************************
C
C     FGPS 25-MAR-88 CHANGED THE DIMENSIONING OF R(1,1) TO R(N2PD,1)
C     FGPS 7-MAR-88 SUBROUTINE RPRINT REPROGRAMMED
C
C     N O T E :  THE INDEXING OF ARRAYS HANDLING VARIABLES REPRESENTED I
C     PHYSICAL SPACE IS USUALLY OF THE FORM
C
C                        A (1<=JY<=N2PD,1<=IX<=N1H)
C
C     WHERE "JY" IS THE COUNT ALONG THE Y-AXIS AND "IX" RUNS ALONG THE X
C     AXIS, AND WHERE
C
C                        N2PD = 2 * (N2+1)                 FOR THE Y-AXI
C     AND
C                         N1H = N1/2                       FOR THE X-AXI
C
C     AND WHERE THE BASIC GRID IS DIMENSIONED 2*(N2+1) BY N1.
C
C
      SUBROUTINE RPRINT(R,NAME,N1H,N2PD,IMODE)
C
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMMON /VERIFY/VDAT(15),CDAT(15)
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      REAL R(N2PD,1)
      CHARACTER*8 NAME
C
C
      WRITE (6,FMT=9000) NSTEPS
C
C     SELECT THE PRINT-OUT MODE
C
      IF (IMODE) 10,20,30
C
   10 CONTINUE
C
      IX1 = 1 + N1H/4
      IX2 = 1 + N1H/2
      JY1 = 1 + N2PD/4
      WRITE (6,FMT=9010) NAME,JY1,IX1,R(JY1,IX1)
C
      WRITE (6,FMT=9020) NAME,JY1,IX2,R(JY1,IX2)
      IF (NSTEPS.EQ.1) THEN
          IF (NAME.EQ.'ZETA PHY') THEN
              CDAT(2) = R(JY1,IX1)

          ELSE IF (NAME.EQ.'UVEL PHY') THEN
              CDAT(3) = R(JY1,IX2)

          ELSE IF (NAME.EQ.'VVEL PHY') THEN
              CDAT(4) = R(JY1,IX2)
          END IF

      ELSE IF (NSTEPS.EQ.NUMBER) THEN
          IF (NAME.EQ.'ZETA PHY') THEN
              CDAT(12) = R(JY1,IX1)

          ELSE IF (NAME.EQ.'UVEL PHY') THEN
              CDAT(13) = R(JY1,IX2)

          ELSE IF (NAME.EQ.'VVEL PHY') THEN
              CDAT(14) = R(JY1,IX2)
          END IF

      END IF
C
C
      NXPRNT = NXPRNT + 1
C
      RETURN
C
   20 CONTINUE
C
C     VALUES OF THE VARIABLE ASSOCIATED WITH NAME ARE PRINTED OUT AT INT
C     VALS AS A FUNCTION OF Y (I.E., 1<=JY<=N2P WITH STRIDE = 4) AT CONS
C     X (I.E., IX = 1+N1H/2)
C
      IX1 = 1 + N1H/2
      N2P = N2PD/2
      WRITE (6,FMT=9030) (JY,NAME,IX1,R(JY,IX1),JY=1,N2P,4)
      WRITE (6,FMT=9040)
      IF (NSTEPS.EQ.1 .AND. NAME.EQ.'TEMP PHY') CDAT(5) = R(65,IX1)
C
      NXPRNT = NXPRNT + 1
C
      RETURN
C
   30 CONTINUE
C
C     VALUES OF THE VARIABLE ASSOCIATED WITH NAME ARE PRINTED OUT AT INT
C     VALS AS A FUNCTION OF X (I.E., 1<=IX<=N1H WITH STRIDE = 2) AT CONS
C     Y (I.E., JY = N2PD/4)
C
      JY1 = 3*N2PD/4
      WRITE (6,FMT=9050) (IX,NAME,JY1,R(JY1,IX),IX=1,N1H,2)
C
      NXPRNT = NXPRNT + 1
C
C
C
 9000 FORMAT (/,2X,8HNSTEPS =,I7)
 9010 FORMAT (2X,A8,4H(JY=,I3,4H,IX=,I3,3H) =,1P,E13.6)
 9020 FORMAT (2X,A8,4H(JY=,I3,4H,IX=,I3,3H) =,1P,E13.6,/)
C
 9030 FORMAT (34 (2X,4HJY =,I5,2X,A8,7H(JY,IX=,I3,3H) =,1P,E13.6,/))
 9040 FORMAT (/,/,/)
C
 9050 FORMAT (/,32 (2X,4HIX =,I5,2X,A8,4H(JY=,I3,6H,IX) =,1P,E13.6,6X,
     +       4HIX =,I5,2X,A8,4H(JY=,I3,6H,IX) =,1P,E13.6,/))

      END
C***********************************************************************
C
C     FGPS 29-MAR-88 MODIFIED RPRINT =====> CPRINT SO AS TO PRINT OUT
C                    COMPLEX ARRAYS
C
C     N O T E :  THE INDEXING OF COMPLEX ARRAYS REPRESENTING PRINCIPAL V
C     IABLES IS OF THE FORM
C
C     COMPLEX  ======>  CA (1<=JY<=N2P,1<=IX<=N1H)
C
C     WHERE "JY" IS THE COUNT ALONG THE Y-AXIS AND "IX" RUNS ALONG THE X
C     AXIS, AND WHERE
C
C                          N2P = (N2+1)                    FOR THE Y-AXI
C     AND
C                           N1H = N1/2                     FOR THE X-AXI
C
C     AND WHERE THE GRID IS NOMINALLY DIMENSIONED (N2+1) BY N1.  MOREOVE
C     THE INDEX PAIR  JY = (N2+1)/2, IX= N1H/2  IS ASSUMED TO IDENTIFY T
C     CENTER OF THE VORTEX CELL.  HOWEVER, IN ACTUAL USE HERE, THE PROGR
C     MING INVOLVES THE REAL EQUIVALENT TO CA; I.E.,
C
C     REAL     ======>  RA (1<=JY<=N2P,1<=IX<=N1)
C
C
      SUBROUTINE CPRINT(RA,NAME,N1H,N2P,IMODE)
C
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /VERIFY/VDAT(15),CDAT(15)
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      REAL RA(N2P,1)
      CHARACTER*8 NAME
C
      WRITE (6,FMT=9000) NSTEPS
C
C     SELECT THE PRINT-OUT MODE
C
c      IF (IMODE) 10,20,30
C
      if(imode.lt.0) then
   10 CONTINUE
C
      IX1 = 1 + N1H/4
      IX2 = 1 + N1H/2
      JY1 = 1 + N2P/2
      WRITE (6,FMT=9010) NAME,JY1,IX1,RA(JY1,IX1)
      WRITE (6,FMT=9020) NAME,JY1,IX2,RA(JY1,IX2)
      IF (NAME.EQ.'PSI PHYS' .AND. NSTEPS.EQ.NUMBER) CDAT(11) = RA(JY1,
     +    IX2)
C
c      RETURN
      elseif(imode.eq.0) then
C
   20 CONTINUE
C
C     VALUES OF THE VARIABLE ASSOCIATED WITH NAME ARE PRINTED OUT AT INT
C     VALS AS A FUNCTION OF Y (I.E., 1<=JY<=N2P WITH STRIDE = 4) AT CONS
C     X (E.G., IX = N1H/2)
C
      IX1 = N1H/2
      WRITE (6,FMT=9030) (JY,NAME,IX1,RA(JY,IX1),JY=1,N2P,4)
C
c      RETURN
      else
C
   30 CONTINUE
C
C     VALUES OF THE VARIABLE ASSOCIATED WITH NAME ARE PRINTED OUT AT INT
C     VALS AS A FUNCTION OF X (I.E., 1<=IX<=N1 WITH STRIDE = 4) AT CONST
C     Y (E.G., JY = N2P/2 + 1)
C
c      N1 = N1H + N1H
      JY1 = N2P/2 + 1
      WRITE (6,FMT=9040) (IX,NAME,JY1,RA(JY1,IX),IX=1,N1H+N1H,4)
      IF (NAME.EQ.'PSI PHYS') THEN
          IF (NSTEPS.EQ.1) THEN
              CDAT(1) = RA(JY1,33)

          ELSE IF (NSTEPS.EQ.2) THEN
              CDAT(6) = RA(JY1,33)
          END IF

      END IF
      endif
C
C
C
 9000 FORMAT (/,2X,8HNSTEPS =,I7)
 9010 FORMAT (2X,A8,4H(JY=,I3,4H,IX=,I3,3H) =,1P,E13.6)
 9020 FORMAT (2X,A8,4H(JY=,I3,4H,IX=,I3,3H) =,1P,E13.6,/)
C
 9030 FORMAT (/,34 (2X,4HJY =,I5,2X,A8,7H(JY,IX=,I3,3H) =,1P,E13.6,6X,
     +       4HJY =,I5,2X,A8,7H(JY,IX=,I3,3H) =,1P,E13.6,/))
C
 9040 FORMAT (/,34 (2X,4HIX =,I5,2X,A8,4H(JY=,I3,6H,IX) =,1P,E13.6,/))

      END
C***********************************************************************
C
C     FGPS  2-APR-88 REWROTE SUBROUTINE SO AS TO RECEIVE THE COMPLEX ARR
C                    CA IN FOURIER SPACE, TRANSFORM THIS ONTO PHYSICAL S
C                    AND CALL CPRINT TO PRINT OUT THE RESULT
C
      SUBROUTINE PSIPHY(CA,N2P,N1H,EX,G,U,WORK,CWORK3,IMODE)
      COMMON /EXSIZE/NMAX2
      COMMON /DATA/ALPHA,BETA,FX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMPLEX CA(N2P,1),CWORK3(1),EX(1)
      REAL U(1),G(1),WORK(1)
C
      N2PD = N2P + N2P
      NPTS = N2PD - 2
      NSKIP = 1
      MTRN = N1H
      MSKIP = N2P
      ISIGN = 1
      CALL XLOG2(NPTS,LOG)
      IEXSHF = NMAX2/NPTS
      CALL ACAC(CA,EX,WORK)
C
      N1 = N1H + N1H
      NPTS = N1
      NSKIP = N2P
      MTRN = N2P
      MSKIP = 1
      ISIGN = 1
      CALL XLOG2(NPTS,LOG)
      IEXSHF = NMAX2/NPTS
      CALL CSR(CA,EX)
C
      CALL TELTIM
      CALL UNSHUF(CA,N2P,N1,WORK)
      CALL CPRINT(CA,'PSI PHYS',N1H,N2P,IMODE)
C      CALL PLOTME (CA,WORK,NCALL)
C
      END
C***********************************************************************
C
C     FGPS 18-FEB-88 RELOCATED "CALL RPRINT ( , ,N1H,N2PD,IMODE)"
C
      SUBROUTINE ZETAPH(A,N2PD,N1H,EX,WORK,IMODE)
      DIMENSION A(N2PD,1),EX(1),WORK(1)
      SAVE NCALL
      DATA NCALL/0/
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2

      CALL CSR(A,EX)
      N2P = N2PD/2
      N1 = N1H + N1H
      CALL UNSHUF(A,N2P,N1,WORK)
      CALL TELTIM
      DO 10 J = 1,N1H
          WORK(J) = 0.0
   10 CONTINUE
      DO 20 I = 1,N2PD
          DO 30 J = 1,N1H
              WORK(J) = WORK(J) + A(I,J)*A(I,J)
   30     CONTINUE
   20 CONTINUE
      SUMSQ = 0.0
      DO 40 J = 1,N1H
          SUMSQ = SUMSQ + WORK(J)
   40 CONTINUE
      ZRMS = SQRT(SUMSQ/ (REAL(N1)*REAL(N2P)))
      WRITE (6,FMT=9000) ZRMS
      NCALL = NCALL + 1
      CALL RPRINT(A,'ZETA PHY',N1H,N2PD,IMODE)
C      CALL PLOTME (A,WORK,NCALL)
 9000 FORMAT (/,' SQRT (AVERAGE ZETASQUARED) =',F30.15)

      END
C***********************************************************************
C
C
C     FGPS  9-APR-88 MADE TO CALL RPRNT2 (NOT RPRINT) WHENEVER IMODE =
C     FGPS 18-FEB-88 RELOCATED "CALL RPRINT ( , ,N1H,N2PD,IMODE)"
C     FGPS 10-FEB-88 ADDED "COMPLEX  EX(1)"
C
      SUBROUTINE TEMPHY(A,N2PD,N1H,EX,WORK,IMODE)
      COMMON /LADDR/LDUMMY,LTN,LDUMY1(10)
      DIMENSION A(N2PD,1),WORK(1)
      COMPLEX EX(1)
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /MAPING/GMAPP(129),GDUMY1(129)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
      SAVE NCALL
      DATA NCALL/0/

      NW = N2PD*N1H
      N2P = N2PD/2
      N1 = N1H + N1H
      CALL IN(A,LTN,NW)
      CALL CSR(A,EX)
      DO 10 J = 1,N1H
          WORK(J) = 0.0
   10 CONTINUE
      DO 20 I = 1,N2PD
          IPH = (I+1)/2
          ADJUST = BETA*GMAPP(IPH)
C WHEN INTERESTED IN TEMPERATURE DISTURBANCE ONLY, USE THIS INSTEAD:
C        ADJUST = 0.0
          DO 30 J = 1,N1H
              WORK(J) = WORK(J) + A(I,J)*A(I,J)
              A(I,J) = 0.5*A(I,J) - ADJUST
   30     CONTINUE
   20 CONTINUE
      SUMSQ = 0.0
      DO 40 J = 1,N1H
          SUMSQ = SUMSQ + WORK(J)
   40 CONTINUE
      ZRMS = 0.5*SQRT(SUMSQ/ (REAL(N1)*REAL(N2P)))
      WRITE (6,FMT=9000) ZRMS
      NCALL = NCALL + 1
      CALL UNSHUF(A,N2P,N1,WORK)
      CALL TELTIM
      IF (IMODE.LT.0) THEN
          CALL RPRNT2(A,'TEMP PHY',N1H,N2PD,IMODE)

      ELSE
          CALL RPRINT(A,'TEMP PHY',N1H,N2PD,IMODE)
      END IF
C      CALL PLOTME (A,WORK,NCALL)
 9000 FORMAT (/,' SQRT (AVERAGE TEMPERATURESQUARED) =',F30.15)

      END
C***********************************************************************
C
C     FGPS 18-FEB-88 RELOCATED "CALL RPRINT ( , ,N1H,N2PD,IMODE)"
C
      SUBROUTINE UVELPH(A,N2PD,N1H,WORK,IMODE)
      COMMON /LADDR/LDUMMY(5),LPY,LDUMY1(6)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      DIMENSION A(N2PD,1),WORK(1)
      COMMON /DATA/DUMY1(2),BOXX,DUMY2(6),IDUMY1

      DO 10 I = 1,N2PD
          WORK(I) = 0.0
   10 CONTINUE
      NW = N2PD*N1H
      N2P = N2PD/2
      N1 = N1H + N1H
      CALL TELTIM
      CALL IN(A,LPY,NW)
      DO 20 J = 1,N1H
          DO 30 I = 1,N2PD
              WORK(I) = MAX(WORK(I),ABS(A(I,J)))
   30     CONTINUE
   20 CONTINUE
      WRKMAX = 0.0
      DO 40 I = 1,N2PD
          WRKMAX = MAX(WRKMAX,WORK(I))
   40 CONTINUE
      CALL UNSHUF(A,N2P,N1,WORK)
      CALL RPRINT(A,'UVEL PHY',N1H,N2PD,IMODE)
C
C AT THIS POINT A(N2P,N1) CONTAINS 2.0*U
C
      UMAX = 0.5*WRKMAX + 1.0E-32
      DTMAX = 0.2* (BOXX/REAL(N1))/UMAX
      WRITE (6,FMT=9000) DTMAX

 9000 FORMAT (/,/,' DTMAX =',F16.10)

      END
C***********************************************************************
C
C     FGPS 18-FEB-88 RELOCATED "CALL RPRINT ( , ,N1H,N2PD,IMODE)"
C
      SUBROUTINE VVELPH(A,N2PD,N1H,WORK,IMODE)
      COMMON /LADDR/LDUMMY(2),LPN,LDUMY1(9)
      DIMENSION A(N2PD,1),WORK(1)
      COMMON /MAPING/GMAPP(129),G(129)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2

      NW = N2PD*N1H
      N2P = N2PD/2
      N1 = N1H + N1H
      CALL TELTIM
      CALL IN(A,LPN,NW)
      CALL UNSHUF(A,N2P,N1,WORK)
      CALL RPRINT(A,'VVEL PHY',N1H,N2PD,IMODE)
C
C AT THIS POINT A(N2P,N1) CONTAINS 2.0*V
C
      CALL TESTDT(A,N2P,N1,WORK,G)
      END
C***********************************************************************
      SUBROUTINE TESTDT(A,N2P,N1,WORK,G)
      COMMON /PI/PI
      DIMENSION A(N2P,N1),WORK(N2P),G(N2P)
C
C FIND MAXIMUM DT:
C
      DO 10 I = 1,N2P
          WORK(I) = 0.0
   10 CONTINUE
      DO 20 J = 1,N1
          DO 30 I = 1,N2P
              WORK(I) = MAX(WORK(I),ABS(G(I)*A(I,J)))
   30     CONTINUE
   20 CONTINUE
C
C G WAS 1.0/(DY/DX), AND X GOES FROM 0.0 TO PI, HENCE IF WE WANT
C ((DY/DI)/V)=((DY/DX)(DX/DI)/V)=PI/(N1*G(I)*V(I))
C REMEMBER, A CONTAINS 2.0*V:
      DO 40 I = 2,N2P
          WORK(1) = MAX(WORK(1),WORK(I))
   40 CONTINUE
C
C SO, WORK(1) = MAX (G(I)*2*V(I,J)):
C DT(MAX) = 0.2* MIN (DY/V) = 0.2*2.0*PI/(N1*WORK(1)):
CSRD  DTMAX = 0.4*PI/(REAL(N1)*WORK(1)+1.0E-1000)
      DTMAX = 0.4*PI/ (REAL(N1)*WORK(1)+1.0E-300)
      WRITE (6,FMT=9000) DTMAX

 9000 FORMAT (/,/,' DTMAX =',F16.10)

      END
C***********************************************************************
      SUBROUTINE PSIFOU(CA,N2P,N1H,CWORK)
      COMMON /LADDR/LDUMMY(2),LPN,LDUMY1(9)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMPLEX CA(N2P,1),CWORK(1)

      NW = N2P*N1H*2
      CALL IN(CA,LPN,NW)
      CALL CPRINT(CA,'PSI FOU',N1H,N2P,-1)
      END
C***********************************************************************
      SUBROUTINE ZETAFO(CA,N2P,N1H,CWORK)
      COMMON /LADDR/LZN,LDUMY1(11)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMPLEX CA(N2P,1),CWORK(1)

      NW = N2P*N1H*2
      CALL IN(CA,LZN,NW)
      CALL CPRINT(CA,'ZETA FOU',N1H,N2P,-1)
      END
C***********************************************************************
      SUBROUTINE TEMPFO(CA,N2P,N1H,CWORK)
      COMMON /LADDR/LDUMMY,LTN,LDUMY1(10)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMPLEX CA(N2P,1),CWORK(1)

      NW = N2P*N1H*2
      CALL IN(CA,LTN,NW)
      CALL CPRINT(CA,'TEMP FOU',N1H,N2P,-1)
      END
C***********************************************************************
      SUBROUTINE UVELFO(CA,N2P,N1H,G,EX,CWORK)
c      COMMON /FT/NDUMMY(4),ISIGN,NDUMY2(2)
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
      COMPLEX CA(N2P,1),CWORK(1),EX(1)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      DIMENSION G(1)

      DO 10 I = 1,N2P
          DO 20 J = 1,N1H
              CA(I,J) = G(I)*CA(I,J)
   20     CONTINUE
   10 CONTINUE
      ISIGN = -1
      CALL SCSC(CA,EX,CWORK)
      ISIGN = 1
      F = 0.125/REAL(N2P-1)
      DO 30 J = 1,N1H
          DO 40 I = 1,N2P
              CA(I,J) = F*CA(I,J)
   40     CONTINUE
   30 CONTINUE
      CALL CPRINT(CA,'UVEL FOU',N1H,N2P,-1)
      END
C***********************************************************************
      SUBROUTINE VVELFO(CA,N2P,N1H,FX,CWORK)
      COMMON /LADDR/LDUMMY(2),LPN,LDUMY1(9)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMPLEX CA(N2P,1),CWORK(1)

      NW = N2P*N1H*2
      CALL IN(CA,LPN,NW)
      F = 0.
      DO 10 J = 1,N1H
          DO 20 I = 1,N2P
              CA(I,J) = CMPLX(-F*AIMAG(CA(I,J)),F*REAL(CA(I,J)))
   20     CONTINUE
          F = F + FX
   10 CONTINUE
      CALL CPRINT(CA,'VVEL FOU',N1H,N2P,-1)
      END
C***********************************************************************
C***********************************************************************
      SUBROUTINE BUG(POINT)
      CHARACTER*10 POINT
      COMMON /DEBUG/NUM1,TLAST

      CALL CPUTIM(TNOW)
      TSINCE = TNOW - TLAST
      WRITE (6,FMT=9000) NUM1,TNOW,TSINCE,POINT
      NUM1 = NUM1 + 1
      TLAST = TNOW

 9000 FORMAT (' NUM1 =',I3,'; TNOW =',F9.3,' S; TSINCE =',F9.3,
     +       ' S; LOCATION IS: ',A10)

      END
C***********************************************************************
      SUBROUTINE TELTIM
      COMMON /DATA/DUMMY(8),TIME,IDUMY1

      WRITE (6,FMT=9000) TIME

 9000 FORMAT (' TIME = ',F20.6)

      END
C***********************************************************************
C
C     FGPS 12-FEB-1988 SUBR. PLOTME TEMPORARILY COMMENTED "C" OUT
C
C      SUBROUTINE PLOTME (Z,WORK,NCALL)
C      COMMON /DATA/ ALPHA,BETA,BOXX,H, RDUMY1(5), IDUMY1
C      COMMON /OPTONS/ NX,NY
C      DIMENSION Z(1), WORK(1)
C      COMMON /PLTDEF/ IDPAT, DXA, DXB, DYA, DYB
CC
CC THIS SUBROUTINE CAN BE USED FOR PLOTTING OF FIELDS IN PHYSICAL
CC SPACE. CONTOUR PLOTS ARE USUALLY BEST.
CC
C      ASPECT = BOXX/H
C      A = (DXB-DXA) / (DYB-DYA)
C      XMIN = DXA
C      XMAX = DXB
C      YMIN = DYA
C      YMAX = DYB
C      IF (A.LT.ASPECT) GO TO 10
C      SPANH = 0.5*(DYB-DYA) * ASPECT
C      RMEAN = 0.5*(DXA+DXB)
C      XMIN = RMEAN-SPANH
C      XMAX = RMEAN+SPANH
C      GO TO 30
C   10 SPANH = 0.5*(DXB-DXA) / ASPECT
C      RMEAN = 0.5*(DYA+DYB)
C      YMIN = RMEAN-SPANH
C      YMAX = RMEAN+SPANH
C   30 CALL SET (XMIN,XMAX,YMIN,YMAX,1.0,REAL(NX+1),1.0,REAL(NY+1),1)
C      CALL EDGES (-1)
C      CALL DRAWBOX (1.0,1.0,REAL(NX),REAL(NY+1))
C      CALL CONREC (Z,NY+1,NY+1,NX,0.0,0.0,0.0,1,0,0)
C      CALL FRAME
C      RETURN
C      END
C***********************************************************************
C      SUBROUTINE DATE (TODAY)
C      TODAY = DATEF(1.0)
C      RETURN
C      END
C***********************************************************************
C      SUBROUTINE TIME (NOW)
C      NOW = 8HNOW
C      RETURN
C      END
C***********************************************************************
C                           E - O - D
C***********************************************************************
C
C FFTFORTRANPACK3
C FFTFTN.5
C 10/16/80
C
      SUBROUTINE EXSET(EX,NPTS)
      COMPLEX EX(1)
      COMPLEX A1,A2

      W = 6.283185307179586477/REAL(NPTS)
      NH = NPTS/2
CVD$ NOVECTOR
CVD$ NOCONCUR
      DO 10 J = 1,NH
C           EX( J ) = EXP( CMPLX( 0., REAL(J - 1)*W ) )
C10         EX( J + NH ) = EXP( CMPLX( 0., REAL(1 - J)*W ) )
          A1 = CMPLX(0.,REAL(J-1)*W)
          EX(J) = EXP(A1)
          EX(J+NH) = EXP(-1.*A1)
   10 CONTINUE
      END
C***********************************************************************
C
C     FGPS 29-APR-88 INTRODUCED "SUBROUTINE XLOG2 (NA,NB)" IN ORDER TO I
C                    PROVE EFFICIENCY.  THIS REPLACES "FUNCTION LOG2 (N)
C
C      FUNCTION LOG2 (N)
C      LOG2 = LOG(REAL(N))/LOG(2.)+0.01
C      RETURN
C      END
C
      SUBROUTINE XLOG2(NA,NB)
C
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
C
      LOGICAL INITL1,INITL2
      DATA INITL1,INITL2/.TRUE.,.TRUE./
      REAL T1,T2,F1,B
      SAVE INITL1,INITL2,F1
C
      IF (INITL1) CALL CPUTIM(T1)
C
      IF (INITL2) THEN
          INITL2 = .FALSE.
          F1 = 1.0/LOG(2.0)
      END IF
C
      B = F1*LOG(REAL(NA)) + 0.01
      NB = INT(B)
C
      IF (INITL1) THEN
          INITL1 = .FALSE.
          CALL CPUTIM(T2)
          TXLOG2 = T2 - T1
      END IF
C
      NXLOG2 = NXLOG2 + 1
C
      END
C***********************************************************************
      FUNCTION LCMREQ(LARGE)
      PARAMETER (LCMSIZ=165117)
      COMMON /MEMORY/MEMSIZ,RM(LCMSIZ)

      IF (LARGE.GT.LCMSIZ) THEN
C
          WRITE (6,FMT=9010) MEMSIZ,LARGE
          STOP

      ELSE
          MEMSIZ = LARGE
          LCMREQ = 1
          WRITE (6,FMT=9000) LARGE
      END IF

 9000 FORMAT (/,/,' LCM REQUESTED:',I7,' WORDS',/,/)
 9010 FORMAT (/,/,/,' LCMREQ: REQUEST EXCEEDS MEMORY SIZE.',/,
     +       ' MEMORY SIZE:',I7,' ;   WORDS REQUESTED:',I22)

      END
C***********************************************************************
C
      SUBROUTINE IN(ARRAY,LOC,NW)
      COMMON /MEMORY/MEMSIZ,RM(165117)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      DIMENSION ARRAY(1)
      LOGICAL LFIRST
      DATA LFIRST/.TRUE./
      SAVE LFIRST
C
      IF (LFIRST) CALL CPUTIM(T1)
C
      IF ((LOC.LT.1) .OR. ((LOC+NW-1).GT.MEMSIZ) .OR. (NW.LT.1)) THEN
C
          WRITE (6,FMT=9000) LOC,NW
          STOP

      ELSE
          LOCM = LOC - 1
          DO 10 I = 1,NW
              ARRAY(I) = RM(LOCM+I)
   10     CONTINUE
C
          IF (LFIRST) THEN
              LFIRST = .FALSE.
              CALL CPUTIM(T2)
              TXXXIN = T2 - T1
          END IF
C
          NXXXIN = NXXXIN + 1
C
      END IF

 9000 FORMAT (/,/,/,' IN: ILLEGAL REQUEST. LOCATION:',I22,
     +       ' ;   NUMBER OF WORDS:',I22,' ;  MEMSIZ:',I7)

      END
C***********************************************************************
C
      SUBROUTINE OUT(ARRAY,LOC,NW)
      COMMON /MEMORY/MEMSIZ,RM(165117)
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      DIMENSION ARRAY(1)
      LOGICAL LFIRST
      DATA LFIRST/.TRUE./
      SAVE LFIRST
C
      IF (LFIRST) CALL CPUTIM(T1)
C
      IF ((LOC.LT.1) .OR. ((LOC+NW-1).GT.MEMSIZ) .OR. (NW.LT.1)) THEN
C
          WRITE (6,FMT=9000) LOC,NW,MEMSIZ
          STOP

      ELSE
          LOCM = LOC - 1
          DO 10 I = 1,NW
              RM(LOCM+I) = ARRAY(I)
   10     CONTINUE
C
          IF (LFIRST) THEN
              LFIRST = .FALSE.
              CALL CPUTIM(T2)
              TXXOUT = T2 - T1
          END IF
C
          NXXOUT = NXXOUT + 1
C
      END IF

 9000 FORMAT (/,/,/,' OUT: ILLEGAL REQUEST. LOCATION:',I22,
     +       ' ;   NUMBER OF WORDS:',I22,' ;  MEMSIZ:',I7)

      END
C
      SUBROUTINE RCS(A,EX)
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMMON /NMBRS/N1H,N1Q,N1E,N2M,N2P,N2DP,N2DM,N2PD
      COMPLEX A(1),EX(1),BJ,BJSTAR
C
C   PERFORM A REAL TO CONJUGATE SYMMETRIC TRANFORM, SAVING ONLY HALF OF
C   THE RESULTING CONJUGATE SYMMETRIC ARRAY, AND PACKING A(0) AND A(N/2)
C   INTO A(0) SINCE THEY ARE BOTH REAL
C
      LOGICAL LFIRST,LXSTEP,LXTIME
      DATA LFIRST,LXSTEP,LXTIME/.TRUE.,.FALSE.,.TRUE./
      SAVE LFIRST,LXSTEP,LXTIME
C
      IF (NSTEPS.EQ.NUMBER) LXSTEP = .TRUE.
      IF (LFIRST) CALL CPUTIM(T1)
      IF (LXSTEP) THEN
          IF (LXTIME) CALL CPUTIM(T1)
      END IF
C
      N = NPTS
      NPTS = NPTS/2
      M = NPTS
      ISHIFT = 0
      IF (ISIGN.LT.0) ISHIFT = M*IEXSHF
      MH = M/2
      MHP = MH + 1
      LOG = LOG - 1
      IEXSHF = 2*IEXSHF
      CALL FTRVMT(A(1),EX(1))
      LOG = LOG + 1
      IEXSHF = IEXSHF/2
      NPTS = N
      DO 10 LOOP = 1,MTRN
          IST = (LOOP-1)*MSKIP
          ISTP = IST + 1
          A(ISTP) = CMPLX(2.* (REAL(A(ISTP))+AIMAG(A(ISTP))),
     +              2.* (REAL(A(ISTP))-AIMAG(A(ISTP))))
          DO 20 I = 2,MH
              J = (I-1)*NSKIP + ISTP
              JSTAR = (M-I+1)*NSKIP + ISTP
              BJ = A(J)
              BJSTAR = A(JSTAR)
              IEXL = IEXSHF* (I-1) + 1 + ISHIFT
              IEXH = IEXSHF* (M-I+1) + 1 + ISHIFT
              A(J) = BJ + CONJG(BJSTAR) +
     +               CMPLX(0.,-1.)*EX(IEXL)* (BJ-CONJG(BJSTAR))
              A(JSTAR) = BJSTAR + CONJG(BJ) +
     +                   CMPLX(0.,-1.)*EX(IEXH)* (BJSTAR-CONJG(BJ))
   20     CONTINUE
          J = MH*NSKIP + ISTP
          A(J) = 2.*A(J)
          IF (ISIGN.LT.0) A(J) = CONJG(A(J))
   10 CONTINUE
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TXXRCS = T2 - T1
      END IF
C
      IF (LXSTEP) THEN
          IF (LXTIME) THEN
              LXTIME = .FALSE.
              CALL CPUTIM(T2)
              EXXRCS = T2 - T1
          END IF

      END IF
C
      NXXRCS = NXXRCS + 1
C
      END
C
      SUBROUTINE CSR(A,EX)
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMMON /NMBRS/N1H,N1Q,N1E,N2M,N2P,N2DP,N2DM,N2PD
      COMPLEX A(1),EX(1),BJ,BJSTAR
C
C   PERFORM A CONJUGATE SYMMETRIC TO REAL TRANSFORM ON A PACKED
C   CONJUGATE SYMMETRIC ARRAY.  ONLY HALF THE ARRAY IS STORED, AND A(0)
C   AND A(N/2) ARE BOTH PACKED TOGETHER IN A(0) SINCE THEY ARE BOTH REAL
C
      LOGICAL LFIRST,LXSTEP,LXTIME
      DATA LFIRST,LXSTEP,LXTIME/.TRUE.,.FALSE.,.TRUE./
      SAVE LFIRST,LXSTEP,LXTIME
C
      IF (NSTEPS.EQ.NUMBER) LXSTEP = .TRUE.
      IF (LFIRST) CALL CPUTIM(T1)
      IF (LXSTEP) THEN
          IF (LXTIME) CALL CPUTIM(T1)
      END IF
C
      N = NPTS
      NPTS = NPTS/2
      M = NPTS
      ISHIFT = 0
      IF (ISIGN.LT.0) ISHIFT = M*IEXSHF
      MH = M/2
      MHP = MH + 1
      DO 10 LOOP = 1,MTRN
          IST = (LOOP-1)*MSKIP
          ISTP = IST + 1
          A(ISTP) = CMPLX(REAL(A(ISTP))+AIMAG(A(ISTP)),
     +              REAL(A(ISTP))-AIMAG(A(ISTP)))
          DO 20 I = 2,MH
              J = (I-1)*NSKIP + ISTP
              JSTAR = (M-I+1)*NSKIP + ISTP
              BJ = A(J)
              BJSTAR = A(JSTAR)
              IEXL = IEXSHF* (I-1) + 1 + ISHIFT
              IEXH = IEXSHF* (M-I+1) + 1 + ISHIFT
              A(J) = BJ + CONJG(BJSTAR) +
     +               CMPLX(0.,1.)*EX(IEXL)* (BJ-CONJG(BJSTAR))
              A(JSTAR) = BJSTAR + CONJG(BJ) +
     +                   CMPLX(0.,1.)*EX(IEXH)* (BJSTAR-CONJG(BJ))
   20     CONTINUE
          J = MH*NSKIP + ISTP
          A(J) = 2.*A(J)
          IF (ISIGN.GT.0) A(J) = CONJG(A(J))
   10 CONTINUE
      LOG = LOG - 1
      IEXSHF = 2*IEXSHF
      CALL FTRVMT(A(1),EX(1))
      LOG = LOG + 1
      IEXSHF = IEXSHF/2
      NPTS = N
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TXXCSR = T2 - T1
      END IF
C
      IF (LXSTEP) THEN
          IF (LXTIME) THEN
              LXTIME = .FALSE.
              CALL CPUTIM(T2)
              EXXCSR = T2 - T1
          END IF

      END IF
C
      NXXCSR = NXXCSR + 1
C
      END
C
      SUBROUTINE SCSC(AC,EX,WORK)
C
C COMPLEX SYMMETRIC TO COMPLEX SYMMETRIC
C
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMPLEX AC(1),WORK(1),TEMP1,TEMP2,X,Y,BJ,BJSYM
      COMMON /NMBRS/N1H,N1Q,N1E,N2M,N2P,N2DP,N2DM,N2PD
      DIMENSION EX(1)
C
      LOGICAL LFIRST,LXSTEP,LXTIME
      DATA LFIRST,LXSTEP,LXTIME/.TRUE.,.FALSE.,.TRUE./
      SAVE LFIRST,LXSTEP,LXTIME
C
      IF (NSTEPS.EQ.NUMBER) LXSTEP = .TRUE.
      IF (LFIRST) CALL CPUTIM(T1)
      IF (LXSTEP) THEN
          IF (LXTIME) CALL CPUTIM(T1)
      END IF
C
      nptsinit = npts
      M = NPTS/2
      NPTS = M
      MH = M/2
      MHP = MH + 1
      ISTOP = 1 + (MTRN-1)*MSKIP
      MNSKIP = M*NSKIP
      LOG = LOG - 1
      NSKIP2 = NSKIP*2
      MP2 = M + 2
      IEXSHF = IEXSHF*2
      FACT = REAL(ISIGN)/2.
      MNSH = MNSKIP/2
      IPROD = MNSH - NSKIP
      IEXSYM = 4 + M*IEXSHF
      DO 10 LOOP = 1,ISTOP,MSKIP
          ISUM = LOOP + MNSKIP
          WORK(1) = AC(LOOP)
          WORK(MHP) = AC(ISUM)
          IAC = LOOP + NSKIP
          Y = AC(IAC)
          DO 20 I = 2,MH
              IAC = IAC + NSKIP2
              Y = Y + AC(IAC)
              TEMP1 = CMPLX(0.,1.)* (AC(IAC)-AC(IAC-NSKIP2))
              TEMP2 = AC(IAC-NSKIP)
              WORK(I) = TEMP2 + TEMP1
              WORK(MP2-I) = TEMP2 - TEMP1
   20     CONTINUE
          IAC = LOOP
          DO 30 I = 1,M
              AC(IAC) = WORK(I)
              IAC = IAC + NSKIP
   30     CONTINUE
          AC(ISUM) = 4.*Y
   10 CONTINUE
      CALL FTRVMT(AC,EX)
      DO 40 LOOP = 1,ISTOP,MSKIP
          ISUM = LOOP + MNSKIP
          JSUM = ISUM + LOOP
          X = 2.*AC(LOOP)
          Y = AC(ISUM)
          AC(LOOP) = X + Y
          AC(ISUM) = X - Y
          AC(LOOP+MNSH) = 2.*AC(LOOP+MNSH)
          ISTART = LOOP + NSKIP
          IFIN = LOOP + IPROD
          I = 2
          DO 50 J = ISTART,IFIN,NSKIP
              I = I + IEXSHF
              JSYM = JSUM - J
              BJ = AC(J)
              BJSYM = AC(JSYM)
              TEMP1 = BJ + BJSYM
              TEMP2 = CMPLX(FACT/EX(I),0.)* (BJ-BJSYM)
              AC(J) = TEMP1 + TEMP2
              AC(JSYM) = TEMP1 - TEMP2
   50     CONTINUE
   40 CONTINUE
c      NPTS = M*2
      npts = nptsinit
      LOG = LOG + 1
      IEXSHF = IEXSHF/2
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TXSCSC = T2 - T1
      END IF
C
      IF (LXSTEP) THEN
          IF (LXTIME) THEN
              LXTIME = .FALSE.
              CALL CPUTIM(T2)
              EXSCSC = T2 - T1
          END IF

      END IF
C
      NXSCSC = NXSCSC + 1
C
      END
C
      SUBROUTINE ACAC(AC,EX,WORK)
C
C COMPLEX ANTISYMMETRIC TO COMPLEX ANTISYMMETRIC
C
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
      COMMON /DATA/ALPHA,BETA,BOXX,H,RNU,RKAPA,DT,DTT,T,NSTEPS
      COMMON /OPTONS/N1,N2,NUMBER,KDATA,LMAPP,NOTEMP,KTA,KTAPE,KSM,KPP,
     +       KZP,KTP,KUP,KVP,KPF,KZF,KTF,KUF,KVF,KPRESS,NOALS
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
      COMMON /NMBRS/N1H,N1Q,N1E,N2M,N2P,N2DP,N2DM,N2PD
      COMPLEX AC(1),WORK(1),TEMP1,TEMP2,X,Y,BJ,BJSYM
      DIMENSION EX(1)
C
      LOGICAL LFIRST,LXSTEP,LXTIME
      DATA LFIRST,LXSTEP,LXTIME/.TRUE.,.FALSE.,.TRUE./
C
      IF (NSTEPS.EQ.NUMBER) LXSTEP = .TRUE.
      IF (LFIRST) CALL CPUTIM(T1)
      IF (LXSTEP) THEN
          IF (LXTIME) CALL CPUTIM(T1)
      END IF
C
      nptsinit = npts
      M = NPTS/2
      NPTS = M
      MH = M/2
      MHP = MH + 1
      ISTOP = 1 + (MTRN-1)*MSKIP
      MNSKIP = M*NSKIP
      LOG = LOG - 1
      NSKIP2 = NSKIP*2
      MP2 = M + 2
      IEXSHF = IEXSHF*2
      FACT = REAL(ISIGN)/2.
      MNSH = MNSKIP/2
      IPROD = MNSH
      DO 10 LOOP = 1,ISTOP,MSKIP
          ISUM = LOOP + MNSKIP
          IAC = LOOP + NSKIP
          WORK(1) = CMPLX(0.,2.)*AC(IAC)
          DO 20 I = 2,MH
              IAC = IAC + NSKIP2
              TEMP1 = CMPLX(0.,1.)* (AC(IAC)-AC(IAC-NSKIP2))
              TEMP2 = AC(IAC-NSKIP)
              WORK(I) = TEMP2 + TEMP1
              WORK(MP2-I) = TEMP1 - TEMP2
   20     CONTINUE
          WORK(MHP) = CMPLX(0.,-2.)*AC(IAC)
          IAC = LOOP
          DO 30 I = 1,M
              AC(IAC) = WORK(I)
              IAC = IAC + NSKIP
   30     CONTINUE
   10 CONTINUE
      CALL FTRVMT(AC,EX)
      DO 40 LOOP = 1,ISTOP,MSKIP
          ISUM = LOOP + MNSKIP
          JSUM = ISUM + LOOP
          AC(LOOP) = 0.
          ISTART = LOOP + NSKIP
          IFIN = LOOP + IPROD
          I = 2
          DO 50 J = ISTART,IFIN,NSKIP
              I = I + IEXSHF
              JSYM = JSUM - J
              BJ = AC(J)
              BJSYM = AC(JSYM)
              TEMP1 = BJ - BJSYM
              TEMP2 = CMPLX(FACT/EX(I),0.)* (BJ+BJSYM)
              AC(J) = TEMP1 + TEMP2
              AC(JSYM) = TEMP2 - TEMP1
   50     CONTINUE
   40 CONTINUE
c      NPTS = M*2
      npts = nptsinit
      LOG = LOG + 1
      IEXSHF = IEXSHF/2
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TXACAC = T2 - T1
      END IF
C
      IF (LXSTEP) THEN
          IF (LXTIME) THEN
              LXTIME = .FALSE.
              CALL CPUTIM(T2)
              EXACAC = T2 - T1
          END IF

      END IF
C
      NXACAC = NXACAC + 1
C
      END
C
      SUBROUTINE FTRVMT(DATA,EX)
C
C THIS IS A FORTRAN VERSION OF THE ASSEMBLY LANGUAGE SUBROUTINE
C FFT2, TO BE USED WITH THE NSSL SUBROUTINE PACKAGE FFTPOW2.
C
      COMMON /FT/NPTS,NSKIP,MTRN,MSKIP,ISIGN,LOG,IEXSHF
C
C DATA IS A COMPLEX ARRAY OF LENGTH NPTS WHICH SERVES FOR BOTH
C INPUT AND OUTPUT.
C
      COMMON /DATA/ALPHA,BETA,BOXX,H1,RNU,RKAPA,DT,DTT,T,NSTEP
      COMMON /TIMING/TXXXIN,TXXOUT,TXXRCS,TXXCSR,TXSCSC,TXACAC,TFTVMT,
     +       TXSHUF,TUSHUF,TXPRNT,TXLOG2,EXXRCS,EXXCSR,EXSCSC,EXACAC,
     +       EFTVMT,NXXXIN,NXXOUT,NXXRCS,NXXCSR,NXSCSC,NXACAC,NFTVMT,
     +       NXSHUF,NUSHUF,NXPRNT,NXLOG2
C
      COMPLEX DATA(99),EXP1,EXK,EXJ,H,EX(1)
      REAL HH(2)
      EQUIVALENCE (HH,H)
      LOGICAL LFIRST,LXSTEP,LXTIME
      DATA LFIRST,LXSTEP,LXTIME/.TRUE.,.FALSE.,.TRUE./
      SAVE LFIRST,LXSTEP,LXTIME
C
      IF (NSTEPS.EQ.NUMBER) LXSTEP = .TRUE.
      IF (LFIRST) CALL CPUTIM(T1)
      IF (LXSTEP) THEN
          IF (LXTIME) CALL CPUTIM(T1)
      END IF
C
      I2K = NPTS
      IF (I2K.NE.1) THEN
          SGN1 = ISIGN
          EXK = EX(1+IEXSHF)
          IF (SGN1.LT.0.) EXK = CONJG(EXK)
   10     CONTINUE
          I2KP = I2K
          I2K = I2K/2
          I2KS = I2K*NSKIP
          IF (2*I2K.NE.I2KP) THEN
              GO TO 20

          ELSE
              JLI = I2K/2 + 1
              DO 30 JL = 1,I2K
                  IF (JL-1.LE.0) THEN
                      EXJ = CMPLX(1.,0.)
                      DO 40 JJ = JL,NPTS,I2KP
                          DO 50 MM = 1,MTRN
                              JS = (JJ-1)*NSKIP + (MM-1)*MSKIP + 1
                              H = DATA(JS) - DATA(JS+I2KS)
                              DATA(JS) = DATA(JS) + DATA(JS+I2KS)
                              DATA(JS+I2KS) = H
   50                     CONTINUE
   40                 CONTINUE

                  ELSE IF (JL-JLI.EQ.0) THEN
                      EXJ = CMPLX(0.,SGN1)
                      DO 60 JJ = JL,NPTS,I2KP
                          DO 70 MM = 1,MTRN
                              JS = (JJ-1)*NSKIP + (MM-1)*MSKIP + 1
                              H = DATA(JS) - DATA(JS+I2KS)
                              DATA(JS) = DATA(JS) + DATA(JS+I2KS)
                              DATA(JS+I2KS) = CMPLX(-SGN1*HH(2),
     +                                        SGN1*HH(1))
   70                     CONTINUE
   60                 CONTINUE

                  ELSE
C
C INCREMENT JL-DEPENDENT EXPONENTIAL FACTOR
C
                      EXJ = EXJ*EXK
                      DO 80 JJ = JL,NPTS,I2KP
                          DO 90 MM = 1,MTRN
                              JS = (JJ-1)*NSKIP + (MM-1)*MSKIP + 1
                              H = DATA(JS) - DATA(JS+I2KS)
                              DATA(JS) = DATA(JS) + DATA(JS+I2KS)
                              DATA(JS+I2KS) = H*EXJ
   90                     CONTINUE
   80                 CONTINUE
                  END IF

   30         CONTINUE
C
C INCREMENT K-DEPENDENT EXPONENTIAL FACTOR
C
              EXK = EXK*EXK
              IF (I2K-1.GT.0) GO TO 10
          END IF

          IF (NPTS.GT.2) THEN
              NPTSD2 = NPTS/2
              JMIN = 0
              JMAX = NPTS - 4
              JREV = 0
              DO 100 J = JMIN,JMAX,2
                  I2K = NPTSD2
                  JREV2 = JREV + I2K
                  IF (JREV2- (J+1).GT.0) THEN
                      DO 110 MM = 1,MTRN
                          JREV2S = (MM-1)*MSKIP + JREV2*NSKIP
                          JS = (MM-1)*MSKIP + (J+1)*NSKIP
                          H = DATA(JREV2S+1)
                          DATA(JREV2S+1) = DATA(JS+1)
                          DATA(JS+1) = H
  110                 CONTINUE
                      IF (JREV-J.GT.0) THEN
                          DO 120 MM = 1,MTRN
                              JREVS = JREV*NSKIP + (MM-1)*MSKIP
                              JS = J*NSKIP + (MM-1)*MSKIP
                              H = DATA(JREVS+1)
                              DATA(JREVS+1) = DATA(JS+1)
                              DATA(JS+1) = H
  120                     CONTINUE
                      END IF

                  END IF

  130             CONTINUE
                  I2K = I2K/2
                  JREV = JREV - I2K
                  IF (JREV.GE.0) GO TO 130
                  JREV = JREV + 2*I2K
  100         CONTINUE
          END IF

          GO TO 140

   20     IERR = 101
      END IF
C
  140 CONTINUE
C
      IF (LFIRST) THEN
          LFIRST = .FALSE.
          CALL CPUTIM(T2)
          TFTVMT = T2 - T1
      END IF
C
      IF (LXSTEP) THEN
          IF (LXTIME) THEN
              LXTIME = .FALSE.
              CALL CPUTIM(T2)
              EFTVMT = T2 - T1
          END IF

      END IF
C
      NFTVMT = NFTVMT + 1
C
      END
C***********************************************************************
C
C     "CPUTIM" RETURNS ELAPSED USER (CPU) TIME, T, IN SECONDS, WHERE "T"
C     A REAL VARIABLE
C
C     ON THE CYBER205/ETA-10, THE INTRINSIC FUNCTION "SECOND()" IS REFER
C
      SUBROUTINE CPUTIM(T)
      REAL T

      T = 1.0

      END
C***********************************************************************
C
C     "ELAPSE" RETURNS ELAPSED WALL-CLOCK TIME, T, IN SECONDS
C
C     ON THE CYBER205 SYSTEM USE IS MADE OF THE SIL SUBROUTINE "Q5DCDMSC
C
C     FGPS 19-APR-88
C
      SUBROUTINE ELAPSE(T)

      REAL T

      T = 1.0

      END
