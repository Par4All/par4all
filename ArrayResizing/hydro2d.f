      PROGRAM HYD2D
*
*
****************************************************************
*
* Astrophysical program for the computation of galactical jets,
* using hydrodynamical Navier Stokes equations
*
* SPEC Benchmark Program 104.hydro2d, adopted from the program
* "hydro2d" (SNI-internal name: ASW02) written by Dr. Koesl,
* Max-Planck-Institut fuer Astrophysik, Garching, Germany
*
* This version of the benchmark evolved from the benchmark
* 090.hydro2d. The parameters MP and NP have been enlarged,
* for a larger problem size and longer execution time.
*
* Contact: Dr. Wilfried Stehling,
*          Siemens Nixdorf, D 552
*          81730 Muenchen, Germany
*
* Modifications for the SPEC version:
*
* - Internal timing statements removed
* - Output modified. Detailed output is now on unit 10.
* - All floating point data changed to double precision
*     (Declaration IMPLICIT DOUBLE PRECISION (A-H, O-Z))
*     Accordingly, replaced all calls to AMIN1, AMAX1
*     by calls to DMIN1, DMAX1
* - Association of Fortran unit number (10) and file HYDRO.DET
*     via an OPEN statement
* - Assigning variables via input file instead of DATA statements.
*     New subroutine (INPDTA) created. Subroutine call inserted in
*     main program.
* - Rearanged variables COMMON areas to avoid alignment problems
* - Changed fp parameter (constant) values in subroutines CUT, VPR,
*   VPS to 1.0D-7, to eliminate numerical differences for runs on
*   different machines
*
* Modifications March, July, Aug., Oct. 1991 by
*   Aenne Scharbert,
*   Marcus Schwankl,
*   Reinhold Weicker,
*   Andrea Wittmann,
*       Siemens Nixdorf, STM OS 323
*
* Modifications Sept./Oct.Nov. 1994 by
*   Reinhold Weicker
*   Siemens Nixdorf, MR PD 214
*   33094 Paderborn
*   Germany
* 
* - Increased grid size (N = 40)
* - Expressed padding of COMMON blocks in terms of MP, NP
*
****************************************************************
*
*          PROGRAMM ZUR BERECHNUNG GALAKTISCHER JETS
*          HERKUNFT: MAX-PLANCK-INSTITUT FUER ASTROPHYSIK
*                    GARCHING
*                    HERR DR. KOESSL
*
* PARAMETER MP, NP WERDEN IN ALLEN UNTERPROGRAMMEN GESETZT
* SIE MUESSEN DIE WERTE ANNEHMEN:
*
*      MP = 10 * N + 2  , NP = 4 * N
*
* SPEICHERBEDARF:
*
*      UNGEFAEHR 134 * MP * NP BYTES
*
* PARAMETERAENDERUNG DURCH CHANGE ALL COMMAND
*
*      C 'MP   =      102' 'MP   =      XXX' ALL
*      C 'NP   =       40' 'NP   =      XXX' ALL
*
* ANZAHL DER ZEITSCHRITTE WIRD IM BLOCK DATA PROGRAMM DURCH DEN
* WERT VON ISTEP FESTGELEGT.
*
* AUSDRUCK VON INFORMATION WIRD AUF UNIT 11 GELENKT.
* DETAILLIERTE MODELLDATEN WERDEN ALLE IPRIN ZEITSCHRITTE AUSGEGEBEN.
* DER WERT VON IPRIN WIRD IM BLOCK DATA PROGRAMM FESTGELEGT.
*
***************************************************************
*
*   English translation of comment:
*
*   The parameters MP, NP are set in all subroutines
*   They must have the values
*      MP = 10 * N + 2  , NP = 4 * N
*   Approximate memory requirements:
*      134 * MP * NP BYTES
*   The number of time steps is controlled in the block data
*   module (hydro2d.data.f) by the value of ISTEP.
*
*   Detailed data are generated every IPRIN time steps; the value
*   of IRPN is taken from the block data module.
*
***********************************************************************
*
*     MAIN PROGRAM:
*
*     1.)  DEFINTION OF COMMON BLOCKS
*     2.)  READ AND TEST BASIC DATA
*          (PARAMETERS READ FROM INPUT DATA SET ARE SIGNED WITH #)
*     3.)  READ LAST MODEL OR CONSTRUCT NEW MODEL
*     4.)  SOLVE HYDRODYNAMIC EQUATIONS
*     5.)  CHECK AND SAVE DATA  (CHANNEL NUMBER USED: 19)
*     6.)  CHECK REMAINING CPU TIME
*
***********************************************************************
*
*     DESCRIPTION OF PARAMETERS:
*
*     - MP(MX#) ...   NUMBER OF GRIDPOINTS IN Z-DIRECTION
*     - NP(NX#) ...   NUMBER OF GRIDPOINTS IN R-DIRECTION
*
*     REMARK:   MX,NX ARE USED TO CHECK THE DIMENSIONS MP,NP
*               OF THE ARRAYS THAT ARE FIXED IN PARAMETER
*               STATEMENTS IN EVERY ROUTINE
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
*     i.e. N = 40 (see comment above)
*
*
***********************************************************************
*
*     DESCRIPTION OF COMMONS:
*
*     - PRMT      ...   BASIC PARAMETERS
*     - DISK      ...   CONTROL BLOCK FOR LOOP AND I/O
*     - BOND      ...   CONTROL BLOCK FOR TREATMENT OF BOUNDARIES
*     - ADVC      ...   CONTROL BLOCK FOR LIMITED ADVECTION
*     - CUTP      ...   CONTROL BLOCK FOR PRESSURE UNDERSHOOT
*     - CONT      ...   CONTROL VARIABLES
*     - GRID      ...   POSITION OF GRIDPOINTS
*     - VAR1      ...   VARIABLES AT OLD TIMESTEP
*     - VAR2      ...   VARIABLES AT NEW TIMESTEP & STAGGERED GRID (1)
*     - VARH      ...   PRESSURE, VELOCITY & FLUXES
*     - VARS      ...   CORIOLIS FORCES
*     - SCRA      ...   SCRATCH MEMORY & VARIBLES AT STAGG. GRID (2)
*
*     ATTENTION:   COMMON VARH USED AS SCRATCH IN ROUTINE STAGF1
*
*     REMARK:      THE MAXIMUM MEMORY SPACE NEEDED FOR COMMON BLOCKS
*                  /GRID/ /VAR1/ /VAR2/ /VARH/ /VARS/ /SCRA/
*                  IS ALLOCATED ONLY VIA ONE DIMENSIONAL DUMMY ARRAYS
*
***********************************************************************
*
*     DESCRIPTION OF VARIABLES IN COMMON /PRMT/
*
*     - IS      ...   NUMBER OF TIMESTEP
*     - TS      ...   TOTAL TIME
*
*     - DT      ...   TIME STEP
*
*     - GAM     ...   GAMMA  =   CP / CV
*     - CFL#    ...   PORTION OF TIMESTEP USED (COURANT-FACTOR)
*     - VCT#    ...   MAGNITUDE OF ARTIFICIAL DIFFUSION COEFFICENT
*
*     - TZ      ...   MINIMAL GRID SPACING IN Z-DIRECTION
*     - TR      ...   MINIMAL GRID SPACING IN R-DIRECTION
*     - FTZ     ...   POSITION OF LOWER Z-BOUNDARY
*     - FTR     ...   POSITION OF LOWER R-BOUNDARY
*     - BSZ     ...   Z- BIAS OF EXPONENTIAL PART OF GRID
*     - BSR     ...   R- BIAS OF EXPONENTIAL PART OF GRID
*     - QTZ     ...   Z- FRACTION OF GRID WHICH IS EQUIDISTANT
*     - QTR     ...   R- FRACTION OF GRID WHICH IS EQUIDISTANT
*
*     DESCRIPTION OF VARIABLES IN COMMON /DISK/
*
*     - ISTEP#  ...   NUMBER OF TIMESTEPS TO BE COMPUTE
*     - ICNT    ...   NUMBER OF TIMESTEPS ACTUALLY COMPUTED
*     - ISTOR#  ...   EACH "ISTOR" TIMESTEPS A MODEL IS STORED
*     - IPRIN#  ...   EACH "IPRIN" TIMESTEPS DATA IS PRINTED
*     - MPROW#  ...   NUMBER OF Z-ROWS TO BE PRINTED
*     - NPROW#  ...   NUMBER OF R-ROWS TO BE PRINTED
*                     SKIP-DISTANCE = M(N)P / M(N)PROW
*
*     DESCRIPTION OF VARIABLES IN COMMON /BOND/
*
*     - ZL#     ...   TYPE OF LOWER Z-BOUNDARY  ( O=OPEN , C=CLOSED )
*     - ZU#     ...   TYPE OF UPPER Z-BOUNDARY
*     - RL#     ...   TYPE OF LOWER R-BOUNDARY
*     - RU#     ...   TYPE OF UPPER R-BOUNDARY
*
*     DESCRIPTION OF VARIABLES IN COMMON /ADVC/
*
*     - MQ      ...   UPPER INDEX FOR Z-ADVECTION
*     - NQ      ...   UPPER INDEX FOR R-ADVECTION
*                     (XQ1= PRED(XQ), XQ2= PRED(PRED(XQ))
*     - MQFLG   ...   LIMIT Z-ADVECTION IF MQFLG=1
*     - NQFLQ   ...   LIMIT R-ADVECTION IF NQFLG=1
*     - ROBMQ   ...   TARGET FOR SEARCH IN Z-DIRECTION
*     - ROBNQ   ...   TARGET FOR SEARCH IN R-DIRECTION
*
*     DESCRIPTION OF VARIABLES IN COMMON /CUTP/
*
*     - NEGFL#  ...   PRINT POSITION OF NEGATIV PRESSURE IF NEGFL=1
*     - NEGCN#  ...   MAXIMUM NUMBER OF CELLS WITH NEGATIVE PRESSURE
*                     (THE PROGRAM STOPS IF NEGPR > NEGCN BECOMES TRUE)
*     - NEGPR   ...   NUMBER OF CELLS WITH NEGATIVE PRESSURE
*     - NEGPO   ...   POSITIONS OF NEGATIVE PRESSURE (DIM=1000)
*     - CUTPR   ...   "FLOOR" OF PRESSURE
*                     (IF PR < PRCUT THEN PR IS SET EQUAL PRCUT)
*
*     DESCRIPTION OF VARIABLES IN COMMON /CONT/
*
*     - ROMIN   ...   MINIMUM OF DENSITY     (POSITION AT MINRO)
*     - ENMIN   ...   MINIMUM OF ENERGY      (POSITION AT MINEN)
*     - GZMIN   ...   MINIMUM OF Z-MOMENTUM  (POSITION AT MINGZ)
*     - GRMIN   ...   MINIMUM OF R-MOMENTUM  (POSITION AT MINGR)
*
*     - ROMAX   ...   MAXIMUM OF DENSITY     (POSITION AT MAXRO)
*     - ENMAX   ...   MAXIMUM OF ENERGY      (POSITION AT MAXEN)
*     - GZMAX   ...   MAXIMUM OF Z-MOMENTUM  (POSITION AT MAXGZ)
*     - GRMAX   ...   MAXIMUM OF R-MOMENTUM  (POSITION AT MAXGR)
*
*     - TOTMA   ...   TOTAL MASS
*     - TOTEN   ...   TOTAL ENERGY
*     - TOTGZ   ...   TOTAL Z-MOMENTUM
*
*     DESCRIPTION OF VARIABLES IN COMMON /GRID/
*
*     - Z       ...   POSITION OF Z-MESH-POINTS
*     - ZB      ...   POSITION OF Z-CELL-BOUNDARIES
*     - DZ      ...   DISTANCE BETWEEN Z-CELL-BOUNDARIES
*     - DBZ     ...   DISTANCE BETWEEN Z-MESH-POINTS
*     - FZ      ...   INTERPOLATION WEIGHTS FOR Z-DIRECTION
*
*     - R       ...   POSITION OF R-MESH-POINTS
*     - RB      ...   POSITION OF R-CELL-BOUNDARIES
*     - DR      ...   DISTANCE BETWEEN R-CELL-BOUNDARIES
*     - DBR     ...   DISTANCE BETWEEN R-MESH-POINTS
*     - FR      ...   INTERPOLATION WEIGHTS FOR R-DIRECTION
*
*     DESCRIPTION OF VARIABLES IN COMMON /VAR1/
*
*     - RO      ...   DENSITY
*     - EN      ...   TOTAL ENERGY
*     - GZ      ...   Z-MOMENTUM
*     - GR      ...   R-MOMENTUM
*     - GP      ...   PHI-MOMENTUM
*     - BZ      ...   Z-FIELD
*     - BR      ...   R-FIELD
*     - BP      ...   PHI-FIELD
*
*     FOR DESCRIPTION OF THE VARIABLES IN THE REMAINING
*     COMMON BLOCKS SEE THE COMMENTS IN THE SUBROUTINES
*
***********************************************************************
*
*
      CHARACTER       ZL,ZU, RL,RU
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /DISK/   ISTEP, ICNT,
     &                ISTOR, IPRIN, MPROW, NPROW
      COMMON /BOND/   ZL,ZU, RL,RU
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
      COMMON /CONT/   ENMAX, ROMAX, ENMIN, ROMIN,
     &                GRMAX, GZMAX, GRMIN, GZMIN,
     &                TOTMA, TOTEN, TOTGZ,
     &                MINRO, MINEN, MAXRO, MAXEN,
     &                MINGZ, MINGR, MAXGZ, MAXGR
      COMMON /GRID/   H0(  5 * (MP+1)  +
     &                     5 * (NP+1)           )
      COMMON /VAR1/   H1(  4 *  MP    *  NP     )
      COMMON /VAR2/   H2(  4 * (MP+1) * (NP+1)  )
      COMMON /VARH/   H3(  4 * (MP+1) * (NP+1)  )
      COMMON /VARS/   H4(  1 *  MP    *  NP     )
      COMMON /SCRA/   H5(  4 * (MP+1) * (NP+1)  )
*
      OPEN (UNIT=10, FILE='HYDRO.DET')
*
*       File for detail output, not included in the comparison
*       of results for SPEC purposes
*
*      read values from file
       CALL INPDTA
*
*      Print the values only for last model computed
       IPRIN = ISTEP
*      If, for diagnostic purposes, more values are to be printed,
*      this should be changed accordingly (e.g., IPRIN = 80 causes
*      the values to be printed every 80th time step).
*
*
***********************************************************************
*        PREPARE THE COMPUTATION
***********************************************************************
*
*
      CALL PREPAR
*
*
***********************************************************************
*        COMPUTE INITAL MODEL OR READ LAST MODEL FROM DISC
***********************************************************************
*
*
*
         CALL INIVAL
*
*
***********************************************************************
*        ADVANCE SOLUTION BY 'ISTEP' TIMESTEPS
***********************************************************************
*
*
      DO 100  ICNT = 1,ISTEP
*
*----------------------------------------------------------------------
*        COMPUTE NEW TIMESTEP AND SOLVE EQUATIONS
*----------------------------------------------------------------------
*
         CALL TISTEP
         CALL ADVNCE
*
*----------------------------------------------------------------------
*        COMPUTE CONTROL VARIABLES AND SAVE DATA
*----------------------------------------------------------------------
*
         CALL CHECK
         CALL OUTPUT
*
*----------------------------------------------------------------------
*        END OF LOOP
*----------------------------------------------------------------------
*
  100 CONTINUE
*
*----------------------------------------------------------------------
*        EXIT IF THE LOOP IS NOT FINISHED
*        BUT THE CPU TIME IS ALREADY CONSUMED
*----------------------------------------------------------------------
*
*  200 CONTINUE
*
*
***********************************************************************
*        TERMINATE THE COMPUTATION
***********************************************************************
*
*
      CALL TERM
*
*
***********************************************************************
*
*
      END
      SUBROUTINE PREPAR
*
*
***********************************************************************
*
*     THIS SUBROUTINE PREPARES THE COMPUTATION
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      CHARACTER       ZL,ZU, RL,RU
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /DISK/   ISTEP, ICNT,
     &                ISTOR, IPRIN, MPROW, NPROW
      COMMON /BOND/   ZL,ZU, RL,RU
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
*
*
***********************************************************************
*        DEFINE DIMENSION CONTROL VARAIABLES
***********************************************************************
*
*
      PARAMETER  ( MX      =         MP )
      PARAMETER  ( NX      =         NP )
*
*
***********************************************************************
*       DEFINE STATEMENT FUNCTION FOR BOUNDARY PARAMTER TEST
***********************************************************************
*
*
      CHARACTER  CH
      LOGICAL    NORCO
      NORCO(CH) =  .NOT. ( CH.EQ.'C' .OR. CH.EQ.'O' )
*
*
***********************************************************************
*        TEST LENGTH OF MEMORY
***********************************************************************
*
*
      IF (  MX .NE. MP  .OR.  NX .NE. NP  ) THEN
       WRITE (10,50)    MX,NX , MP,NP
   50    FORMAT (1H1//////6X,'REQUESTED DIMENSION'
     &                   /6X,'MX   =  ',I15
     &                   /6X,'NX   =  ',I15
     &                 ///6X,'NOT EQUAL'
     &                   /6X,'DEFINED DIMENSION'
     &                   /6X,'MP   =  ',I15
     &                   /6X,'NP   =  ',I15
     &              //////6X,'STOP IN ROUTINE:   PREPAR')
         STOP
      END IF
*
*
      IF (  MP .LT. 3  .OR.  NP .LT. 3  ) THEN
	 WRITE(10,60) MP , NP
   60    FORMAT (1H1//////6X,'DEFINED DIMENSION'
     &                   /6X,'TOO SMALL (LESS THAN 3)'
     &                   /6X,'MP   =  ',I15
     &                   /6X,'NP   =  ',I15
     &              //////6X,'STOP IN ROUTINE:   PREPAR')
         STOP
      END IF
*
*
***********************************************************************
*        CHECK BOUNDARY PARAMETERS
***********************************************************************
*
*
      IF (  NORCO(ZL)  .OR.
     &      NORCO(ZU)  .OR.
     &      NORCO(RL)  .OR.
     &      NORCO(RU)       ) THEN
	 WRITE(10,70)  ZL,ZU, RL,RU
   70    FORMAT (1H1//////6X,'ILLEGAL BOUNDARY PARAMETER SPECIFCATION'
     &                   /6X,'ONLY ''C'' AND ''O'' ARE ACCEPTED'
     &                   /6X,'Z-DIR. (LOWER)  = ',A
     &                   /6X,'Z-DIR. (UPPER)  = ',A
     &                   /6X,'R-DIR. (LOWER)  = ',A
     &                   /6X,'R-DIR. (UPPER)  = ',A
     &              //////6X,'STOP IN ROUTINE:   PREPAR')
         STOP
      END IF
*
*
***********************************************************************
*        CHECK SOME I/O AND THE Z-ADVECTION PARAMETERS
***********************************************************************
*
*
      IF ( MPROW .LT.    1 )   MPROW =    1
      IF ( MPROW .GT.   MP )   MPROW =   MP
      IF ( NPROW .LT.    1 )   NPROW =    1
      IF ( NPROW .GT.   NP )   NPROW =   NP
*
      IF ( NEGFL .LT.    0 )   NEGFL =    0
      IF ( NEGFL .GT.    1 )   NEGFL =    1
      IF ( NEGCN .LT.    0 )   NEGCN =    0
      IF ( NEGCN .GT. 1000 )   NEGCN = 1000
*
      IF ( MQFLG .LT.    0 )   MQFLG =    0
      IF ( MQFLG .GT.    1 )   MQFLG =    1
      IF ( NQFLG .LT.    0 )   NQFLG =    0
      IF ( NQFLG .GT.    1 )   NQFLG =    1
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE INIVAL
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE PROGRAM IS STARTED
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /DISK/   ISTEP, ICNT,
     &                ISTOR, IPRIN, MPROW, NPROW
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
*
*
***********************************************************************
*     SET TIME TO ZERO AND COMPUTE INITIAL MODEL AND GRID
***********************************************************************
*
*
      IS = 0
      TS = 0.0D0
      DT = 0.0D0
*
*
      CALL INIMOD
      CALL GRIDCO
*
*
***********************************************************************
*     PRESET BOUNDARIES AND DETERMINE TARGETS FOR LIMITED ADVECTION
***********************************************************************
*
*
      ROBMQ = RO(MP,1)
      ROBNQ = RO(1,NP)

*
*
      CALL ADLEN ('H')
*
***********************************************************************
*     COMPUTE CONTROL VARIABLES, PRESSURE AND VELOCITY,
*     PRINT BASIC PARAMETERS OF INITIAL MODEL
*     AND WRITE INITIAL MODEL ON DISC
***********************************************************************
*
*
      CALL CHECK
C     CALL OUTPUT
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE INIMOD
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE INITIAL MODEL IS COMPUTED
*
***********************************************************************
*
*     VALUES SET OR COMPUTED:
*        -  GAM
*        -  RO,EN, GZ,GR
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,   NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
*
*
      PARAMETER  ( XLZ =       30.0D0 )
      PARAMETER  ( XLR =        8.0D0 )
*
C     PARAMETER  ( XCZ =       30.0D0 )
      PARAMETER  ( XCZ =       15.0D0 )
      PARAMETER  ( XSZ =       30.0D0 )
*
      PARAMETER  ( RO0 =        1.0D0 )
      PARAMETER  ( PR0 =        0.6D0 )
*
      PARAMETER  ( XLJET =      1.0D0 )
      PARAMETER  ( ROJET =      0.1D0 )
      PARAMETER  ( PRJET =      0.6D0 )
      PARAMETER  (  VJET =    18.97D0 )

      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
*
*
***********************************************************************
*     OPEN MODEL INPUT
***********************************************************************
      OPEN(11,FILE='HYDRO2D.MODEL',STATUS='OLD')
***********************************************************************
*     SET BASIC PARAMETERS
***********************************************************************
*
*
      GAM =  DBLE (5.0D0) / DBLE (3.0D0)
*
      TZ  =   XLZ / (MP-2)
      TR  =   XLR / NP
      FTZ =  DBLE (-2.0D0) * TZ
      FTR =   0.0D0
      BSZ =   1.0D0
      BSR =   1.0D0
      QTZ =   1.0D0
      QTR =   1.0D0
*
*
***********************************************************************
*     SET VARIABLES
***********************************************************************
*
*
      CON =  GAM - 1.0D0
      EN0 =  PR0 / CON
*
      MPCON =  NINT( MP * XCZ / XLZ )
      MPSTR =  NINT( MP * XSZ / XLZ )
*
      DO 5 J=1,NP
      DO 5 I=1,MP
      READ(11,1000,END=99) RO(I,J)
  5   CONTINUE
      DO 10 J =  1 , NP
      DO 10 I =  1 , MPCON
C        RO(I,J) =  RO0
         EN(I,J) =  EN0
         GZ(I,J) =  0.0D0
         GR(I,J) =  0.0D0
   10 CONTINUE
*
      DO 12 J =        1 , NP
      DO 12 I =  MPCON+1 , MP
*        RO(I,J) =  RO0  *  10 ** ( REAL(MPCON-I) / REAL(MPSTR) )
C        RO(I,J) =  RO0  *  10 ** ( DBLE(MPCON-I) / DBLE(MPSTR) )
         EN(I,J) =  EN0
         GZ(I,J) =  0.0D0
         GR(I,J) =  0.0D0
   12 CONTINUE
*
*
      ENJET =  PRJET / CON  +  0.5D0 * ROJET * VJET**2
      GZJET =  ROJET * VJET
*
      NPJET =  NINT( NP * XLJET / XLR )
*
      DO 20 J =  1 , NPJET
      DO 20 I =  1 , 2
C        RO(I,J) =  ROJET
         EN(I,J) =  ENJET
         GZ(I,J) =  GZJET
   20 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
   99 WRITE(6,1099)
      STOP
 1000 FORMAT(E20.14)
 1099 FORMAT(' NO FURTHER MODEL DATA AVAILABLE, STOP')
      END
      SUBROUTINE BBH
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE "BOTTOM"-BOUNDARY-VALUES
*     FOR THE HALF TIME-STEP ARE COMPUTED
*
***********************************************************************
*
*     VALUES COMPUTED:
*        -  RO,EN           ( VALUES AT STAGGERED GRID (1)    )
*        -  GZ,GR
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,   NP   =       160 )
*
*
      PARAMETER  ( XLZ =       30.0D0 )
      PARAMETER  ( XLR =        8.0D0 )
      PARAMETER  ( XLJET =      1.0D0 )
      PARAMETER  ( ROJET =      0.1D0 )
      PARAMETER  ( PRJET =      0.6D0 )
      PARAMETER  (  VJET =    18.97D0 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
*
*
***********************************************************************
*     COMPUTE LOWER Z-BOUNDARY
***********************************************************************
*
*
      CON = GAM - 1.0D0
      ENJET =  PRJET / CON  +  0.5D0 * ROJET * VJET**2
      GZJET =  ROJET * VJET
*
      NPJET =  NINT( NP * XLJET / XLR )
*
      DO 100  J = 1 , NPJET
      DO 100  I = 1 , 2
*
         RO(I,J) =  ROJET
         EN(I,J) =  ENJET
         GZ(I,J) =  GZJET
         GR(I,J) =  0.0D0
*
  100 CONTINUE
*
      DO 200  J = NPJET+1 , NQ
*
         RO(1,J) =   RO(3,J)
         EN(1,J) =   EN(3,J)
         GZ(1,J) = - GZ(3,J)
         GR(1,J) =   GR(3,J)
*
         RO(2,J) =   RO(3,J)
         EN(2,J) =   EN(3,J)
         GZ(2,J) = - GZ(3,J)
         GR(2,J) =   GR(3,J)
*
  200 CONTINUE
*
*
***********************************************************************
*     COMPUTE LOWER R-BOUNDARY
***********************************************************************
*
*
      DO 300  I = 1 , MQ
*
         RO(I,1) =   RO(I,2)
         EN(I,1) =   EN(I,2)
         GZ(I,1) =   GZ(I,2)
	 GR(I,1) =   GR(I,2)  /  DBLE(3.0D0)
*
  300 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE BBF
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE "BOTTOM"-BOUNDARY-VALUES
*     FOR THE FULL TIME-STEP ARE COMPUTED
*
***********************************************************************
*
*     VALUES COMPUTED:
*        -  RON,ENN           ( VALUES AT STAGGERED GRID (2)    )
*        -  GZN,GRN
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,   NP   =       160 )
*
*
      PARAMETER  ( XLZ =       30.0D0 )
      PARAMETER  ( XLR =        8.0D0 )
      PARAMETER  ( XLJET =      1.0D0 )
      PARAMETER  ( ROJET =      0.1D0 )
      PARAMETER  ( PRJET =      0.6D0 )
      PARAMETER  (  VJET =    18.97D0 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
* DU2 is a dummy              a.w.
      COMMON /VAR2/   RON(MP,NP), ENN(MP,NP),
     &                GZN(MP,NP), GRN(MP,NP),
     &                DU2(4*MP + 4*NP + 4)
*
*
***********************************************************************
*     COMPUTE LOWER Z-BOUNDARY
***********************************************************************
*
*
      CON = GAM - DBLE(1.0D0)
      ENJET =  PRJET / CON  +  DBLE(0.5D0) * ROJET * VJET**2
      GZJET =  ROJET * VJET
*
      NPJET =  NINT( NP * XLJET / XLR )
*
      DO 100  J = 1 , NPJET
      DO 100  I = 1 , 2
*
         RON(I,J) =  ROJET
         ENN(I,J) =  ENJET
         GZN(I,J) =  GZJET
         GRN(I,J) =  0.0D0
*
  100 CONTINUE
*
      DO 200  J = NPJET+1 , NQ
*
         RON(1,J) =   RON(3,J)
         ENN(1,J) =   ENN(3,J)
         GZN(1,J) = - GZN(3,J)
         GRN(1,J) =   GRN(3,J)
*
         RON(2,J) =   RON(3,J)
         ENN(2,J) =   ENN(3,J)
         GZN(2,J) = - GZN(3,J)
         GRN(2,J) =   GRN(3,J)
*
  200 CONTINUE
*
*
***********************************************************************
*     COMPUTE LOWER R-BOUNDARY
***********************************************************************
*
*
      DO 300  I = 1 , MQ
*
         RON(I,1) =   RON(I,2)
         ENN(I,1) =   ENN(I,2)
         GZN(I,1) =   GZN(I,2)
	 GRN(I,1) =   GRN(I,2)  /  DBLE(3.0D0)
*
  300 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE GRIDCO
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE GRID IS CONSTRUCTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  TX      ( MINIMAL GRID SPACING                  )
*        -  FTX     ( POSITION OF LOWER BOUNDARY            )
*        -  BSX     ( BIAS OF EXPONENTIAL PART OF GRID      )
*        -  QTX     ( FRACTION OF GRID WHICH IS EQUIDISTANT )
*     VALUES COMPUTED:
*        -  X       ( POSITION OF MESH-POINTS               )
*        -  XB      ( POSITION OF CELL-BOUNDARIES           )
*        -  DX      ( DISTANCE BETWEEN CELL-BOUNDARIES      )
*        -  DBX     ( DISTANCE BETWEEN MESH-POINTS          )
*        -  FX      ( INTERPOLATION WEIGHTS                 )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
*
*
***********************************************************************
*     COMPUTE AND CHECK VALUES OF GRID-PARAMETERS
***********************************************************************
*
*
      MH =  INT( QTZ * MP )
      NH =  INT( QTR * NP )
*
      IF ( MH .LT.  1 )   MH =   1
      IF ( NH .LT.  1 )   NH =   1
      IF ( MH .GT. MP )   MH =  MP
      IF ( NH .GT. NP )   NH =  NP
      IF ( BSZ .LE. 0.0 )   BSZ =  1.0D0
      IF ( BSR .LE. 0.0 )   BSR =  1.0D0
*
*
***********************************************************************
*     COMPUTE Z-GRID
***********************************************************************
*
*
      DO 100  I = 1,MH
	 Z(I) =  TZ * ( I - DBLE(0.5D0) )  +  FTZ
  100 CONTINUE
      WIDO = DBLE(0.5D0) * TZ
      DO 105  I = MH+1,MP
	 WIDN =  DBLE(0.5D0)  *  TZ  *  BSZ ** ( I - MH )
         Z(I) =  Z(I-1) + WIDO + WIDN
         WIDO = WIDN
  105 CONTINUE
*
      ZB(0) = FTZ
      DO 110  I = 1,MP
         ZB(I) =  Z(I)  +   Z(I) - ZB(I-1)
  110 CONTINUE
*
      DO 120  I = 1,MP
         DZ(I) =  ZB(I) - ZB(I-1)
  120 CONTINUE
*
      DO 130  I = 1,M1
         DBZ(I) =  Z(I+1) - Z(I)
         FZ(I)  =  ( ZB(I) - Z(I) )  /  ( Z(I+1) - Z(I) )
  130 CONTINUE
      DBZ( 0) =  DBZ( 1)
      DBZ(MP) =  DBZ(M1)
*
*
***********************************************************************
*     COMPUTE R-GRID
***********************************************************************
*
*
      DO 200  J = 1,NH
	 R(J) =  TR * ( J - DBLE(0.5D0) )  +  FTR
  200 CONTINUE
      WIDO = DBLE(0.5D0) * TR
      DO 205  J = NH+1,NP
	 WIDN =  DBLE(0.5D0)  *  TR  *  BSR ** ( J - NH )
         R(J) =  R(J-1) + WIDO + WIDN
         WIDO = WIDN
  205 CONTINUE
*
      RB(0) = FTR
      DO 210  J = 1,NP
         RB(J) =  R(J)  +   R(J) - RB(J-1)
  210 CONTINUE
*
      DO 220  J = 1,NP
         DR(J) =  RB(J) - RB(J-1)
  220 CONTINUE
*
      DO 230  J = 1,N1
         DBR(J) =  R(J+1) - R(J)
         FR(J)  =  ( RB(J) - R(J) )  /  ( R(J+1) - R(J) )
  230 CONTINUE
      DBR( 0) =  DBR( 1)
      DBR(NP) =  DBR(N1)
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE TISTEP
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE TIMESTEP IS COMPUTED
*     (CFL-CONDITION & DIFFUSION TIME SCALE & PRESSURE GRADIENT)
*
***********************************************************************
*
*     VALUES REQUIRED:
*        - GAM, CFL,VCT, IS,TS, DZ,DR
*        - RO,PR, VZ,VR
*     VALUES USED AS TEMPORARY STORAGE:
*        - DPZ,DPR   ( PRESSURE DIFFERENCES     )
*        - DVZ,DVR   ( VELOCITY DIFFERENCES     )
*        - TST       ( TIME STEPS FOR EACH CELL )
*     VALUES COMPUTED:
*        - DT, IS,TS
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( MN   =   MP * NP )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
      COMMON /VAR1/   RO (MP,NP), EN (MP,NP),
     &                GZ (MP,NP), GR (MP,NP)
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ (MP,NP), VR (MP,NP),
     &                PR (MP,NP), TST(MN),
     &                DU3(4*MP + 4*NP + 4)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   DPZ(MP,NP), DPR(MP,NP),
     &                DVZ(MP,NP), DVR(MP,NP),
     &                DU4(4*MP + 4*NP + 4)
*
*
***********************************************************************
*     COMPUTE DIFFERENCES OF PRESSURE
***********************************************************************
*
*
      DO 200  J = 1,NQ
      DO 200  I = 2,MQ1
	 DPZ(I,J) =DBLE(0.5D0) * ABS( PR(I+1,J) - PR(I-1,J) ) / RO(I,J)
  200 CONTINUE
*
      DO 202  J = 1,NQ
         DPZ( 1,J) =  ABS( PR( 2,J) - PR(  1,J) )  /  RO( 1,J)
         DPZ(MQ,J) =  ABS( PR(MQ,J) - PR(MQ1,J) )  /  RO(MQ,J)
  202 CONTINUE
*
*
      DO 210  J = 2,NQ1
      DO 210  I = 1,MQ
	 DPR(I,J) =DBLE(0.5D0) * ABS( PR(I,J+1) - PR(I,J-1)) / RO(I,J)
  210 CONTINUE
*
      DO 212  I = 1,MQ
         DPR(I, 1) =  ABS( PR(I, 2) - PR(I,  1) )  /  RO(I, 1)
         DPR(I,NQ) =  ABS( PR(I,NQ) - PR(I,NQ1) )  /  RO(I,NQ)
  212 CONTINUE
*
*
***********************************************************************
*     COMPUTE DIFFERENCES OF VELOCITY
***********************************************************************
*
*
      VC2 =  DBLE(2.0D0) * VCT
*
*
      DO 300  J = 1,NQ
      DO 300  I = 2,MQ1
         DVZ(I,J) =  VCT  *  ABS( VZ(I+1,J) - VZ(I-1,J) )
  300 CONTINUE
*
      DO 302  J = 1,NQ
         DVZ( 1,J) =  VC2  *  ABS( VZ( 2,J) - VZ(  1,J) )
         DVZ(MQ,J) =  VC2  *  ABS( VZ(MQ,J) - VZ(MQ1,J) )
  302 CONTINUE
*
*
      DO 310  J = 2,NQ1
      DO 310  I = 1,MQ
         DVR(I,J) =  VCT  *  ABS( VR(I,J+1) - VR(I,J-1) )
  310 CONTINUE
*
      DO 312  I = 1,MQ
         DVR(I, 1) =  VC2  *  ABS( VR(I, 2) - VR(I,  1) )
         DVR(I,NQ) =  VC2  *  ABS( VR(I,NQ) - VR(I,NQ1) )
  312 CONTINUE
*
*
***********************************************************************
*     COMPUTE TIMESTEP FOR EACH CELL
***********************************************************************
*
*
      K = 0
*
      DO 400  J = 1,NQ
      DO 400  I = 1,MQ
*
         VSD =  SQRT ( GAM * PR(I,J) / RO(I,J) )
*
         VZ1 =  VSD  +  ABS(VZ(I,J))  +  DVZ(I,J)
         VZ2 =  VZ1 ** 2
         VR1 =  VSD  +  ABS(VR(I,J))  +  DVR(I,J)
         VR2 =  VR1 ** 2
*
	 XPZ =  DBLE(0.25D0) * DPZ(I,J) / VZ2
	 XPR =  DBLE(0.25D0) * DPR(I,J) / VR2
*
         IF ( XPZ .LT. 1.0D-3 )   DPZ(I,J) = 1.0D0
         IF ( XPR .LT. 1.0D-3 )   DPR(I,J) = 1.0D0
*
	 TCZ =  DBLE(0.5D0) * DZ(I) / DPZ(I,J) *
     &             ( SQRT( VZ2 + DBLE(4.0D0)*DPZ(I,J) ) - VZ1 )
	 TCR =  DBLE(0.5D0) * DR(J) / DPR(I,J) *
     &             ( SQRT( VR2 + DBLE(4.0D0)*DPR(I,J) ) - VR1 )
*
         IF ( XPZ .LT. 1.0D-3 )
     &         TCZ =  DZ(I) / VZ1  *  ( DBLE(1.0D0) - XPZ )
         IF ( XPR .LT. 1.0D-3 )
     &         TCR =  DR(J) / VR1  *  ( DBLE(1.0D0) - XPR )
*
         K =  K + 1
	 TST(K) =  DMIN1( TCZ , TCR )
*
  400 CONTINUE
*
*
***********************************************************************
*     COMPUTE NEW TIMESTEP
*     ADVANCE TIME AND STEP-COUNTER
***********************************************************************
*
*
      MI =  ISMIN( MQ*NQ , TST , 1 )
      TC =  CFL * TST(MI)
      DT =  DBLE(2.0D0) * DT
      IF ( TC .LT. DT )   DT =  TC
*
      IF ( IS .EQ.  0 )   DT =  DBLE(0.01D0) * TC
*
      IS =  IS + 1
      TS =  TS + DT
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE ADVNCE
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE SOLUTION IS ADVANCED BY ONE TIME-STEP
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  DT
*        -  RO, EN             ( "OLD" - VALUES AT TIMESTEP N      )
*        -  GZ, GR
*     VALUES USED AS TEMPORARY STORAGE:
*        -  RON,ENN            ( "NEW" - VALUES AT HALF TIMESTEP
*        -  GZN,GRN              T-VALUES AT FULL & HALF TIMESTEP  )
*     VALUES COMPUTED:
*        -  RO, EN             ( "NEW" - VALUES AT FULL TIMESTEP   )
*        -  GZ, GR
*
*     REMARKS:  THE CALLING SEQUENCE OF THE SUBROUTINES
*                         M U S T   N O T
*               BE CHANGED IN ORDER TO AVOID CATASTROPHIC
*               MEMORY CONFLICTS IN COMMON /VAR2/ & /VARS/
*     /VAR2/ :  1.)  UNDEFINED ON ENTRY
*               2.)  OLD VALUES ON STAGGERED GRID (2) AFTER STAGH1
*               3.)  HALF TRANSPORTED VALUES AFTER TRANS2 (1ST CALL)
*               4.)  HALF VALUES AFTER HALF TIMESTEP
*               5.)  HALF VALUES ON STAGGERED GRID (2) AFTER STAGF1
*               6.)  NEW TRANSPORTED VALUES AFTER TRANS2 (2ND CALL)
*               7.)  UNDEFINED ON EXIT
*     /VARS/ :  1.)  UNDEFINED ON ENTRY
*               2.)  OLD CORIOLIS FORCES AFTER CORIH
*               3.)  HALF CORIOLIS FORCES AFTER CORIF
*               4.)  UNDEFINED ON EXIT
*     ORDER  :  CALL STAGF1 AFTER  STAGF2
*               CALL TRANS1 BEFORE TRANS2
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /VAR1/   RO (MP,NP), EN (MP,NP),
     &                GZ (MP,NP), GR (MP,NP)
* DU2 is a dummy              a.w.
      COMMON /VAR2/   RON(MP,NP), ENN(MP,NP),
     &                GZN(MP,NP), GRN(MP,NP),
     &                DU2(4*MP + 4*NP + 4)
*
*
***********************************************************************
*     COMPUTE HALF TIMESTEP
***********************************************************************
*
*
*----------------------------------------------------------------------
*     COMPUTE TIMESTEP AND REGION OF ADVECTION
*----------------------------------------------------------------------
*
      DT = DBLE(0.5D0) * DT
      CALL ADLEN ('H')
*
*----------------------------------------------------------------------
*     COMPUTE VALUES OF VARAIABLES AT CELL BOUNDARIES
*----------------------------------------------------------------------
*
      CALL BBH
      CALL CORIH
      CALL STAGH2
      CALL STAGH1
*
*----------------------------------------------------------------------
*     COMPUTE TRANSPORT
*----------------------------------------------------------------------
*
      CALL TRANS1
      CALL TRANS2
*
*----------------------------------------------------------------------
*     COMPUTE FLUX-CORRECTION
*----------------------------------------------------------------------
*
      CALL FCT   ( RON , RON , RO )
      CALL FCT   ( ENN , ENN , EN )
      CALL FCT   ( GZN , GZN , GZ )
      CALL FCT   ( GRN , GRN , GR )
*
*
***********************************************************************
*     COMPUTE FULL TIMESTEP
***********************************************************************
*
*
*----------------------------------------------------------------------
*     COMPUTE TIMESTEP AND REGION OF ADVECTION
*----------------------------------------------------------------------
*
      DT = DBLE(2.0D0) * DT
      CALL ADLEN ('F')
*
*----------------------------------------------------------------------
*     COMPUTE VALUES OF VARAIABLES AT CELL BOUNDARIES
*----------------------------------------------------------------------
*
      CALL BBF
      CALL CORIF
      CALL STAGF2
      CALL STAGF1
*
*----------------------------------------------------------------------
*     COMPUTE TRANSPORT
*----------------------------------------------------------------------
*
      CALL TRANS1
      CALL TRANS2
*
*----------------------------------------------------------------------
*     COMPUTE FLUX-CORRECTION
*----------------------------------------------------------------------
*
      CALL FCT   ( RO , RON , RO )
      CALL FCT   ( EN , ENN , EN )
      CALL FCT   ( GZ , GZN , GZ )
      CALL FCT   ( GR , GRN , GR )
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE ADLEN (STEP)
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE UPPER BOUNDARY
*     (MAXIMUM INDEX) FOR THE Z- AND R-ADVECTION ARE COMPUTED
*     IF MQFLG=1 THE Z-ADVECTION IS SKIPPED IN THE RIGHT PART
*     AND IF NQFLG=1 THE R-ADVECTION IS SKIPPED IN THE
*     UPPER PART OF THE GRID TO SAVE COMPUTATION TIME
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  MQFLG, NQFLG
*        -  ROBMQ, ROBNQ
*        -  RO
*     VALUES COMPUTED:
*        -  MQ,MQ1,MQ2
*        -  NQ,NQ1,NQ2
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
      PARAMETER  ( M2   =   MP - 2 ,   N2   =   NP - 2 )
      PARAMETER  ( MN   =   MP * NP )
*
      CHARACTER  STEP
*
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   H1(MP), H2(NP),
     &                DU4(4*MP*NP + 3*MP + 3*NP + 4)
*
***********************************************************************
*     DETERMINE IF CALLED FOR HALF OF FULL TIMESTEP
***********************************************************************
*
*
      IF ( STEP .NE. 'H' )  GOTO 100
*
*
***********************************************************************
*     HALF TIMESTEP
***********************************************************************
*
*
*----------------------------------------------------------------------
*     LIMIT Z-ADVECTION
*----------------------------------------------------------------------
*
      IF ( MQFLG .EQ. 1 ) THEN
         DO  10  I = 1,MP
            H1(I) = RO(MP-I+1,1)
   10    CONTINUE
         MQ  =  ISFNE ( MP , H1 , 1 , ROBMQ )
         MQ  =  MIN0  ( MP , MP-MQ+11 )
         MQ1 =  MQ - 1
         MQ2 =  MQ - 2
      ELSE
         MQ  =  MP
         MQ1 =  M1
         MQ2 =  M2
      ENDIF
*
*----------------------------------------------------------------------
*     LIMIT R-ADVECTION
*----------------------------------------------------------------------
*
      IF ( NQFLG .EQ. 1 ) THEN
         DO  20  J = 1,NP
            H2(J) = RO(1,NP-J+1)
   20    CONTINUE
         NQ  =  ISFNE ( NP , H2 , 1 , ROBNQ )
         NQ  =  MIN0  ( NP , NP-NQ+11 )
         NQ1 =  NQ - 1
         NQ2 =  NQ - 2
      ELSE
         NQ  =  NP
         NQ1 =  N1
         NQ2 =  N2
      ENDIF


*
*----------------------------------------------------------------------
*     RETURN TO ROUTINE ADVNCE
*----------------------------------------------------------------------
*
      RETURN
*
*
***********************************************************************
*     FULL TIMESTEP
***********************************************************************
*
*
  100 CONTINUE
*
*----------------------------------------------------------------------
*     DIMINISH ADVECTION LENGTHES BY ONE
*----------------------------------------------------------------------
*
      IF ( MQFLG.EQ.1 .AND. MQ.LT.MP ) THEN
         MQ  =  MQ  - 1
         MQ1 =  MQ1 - 1
         MQ2 =  MQ2 - 1
      ENDIF
*
      IF ( NQFLG.EQ.1 .AND. NQ.LT.NP ) THEN
         NQ  =  NQ  - 1
         NQ1 =  NQ1 - 1
         NQ2 =  NQ2 - 1
      ENDIF
*
*----------------------------------------------------------------------
*     RETURN TO ROUTINE ADVNCE
*----------------------------------------------------------------------
*
      RETURN
*
*
***********************************************************************
*
*
      END
      SUBROUTINE CORIH
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE CORIOLIS FORCES ARE COMPUTED
*     FROM THE "OLD" VALUES FOR THE HALF TIME STEP
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RO,EN             ( VALUES BEFORE TRANSPORT         )
*        -  GZ,GR
*     VALUES COMPUTED:
*        -  CHGR              ( CORIOLIS FORCES                 )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
      COMMON /VARS/   CHGR(MP,NP)
*
*
***********************************************************************
*
*
      CON =  GAM - DBLE(1.0D0)
*
*
      DO 100  J = 1,NQ
      DO 100  I = 1,MQ
*
         PR  =  CON  *  ( EN(I,J)  -
     &                     DBLE(0.5D0) * ( GZ(I,J)**2 +
     &                             GR(I,J)**2   ) / RO(I,J)  )
*
         IF ( PR .LT. PRCUT )   PR  =  PRCUT
*
         CHGR(I,J) =  - PR / R(J)
*
  100 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE STAGH1
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VALUES ON THE STAGGERED GRID (1)
*     ARE COMPUTED FROM THE "OLD" VALUES FOR THE HALF TIME STEP
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RO ,EN            ( VALUES BEFORE TRANSPORT         )
*        -  GZ ,GR
*     VALUES COMPUTED:
*        -  RO1,EN1           ( VALUES AT STAGGERED GRID (1)    )
*        -  GZ1,GR1
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU2 is a dummy              a.w.
      COMMON /VAR2/   RO1(0:MP,NP), EN1(0:MP,NP),
     &                GZ1(0:MP,NP), GR1(0:MP,NP),
     &                DU2(4*MP + 4)
*
*
***********************************************************************
*     COMPUTE VALUES ON STAGGERED GRID (1)
***********************************************************************
*
*
      CALL S1 ( RO1 , RO )
      CALL S1 ( EN1 , EN )
      CALL S1 ( GZ1 , GZ )
      CALL S1 ( GR1 , GR )
*
*
***********************************************************************
*     COMPUTE BOUNDARY-VALUES ON STAGGERED GRID (1)
***********************************************************************
*
*
      CALL B1 ( RO1 , RO , 'RO' )
      CALL B1 ( EN1 , EN , 'EN' )
      CALL B1 ( GZ1 , GZ , 'GZ' )
      CALL B1 ( GR1 , GR , 'GR' )
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE STAGH2
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VALUES ON THE STAGGERED GRID (2)
*     ARE COMPUTED FROM THE "OLD" VALUES FOR THE HALF TIME STEP
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RO ,EN            ( VALUES BEFORE TRANSPORT         )
*        -  GZ ,GR
*     VALUES COMPUTED:
*        -  RO2,EN2           ( VALUES AT STAGGERED GRID (2)    )
*        -  GZ2,GR2
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   RO2(MP,0:NP), EN2(MP,0:NP),
     &                GZ2(MP,0:NP), GR2(MP,0:NP),
     &                DU4(4*NP + 4)
*
*
***********************************************************************
*     COMPUTE VALUES ON STAGGERED GRID (2)
***********************************************************************
*
*
      CALL S2 ( RO2 , RO )
      CALL S2 ( EN2 , EN )
      CALL S2 ( GZ2 , GZ )
      CALL S2 ( GR2 , GR )
*
*
***********************************************************************
*     COMPUTE BOUNDARY-VALUES ON STAGGERED GRID (2)
*     LOWER BOUNDARY:   Z-AXIS
***********************************************************************
*
*
      CALL B2 ( RO2 , RO , 'RO' )
      CALL B2 ( EN2 , EN , 'EN' )
      CALL B2 ( GZ2 , GZ , 'GZ' )
      CALL B2 ( GR2 , GR , 'GR' )
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE CORIF
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE CORIOLIS FORCES ARE COMPUTED
*     FROM THE "NEW" VALUES FOR THE FULL TIME STEP
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RON,ENN           ( VALUES AFTER HALF TIME STEP     )
*        -  GZN,GRN
*     VALUES COMPUTED:
*        -  CHGR              ( CORIOLIS FORCES                 )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
* DU2 is a dummy              a.w.
      COMMON /VAR2/   RON(MP,NP), ENN(MP,NP),
     &                GZN(MP,NP), GRN(MP,NP),
     &                DU2(4*MP + 4*NP + 4)
      COMMON /VARS/   CHGR(MP,NP)
*
*
***********************************************************************
*
*
      CON =  GAM - DBLE(1.0D0)
*
*
      DO 100  J = 1,NQ
      DO 100  I = 1,MQ
*
         PRN  =  CON  *  ( ENN(I,J)  -
     &                     DBLE(0.5D0) * ( GZN(I,J)**2 +
     &                             GRN(I,J)**2   ) / RON(I,J)  )
*
         IF ( PRN .LT. PRCUT )   PRN  =  PRCUT
*
         CHGR(I,J) =  - PRN / R(J)
*
  100 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE STAGF1
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VALUES ON THE STAGGERED GRID (1)
*     ARE COMPUTED FROM THE "NEW" VALUES FOR THE FULL TIME STEP
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RON,ENN           ( VALUES AFTER HALF TIME STEP
*        -  GZN,GRN             -> OVERWITTEN BY THE COMPUTED VALUES )
*     VALUES USED AS TEMPORARY STORAGE:
*        -  HH                ( TEMPORARY STORAGE                    )
*     VALUES COMPUTED:
*        -  RO1,EN1           ( VALUES AT STAGGERED GRID (1)         )
*        -  GZ1,GR1
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
* DU3 is a dummy             a.w.
      COMMON /VARH/   HH(0:MP,NP),
     &                DU3(3*MP*NP + 4*MP + 3*NP + 4)
*
***********************************************************************
*     SHARE COMMON /VAR2/ BETWEEN "NEW" AND "STAGGERED" VALUES
*     ATTENTION:   THE ARRAYS FOR "NEW" VALUES ARE SHORTER
*                  THAN THE ARRAYS FOR THE "STAGGERED" VALUES
*                  THUS NO MEMORY CONFLICT CAN OCCUR
*                  IF A "NEW" VALUE ISN'T USED AFTER AVERAGING
*                  AND IF THE SEQUENCE OF THE OPERATIONS ISN'T CHANGED
*                  BECAUSE THE "STAGGERED" VALUES ARE WRITTEN INTO
*                  COMMON /CONT/ "BOTTOM UP"
*                  TO PREVENT LOSS OF INFORMATION AN INTERMEDIATE
*                  ARRAY HH IS USED TO ASSEMBLE THE "STAGGERED" VALUES
***********************************************************************
*
*
      COMMON /VAR2/   H( 4 * (MP+1) * (NP+1) )
*
      DIMENSION       RO1(0:MP,NP), EN1(0:MP,NP),
     &                GZ1(0:MP,NP), GR1(0:MP,NP)
      DIMENSION       RON(MP,NP), ENN(MP,NP),
     &                GZN(MP,NP), GRN(MP,NP)
*
      EQUIVALENCE  (  H(             1 )  ,  RO1(0,1)  )    ,
     &             (  H(   (MP+1)*NP+1 )  ,  EN1(0,1)  )    ,
     &             (  H( 2*(MP+1)*NP+1 )  ,  GZ1(0,1)  )    ,
     &             (  H( 3*(MP+1)*NP+1 )  ,  GR1(0,1)  )
      EQUIVALENCE  (  H(             1 )  ,  RON(1,1)  )    ,
     &             (  H(    MP*   NP+1 )  ,  ENN(1,1)  )    ,
     &             (  H( 2* MP*   NP+1 )  ,  GZN(1,1)  )    ,
     &             (  H( 3* MP*   NP+1 )  ,  GRN(1,1)  )
*
*
***********************************************************************
*     COMPUTE INTERPOLATED VALUES OF R-MOMENTUM
***********************************************************************
*
*
      CALL S1 ( HH , GRN )
      CALL B1 ( HH , GRN , 'GR' )
*
      DO 400  J = 1,NQ
      DO 400  I = 0,MQ
         GR1(I,J) =  HH(I,J)
  400 CONTINUE
*
*
***********************************************************************
*     COMPUTE INTERPOLATED VALUES OF Z-MOMENTUM
***********************************************************************
*
*
      CALL S1 ( HH , GZN )
      CALL B1 ( HH , GZN , 'GZ' )
*
      DO 300  J = 1,NQ
      DO 300  I = 0,MQ
         GZ1(I,J) =  HH(I,J)
  300 CONTINUE
*
*
***********************************************************************
*     COMPUTE INTERPOLATED VALUES OF ENERGY
***********************************************************************
*
*
      CALL S1 ( HH , ENN )
      CALL B1 ( HH , ENN , 'EN' )
*
      DO 200  J = 1,NQ
      DO 200  I = 0,MQ
         EN1(I,J) =  HH(I,J)
  200 CONTINUE
*
*
***********************************************************************
*     COMPUTE INTERPOLATED VALUES OF DENSITY
***********************************************************************
*
*
      CALL S1 ( HH , RON )
      CALL B1 ( HH , RON , 'RO' )
*
      DO 100  J = 1,NQ
      DO 100  I = 0,MQ
         RO1(I,J) =  HH(I,J)
  100 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE STAGF2
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VALUES ON THE STAGGERED GRID (2)
*     ARE COMPUTED FROM THE "NEW" VALUES FOR THE FULL TIME STEP
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RON,ENN           ( VALUES AFTER HALF TIME STEP     )
*        -  GZN,GRN
*     VALUES COMPUTED:
*        -  RO2,EN2           ( VALUES AT STAGGERED GRID (2)    )
*        -  GZ2,GR2
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
* DU2 is a dummy              a.w.
      COMMON /VAR2/   RON(MP,NP), ENN(MP,NP),
     &                GZN(MP,NP), GRN(MP,NP),
     &                DU2(4*MP + 4*NP + 4)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   RO2(MP,0:NP), EN2(MP,0:NP),
     &                GZ2(MP,0:NP), GR2(MP,0:NP),
     &                DU4(4*NP + 4)
*
*
***********************************************************************
*     COMPUTE VALUES ON STAGGERED GRID (2)
***********************************************************************
*
*
      CALL S2 ( RO2 , RON )
      CALL S2 ( EN2 , ENN )
      CALL S2 ( GZ2 , GZN )
      CALL S2 ( GR2 , GRN )
*
*
***********************************************************************
*     COMPUTE BOUNDARY-VALUES ON STAGGERED GRID (2)
*     LOWER BOUNDARY:   Z-AXIS
***********************************************************************
*
*
      CALL B2 ( RO2 , RON , 'RO' )
      CALL B2 ( EN2 , ENN , 'EN' )
      CALL B2 ( GZ2 , GZN , 'GZ' )
      CALL B2 ( GR2 , GRN , 'GR' )
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE S1 ( U1 , U )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VALUES
*     ON THE STAGGERED GRID (1) ARE COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  U      ( VALUES AT CENTER OF CELL          )
*        -  FZ     ( INTERPOLATION WEIGHTS             )
*     VALUES COMPUTED:
*        -  U1     ( INTERPOLATED VALUES ON GRID (1)   )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      DIMENSION  U1(0:MP,NP), U(MP,NP)
*
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
*
*
***********************************************************************
*
*
      MH =  MIN0 ( MQ , M1 )
*
      DO 100  J = 1,NQ
      DO 100  I = 1,MH
         U1(I,J) =  FZ(I) * ( U(I+1,J) - U(I,J) )  +  U(I,J)
  100 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE S2 ( U2 , U )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VALUES
*     ON THE STAGGERED GRID (2) ARE COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  U      ( VALUES AT CENTER OF CELL          )
*        -  FR     ( INTERPOLATION WEIGHTS             )
*     VALUES COMPUTED:
*        -  U2     ( INTERPOLATED VALUES ON GRID (2)   )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      DIMENSION  U2(MP,0:NP), U(MP,NP)
*
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
*
*
***********************************************************************
*
*
      NH =  MIN0 ( NQ , N1 )
*
      DO 100  J = 1,NH
      DO 100  I = 1,MQ
         U2(I,J) =  FR(J) * ( U(I,J+1) - U(I,J) )  +  U(I,J)
  100 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE B1 ( U1 , U , VAR )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE BOUNDARY VALUES
*     ON THE STAGGERED GRID (1) ARE COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  ZL,ZU  ( TYPE OF BOUNDARIES            )
*        -  U      ( VALUES AT CENTER OF CELL      )
*     VALUES USED AS TEMPORARY STORAGE:
*        -  TYPX   ( TYPE OF BOUNDARY:
*                    C  ...  ZERO VALUE
*                    O  ...  ZERO DERIVATIVE       )
*                    L=LOWER & U=UPPER BOUNDARY    )
*     VALUES COMPUTED:
*        -  U1     ( BOUNDARY VALUES ON GRID (1)   )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      CHARACTER       ZL,ZU, RL,RU
      COMMON /BOND/   ZL,ZU, RL,RU
*
*
      CHARACTER*2  VAR
      CHARACTER    TYPL, TYPU
      DIMENSION    U1(0:MP,NP), U(MP,NP)
*
*
***********************************************************************
*     DETERMINE TYPE OF BOUNDARY
***********************************************************************
*
*
      IF ( VAR .EQ. 'RO'  .OR.
     &     VAR .EQ. 'EN'  .OR.
     &     VAR .EQ. 'GR'  .OR.
     &     VAR .EQ. 'GP'  .OR.
     &     VAR .EQ. 'BZ'       ) THEN
         TYPL =  'O'
         TYPU =  'O'
      ELSE
         TYPL =  ZL
         TYPU =  ZU
      ENDIF
*
*
***********************************************************************
*     LOWER BOUNDARY
***********************************************************************
*
*
      IF ( TYPL .EQ. 'C' ) THEN
*
         DO 100  J = 1,NP
	    U1(0,J) =  DBLE(0.0D0)
  100    CONTINUE
*
      ELSE
*
         DO 200  J = 1,NP
            U1(0,J) =  U(1,J)
  200    CONTINUE
*
      END IF
*
*
***********************************************************************
*     UPPER BOUNDARY
***********************************************************************
*
*
      IF ( TYPU .EQ. 'C' ) THEN
*
         DO 120  J = 1,NP
	    U1(MP,J) =  DBLE(0.0D0)
  120    CONTINUE
*
      ELSE
*
         DO 220  J = 1,NP
            U1(MP,J) =  U(MP,J)
  220    CONTINUE
*
      END IF
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE B2 ( U2 , U , VAR )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE BOUNDARY VALUES
*     ON THE STAGGERED GRID (2) ARE COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RL,RU  ( TYPE OF BOUNDARIES            )
*        -  U      ( VALUES AT CENTER OF CELL      )
*     VALUES USED AS TEMPORARY STORAGE:
*        -  TYPX   ( TYPE OF BOUNDARY:
*                    C  ...  ZERO VALUE
*                    O  ...  ZERO DERIVATIVE       )
*                    L=LOWER & U=UPPER BOUNDARY    )
*     VALUES COMPUTED:
*        -  U2     ( BOUNDARY VALUES ON GRID (2)   )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      CHARACTER       ZL,ZU, RL,RU
      COMMON /BOND/   ZL,ZU, RL,RU
*
*
      CHARACTER*2  VAR
      CHARACTER    TYPL, TYPU
      DIMENSION    U2(MP,0:NP), U(MP,NP)
*
*
***********************************************************************
*     DETERMINE TYPE OF BOUNDARY
***********************************************************************
*
*
      IF ( VAR .EQ. 'RO'  .OR.
     &     VAR .EQ. 'EN'  .OR.
     &     VAR .EQ. 'GZ'  .OR.
     &     VAR .EQ. 'BZ'       ) THEN
         TYPL =  'O'
         TYPU =  'O'
      ELSE
         TYPL =  RL
         TYPU =  RU
      ENDIF
*
*
***********************************************************************
*     LOWER BOUNDARY
***********************************************************************
*
*
      IF ( TYPL .EQ. 'C' ) THEN
*
         DO 100  I = 1,MP
	    U2(I,0) =  DBLE(0.0D0)
  100    CONTINUE
*
      ELSE
*
         DO 200  I = 1,MP
            U2(I,0) =  U(I,1)
  200    CONTINUE
*
      END IF
*
*
***********************************************************************
*     UPPER BOUNDARY
***********************************************************************
*
*
      IF ( TYPU .EQ. 'C' ) THEN
*
         DO 120  I = 1,MP
            U2(I,NP) =  0.0D0
  120    CONTINUE
*
      ELSE
*
         DO 220  I = 1,MP
            U2(I,NP) =  U(I,NP)
  220    CONTINUE
*
      END IF
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE TRANS1
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE TRANSPORT IN Z-DIRECTION IS COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  GAM, MQ,NQ
*        -  RO ,EN            ( VALUES BEFORE TRANSPORT               )
*        -  GZ ,GR
*        -  RO1,EN1           ( VALUES AT STAGGERED GRID (1)
*        -  GZ1,GR1             -> OVERWRITTEN BY THE COMPUTED VALUES )
*     VALUES USED AS TEMPORARY STORAGE:
*        -  VR1               ( VELOCITIES AT STAGGERED GRID (1)      )
*        -  PR1               ( PRESSURE   AT STAGGERED GRID (1)      )
*        -  FL1               ( FLUXES     AT STAGGERED GRID (1)      )
*     VALUES COMPUTED:
*        -  RON,ENN           ( VALUES AFTER Z-TRANSPORT              )
*        -  GZN,GRN
*        -  VZ1               ( Z-VELOCITY AT STAGGERED GRID (1)      )
*
*     REMARK:  Z-VELOCITY MUST NOT BE CHANGED (INPUT FOR FCT ROUTINES )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ1(0:MP,NP), VR1(0:MP,NP),
     &                PR1(0:MP,NP), FL1(0:MP,NP),
     &                DU3(4*MP + 4)
*
*
***********************************************************************
*     SHARE COMMON /VAR2/ BETWEEN "NEW" AND "STAGGERED" VALUES
*     ATTENTION:   THE ARRAYS FOR "STAGGERED" VALUES ARE LONGER
*                  THAN THE ARRAYS FOR THE "NEW" VALUES
*                  THUS NO MEMORY CONFLICT CAN OCCUR
*                  IF A "STAGGERED" VALUE ISN'T USED AFTER TRANSPORT
*                  AND IF THE SEQUENCE OF THE OPERATIONS ISN'T CHANGED
*                  BECAUSE THE "NEW" VALUES ARE WRITTEN INTO
*                  COMMON /CONT/ "TOP DOWN"
***********************************************************************
*
*
      COMMON /VAR2/   H( 4 * (MP+1) * (NP+1) )
*
      DIMENSION       RO1(0:MP,NP), EN1(0:MP,NP),
     &                GZ1(0:MP,NP), GR1(0:MP,NP)
      DIMENSION       RON(MP,NP), ENN(MP,NP),
     &                GZN(MP,NP), GRN(MP,NP)
*
      EQUIVALENCE  (  H(             1 )  ,  RO1(0,1)  )    ,
     &             (  H(   (MP+1)*NP+1 )  ,  EN1(0,1)  )    ,
     &             (  H( 2*(MP+1)*NP+1 )  ,  GZ1(0,1)  )    ,
     &             (  H( 3*(MP+1)*NP+1 )  ,  GR1(0,1)  )
      EQUIVALENCE  (  H(             1 )  ,  RON(1,1)  )    ,
     &             (  H(    MP*   NP+1 )  ,  ENN(1,1)  )    ,
     &             (  H( 2* MP*   NP+1 )  ,  GZN(1,1)  )    ,
     &             (  H( 3* MP*   NP+1 )  ,  GRN(1,1)  )
*
*
***********************************************************************
*     COMPUTE PRESSURE AND VELOCITIES ON STAGGERED GRID (1)
*     THESE QUANTITIES ARE USED SEVERAL TIMES!
***********************************************************************
*
*
      CON =  GAM - DBLE(1.0D0)
*
      DO 010  J = 1,NQ
      DO 010  I = 0,MQ
*
         VZ1(I,J) =  GZ1(I,J) / RO1(I,J)
         VR1(I,J) =  GR1(I,J) / RO1(I,J)
         PR1(I,J) =  CON  *  ( EN1(I,J)  -
     &                         0.5D0 * ( GZ1(I,J)**2 +
     &                                 GR1(I,J)**2   ) / RO1(I,J)  )
*
         IF ( PR1(I,J) .LT. PRCUT )   PR1(I,J) =  PRCUT
*
  010 CONTINUE
*
*
***********************************************************************
*     COMPUTE MASS-TRANSPORT
***********************************************************************
*
*
      DO 100  J = 1,NQ
      DO 100  I = 0,MQ
         FL1(I,J) =  RO1(I,J) * VZ1(I,J)
  100 CONTINUE
*
      CALL T1 ( RON , RO , FL1 )
*
*
***********************************************************************
*     COMPUTE ENERGY-TRANSPORT
***********************************************************************
*
*
      DO 200  J = 1,NQ
      DO 200  I = 0,MQ
         FL1(I,J) =  ( EN1(I,J) + PR1(I,J) )  *  VZ1(I,J)
  200 CONTINUE
*
      CALL T1 ( ENN , EN , FL1 )
*
*
***********************************************************************
*     COMPUTE MOMENTUM-TRANSPORT
***********************************************************************
*
*
*----------------------------------------------------------------------
*     1.) Z-MOMENTUM
*----------------------------------------------------------------------
*
      DO 300  J = 1,NQ
      DO 300  I = 0,MQ
         FL1(I,J) =  GZ1(I,J) * VZ1(I,J)  +  PR1(I,J)
  300 CONTINUE
*
      CALL T1 ( GZN , GZ , FL1 )
*
*----------------------------------------------------------------------
*     2.) R-MOMENTUM
*----------------------------------------------------------------------
*
      DO 400  J = 1,NQ
      DO 400  I = 0,MQ
         FL1(I,J) =  GR1(I,J) * VZ1(I,J)
  400 CONTINUE
*
      CALL T1 ( GRN , GR , FL1 )
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE TRANS2
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE TRANSPORT IN R-DIRECTION IS COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  GAM, MQ,NQ
*        -  RON,ENN           ( VALUES AFTER Z-TRANSPORT        )
*        -  GZN,GRN
*        -  RO2,EN2           ( VALUES AT STAGGERED GRID (2)    )
*        -  GZ2,GR2
*        -  CHGR              ( CORIOLIS FORCES                 )
*     VALUES USED AS TEMPORARY STORAGE:
*        -  PR2               ( PRESSURE AT STAGGERED GRID (2)  )
*        -  FL2               ( FLUXES   AT STAGGERED GRID (2)  )
*     VALUES COMPUTED:
*        -  RON,ENN           ( VALUES AFTER R-TRANSPORT        )
*        -  GZN,GRN
*        -  VR2               ( VELOCITY AT STAGGERED GRID (2)  )
*     DEFINED BUT NOT USED:
*        -  VZ1               ( VELOCITY AT STAGGERED GRID (1)  )
*
*     REMARKS: Z-VELOCITY MUST NOT BE CHANGED (INPUT FOR FCT ROUTINES )
*              R-VELOCITY MUST NOT BE CHANGED (INPUT FOR FCT ROUTINES )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
* DU2 is a dummy              a.w.
      COMMON /VAR2/   RON(MP,NP), ENN(MP,NP),
     &                GZN(MP,NP), GRN(MP,NP),
     &                DU2(4*MP + 4*NP + 4)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   RO2(MP,0:NP), EN2(MP,0:NP),
     &                GZ2(MP,0:NP), GR2(MP,0:NP),
     &                DU4(4*NP + 4)
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ1(0:MP,NP), VR2(MP,0:NP),
     &                PR2(MP,0:NP), FL2(MP,0:NP),
     &                DU3(1*MP + 3*NP + 4)
      COMMON /VARS/   CHGR(MP,NP)
*
*
***********************************************************************
*     COMPUTE PRESSURE AND R-VELOCITY ON STAGGERED GRID (2)
*     THESE QUANTITIES ARE USED SEVERAL TIMES!
***********************************************************************
*
*
      CON =  GAM - 1.0D0
*
      DO 010  J = 0,NQ
      DO 010  I = 1,MQ
*
         VR2(I,J) =  GR2(I,J) / RO2(I,J)
         PR2(I,J) =  CON  *  ( EN2(I,J)  -
     &                         0.5D0 * ( GZ2(I,J)**2 +
     &                                 GR2(I,J)**2   ) / RO2(I,J)  )
*
         IF ( PR2(I,J) .LT. PRCUT )   PR2(I,J) =  PRCUT
*
  010 CONTINUE
*
*
***********************************************************************
*     COMPUTE MASS-TRANSPORT
***********************************************************************
*
*
      DO 100  J = 0,NQ
      DO 100  I = 1,MQ
         FL2(I,J) =  RO2(I,J) * VR2(I,J)
  100 CONTINUE
*
      CALL T2 ( RON , FL2 )
*
*
***********************************************************************
*     COMPUTE ENERGY-TRANSPORT
***********************************************************************
*
*
      DO 200  J = 0,NQ
      DO 200  I = 1,MQ
         FL2(I,J) =  ( EN2(I,J) + PR2(I,J) )  *  VR2(I,J)
  200 CONTINUE
*
      CALL T2 ( ENN , FL2 )
*
*
***********************************************************************
*     COMPUTE MOMENTUM-TRANSPORT
***********************************************************************
*
*
*----------------------------------------------------------------------
*     1.) Z-MOMENTUM
*----------------------------------------------------------------------
*
      DO 300  J = 0,NQ
      DO 300  I = 1,MQ
         FL2(I,J) =  GZ2(I,J) * VR2(I,J)
  300 CONTINUE
*
      CALL T2 ( GZN , FL2 )
*
*----------------------------------------------------------------------
*     2.) R-MOMENTUM   (CORIOLIS FORCES!)
*----------------------------------------------------------------------
*
      DO 400  J = 0,NQ
      DO 400  I = 1,MQ
         FL2(I,J) =  GR2(I,J) * VR2(I,J)  +  PR2(I,J)
  400 CONTINUE
*
      CALL T2 ( GRN , FL2 )
*
      DO 410  J = 1,NQ
      DO 410  I = 1,MQ
         GRN(I,J) =  GRN(I,J)  -  DT  *  CHGR(I,J)
  410 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE T1 ( UN , U , FL )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE TRANSPORT IN Z-DIRECTION IS COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  DT     ( TIMESTEP                 )
*        -  DZ     ( GRIDSPACING              )
*        -  U      ( VALUES BEFORE TRANSPORT  )
*        -  FL     ( FLUX                     )
*     VALUES COMPUTED:
*        -  UN     ( VALUES AFTER Z-TRANSPORT )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      DIMENSION  UN(MP,NP), U(MP,NP), FL(0:MP,NP)
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
*
*
***********************************************************************
*
*
      DO 10  J = 1,NQ
      DO 10  I = 1,MQ
         UN(I,J) =  U(I,J)  -  DT / DZ(I) * ( FL(I,J) - FL(I-1,J) )
   10 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE T2 ( UN , FL )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE TRANSPORT IN R-DIRECTION IS COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  DT     ( TIMESTEP                   )
*        -  DR     ( GRIDSPACING                )
*        -  R      ( POSITION OF GRIDPOINT      )
*        -  RB     ( POSITION OF CELL-BOUNDARY  )
*        -  UN     ( VALUES AFTER Z-TRANSPORT   )
*        -  FL     ( FLUX                       )
*     VALUES COMPUTED:
*        -  UN     ( VALUES AFTER R-TRANSPORT   )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      DIMENSION  UN(MP,NP), FL(MP,0:NP)
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
*
*
***********************************************************************
*
*
      DO 10  J = 1,NQ
      DO 10  I = 1,MQ
         UN(I,J) =  UN(I,J)  -  DT  /  ( R(J) * DR(J) )  *
     &                          (  RB(J  ) * FL(I,J  )  -
     &                             RB(J-1) * FL(I,J-1)     )
   10 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE ARTDIF
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE ARTIFICIAL
*     DIFFUSION COEFFICENT IS COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  VCT
*        -  DT, DZ,DR
*        -  VZ1,VR2       ( VELOCITY AT STAGGERED GRIDS     )
*     VALUES COMPUTED:
*        -  VC1,VC2       ( AVERAGED DIFFUSION COEFFICENTS  )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ1(0:MP,NP), VR2(MP,0:NP),
     &                VC1(0:MP,NP), VC2(MP,0:NP),
     &                DU3(2*MP +2*NP + 4)
C$$$NEW BY O.HAAN, 6.4.87
      DIMENSION DRIN(NP),DZIN(MP)
C$$$NEW
C$$$ MOVE VCT*DT/DZ(I) AND VCT*DT/DR(J) OUT OF LOOPS
C$$$
*
*
***********************************************************************
*     COMPUTE Z-COMPONENT
***********************************************************************
*
*
C$$$
      DO 20 I=1,MQ1
      DZIN(I)=VCT*DT/DZ(I)
   20 CONTINUE
      DO 40 J=1,NQ1
      DRIN(J)=VCT*DT/DR(J)
   40 CONTINUE
C$$$
      DO  100  J = 1,NQ
      DO  100  I = 1,MQ1
         VC1(I,J) =        DZIN(I) *
     &                  (  ABS( VZ1(I+1,J) - VZ1(I-1,J) )     +
     &                     0.5D0 * ABS(   VR2(I+1,J) + VR2(I+1,J-1)
     &                                - VR2(I  ,J) - VR2(I  ,J-1) )  )
  100 CONTINUE
*
*
***********************************************************************
*     COMPUTE R-COMPONENT
***********************************************************************
*
*
      DO  200  J = 1,NQ1
      DO  200  I = 1,MQ
         VC2(I,J) =        DRIN(J)    *
     &                  (  ABS( VR2(I,J+1) - VR2(I,J-1) )     +
     &                     0.5D0 * ABS(   VZ1(I,J+1) + VZ1(I-1,J+1)
     &                                - VZ1(I,J  ) - VZ1(I-1,J  ) )  )
  200 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE FCT ( UNEW, UTRA, UOLD )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE FLUX-CORRECTION-PROCESS IS COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  DT
*        -  DZ,DR, FZ,FR, R,RB, DBZ,DBR
*        -  UOLD          ( OLD- VALUES OF VARIABLE         )
*        -  UTRA          ( TRANSPORTED VALUES OF VARIABLE  )
*        -  VZ1,VR2       ( VELOCITY AT STAGGERED GRIDS     )
*        -  VC            ( DIFFUSION COEFFICENTS           )
*     VALUES USED AS TEMPORARY STORAGE:
*        -  DK,AK, VK     ( DIFFUSION COEFFICENTS           )
*        -  AZ1,AR2       ( RAW AND CORRECTED
*                           ANTIDIFFUSION FLUXES            )
*        -  DZ1,DR2       ( DIFFUSION FLUXES                )
*        -  UTRA          ( TD- VALUES OF VARIABLES         )
*     VALUES COMPUTED:
*        -  UNEW          ( VALUES OF VARIABLE AT NEW STEP  )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      DIMENSION  UNEW(MP,NP), UTRA(MP,NP), UOLD(MP,NP)
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ1(0:MP,NP), VR2(MP,0:NP),
     &                VC1(0:MP,NP), VC2(MP,0:NP),
     &                DU3(2*MP +2*NP + 4)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   AZ1(0:MP,NP), AR2(MP,0:NP),
     &                DZ1(0:MP,NP), DR2(MP,0:NP),
     &                DU4(2*MP +2*NP + 4)
C$$$NEW BY O.HAAN, 6.4.87
      DIMENSION DRIN(NP),DZIN(MP)
C$$$NEW
C$$$ MOVE 1/DZ(I) AND 1/(DR(J)*R(J)) OUT OF LOOPS
C$$$
*
*
***********************************************************************
*     COMPUTE DIFFUSION COEFFICENTS AS STATEMENT FUNCTIONS
***********************************************************************
*
*
*----------------------------------------------------------------------
*     DEFINE DIFFUSION COEFFICENTS AS STATEMENT FUNCTIONS
*----------------------------------------------------------------------
*
      DC(E,V) =  DBLE(0.125D0)
      AC(E,V) =  DBLE(0.125D0)
*
*----------------------------------------------------------------------
*     COMPUTE ARTIFICIAL DIFFUSION COEFFICENTS
*----------------------------------------------------------------------
*
      CALL ARTDIF
*
*
***********************************************************************
*     COMPUTE DIFFUSION AND RAW ANTIDIFFUSION FLUXES
***********************************************************************
*
*
*----------------------------------------------------------------------
*     COMPUTE FLUXES IN Z-DIRECTION
*----------------------------------------------------------------------
*
      DO 100  J = 1,NQ
      DO 100  I = 1,MQ1
*
         EK =  DT / DZ(I)
         DK =  DC( EK , VZ1(I,J) )
         AK =  AC( EK , VZ1(I,J) )
*
         DZ1(I,J) =  ( DK + VC1(I,J) )  *  ( UOLD(I+1,J) - UOLD(I,J) )
         AZ1(I,J) =    AK               *  ( UTRA(I+1,J) - UTRA(I,J) )
*
  100 CONTINUE
*
*----------------------------------------------------------------------
*     COMPUTE FLUXES IN R-DIRECTION
*----------------------------------------------------------------------
*
      DO 110  J = 1,NQ1
      DO 110  I = 1,MQ
*
         EK =  DT / DR(J)
         DK =  DC( EK , VR2(I,J) )
         AK =  AC( EK , VR2(I,J) )
*
         DR2(I,J) =  ( DK + VC2(I,J) )  *  ( UOLD(I,J+1) - UOLD(I,J) )
         AR2(I,J) =    AK               *  ( UTRA(I,J+1) - UTRA(I,J) )
*
  110 CONTINUE
*
*----------------------------------------------------------------------
*     SET FLUXES ACROSS BOUNDARY TO ZERO
*----------------------------------------------------------------------
*
      DO 120  J = 1,NQ
         DZ1( 0,J) =  0.0D0
         DZ1(MQ,J) =  0.0D0
         AZ1( 0,J) =  0.0D0
         AZ1(MQ,J) =  0.0D0
  120 CONTINUE
*
      DO 130  I = 1,MQ
         DR2(I, 0) =  0.0D0
         DR2(I,NQ) =  0.0D0
         AR2(I, 0) =  0.0D0
         AR2(I,NQ) =  0.0D0
  130 CONTINUE
*
*
***********************************************************************
*     COMPUTE DIFFUSION
***********************************************************************
*
*
C$$$
      DO 20 I=1,MQ
      DZIN(I)=1.0D0/DZ(I)
   20 CONTINUE
      DO 40 J=1,NQ
      DRIN(J)=1.0D0/(DR(J)*R(J))
   40 CONTINUE
C$$$
      DO 200  J = 1,NQ
      DO 200  I = 1,MQ
         UTRA(I,J) =  UTRA(I,J)
     &                   +   ( DBZ(I  ) * DZ1(I  ,J)  -
     &                         DBZ(I-1) * DZ1(I-1,J)     )
     &                            *  DZIN(I)
     &                   +   ( DBR(J  ) * RB(J  ) * DR2(I,J  )  -
     &                         DBR(J-1) * RB(J-1) * DR2(I,J-1)     )
     &                            *  DRIN(J)
  200 CONTINUE
*
*
***********************************************************************
*     LIMIT ANTIDIFFUSION FLUXES
***********************************************************************
*
*
      CALL FILTER ( UTRA , UOLD )
*
*
***********************************************************************
*     COMPUTE ANTIDIFFUSION
***********************************************************************
*
*
      DO 400  J = 1,NQ
      DO 400  I = 1,MQ
         UNEW(I,J) =  UTRA(I,J)
     &                   -   ( DBZ(I  ) * AZ1(I  ,J)  -
     &                         DBZ(I-1) * AZ1(I-1,J)     )
     &                            *  DZIN(I)
     &                   -   ( DBR(J  ) * RB(J  ) * AR2(I,J  )  -
     &                         DBR(J-1) * RB(J-1) * AR2(I,J-1)     )
     &                            *  DRIN(J)
  400 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE FILTER ( UTDF , UOLD )
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE ANTI-DIFFUSION FLUXES ARE LIMITED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  AZ1,AR2       ( RAW ANTI-DIFFUSION FLUXES       )
*        -  UOLD,UTDF     ( OLD/TD- VALUES OF VARIABLE      )
*     VALUES USED AS TEMPORARY STORAGE:
*        -  TZ1,TR2       ( DIFFERENCES OF TD-VALUES        )
*        -  XMAX,XMIN     ( MAX/MIN OF OLD & TD-VALUES      )
*        -  VMAX,VMIN     ( MAX/MIN VALUE USED IN FILTER    )
*        -  PP,PM         ( SUM OF FLUXES INTO/OUT OF CELL  )
*        -  QP,QM         ( MAX/MIN DIFFERENCES OF VALUES   )
*        -  RP,RM         ( FILTER- FUNCTION                )
*     VALUES COMPUTED:
*        -  AZ1,AR2       ( CORRECTED ANTI-DIFFUSION FLUXES )
*     DEFINED BUT NOT USED:
*        -  VZ1,VR2       ( VELOCITIES AT STAGGERED GRIDS   )
*
*     REMARKS: Z-VELOCITY MUST NOT BE CHANGED (INPUT FOR FCT ROUTINES )
*              R-VELOCITY MUST NOT BE CHANGED (INPUT FOR FCT ROUTINES )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      DIMENSION  UOLD(MP,NP), UTDF(MP,NP)
*
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
* DU4 is a dummy               a.w.
      COMMON /SCRA/   AZ1(0:MP,NP), AR2(MP,0:NP),
     &                XMAX (MP,NP), XMIN (MP,NP),
     &                DU4(3*MP + 3*NP + 4)
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ1(0:MP,NP), VR2(MP,0:NP),
     &                VMAX (MP,NP), VMIN (MP,NP),
     &                DU3(3*MP + 3*NP + 4)
*
*
***********************************************************************
*     SHARE PART OF COMMON /SCRA/ BETWEEN SCRATCH-VECTORS
*     REMARK:   NO MEMORY CONFLICT POSSIBLE
***********************************************************************
*
*
      DIMENSION  TZ1(MP,NP), TR2(MP,NP)
      DIMENSION  RP (MP,NP), RM (MP,NP)
*
      EQUIVALENCE  (  XMAX , RP , TZ1 , TR2 )
      EQUIVALENCE  (  XMIN , RM )
*
*
***********************************************************************
*     COMPUTE DIFFERENCES OF TD-VALUES AND
*     ELIMINATE FLUXES IN THE WRONG DIRECTION
***********************************************************************
*
*
*----------------------------------------------------------------------
*     1.) Z-DIRECTION
*----------------------------------------------------------------------
*
      DO 100  J = 1,NQ
      DO 100  I = 1,MQ1
         TZ1(I,J) =  UTDF(I+1,J) - UTDF(I,J)
  100 CONTINUE
*
      DO 110  J = 1,NQ
      DO 110  I = 2,MQ2
         IF ( AZ1(I,J) * TZ1(I-1,J) .LT. 0.0D0   .OR.
     &        AZ1(I,J) * TZ1(I  ,J) .LT. 0.0D0   .OR.
     &        AZ1(I,J) * TZ1(I+1,J) .LT. 0.0D0          )
     &   AZ1(I,J) = 0.0D0
  110 CONTINUE
*
      DO 112  J = 1,NQ
         IF ( AZ1(1,J) * TZ1(1,J) .LT. 0.0D0   .OR.
     &        AZ1(1,J) * TZ1(2,J) .LT. 0.0D0          )
     &   AZ1( 1,J) = 0.0D0
         IF ( AZ1(MQ1,J) * TZ1(MQ1,J) .LT. 0.0D0   .OR.
     &        AZ1(MQ1,J) * TZ1(MQ2,J) .LT. 0.0D0          )
     &   AZ1(MQ1,J) = 0.0D0
  112 CONTINUE
*
*----------------------------------------------------------------------
*     2.) R-DIRECTION
*----------------------------------------------------------------------
*
      DO 150  J = 1,NQ1
      DO 150  I = 1,MQ
         TR2(I,J) =  UTDF(I,J+1) - UTDF(I,J)
  150 CONTINUE
*
      DO 160  J = 2,NQ2
      DO 160  I = 1,MQ
         IF ( AR2(I,J) * TR2(I,J-1) .LT. 0.0D0   .OR.
     &        AR2(I,J) * TR2(I,J  ) .LT. 0.0D0   .OR.
     &        AR2(I,J) * TR2(I,J+1) .LT. 0.0D0          )
     &   AR2(I,J) = 0.0D0
  160 CONTINUE
*
      DO 162  I = 1,MQ
         IF ( AR2(I,1) * TR2(I,1) .LT. 0.0D0   .OR.
     &        AR2(I,1) * TR2(I,2) .LT. 0.0D0          )
     &   AR2(I, 1) = 0.0D0
         IF ( AR2(I,NQ1) * TR2(I,NQ1) .LT. 0.0D0   .OR.
     &        AR2(I,NQ1) * TR2(I,NQ2) .LT. 0.0D0          )
     &   AR2(I,NQ1) = 0.0D0
  162 CONTINUE
*
*
***********************************************************************
*     COMPUTE MAXIMUM VALUE OF VARIABLE TO USE IN FILTER
***********************************************************************
*
*
*----------------------------------------------------------------------
*     1.) COMPUTE MAX/MIN OF OLD AND TD VALUE
*----------------------------------------------------------------------
*
      DO 200  J = 1,NQ
      DO 200  I = 1,MQ
	 XMAX(I,J) =  DMAX1( UOLD(I,J) , UTDF(I,J) )
	 XMIN(I,J) =  DMIN1( UOLD(I,J) , UTDF(I,J) )
  200 CONTINUE
*
*----------------------------------------------------------------------
*     2.) COMPUTE MAX/MIN VALUE OF VARIABLE FOR FILTER
*         A.) Z-SWEEP & BOUNDARY
*----------------------------------------------------------------------
*
      DO 220  J = 1,NQ
      DO 220  I = 2,MQ1
	 VMAX(I,J) =  DMAX1( XMAX(I,J)   ,
     &                       XMAX(I+1,J) , XMAX(I-1,J)  )
	 VMIN(I,J) =  DMIN1( XMIN(I,J)   ,
     &                       XMIN(I+1,J) , XMIN(I-1,J)  )
  220 CONTINUE
*
      DO 222  J = 1,NQ
	 VMAX( 1,J) =  DMAX1( XMAX(  1,J) , XMAX( 2,J) )
	 VMAX(MQ,J) =  DMAX1( XMAX(MQ1,J) , XMAX(MQ,J) )
	 VMIN( 1,J) =  DMIN1( XMIN(  1,J) , XMIN( 2,J) )
	 VMIN(MQ,J) =  DMIN1( XMIN(MQ1,J) , XMIN(MQ,J) )
  222 CONTINUE
*
*----------------------------------------------------------------------
*     2.) COMPUTE MAX/MIN VALUE OF VARIABLE FOR FILTER
*         B.) R-SWEEP & BOUNDARY
*----------------------------------------------------------------------
*
      DO 240  J = 2,NQ1
      DO 240  I = 1,MQ
	 VMAX(I,J) =  DMAX1( VMAX(I,J)   ,
     &                       XMAX(I,J+1) , XMAX(I,J-1)  )
	 VMIN(I,J) =  DMIN1( VMIN(I,J)   ,
     &                       XMIN(I,J+1) , XMIN(I,J-1)  )
  240 CONTINUE
*
      DO 242  I = 1,MQ
	 VMAX(I, 1) =  DMAX1( VMAX(I,  1) ,
     &                        XMAX(I,  1) , XMAX(I, 2)  )
	 VMAX(I,NQ) =  DMAX1( VMAX(I, NQ) ,
     &                        XMAX(I,NQ1) , XMAX(I,NQ)  )
	 VMIN(I, 1) =  DMIN1( VMIN(I,  1) ,
     &                        XMIN(I,  1) , XMIN(I, 2)  )
	 VMIN(I,NQ) =  DMIN1( VMIN(I, NQ) ,
     &                        XMIN(I,NQ1) , XMIN(I,NQ)  )
  242 CONTINUE
*
*
***********************************************************************
*     COMPUTE POSITIVE AND NEGITVE FILTER FUNCTION
***********************************************************************
*
*
      DO 300  J = 1,NQ
      DO 300  I = 1,MQ
	 PP =     DMAX1( DBLE(0.0D0) , AZ1(I-1,J) )
     &         +  DMAX1( DBLE(0.0D0) , AR2(I,J-1) )
     &         -  DMIN1( DBLE(0.0D0) , AZ1(I,J)   )
     &         -  DMIN1( DBLE(0.0D0) , AR2(I,J)   )
	 PM =     DMAX1( DBLE(0.0D0) , AZ1(I,J)   )
     &         +  DMAX1( DBLE(0.0D0) , AR2(I,J)   )
     &         -  DMIN1( DBLE(0.0D0) , AZ1(I-1,J) )
     &         -  DMIN1( DBLE(0.0D0) , AR2(I,J-1) )
         QP =  VMAX(I,J) - UTDF(I,J)
         QM =  UTDF(I,J) - VMIN(I,J)
         XP = PP
         XM = PM
            IF ( XP .EQ. 0.0D0 )   PP = 1.0D0
            IF ( XM .EQ. 0.0D0 )   PM = 1.0D0
	 RP(I,J) =  DMIN1( DBLE(1.0D0) , QP / PP )
	 RM(I,J) =  DMIN1( DBLE(1.0D0) , QM / PM )
            IF ( XP .EQ. 0.0D0 )   RP(I,J) = 0.0D0
            IF ( XM .EQ. 0.0D0 )   RM(I,J) = 0.0D0
  300 CONTINUE
*
*
***********************************************************************
*     LIMIT FLUXES
***********************************************************************
*
*
*----------------------------------------------------------------------
*     1.) Z-DIRECTION
*----------------------------------------------------------------------
*
      DO 400  J = 1,NQ
      DO 400  I = 1,MQ1
	    C =  DMIN1( RP(I+1,J) , RM(I,J) )
         IF ( AZ1(I,J) .LT. 0.0D0 )
     &      C =  DMIN1( RP(I,J) , RM(I+1,J) )
         AZ1(I,J) =  C * AZ1(I,J)
  400 CONTINUE
*
*----------------------------------------------------------------------
*     2.) R-DIRECTION
*----------------------------------------------------------------------
*
      DO 420  J = 1,NQ1
      DO 420  I = 1,MQ
	    C =  DMIN1( RP(I,J+1) , RM(I,J) )
         IF ( AR2(I,J) .LT. 0.0D0 )
     &      C =  DMIN1( RP(I,J) , RM(I,J+1) )
         AR2(I,J) =  C * AR2(I,J)
  420 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE CHECK
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE SOME IMPORTANT CHECKS ARE PERFORMED
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /DISK/   ISTEP, ICNT,
     &                ISTOR, IPRIN, MPROW, NPROW
*
*
***********************************************************************
*     COMPUTE MAXIMUM, MINIMUM AND FLOOR VALUES OF VARIABLES
*     COMPUTE PRESSURE, VELOCITY AND FLOOR OF PRESSURE
***********************************************************************
*
*
      CALL CUT
      CALL VPR
*
*
***********************************************************************
*     CHECK TIMESTEP AND RETURN IF NOTHING IS PRINTED OR STORED
***********************************************************************
*
*
      IF ( .NOT. ( MOD(IS,ISTOR) .EQ. 0   .OR.
     &             MOD(IS,IPRIN) .EQ. 0         )  )   RETURN
*
*
***********************************************************************
*     COMPUTE PRESSURE, VELOCITY AND FLOOR OF PRESSURE,
*     AND CHECK THE CONSERVATITION PROPERITES OF THE EQUATIONS
***********************************************************************
*
*
      CALL VPS
      CALL CON
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE CUT
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE EXTREMAL VALUES OF THE VARIABLES
*     ARE COMPUTED AND VERY SMALL VALUES ARE CUT OFF
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  RO,EN, GZ,GR
*     VALUES COMPUTED:
*        -  RO,EN, GZ,GR
*        -  ROMIN,ENMIN         ( AND POSITION )
*        -  GZMIN,GRMIN         ( AND POSITION )
*        -  ROMAX,ENMAX         ( AND POSITION )
*        -  GZMAX,GRMAX         ( AND POSITION )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( MN   =   MP * NP )
*
*     Original values for the next 4 constants: 1.0D-14
*     Changed 10/18/91, Reinhold Weicker
      PARAMETER  ( ROFLOR  =    1.0D-7 )
      PARAMETER  ( ENFLOR  =    1.0D-7 )
      PARAMETER  ( GZFLOR  =    1.0D-7 )
      PARAMETER  ( GRFLOR  =    1.0D-7 )
*
*
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CONT/   ENMAX, ROMAX, ENMIN, ROMIN,
     &                GRMAX, GZMAX, GRMIN, GZMIN,
     &                TOTMA, TOTEN, TOTGZ,
     &                MINRO, MINEN, MAXRO, MAXEN,
     &                MINGZ, MINGR, MAXGZ, MAXGR
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   ROH(MN), ENH(MN),
     &                GZH(MN), GRH(MN),
     &                DU4(4*MP + 4*NP +4)
*
*****************************************************************
*   Declaration of variables.
*   This is done additionally to the IMPLICIT declaration,
*   because one of the compilers reported strange
*   error messages.
*   We have no idea, what difference an explicit declaration makes,
*   but it seems to work.
*                                   m.s. 07/22/91
*****************************************************************
* integer
       INTEGER*4 MQ, MQ1, MQ2, NQ, NQ1, NQ2, MQFLG, NQFLG,
     &           MINRO, MINEN, MAXRO, MAXEN,
     &           MINGZ, MINGR, MAXGZ, MAXGR
*
* double precision
       DOUBLE PRECISION ROBMQ, ROBNQ, ENMAX, ROMAX, ENMIN, ROMIN,
     &                  GRMAX, GZMAX, GRMIN, GZMIN,
     &                  TOTMA, TOTEN, TOTGZ, RO, EN, GZ, GR,
     &                  ROH, ENH, GZH, GRH
****************************************************************
*
***********************************************************************
*     PACK VARAIABLES INTO SCRATCH FOR SEARCH
***********************************************************************
*
*
      MNQ =  MQ * NQ
*
      K = 0
      DO  100  J = 1,NQ
      DO  100  I = 1,MQ
         K =  K + 1
         ROH(K) =  RO(I,J)
         ENH(K) =  EN(I,J)
         GZH(K) =  GZ(I,J)
         GRH(K) =  GR(I,J)
  100 CONTINUE
*
*
***********************************************************************
*     COMPUTE MINIMUM VALUES OF VARIABLES
***********************************************************************
*
*
      MINRO =  ISMIN ( MNQ , ROH , 1 )
      MINEN =  ISMIN ( MNQ , ENH , 1 )
      MINGZ =  ISMIN ( MNQ , GZH , 1 )
      MINGR =  ISMIN ( MNQ , GRH , 1 )
*
      ROMIN =  ROH( MINRO )
      ENMIN =  ENH( MINEN )
      GZMIN =  GZH( MINGZ )
      GRMIN =  GRH( MINGR )
*
*
***********************************************************************
*     COMPUTE MAXIMUM VALUES OF VARIABLES
***********************************************************************
*
*
      MAXRO =  I0MAX ( MNQ , ROH , 1 )
      MAXEN =  I0MAX ( MNQ , ENH , 1 )
      MAXGZ =  I0MAX ( MNQ , GZH , 1 )
      MAXGR =  I0MAX ( MNQ , GRH , 1 )
*
      ROMAX =  ROH( MAXRO )
      ENMAX =  ENH( MAXEN )
      GZMAX =  GZH( MAXGZ )
      GRMAX =  GRH( MAXGR )
*
*
***********************************************************************
*     CUT OFF VERY SMALL VALUES
***********************************************************************
*
*
      ROCUT =  ROFLOR * ROMAX
      ENCUT =  ENFLOR * ENMAX
      GZCUT =  GZFLOR * DMAX1 ( DABS(GZMIN) , DABS(GZMAX) )
      GRCUT =  GRFLOR * DMAX1 ( DABS(GRMIN) , DABS(GRMAX) )
*
      GTCUT =  DMAX1 ( GZCUT , GRCUT )
*
*
      DO 200  J = 1,NQ
      DO 200  I = 1,MQ
         IF ( RO(I,J)      .LT. ROCUT )   RO(I,J) = ROCUT
         IF ( EN(I,J)      .LT. ENCUT )   EN(I,J) = ENCUT
         IF ( ABS(GZ(I,J)) .LT. GTCUT )   GZ(I,J) = 0.0D0
         IF ( ABS(GR(I,J)) .LT. GTCUT )   GR(I,J) = 0.0D0
  200 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE VPR
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VELOCITIES, THE PRESSURE
*     AND THE FLOOR OF THE PRESSURE IS COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  GAM
*        -  RO,EN, GZ,GR
*     VALUES USED AS TEMPORARY STORAGE:
*        -  ID,PH
*     VALUES COMPUTED:
*        -  PR, VZ,VR
*        -  NEGPR,NEGPO,PRCUT
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( MN   =   MP * NP )
*
*     Original value for the next constant: 1.0D-14
*     Changed 10/18/91, Reinhold Weicker
      PARAMETER  ( PRFLOR  =    1.0D-7 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ(MP,NP), VR(MP,NP),
     &                PR(MP,NP),
     &                DU3(1*MP*NP + 4*MP + 4*NP + 4)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   ID(MN), PH(MN),
     &                DU4(2*MP*NP + MP*NP/2 + 4*MP + 4*NP + 4)
* MP*NP/2 because integer array ID takes only half the space
* of an array of REAL*8 data
*
***********************************************************************
*     COMPUTE VELOCITIES AND THERMAL PRESSURE
***********************************************************************
*
*
      CON =  GAM - DBLE(1.0D0)
*
      DO 100  J = 1,NQ
      DO 100  I = 1,MQ
*
         VZ(I,J) =  GZ(I,J) / RO(I,J)
         VR(I,J) =  GR(I,J) / RO(I,J)
*
         PR(I,J) =  CON  *  ( EN(I,J)  -
     &                        0.5D0 * ( GZ(I,J)**2 +
     &                                GR(I,J)**2   ) / RO(I,J)  )
*
  100 CONTINUE
*
*
***********************************************************************
*     COUNT NEGATIVE VALUES OF THERMAL PRESSURE
*     AND STORE POSITION OF NEGATIVE VALUES
***********************************************************************
*
*
      K = 0
      DO  200  J = 1,NQ
      DO  200  I = 1,MQ
         K =  K + 1
         PH(K) =  PR(I,J)
  200 CONTINUE
*
      CALL WNFLE ( MQ*NQ , PH , 1 , DBLE(0.0D0) , ID , NEGPR )
*
      IF ( NEGPR .GE. 1  .AND.  NEGPR .LE. NEGCN ) THEN
         DO 300  I = 1,NEGPR
            NEGPO(I) =  ID(I)
  300    CONTINUE
      END IF
*
*
***********************************************************************
*     COMPUTE FLOOR OF THERMAL PRESSURE,
*     AND CUT OFF SMALL OR NEGATIVE VALUES
***********************************************************************
*
*
      MAXPR =  I0MAX ( MQ*NQ , PH , 1 )
      PRCUT =  PRFLOR * PH(MAXPR)
*
      DO 400  J = 1,NQ
      DO 400  I = 1,MQ
         IF ( PR(I,J) .LT. PRCUT )   PR(I,J) = PRCUT
  400 CONTINUE

*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE VPS
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE VELOCITIES, THE PRESSURE
*     AND THE FLOOR OF THE PRESSURE IS COMPUTED
*     FOR ALL GRIDPOINTS FOR THE OUTPUT ROUTINE
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  GAM
*        -  RO,EN, GZ,GR
*     VALUES USED AS TEMPORARY STORAGE:
*        -  ID
*     VALUES COMPUTED:
*        -  PR, VZ,VR
*        -  NEGPR,NEGPO,PRCUT
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( MN   =   MP * NP )
*
*     Original value for the next constant: 1.0D-14
*     Changed 10/18/91, Reinhold Weicker
      PARAMETER  ( PRFLOR  =    1.0D-7 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ(MP,NP), VR(MP,NP),
     &                PR(MP,NP),
     &                DU3(1*MP*NP + 4*MP + 4*NP + 4)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   ID(MN),
     &                DU4(3*MP*NP + MP*NP/2 + 4*MP + 4*NP + 4)
* MP*NP/2 because integer array ID takes only half the space
* of an array of REAL*8 data
*
***********************************************************************
*     COMPUTE VELOCITIES AND THERMAL PRESSURE
***********************************************************************
*
*
      CON =  GAM - 1.0D0
*
      DO 100  J = 1,NP
      DO 100  I = 1,MP
*
         VZ(I,J) =  GZ(I,J) / RO(I,J)
         VR(I,J) =  GR(I,J) / RO(I,J)
*
         PR(I,J) =  CON  *  ( EN(I,J)  -
     &                        0.5D0 * ( GZ(I,J)**2 +
     &                                GR(I,J)**2   ) / RO(I,J)  )
*
  100 CONTINUE
*
*
***********************************************************************
*     COUNT NEGATIVE VALUES OF THERMAL PRESSURE
*     AND STORE POSITION OF NEGATIVE VALUES
***********************************************************************
*
*
      CALL WNFLE ( MN , PR , 1 ,DBLE(0.0D0) , ID , NEGPR )
*
      IF ( NEGPR .GE. 1  .AND.  NEGPR .LE. NEGCN ) THEN
         DO 200  I = 1,NEGPR
            NEGPO(I) =  ID(I)
  200    CONTINUE
      END IF
*
*
***********************************************************************
*     COMPUTE FLOOR OF THERMAL PRESSURE,
*     AND CUT OFF SMALL OR NEGATIVE VALUES
***********************************************************************
*
*
      MAXPR =  I0MAX ( MN , PR , 1 )
      M01   =  MOD(MAXPR-1,MP) + 1
      N01   =  (MAXPR-1)/MP + 1
CSTE  PRCUT =  PRFLOR * PR(MAXPR,1)
      PRCUT =  PRFLOR * PR(M01,N01)
*
      DO 300  J = 1,NP
      DO 300  I = 1,MP
         IF ( PR(I,J) .LT. PRCUT )   PR(I,J) = PRCUT
  300 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE CON
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE THE CONTROL VARIABLES ARE COMPUTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  DZ,R,DR
*        -  RO,EN, GZ,GR
*     VALUES USED AS TEMPORARY STORAGE:
*        -  DF
*     VALUES COMPUTED:
*        -  TOTMA,TOTEN,TOTGZ
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
      PARAMETER  ( MN   =   MP * NP )
*
      COMMON /CONT/   ENMAX, ROMAX, ENMIN, ROMIN,
     &                GRMAX, GZMAX, GRMIN, GZMIN,
     &                TOTMA, TOTEN, TOTGZ,
     &                MINRO, MINEN, MAXRO, MAXEN,
     &                MINGZ, MINGR, MAXGZ, MAXGR
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU4 is a dummy               a.w.
      COMMON /SCRA/   DF(MP,NP),
     &                DU4(3*MP*NP + 4*MP + 4*NP + 4)
*
***********************************************************************
*     COMPUTE TOTAL MASS, ENERGY AND Z-MOMENTUM
***********************************************************************
*
*
      DO 100  J = 1,NP
         RDR =  R(J) * DR(J)
         DO 120  I = 1,MP
            DF(I,J) =  RDR * DZ(I)
  120    CONTINUE
  100 CONTINUE
*
      TOTMA =  SDOT( MN, DF, 1, RO , 1 )
      TOTEN =  SDOT( MN, DF, 1, EN , 1 )
      TOTGZ =  SDOT( MN, DF, 1, GZ , 1 )
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE OUTPUT
*
*
***********************************************************************
*
*     IN THIS SUBROUTINE A MODEL IS WRITEN ON DISC
*     AND / OR ASSORTED DATA IS PRINTED
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      PARAMETER  ( M1   =   MP - 1 ,   N1   =   NP - 1 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /DISK/   ISTEP, ICNT,
     &                ISTOR, IPRIN, MPROW, NPROW
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /CUTP/   PRCUT, NEGFL, NEGCN, NEGPR, NEGPO(1000)
      COMMON /CONT/   ENMAX, ROMAX, ENMIN, ROMIN,
     &                GRMAX, GZMAX, GRMIN, GZMIN,
     &                TOTMA, TOTEN, TOTGZ,
     &                MINRO, MINEN, MAXRO, MAXEN,
     &                MINGZ, MINGR, MAXGZ, MAXGR
      COMMON /GRID/   Z(MP), ZB(0:MP), DZ(MP), DBZ(0:MP), FZ(M1),
     &                R(NP), RB(0:NP), DR(NP), DBR(0:NP), FR(N1),
     &                DUM(8)
* Dum is a dummy             a.w.
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
* DU3 is a dummy             a.w.
      COMMON /VARH/   VZ(MP,NP), VR(MP,NP),
     &                PR(MP,NP),
     &                DU3(1*MP*NP + 4*MP + 4*NP + 4)
*
*
***********************************************************************
*     PRINT INFORMATION ABOUT NEGATIVE PRESSURE
***********************************************************************
*
*
      IF ( MOD(IS-1,IPRIN) .EQ. 0 ) THEN
         IFLAG = 0
      ENDIF
*
      IF ( NEGPR .GT. 0 ) THEN
*
         IFLAG = 1
*
         IF ( NEGPR .GT. NEGCN ) THEN
	    WRITE(10,500) IS, NEGPR, NEGCN
  500       FORMAT (1H1/////6X,'TOO MANY CELLS WITH NEGATIVE PRESSURE:'
     &                     /6X,'AT TIMESTEP NUMBER: ',I6
     &                     /6X,'NEGPR  =  ',I6
     &                     /6X,'NEGCN  =  ',I6
     &                     /6X,'STOP IN ROUTINE:    OUTPUT'            )
            STOP
         ENDIF
*
         IF ( NEGFL .EQ. 0 ) THEN
	    WRITE(10,510) IS, NEGPR
  510       FORMAT(4X,'TIMESTEP: ',I6,4X,I4,'  CELLS WITH P<0')
         ELSE
            IF ( MOD(IS,IPRIN) .EQ. 0 ) THEN
               MH = MP
            ELSE
               MH = MQ
            ENDIF
            DO 50  I = 1,NEGPR
               JNEG =  (NEGPO(I)-1) / MH  +  1
               INEG =  NEGPO(I)  -  MH * (JNEG-1)
               IF ( I .EQ. 1 ) THEN
		  WRITE(10,520) IS, NEGPR, I, INEG, JNEG
  520             FORMAT (4X,' '/
     &                    4X,'TIMESTEP: ',I6,4X,I4,'  CELLS WITH P<0',
     &                    6X,I2,4X,'I=',I4,4X,'J=',I4                 )
               ELSE
		  WRITE(10,530) I, INEG, JNEG
  530             FORMAT (50X,I2,4X,'I=',I4,4X,'J=',I4)
               ENDIF
   50       CONTINUE
         ENDIF
*
      ENDIF
*
*
***********************************************************************
*     CHECK TIMESTEP AND STORE MODEL ON DISK
***********************************************************************
*
*
      IF ( MOD(IS,ISTOR) .NE. 0 )   GOTO 100
*
*
*
*
***********************************************************************
*     CHECK TIMESTEP,
*     COMPUTE POSITION OF CONTROL VARIABLES
*     AND PRINT CONTROL VARIABLES
***********************************************************************
*
*
  100 IF ( MOD(IS,IPRIN) .NE. 0 )   RETURN
*
*
      JMIRO =  (MINRO-1) / MQ  +  1
      JMIEN =  (MINEN-1) / MQ  +  1
      JMIGZ =  (MINGZ-1) / MQ  +  1
      JMIGR =  (MINGR-1) / MQ  +  1
*
      JMARO =  (MAXRO-1) / MQ  +  1
      JMAEN =  (MAXEN-1) / MQ  +  1
      JMAGZ =  (MAXGZ-1) / MQ  +  1
      JMAGR =  (MAXGR-1) / MQ  +  1
*
*
      IMIRO =  MINRO  -  MQ * (JMIRO-1)
      IMIEN =  MINEN  -  MQ * (JMIEN-1)
      IMIGZ =  MINGZ  -  MQ * (JMIGZ-1)
      IMIGR =  MINGR  -  MQ * (JMIGR-1)
*
      IMARO =  MAXRO  -  MQ * (JMARO-1)
      IMAEN =  MAXEN  -  MQ * (JMAEN-1)
      IMAGZ =  MAXGZ  -  MQ * (JMAGZ-1)
      IMAGR =  MAXGR  -  MQ * (JMAGR-1)
*
*
*     IF ( IFLAG .EQ. 1 )    PRINT '(1H1)'
*   PRINT changed to WRITE - R.W., 4/30/91
      IF ( IFLAG .EQ. 1 )    WRITE (10, '(1H1)')
*
      PRINT 210,  IS,TS, DT,
     &            TZ,TR, BSZ,BSR, QTZ,QTR,
     &            GAM, CFL,VCT,
     &            MQ,NQ,
     &            TOTMA, TOTEN, TOTGZ
* old version:
*      PRINT 230,  ROMIN, IMIRO, JMIRO,
*     &            ENMIN, IMIEN, JMIEN,
*     &            GZMIN, IMIGZ, JMIGZ,
*     &            GRMIN, IMIGR, JMIGR
* new version:
      PRINT 230,  ROMIN,
     &            ENMIN,
     &            GZMIN,
     &            GRMIN
* old version:
*      PRINT 240,  ROMAX, IMARO, JMARO,
*     &            ENMAX, IMAEN, JMAEN,
*     &            GZMAX, IMAGZ, JMAGZ,
*     &            GRMAX, IMAGR, JMAGR
* new version:
      PRINT 240,  ROMAX,
     &            ENMAX,
     &            GZMAX,
     &            GRMAX
*
  210 FORMAT (////6X,'CONTROL VARIABLES OF MODEL NO.:',I13
     &        ////6X,'TOTAL TIME:            TS = ',1PE16.9
     &           /6X,'LAST TIMESTEP:         DT = ',E16.9
     &        ////6X,'GRIDSPACING:           TZ = ',E16.9
     &           /6X,'                       TR = ',E16.9
     &           /6X,'                      BSZ = ',E16.9
     &           /6X,'                      BSR = ',E16.9
     &           /6X,'                      QTZ = ',E16.9
     &           /6X,'                      QTR = ',E16.9
     &        ////6X,'GAMMA:                GAM = ',E16.9
     &        ////6X,'COURANT FACTOR:       CFL = ',E16.9
     &           /6X,'VISCOSITY FACTOR:     VCT = ',E16.9
     &        ////6X,'ADVECTION LENGTH:      MQ = ',I16
     &           /6X,'                       NQ = ',I16
     &        ////6X,'TOTAL MASS:         TOTMA = ',E16.9
     &           /6X,'TOTAL ENERGY:       TOTEN = ',E16.9
     &           /6X,'TOTAL Z-MOMENTUM:   TOTGZ = ',E16.9   )
* I had to change the FORMAT            a.w.
*  230 FORMAT (////6X,'DENSITY MIN:   ',5X,'ROMIN = ',1PE16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4
*     &           /6X,'ENERGY MIN:    ',5X,'ENMIN = ',E16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4
*     &           /6X,'Z-MOMENTUM MIN:',5X,'GZMIN = ',E16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4
*     &           /6X,'R-MOMENTUM MIN:',5X,'GRMIN = ',E16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4 )
*  240 FORMAT (////6X,'DENSITY MAX:   ',5X,'ROMAX = ',1PE16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4
*     &           /6X,'ENERGY MAX:    ',5X,'ENMAX = ',E16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4
*     &           /6X,'Z-MOMENTUM MAX:',5X,'GZMAX = ',E16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4
*     &           /6X,'R-MOMENTUM MAX:',5X,'GRMAX = ',E16.9
*     &          ,12X,'POSITION:     I = ',I4,'     J = ',I4 )
  230 FORMAT (////6X,'DENSITY MIN:   ',5X,'ROMIN = ',1PE16.9
     &           /6X,'ENERGY MIN:    ',5X,'ENMIN = ',E16.9
     &           /6X,'Z-MOMENTUM MIN:',5X,'GZMIN = ',E16.9
     &           /6X,'R-MOMENTUM MIN:',5X,'GRMIN = ',E16.9)
  240 FORMAT (////6X,'DENSITY MAX:   ',5X,'ROMAX = ',1PE16.9
     &           /6X,'ENERGY MAX:    ',5X,'ENMAX = ',E16.9
     &           /6X,'Z-MOMENTUM MAX:',5X,'GZMAX = ',E16.9
     &           /6X,'R-MOMENTUM MAX:',5X,'GRMAX = ',E16.9)
*
*
***********************************************************************
*     PRINT ASSORTED DATA
***********************************************************************
*
*
      DO 10  J = 1,NP, NP/NPROW
*
	 WRITE(10,300) J, R(J)
*
  300    FORMAT (1H1//1X,'VALUES OF VARIABLES AT ',
     &                   'RO-DISTANCE     J  = ',I10
     &                            /38X,'R(J) = ',1PE10.3E2
     &            ////1X,'Z-IN',4X,'Z-DIST',9X,'DENSITY',
     &                7X,'PRESSURE',5X,'Z-VELOCITY',4X,'R-VELOCITY')
*
         DO 20  I = 1,MP, MP/MPROW
	    WRITE(10,350)  I, Z(I),  RO(I,J), PR(I,J),
     &                            VZ(I,J), VR(I,J)
  350       FORMAT (1X,I4,2X,1PE10.2E2,2X,4(2X,E12.5))
   20    CONTINUE
*
   10 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE TERM
*
*
***********************************************************************
*
*     THIS SUBROUTINE TERMINATES THE COMPUTATION
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
*
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
*
      COMMON /PRMT/   TS, DT, GAM, CFL,VCT,
     &                TZ,TR, FTZ,FTR, BSZ,BSR, QTZ,QTR, IS
      COMMON /ADVC/   ROBMQ, ROBNQ,
     &                MQ,MQ1,MQ2, NQ,NQ1,NQ2,
     &                MQFLG, NQFLG
      COMMON /DISK/   ISTEP, ICNT,
     &                ISTOR, IPRIN, MPROW, NPROW
      COMMON /VAR1/   RO(MP,NP), EN(MP,NP),
     &                GZ(MP,NP), GR(MP,NP)
*
*
***********************************************************************
*        WRITE LAST MODEL TO DISK IF THIS HASN'T ALREADY BEEN DONE
***********************************************************************
*
*
*
*
***********************************************************************
*        PRINT SOME INFORMATION ABOUT THE COMPUTATION
***********************************************************************
*
*
      IF ( ICNT .GT. ISTEP ) THEN
         ICNT = ISTEP
         PRINT 10
      ELSE
         PRINT 20
      END IF
*
      PRINT 100,   IS, TS, DT,
     &             ICNT, ICNT/ISTOR+1, ICNT/IPRIN+1, ISTEP-ICNT
*
   10 FORMAT (' 1',/////6X,'NORMAL PROGRAM STOP')
   20 FORMAT (' 1',/////6X,'EMERGENCY HALT'     )
*
  100 FORMAT (    /6X,'AT TIMESTEP NO.     :  ',I10
     &            /6X,'TOTAL TIME IS       :  ',1PE10.3E2
     &            /6X,'CURRENT TIMESTEP IS :  ',1PE10.3E2
     &          ///6X,I6,' TIMESTEPS HAVE BEEN COMPUTED'
     &            /6X,I6,' TIMESTEPS HAVE BEEN STORED'
     &            /6X,I6,' TIMESTEPS HAVE BEEN PRINTED'
     &            /6X,I6,' TIMESTEPS HAVE BEEN CANCELED'
     &        /////6X,'STOP IN ROUTINE:   TERM'          )
*
*
***********************************************************************
*
*
      S T O P
*
*
***********************************************************************
*
*
      END
      DOUBLE PRECISION FUNCTION SDOT ( N, SX,INCX, SY,INCY )
*
*
***********************************************************************
*
*     THIS FUNCTION COMPUTES THE SCALAR PRODUCT OF TWO VECTORS
*
*     STATUS:  TESTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  N           ( DIMENSION OF ARRAYS )
*        -  SX(Y)       ( VECTORS             )
*        -  INCX(Y)     ( SKIP DISTANCES      )
*     VALUES COMPUTED:
*        -  SDOT
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      DIMENSION  SX(*), SY(*)
*
*
***********************************************************************
*
*
      INX = 1
      INY = 1
      IF ( INCX .LT. 0 )  INX =  (-INCX) * (N-1)  +  1
      IF ( INCY .LT. 0 )  INY =  (-INCY) * (N-1)  +  1
*
      DOT = 0.0D0
*
      DO  10  I = 0,N-1
         DOT  =  DOT   +   SX( INCX*I + INX )  *  SY( INCY*I + INY )
   10 CONTINUE
*
      SDOT = DOT
*
*
***********************************************************************
*
*
      RETURN
      END
      FUNCTION ISMIN ( N, SX,INCX )
*
*
***********************************************************************
*
*     THIS FUNCTION COMPUTES THE INDEX OF
*     THE MINIMUM ELEMENT OF A VECTOR
*
*     STATUS:  TESTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  N           ( DIMENSION OF ARRAY )
*        -  SX          ( VECTOR             )
*        -  INCX        ( SKIP DISTANCE      )
*     VALUES COMPUTED:
*        -  ISMIN
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      DIMENSION  SX(*)
*
*
***********************************************************************
*
*
      INX = 1
      IF ( INCX .LT. 0 )  INX =  (-INCX) * (N-1)  +  1
*
      IMIN = INX
      SMIN = SX(INX)
*
      DO  10  I = 2,N
         INX = INX + INCX
         IF ( SX(INX) .LT. SMIN )  THEN
         IMIN=INX
         SMIN=SX(INX)
         ENDIF
   10 CONTINUE
*
      ISMIN = IMIN
*
*
***********************************************************************
*
*
      RETURN
      END
      FUNCTION I0MAX ( N, SX,INCX )
*
*
***********************************************************************
*
*     THIS FUNCTION COMPUTES THE INDEX OF
*     THE MAXIMUM ELEMENT OF A VECTOR
*
*     STATUS:  TESTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  N           ( DIMENSION OF ARRAY )
*        -  SX          ( VECTOR             )
*        -  INCX        ( SKIP DISTANCE      )
*     VALUES COMPUTED:
*        -  I0MAX
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      DIMENSION  SX(*)
*
*
***********************************************************************
*
*
      INX = 1
      IF ( INCX .LT. 0 )  INX =  (-INCX) * (N-1)  +  1
*
      IMAX = INX
      SMAX=SX(INX)
*
      DO  10  I = 2,N
         INX = INX + INCX
         IF ( SX(INX) .GT. SMAX )  THEN
         IMAX=INX
         SMAX=SX(INX)
         ENDIF
   10 CONTINUE
*
      I0MAX = IMAX
*
*
***********************************************************************
*
*
      RETURN
      END
      FUNCTION ISAMAX ( N, SX,INCX )
*
*
***********************************************************************
*
*     THIS FUNCTION COMPUTES THE INDEX OF
*     THE ABSOLUTE MAXIMUM ELEMENT OF A VECTOR
*
*     STATUS:  TESTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  N           ( DIMENSION OF ARRAY )
*        -  SX          ( VECTOR             )
*        -  INCX        ( SKIP DISTANCE      )
*     VALUES COMPUTED:
*        -  ISAMAX
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      DIMENSION  SX(*)
*
*
***********************************************************************
*
*
      INX = 1
      IF ( INCX .LT. 0 )  INX =  (-INCX) * (N-1)  +  1
*
      IMAX = INX
      SMAX = ABS(SX(INX))
*
      DO  10  I = 2,N
         INX = INX + INCX
         ASX = ABS(SX(INX))
         IF ( ASX .GT. SMAX ) THEN
            IMAX = INX
            SMAX = ASX
         ENDIF
   10 CONTINUE
*
      ISAMAX = IMAX
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE WNFLE ( N, ARRAY,INC, TARGET, INDEX,NVAL )
*
*
***********************************************************************
*
*     THIS ROUTINE COMPUTES ALL LOCATIONS
*     THAT ARE LESS THAN OR EQUAL TO THE TARGET
*
*     STATUS:  TESTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  N           ( DIMENSION OF ARRAY      )
*        -  ARRAY       ( ARRAY TO BE SEARCHED    )
*        -  INC         ( SKIP DISTANCE           )
*        -  TARGET      ( TARGET TO BE FOUND      )
*     VALUES COMPUTED:
*        -  INDEX       ( ARRAY OF FOUND INDICES  )
*        -  NVAL        ( NUMBER OF FOUND INDICES )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      DIMENSION  ARRAY(*), INDEX(*)
*
*
***********************************************************************
*
*
      INA = 1
      IF ( INC .LT. 0 )  INA =  (-INC) * (N-1)  +  1
*
      NVAL = 0
*
      DO  10  I = 1,N
	 IF ( ARRAY(INA) .LE. TARGET ) THEN
	    NVAL = NVAL + 1
            INDEX(NVAL) = I
         ENDIF
         INA = INA + INC
   10 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      SUBROUTINE WNFGT ( N, ARRAY,INC, TARGET, INDEX,NVAL )
*
*
***********************************************************************
*
*     THIS ROUTINE COMPUTES ALL LOCATIONS
*     THAT ARE GREATER THAN THE TARGET
*
*     STATUS:  TESTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  N           ( DIMENSION OF ARRAY      )
*        -  ARRAY       ( ARRAY TO BE SEARCHED    )
*        -  INC         ( SKIP DISTANCE           )
*        -  TARGET      ( TARGET TO BE FOUND      )
*     VALUES COMPUTED:
*        -  INDEX       ( ARRAY OF FOUND INDICES  )
*        -  NVAL        ( NUMBER OF FOUND INDICES )
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      DIMENSION  ARRAY(*), INDEX(*)
*
*
***********************************************************************
*
*
      INA = 1
      IF ( INC .LT. 0 )  INA =  (-INC) * (N-1)  +  1
*
      NVAL = 0
*
      DO  10  I = 1,N
         IF ( ARRAY(INA) .GT. TARGET ) THEN
            NVAL = NVAL + 1
            INDEX(NVAL) = I
         ENDIF
         INA = INA + INC
   10 CONTINUE
*
*
***********************************************************************
*
*
      RETURN
      END
      FUNCTION ISFNE ( N, ARRAY,INC, TARGET )
*
*
***********************************************************************
*
*     THIS FUNCTION COMPUTES THE INDEX OF THE FIRST
*     LOCATION THAT IS EQUAL TO THE TARGET
*
*     STATUS:  TESTED
*
***********************************************************************
*
*     VALUES REQUIRED:
*        -  N           ( DIMENSION OF ARRAY    )
*        -  ARRAY       ( ARRAY TO BE SEARCHED  )
*        -  INC         ( SKIP DISTANCE         )
*        -  TARGET      ( TARGET TO BE FOUND    )
*     VALUES COMPUTED:
*        -  ISFEQ
*
***********************************************************************
*
      IMPLICIT REAL*8 (A-H, O-Z)
*
      DIMENSION  ARRAY(*)
*
*
***********************************************************************
*
*
      INA = 1
      IF ( INC .LT. 0 )  INA =  (-INC) * (N-1)  +  1
*
      DO  10  I = 1,N
*
         IF ( ABS(ARRAY(INA) - TARGET) .GT. 1D-4 )  GOTO 20
*
* Original statement, leading to different results on different
* machines, due to small numerical differences
* Testing for equality should not be done for floating point numbers
*        IF ( ARRAY(INA) .NE. TARGET )  GOTO 20
*
         INA = INA + INC
   10 CONTINUE
*
   20 CONTINUE
*
      ISFNE = I
*
*
***********************************************************************
*
*
      RETURN
      END
