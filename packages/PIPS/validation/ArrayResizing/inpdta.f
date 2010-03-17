      SUBROUTINE INPDTA
      IMPLICIT REAL*8 (A-H, O-Z)
*
*     INPUT UNIT = STDIN
      INTEGER*4 CNTUNT
      PARAMETER ( CNTUNT  =  5 )
*
*
***********************************************************************
*        READ BASIC DATA INTO COMMONS
*        /DISK/ , /CUTP/ , /PRMT/ , /ADVC/ , /BOND/
***********************************************************************
*
*
* DECLARATION OF DUMMY CHARACTER STRING
      CHARACTER*80 CDUM
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
* INPUT OF CONTROL PARAMETERS AND PRINT THEM FOR COMPARISON
 1000 FORMAT(A80)
      READ(CNTUNT,1000) CDUM
      READ(CNTUNT,*) ISTEP
      PRINT *, ' ISTEP      =         ',ISTEP
*
      READ(CNTUNT,1000) CDUM
      READ(CNTUNT,*) ISTOR, MPROW, NPROW
      PRINT *, ' ISTOR      =         ',ISTOR
      PRINT *, ' MPROW      =         ',MPROW
      PRINT *, ' NPROW      =         ',NPROW
*
      READ(CNTUNT,1000) CDUM
      READ(CNTUNT,*) NEGFL, NEGCN
      PRINT *, ' NEGFL      =         ',NEGFL
      PRINT *, ' NEGCN      =         ',NEGCN
*
      READ(CNTUNT,1000) CDUM
      READ(CNTUNT,*) CFL, VCT
      PRINT *, ' CFL        =         ',CFL
      PRINT *, ' VCT        =         ',VCT
*
      READ(CNTUNT,1000) CDUM
      READ(CNTUNT,*) MQFLG, NQFLG
      PRINT *, ' MQFLG     =         ',MQFLG
      PRINT *, ' NQFLG     =         ',NQFLG
*
      READ(CNTUNT,1000) CDUM
      READ(CNTUNT,*) ZL, ZU, RL, RU
      PRINT *, ' ZL         =         ',ZL
      PRINT *, ' ZU         =         ',ZU
      PRINT *, ' RL         =         ',RL
      PRINT *, ' RU         =         ',RU
*
      END
