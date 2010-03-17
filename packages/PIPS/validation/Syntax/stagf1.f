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
