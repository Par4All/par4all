      PROGRAM INIMOD3

C     Same as inimod.f, but let's analyze the floating point variables!

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
