                    SUBROUTINE  CONDV
C                   *****************
C
     * ( TR , DIM , IUO2 , IZ )
C
C***********************************************************************
C
C  PROGICIEL : COCCINELLE      AU   01/07/87  PAR F.BLANCHON (MMN)
C                                                TEL : 47 65 41 46
C
C                    MODIFIE   AU   31/01/92  PAR S.MARGUET  (PHR)
C                                                TEL : 47 65 40 08
C                                   CORRECTIONS DES ERREURS DECRITES
C                                   DANS LA NOTE MARGUET-BARON
C                                   + FORMULATION PLUTONIUM
C
C
C                    MODIFIE NOVEMBRE 1994 PAR E.ROYER (ADULIS)
C
C***********************************************************************
C
C  FONCTION : CALCULE L'INTEGRALE DE CONDUCTIVITE DU COMBUSTIBLE (UO2
C             OU MOX)
C
C  CADRE    : MODE CRAYON PAR CRAYON
C
C-----------------------------------------------------------------------
C                         ARGUMENTS
C .____________.____.__________________________________________________.
C ^    NOM     ^MODE^                     ROLE                         ^
C ^____________^____^__________________________________________________^
C ^            ^    ^                                                  ^
C ^ TR         ^ -->^ VECTEUR DE TEMPERATURES EN   C                   ^
C ^            ^    ^                                                  ^
C ^ DIM        ^ -->^ NOMBRE D'INTEGRALES A EVALUER SUR TR EN RADIAL   ^
C ^            ^    ^ SI DIM=1 UNE SEULE INTEGRALE EST EVALUEE POUR    ^
C ^            ^    ^ CHACUN DES CRAYONS "ACTIF"                       ^
C ^            ^    ^                                                  ^
C ^ IUO2       ^<-- ^ VECTEUR RESULTAT INTEGRALE DE LA CONDUCTIBILITE  ^
C ^            ^    ^                                                  ^
C ^ IZ         ^ -> ^ MAILLE TERMO. AXIALE COURANTE                    ^
C ^____________^____^__________________________________________________^
C  MODE: -->(DONNEE NON MODIFIEE).<--(RESULTAT).<-->(DONNEE MODIFIEE)
C-----------------------------------------------------------------------
C%
C  SOUS-PROGRAMMES APPELES   :
C  SOUS-PROGRAMMES APPELANTS : TEMPV
C%
C-----------------------------------------------------------------------
C            DECLARATION DES VARIABLES
C-----------------------------------------------------------------------
C  ZONES COMMUNES
C  **************
! include "cstmax.h"
C   CONSTANTES DE DIMENSIONNEMENT ASSEMBLAGE
C     NMARAD : NOMBRE MAXIMUM D'ASSEMBLAGES DANS LES DIRECTIONS I ET J
C     NMARAD2: NOMBRE MAXIMUM D'ASSEMBLAGES DANS LA CONFIGURATION
C     NASPA  : NOMBRE MAXIMUM D'ASSEMBLAGES ACTIFS DANS LA CONFIGURATION
C   CONSTANTES DE DIMENSIONNEMENT NEUTRONIQUE
C     NMNRAD : NOMBRE MAXIMUM DE MAILLES NEUTRONIQUES DANS LES
C              DIRECTIONS I ET J
C     NMNAX  : NOMBRE MAXIMUM DE MAILLES NEUTRONIQUES AXIALES
C     NMNACR : NOMBRE COURANT DE MAILLES NEUTRONIQUES AXIALES
C   CONSTANTES DE DIMENSIONNEMENT THERMOHYDRAULIQUE
C     NTHERM : NOMBRE MAXIMUM DE MAILLES THERMOHYDRAULIQUES PAR
C              ASSEMBLAGE DANS LES DIRECTIONS I ET J POUR UN MAILLAGE
C              UNIFORME SUR UN COEUR COMPLET
C     NMTRAD : NOMBRE MAXIMUM DE MAILLES THERMOHYDRAULIQUES DANS LES
C              DIRECTIONS I ET J
C     NPAMAX : NOMBRE MAXIMUM DE MAILLES THERMOHYDRAULIQUES RADIALES
C              ACTIVES SUR LE COEUR COMPLET
C     NMTRAC : NOMBRE MAXIMUM DE MAILLES THERMOHYDRAULIQUES DANS LES
C              DIRECTIONS I ET J (POUR LES MAILLAGES CRAYON PAR CRAYON)
C              AU MAXIMUM 5 MAILLES THERMOHYDRAULIQUES PAR ASSEMBLAGE
C              (BLOTHER 5)
C     NMTAX  : NOMBRE MAXIMUM DE MAILLES THERMOHYDRAULIQUES AXIALES
C   CONSTANTES DE DIMENSIONNEMENT CRAYON
C     NMCRA  : NOMBRE MAXIMUM DE CRAYONS PAR COTE DE GRILLE D'ASSEMBLAGE
C     NMCRA1 : (NMCRA+1)*(NMCRA+1)
C     NMCRA2 : NOMBRE MAXIMUM DE CRAYONS PAR ASSEMBLAGE (NMCRA*NMCRA)
C     NMZRAC1: NOMBRE MAXIMUM DE MAILLES POUR LA THERMIQUE CRAYON
      INTEGER NMARAD,
     &        NMARAD2,
     &        NASPA,
     &        NTHERM,
     &        NMTRAD,
     &        NPAMAX,
     &        NMTRAC,
     &        NMTAX,
     &        NMCRA,
     &        NMCRA1,
     &        NMCRA2,
     &        NMNRAD,
     &        NMNAX, NMZRAC, NMNACR, NMZRAC1
      PARAMETER (NMARAD  = 23)
      PARAMETER (NMARAD2 = NMARAD*NMARAD)
      PARAMETER (NASPA   = 250)
      PARAMETER (NTHERM  = 4)
      PARAMETER (NMTRAD  = NTHERM*NMARAD)
      PARAMETER (NPAMAX  = NTHERM*NTHERM*NASPA)
      PARAMETER (NMTRAC  = 5*NMARAD)
      PARAMETER (NMCRA   = 17)
      PARAMETER (NMCRA1  = (NMCRA+1)*(NMCRA+1))
      PARAMETER (NMCRA2  = NMCRA*NMCRA)
      PARAMETER (NMNRAD  = NMCRA*17)
      PARAMETER (NMZRAC1  = 15)
C
      COMMON /CSTMAX/ NMTAX, NMNACR, NMNAX,NMZRAC
! end include "cstmax.h"
!       INCLUDE "cstmax.h"
! include "milpa.h"
      INTEGER MILPA2,MILPAC2
      INTEGER MILPA, NMCMAX, NBXREF
      PARAMETER(MILPA=100)
      PARAMETER (NMCMAX=50)
      PARAMETER (NBXREF=50)

      COMMON /MILPAI/ MILPA2,MILPAC2
! end include "milpa.h"
!       INCLUDE "milpa.h"
! include "cxm.h"
       INTEGER IXM
       REAL    POROS
      COMMON / CXM /  POROS,IXM
! end include "cxm.h"
!       INCLUDE "cxm.h"
! include "ccond.h"
       INTEGER ICOND
      COMMON / CCOND /  ICOND
! end include "ccond.h"
!       INCLUDE "ccond.h"
! include "thpin.h"
C
C CALCULS DE THERMIQUE DU CRAYON
      INTEGER IPLUT,LMC
      REAL TPU(MILPA),TENPU
      pointer (ENRU_ptr, ENRU)
      REAL ENRU(NMCMAX,*)
      pointer (ENRPU_ptr, ENRPU)
      REAL ENRPU(NMCMAX,*)
      COMMON /THPIN/ IPLUT,TPU,TENPU,LMC
      COMMON /THPIN_PTR/ ENRU_ptr,ENRPU_ptr




! end include "thpin.h"
!       INCLUDE "thpin.h"
! include "unites.h"
      REAL ZMCM, ZMMM, ZM2CM2, ZM3CM3,
     $                ZKILO, ZBARPA, ZINCHM, ZPOISE,
     $                ZHEURS, ZPCENT
      COMMON /UNITES/ ZMCM, ZMMM, ZM2CM2, ZM3CM3,
     $                ZKILO, ZBARPA, ZINCHM, ZPOISE,
     $                ZHEURS, ZPCENT
! end include "unites.h"
!       INCLUDE "unites.h"
! include "cinive.h"
C     GRANDEURS MAJ DANS INIVEC ET UTILISEES DANS TEMPV :
C     ***************************************************
C     RN1I    : RAYON CORRESPONDANT AU POINT NP4-1
C               =RC = RAYON EXTERNE DU COMBUSTIBLE
C     RN1RGI  : RAYON CRAYON COMBUSTIBLE / RAYON EXTERNE GAINE
C     RGGAPI  : VARIABLE INTERMEDIAIRE
C     DROIC   : DELTA RO DANS LE CAS DES PASTILLES CREUSES
C               =ROIC(I)-ROIC(I-1)
C     DROI2C  : 2./DROIC
C     DROI4C  : 4./DROIC
C     DROIRC  : VARIABLE INTERMEDIAIRE
C     ROIC(IR): RO AU PT RADIAL IR POUR LES PASTILLES CREUSES
C               =(DISTANCE DU CENTRE AU POINT IR)**2 / 2
C               =R(IR)**2 / 2
C     DROIP   : DELTA RO DANS LE CAS DES PASTILLES PLEINES
C               =ROIP(I)-ROIP(I-1)
C     DROI2P  : 2./DROIP
C     DROI4P  : 4./DROIP
C     DROIRP  : VARIABLE INTERMEDIAIRE
C     ROIP(IR): RO AU PT RADIAL IR POUR LES PASTILLES PLEINES
C               =(DISTANCE DU CENTRE AU POINT IR)**2 / 2
C               =R(IR)**2 / 2
C     TCBI    : VARIABLE INTERMEDIAIRE UTILISEE DANS TEMPV
C     TEMPIC(IR): IDEM
C     TEMPIP(IR): IDEM
C
C     GRANDEURS MAJ DANS INIVEC ET UTILISEES DANS CONDV :
C     ***************************************************
C     TPOROS  : = (1.-POROS)/(1.+0.5*POROS)
C               POROS DU COMMON /CXM/
C
      REAL RN1I,RN1RGI,RGGAPI,TCBI
      REAL DROIC,DROI2C,DROI4C,DROIRC
      REAL DROIP,DROI2P,DROI4P,DROIRP
      REAL TPOROS
      REAL ROIC(NMZRAC1),TEMPIC(NMZRAC1)
      REAL ROIP(NMZRAC1),TEMPIP(NMZRAC1)
      COMMON /CINIVE/ RN1I,RN1RGI,RGGAPI,TCBI,
     .       DROIC,DROI2C,DROI4C,DROIRC,ROIC,TEMPIC,
     .       DROIP,DROI2P,DROI4P,DROIRP,ROIP,TEMPIP,
     &       TPOROS
C
! end include "cinive.h"
!       INCLUDE "cinive.h"
! include "cvecto.h"
C CONSTANTES DE DIMENSIONNEMENT DES TABLEAUX EN CRAYON PAR CRAYON
      INTEGER NCACV, NCACV2, MNACV
      PARAMETER (NCACV=17)
      PARAMETER (NCACV2=NCACV*NCACV)
C
c
c.. Pointer Declarations ..
      pointer ( XRCV_PTR, XRCV )
      real  XRCV(NCACV,NCACV,1)
COLD
COLD        real,pointer , XRCV
COLD
      pointer ( QCV1_PTR, QCV1 )
      real  QCV1(NCACV,NCACV,1)
COLD
COLD        real,pointer , QCV1
COLD
      pointer ( HGAPCV_PTR, HGAPCV )
      real  HGAPCV(NCACV,NCACV,1)
COLD
COLD        real,pointer , HGAPCV
COLD
      pointer ( TCENCV_PTR, TCENCV )
      real  TCENCV(NCACV,NCACV,1)
COLD
COLD        real,pointer , TCENCV
COLD
      pointer ( ECV_PTR, ECV )
      real  ECV(NCACV,1)
COLD
COLD        real,pointer , ECV
COLD
      pointer ( TENPCV_PTR, TENPCV )
      real  TENPCV(NCACV,1)
COLD
COLD        real,pointer , TENPCV
COLD
      pointer ( TRE1CV_PTR, TRE1CV )
      real  TRE1CV(NCACV,NCACV,NMZRAC,1)
COLD
COLD        real,pointer , TRE1CV
COLD
      pointer ( HCONVV_PTR, HCONVV )
      real  HCONVV(NCACV,NCACV,1)
COLD
COLD        real,pointer , HCONVV
COLD
      pointer ( TFV_PTR, TFV )
      real  TFV(NCACV,NCACV,1)
COLD
COLD        real,pointer , TFV
COLD
      pointer ( ROMV_PTR, ROMV )
      real  ROMV(NCACV,NCACV,1)
COLD
COLD        real,pointer , ROMV
COLD
      LOGICAL CDVECV(NCACV2)
      LOGICAL CDCRAV(NCACV2)
C
      COMMON /CVECTO/ MNACV,CDCRAV,CDVECV
      COMMON /CVECTO_PTR/ XRCV_PTR,QCV1_PTR,HGAPCV_PTR,
     &                TCENCV_PTR,ECV_PTR,
     &                TENPCV_PTR,TRE1CV_PTR,HCONVV_PTR,
     &                TFV_PTR,ROMV_PTR

! end include "cvecto.h"
!       INCLUDE "cvecto.h"
! include "validi.h"
      INTEGER IHLP1, IHLP2, IHLT1, IDIPHA, ICONPU, IEBUL, IIRRAD
      REAL    HMRD,HLRD,HGRD,PHCD
      INTEGER IRD,JRD,KRD,ICD,JCD,KCD
      COMMON /VALIDI/ IHLP1,IHLP2,IHLT1,IDIPHA,ICONPU,IEBUL,IIRRAD
     &                ,HMRD,HLRD,HGRD,IRD,JRD,KRD,PHCD,ICD,JCD,KCD
! end include "validi.h"
!       INCLUDE "validi.h"
! include "acsoir.h"
       REAL    ZKELV, BOLTZM, TCK, T0K, XEMOY, XM
      COMMON / ACSOIR / ZKELV,BOLTZM,TCK,T0K,XEMOY,XM
! end include "acsoir.h"
!       INCLUDE "acsoir.h"
C
C  ARGUMENTS
C  *********
      INTEGER DIM, IZ
      REAL TR(NCACV2,MNACV),IUO2(NCACV2,DIM)
C
C  VARIABLES LOCALES
C  *****************
C  NPAS   : NOMBRE D'INTERVALLES POUR LA METHODE DES RECTANGLES (DONNE EN
C  DT     : VALEUR DU PAS POUR LA METHODE DES RECTANGLES
C  T2T1   : DIFFERENCE DE TEMPERATURE (T2-T1) EN CELCIUS
C  TM     : VALEUR MOYENNE DE LA TEMPERATURE EN CELCIUS (CALCUL SANS INTE
C  DTMIN  : ECART DE TEMPERATURE MINIMAL POUR CALCULER UNE INTEGRALE
C           (EN CELCIUS)
C  FPI    : FACTEUR CORRECTIF IRRADIATION
C  FP, FP1, FP2 : FACTEUR CORRECTIF POROSITE
C  HV1,HV2,HV3 : COEFFICIENTS DE LA FORMULE "HGAP VARIABLE 88"
C  STORA1,STORA2,STORA3 : COEFFICIENTS DE LA FORMULE STORA
C  CIRRA  : CONSTANTE DE CORRECTION POUR L'IRRADIATION
C  HK1,HK2,HK4,HK5 : COEFFICIENTS DE LA FORMULE COMETHE
C  WEST1,WEST2,WEST3,WEST4 : COEFFICIENTS DE LA FORMULE WESTINGHOUSE
      INTEGER I, J, IC, NPAS
      REAL DT, TM, DTMIN, T2T1
      REAL TT, TMK
      REAL TEMP, FTP, CINT, TTK
      REAL FP, C1, C2
      REAL HV1, HV2, HV3
      REAL HK1, HK2, HK4, HK5
      REAL WEST1, WEST2, WEST3, WEST4
      REAL INVN, CIRRA
      REAL STORA1, STORA2, STORA3
      REAL FPI(NCACV2)
      integer ALLOUER
      external ALLOUER
      integer IER
      pointer (T2T1I_ptr, T2T1I)
      REAL T2T1I(NCACV2,NMZRAC)
C
C On affecte le pointeur sur XRCV a XRV
C On affecte le pointeur sur TENPCV a TENPUV
C
      Pointer (XRCV_PTR, XRV)
      Real XRV (NCACV2,MNACV)
      Pointer (TENPCV_PTR, TENPUV)
      Real TENPUV (NCACV2)
C
      PARAMETER ( WEST1 = 100./0.92683 , WEST2 = 1./495.8 ,
     *            WEST3 = 42.017*WEST1 , WEST4 = 2.19375E-13*WEST1 )
      PARAMETER ( HV1= 1.3324E-08 , HV2 = -4.3554E-05 ,
     * HV3 = 5.8915E-02 )
      PARAMETER ( HK1= 40.05 , HK2 = 129.4 , HK4 = 0.8 ,
     * HK5 = 0.6416E-12 )
      PARAMETER ( NPAS = 10 )
      PARAMETER ( INVN = 1./NPAS )
      PARAMETER ( CIRRA= 0.124E-02 )
      PARAMETER ( STORA1=0.05693 , STORA2=0.2E-04 ,
     *            STORA3=0.3997E-08 )
C
      REAL    A, A1, A2
      DATA DTMIN /10./DT/0.0/
C-----------------------------------------------------------------------
C             CORPS DU SOUS-PROGRAMME
C-----------------------------------------------------------------------
C
C
C  ALLOCATION DYNAMIQUE
C
      T2T1I_ptr=ALLOUER(NCACV2*NMZRAC,0,IER,"CONDV")
C
C  SEQUENCE 1
C  CALCUL DE COND EN FONCTION DU COMBUSTIBLE OU DE LA CORRELATION
C***********************************************************************
C
      IF (ICOND.EQ.2) THEN
C
C        CHOIX DE LA CORRELATION UO2 N0 2 : FORMULATION WESTINGHOUSE
C
C
         DO 14 I=1,DIM
            DO 13 IC=1,NCACV2
               IF (CDVECV(IC)) THEN
                  T2T1=TR(IC,I+1)-TR(IC,I)
C
                  IF (ABS(T2T1) .GE. DTMIN) THEN
C
C                    CALCUL AVEC INTEGRALE
C
                     IF ( (TENPUV(IC).EQ.0.) .OR. (IPLUT.EQ.0) ) THEN
C
C                       PAS DE PLUTONIUM : FORMULATION UO2
C
                        IUO2(IC,I)=(WEST3*ALOG((1.+TR(IC,I+1)*WEST2)
     *                             /(1.+TR(IC,I)*WEST2) )
     *                           +WEST4 * (TR(IC,I+1)**4-TR(IC,I)**4) )
     *                           * TPOROS / T2T1
                     ELSE
C
C                       PLUTONIUM : FORMULATION MOX COMETHE
C
                        ICONPU=1
                        FPI(IC)=0.124*XRV(IC,IZ-1)*1.E-02
                        DT=T2T1*INVN
                        TT=TR(IC,I)-DT*0.5
                        CINT=0.
                        DO 12 J=1,NPAS
                           TT=TT+DT
                           IF (TT.GT.1000.) THEN
                              A=2.
                           ELSE
                              A=2.58-0.58E-03*TT
                           ENDIF
                           FP=(1.-A*POROS)/(1.-A*0.05)
                           TTK = TT + ZKELV
                           TEMP = HK2 + (1 + HK4*TENPUV(IC)*1.E-02)*TTK
                           FTP = FP*ZMCM*(HK1/TEMP + HK5*TTK*TTK*TTK)
                           IF (TT.EQ.0.) THEN
                              CINT=CINT+FTP
                           ELSE
                              CINT=CINT+1./(1./FTP+FPI(IC)/TT)
                           ENDIF
 12                     CONTINUE
                        IUO2(IC,I)=CINT*INVN
C
                     ENDIF
C
                  ELSE
C
C                    CALCUL AVEC LA VALEUR MOYENNE
C
                     TM=.5*(TR(IC,I+1)+TR(IC,I))
                     IF ( (TENPUV(IC).EQ.0.) .OR. (IPLUT.EQ.0) ) THEN
C
C                       PAS DE PLUTONIUM : FORMULATION UO2
C
                        IUO2(IC,I)=(1./(11.8+0.0238*TM)+8.775E-13*TM**3)
     *                             *WEST1 * TPOROS
C
                     ELSE
C
C                       PLUTONIUM : FORMULATION MOX COMETHE
C
                        ICONPU=1
                        FPI(IC)=0.124*XRV(IC,IZ-1)*1.E-02
                        IF (TM.GT.1000.) THEN
                           A=2.
                        ELSE
                           A=2.58-0.58E-03*TM
                        ENDIF
                        FP=(1.-A*POROS)/(1.-A*0.05)
                        TMK = TM + ZKELV
                        TEMP = HK2 + (1+HK4*TENPUV(IC)*1.E-02) * TMK
                        FTP = FP*ZMCM*(HK1/TEMP + HK5*TMK*TMK*TMK)
                        IF (TM.EQ.0.) THEN
                           IUO2(IC,I)=FTP
                        ELSE
                           IUO2(IC,I)=1./(1./FTP+FPI(IC)/TM)
                        ENDIF
C
                     ENDIF
C
                  ENDIF
C
               ENDIF
 13         CONTINUE
 14      CONTINUE
C
      ELSE IF (ICOND.EQ.1) THEN
C
C           CHOIX DE LA CORRELATION UO2 N0 1 : FORMULATION STORA
C
         DO 24 I=1,DIM
            DO 23 IC=1,NCACV2
               IF (CDVECV(IC)) THEN
                  T2T1=TR(IC,I+1)-TR(IC,I)
                  IF (ABS(T2T1) .GE. DTMIN) THEN
C
C                       CALCUL AVEC INTEGRALE
C
                     IF ((TENPUV(IC).EQ.0.).OR.(IPLUT.EQ.0)) THEN
C
C                          PAS DE PLUTONIUM : FORMULATION UO2
C
                        C1 =(STORA1-(STORA2-STORA3
     *                      *TR(IC,I))*TR(IC,I))*TR(IC,I)
                        C2 =(STORA1-(STORA2-STORA3*TR(IC,I+1))
     *                       *TR(IC,I+1))*TR(IC,I+1)
                        IF (TR(IC,I).GT.1000.) THEN
                           A1=2.
                        ELSE
                           A1=2.58-0.58E-03*TR(IC,I)
                        ENDIF
                        IF (TR(IC,I+1).GT.1000.) THEN
                           A2=2.
                        ELSE
                           A2=2.58-0.58E-03*TR(IC,I+1)
                        ENDIF
                        IUO2(IC,I)=( (1.-A2*POROS)/(1.-A2*0.036)*C2
     *                       - (1.-A1*POROS)/(1.-A1*0.036)*C1 ) *ZMCM
     *                      / T2T1
                     ELSE
C
C                          PLUTONIUM : FORMULATION MOX COMETHE
C
                        ICONPU=1
                        FPI(IC)=0.124*XRV(IC,IZ-1)*1.E-02
                        TT=TR(IC,I)-DT*0.5
                        CINT=0.
                        DO 22 J=1,NPAS
                           TT=TT+DT
                           IF (TT.GT.1000.) THEN
                              A=2.
                           ELSE
                              A=2.58-0.58E-03*TT
                           ENDIF
                           FP=(1.-A*POROS)/(1.-A*0.05)
                           TTK = TT + ZKELV
                           TEMP = HK2 + (1 + HK4*TENPUV(IC)*1.E-02)
     *                            *TTK
                           FTP = FP*ZMCM*(HK1/TEMP + HK5*TTK*TTK*TTK)
                           IF (TT.EQ.0.) THEN
                              CINT=CINT+FTP
                           ELSE
                              CINT=CINT+1./(1./FTP+FPI(IC)/TT)
                           ENDIF
 22                     CONTINUE
                        IUO2(IC,I)=CINT*INVN
                     ENDIF
C
                  ELSE
C
C                       CALCUL AVEC LA VALEUR MOYENNE
C
                     TM=.5*(TR(IC,I+1)+TR(IC,I))
                     IF ((TENPUV(IC).EQ.0.).OR.(IPLUT.EQ.0)) THEN
C
C                          PAS DE PLUTONIUM : FORMULATION UO2
C
                        IF (TM.GT.1000.) THEN
                           A=2.
                        ELSE
                           A=2.58-0.58E-03*TM
                        ENDIF
                        IUO2(IC,I) =(0.057-(0.4E-4 - 0.12E-7*TM )*TM)
     *                              *(1.-A*POROS)/(1.-A*0.036)*ZMCM
                     ELSE
C
C                          PLUTONIUM : FORMULATION MOX COMETHE
C
                        ICONPU=1
                        FPI(IC)=0.124*XRV(IC,IZ-1)*1.E-02
                        TMK = TM + ZKELV
                        TEMP = HK2 + (1+HK4*TENPUV(IC)*1.E-02) * TMK
                        IF (TM.GT.1000.) THEN
                           A=2.
                        ELSE
                           A=2.58-0.58E-03*TM
                        ENDIF
                        FP=(1.-A*POROS)/(1.-A*0.05)
                        FTP = FP*ZMCM*(HK1/TEMP + HK5*TMK*TMK*TMK)
                        IF (TM.EQ.0.) THEN
                           IUO2(IC,I)=FTP
                        ELSE
                           IUO2(IC,I)=1./(1./FTP+FPI(IC)/TM)
                        ENDIF
                     ENDIF
C
                  ENDIF
C
               ENDIF
C
 23         CONTINUE
 24      CONTINUE
C
      ELSE
C
C           CHOIX DE LA CORRELATION UO2 N0 3 : FORMULATION "HGAP VARIABLE
C
         DO 55 IC=1,NCACV2
            IF (CDVECV(IC)) THEN
               FPI(IC)=0.124*XRV(IC,IZ-1)
            ENDIF
 55      CONTINUE
         DO 57 I=1,DIM
            DO 56 IC=1,NCACV2
               IF (CDVECV(IC)) THEN
                  T2T1I(IC,I)=TR(IC,I+1)-TR(IC,I)
               ENDIF
 56         CONTINUE
 57      CONTINUE
         DO 65 I=1,DIM
            DO 64 IC=1,NCACV2
               IF (CDVECV(IC)) THEN
                  IF (ABS(T2T1I(IC,I)) .GE. DTMIN) THEN
C
C                       CALCUL AVEC INTEGRALE
C
                     DT=T2T1I(IC,I)*INVN
                     TT=TR(IC,I)-DT*.5
                     IF ((TENPUV(IC).EQ.0.).OR.(IPLUT.EQ.0)) THEN
C
C                          PAS DE PLUTONIUM : FORMULATION UO2
C
                        CINT=0.
                        DO 62 J=1,NPAS
                           TT=TT+DT
                           IF (TT.GT.1000.) THEN
                              A=2.
                           ELSE
                              A=2.58-0.58E-03*TT
                           ENDIF
                           FP= (1.-A*POROS)/(1.-A*0.034)
                           TTK=TT+ZKELV
                           IF (TT.EQ.0.) THEN
                              CINT=CINT+ZMCM*FP*HV3
                           ELSE
                              CINT=CINT+
     *                            ZMCM / (
     *                           1. / (  FP
     *                                  * (HV3+(HV2+HV1*TT)*TT) )
     *                          + FPI(IC)/TT)
                           ENDIF
 62                     CONTINUE
                        IUO2(IC,I)=CINT*INVN
C
                     ELSE
C
C                          PLUTONIUM : FORMULATION MOX COMETHE
C
                        ICONPU=1
                        CINT=0.
                        DO 63 J=1,NPAS
                           TT=TT+DT
                           IF (TT.GT.1000.) THEN
                              A=2.
                           ELSE
                              A=2.58-0.58E-03*TT
                           ENDIF
                           FP=(1.-A*POROS)/(1.-A*0.05)
                           TTK = TT + ZKELV
                           TEMP = HK2 + (1 + HK4*TENPUV(IC)*1.E-02)
     *                            *TTK
                           FTP = FP*ZMCM*(HK1/TEMP + HK5*TTK*TTK*TTK)
                           IF (TT.EQ.0.) THEN
                              CINT=CINT+FTP
                           ELSE
                              CINT=CINT+1./(1./FTP+0.01*FPI(IC)/TT)
                           ENDIF
 63                     CONTINUE
                        IUO2(IC,I)=CINT*INVN
                     ENDIF
C
                  ELSE
C
C                       CALCUL AVEC LA VALEUR MOYENNE
C
                     TM=.5*(TR(IC,I+1)+TR(IC,I))
                     IF (TM.GT.1000.) THEN
                        A=2.
                     ELSE
                        A=2.58-0.58E-03*TM
                     ENDIF
                     IF ((TENPUV(IC).EQ.0.).OR.(IPLUT.EQ.0)) THEN
C
C                             PAS DE PLUTONIUM : FORMULATION UO2
C
                        TMK=TM+ZKELV
                        FP= (1.-A*POROS)/(1.-A*0.034)
                        IF (TM.EQ.0.) THEN
                           IUO2(IC,I)= ZMCM *FP*HV3
                        ELSE
                           IUO2(IC,I)= ZMCM / (
     *                        1. / (  FP
     *                                * (HV3+(HV2+HV1*TM)*TM) )
     *                        + FPI(IC)/TM)
                        ENDIF
                     ELSE
C
C                          PLUTONIUM : FORMULATION MOX COMETHE
C
                        ICONPU=1
                        FPI(IC)=FPI(IC)*1.E-02
                        TMK = TM + ZKELV
                        TEMP = HK2 + (1+HK4*TENPUV(IC)*1.E-02) * TMK
                        FP= (1.-A*POROS)/(1.-A*0.05)
                        FTP = FP*ZMCM*(HK1/TEMP + HK5*TMK*TMK*TMK)
                        IF (TM.EQ.0.) THEN
                           IUO2(IC,I)=FTP
                        ELSE
                           IUO2(IC,I)=1./(1./FTP+FPI(IC)/TM)
                        ENDIF
                     ENDIF
C
                  ENDIF
C
               ENDIF
 64         CONTINUE
 65      CONTINUE
C
      ENDIF
C
C***********************************************************************
C
C  LIBERATION
C
      call Libere(T2T1I_ptr)
C
      RETURN
      END
