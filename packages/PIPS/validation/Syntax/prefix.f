C       Bug Cachan n. 7: executable statements in the middle of 
C       declarations

C -----------------------------------------------------------
C		PREPARATION DE LA PARTIE FIXE		
C -----------------------------------------------------------

C     KER=11: DEPASSEMENT DU NOMBRE MAX DE SOMMETS
C     KER=12: DEPASSEMENT DU NOMBRE MAX DE LIAISONS
C     KER=13: DEPASSEMENT DU NOMBRE MAX DE GROUPES
C     KER=14: DEPASSEMENT DU NOMBRE MAX DE CONSOMMATIONS
C     KER=15: DEPASSEMENT DU NOMBRE MAX DE TENSIONS
C     KER=16: DEPASSEMENT DU NOMBRE MAX DE SAXBY

	SUBROUTINE PREFIX (CARL, NA1, NA7, NDTM, NDOB, INTM)

	IMPLICIT LOGICAL *1 (B)
c
c********************
C       character*40 sccs_id/'@(#)prefix.f 1.1 94/05/13\n'/
c********************
c

C	en sortie :

C	CARL   CARACTERISTIQUES DES LIAISONS
C	NA1    IPCO,IPCE,ICCO,ICCE,NRLO,NRLE,NRGR,NRCO,NRTJ,
C	NDOB
C	NA7    KZBY,NZBY ; LOCALISATION DES MESURES SAXBY
C	NDTM   TABLE DES INJECTIONS ENTIEREMENT DEFINIES
C	INTM   INDICATION POUR RETABLIR L'OBSERVABILITE

! include "parametre1.inc"

C	include de declaration des parametres de dimensionnement maximum
C	des variables.
        integer*4 NTAZ
        INTEGER*2  NMP,NMS,NML,NMG,NMC,NMT,
     &             NMXJ,NBY1,IRDO,IRIN,IRDH,IRDG,
     &             NTF,NBTMX,NBSX,
     &             KCMM,NMVG,NNLX,NELX
c
c       sccs_id  @(#)parametre.inc 1.1 94/05/19
c
C	NMS  =  NOMBRE MAX DE SOMMETS
C	NML  =  NOMBRE MAX DE LIAISONS
C	NMG  =  NOMBRE MAX DE GROUPES
C	NMC  =  NOMBRE MAX DE CONSOMMATIONS
C	NMT  =  NOMBRE MAX DE CAPTEURS DE TENSION
C	NBY1 =  NOMBRE MAX DE TM SAXBY
C	NTF  =  NOMBRE MAX DE T.M. FAUSSES
C       NMXJ =  NOMBRE DE TRONCONS DE JEU DE BARRE
C       NMP  =  NOMBRE MAX DE POSTES                            
C       NNLX  =  NOMBRE MAX D'ELEMENT DU JACOBIEN DANS FARA
C       NELX  =  NOMBRE MAX D'ELEMENT DU JACOBIEN DANS LAMBDA
C       NBTMX =  NOMBRE MAX DE TELEMESURE
C       NBSX = NOMBRE MAX D'ENTREE DU FICHIER DE LOCALISATION DES SAXBY
c     (2*NBY1)
C	IRDO =  10 X REDONDANCE EN TELE INFORMATION
C	IRIN =  10 X NB MOYEN DE T.M. POUR UNE INJECTION
C	IRDH =  10 X NB MOYEN D'ELEMENT DE JACOBIEN PAR SOMMET
C	IRDG =  10 X NB MOYEN D'ELEMENT DE LA MATRICE DE GAIN PAR SOMMET
c       NTAZ =  45000    dimension de la table Z de FARA      

C	Include parametre.inc

C	PARAMETER (NMP=500,  NMS=700, NML=1000, NMG=300, NMC=600,
C     $       NMT=800)
C	PARAMETER (NMXJ=1500, NBY1=50, IRDO=40, IRIN=15, IRDH=80,
C     $       IRDG=108)
C	PARAMETER (NTF=200,NBTMX=4200,NBSX=100,NTAZ=45000)
C        PARAMETER (KCMM=30, NMVG=800,NNLX=(2*NML+NMS),NELX=(4*NML+NMS))

C	garder parametres symboliques
 	NNLX = 2*NML+NMS
 	NELX = 4*NML+NMS
! end include "parametre1.inc"
! include "indice.inc"
C	--- -----------------------------------------------------------
C	--- Fichier d'include de declaration des "common" de la taille
C	--- des zones memoires et des indices des tableaux dans 
C	--- ces zones.						
C	--- -----------------------------------------------------------

C	--- -----------------------------------------------------------
C	CARL	ZIMP, ATG, SUSO, SUSE
C	Nx1	IPCO, IPCE, ICCO, ICCE, NRLO, NRLE, NRGR, NRCO, NRTJ
C	NB1	NRLO, NRLE, NRTJ
C	Nx2	IPTAI, NTAI
C	Nx3	LHAC, NCHA, EHA
C	Nx4	IVOL, IVEL, IVGR, IVCO, IVTM
C	Nx5	IVAC, LITA, ITMF, NXUT
C	Nx6	MVOL, MVEL, MVGR, MVCO, MVTV
C	M_GAIN	LCHBG, LORD, IDEB, NONU, ISUI, INDL, ELM
C	NA7	KZBY, NZBY
C	--- -----------------------------------------------------------
c
c********************
c       character*40 sccs_id/'@(#)indice.inc 1.1 94/05/13\n'/
c********************
c

	Integer*4	I_ATG, I_SUSO, I_SUSE
	Integer*4	I_IPCO, I_IPCE, I_ICCO, I_ICCE
	Integer*4	I_NRLO, I_NRLE, I_NRGR, I_NRCO, I_NRTJ
	Integer*4	IB_NRLO, IB_NRLE, IB_NRTJ
	Integer*4	I_IPTAI, I_NTAI
	Integer*4	I_LHAC, I_NCHA, I_EHA
	Integer*4	I_IVOL, I_IVEL, I_IVGR, I_IVCO, I_IVTM
	Integer*4 	I_LCHBG, I_LORD, I_IDEB, I_NONU, I_ISUI
	Integer*4	I_INDL
	Integer*4	I_ELM
	Integer*4	I_IVAC, I_LITA, I_ITMF, I_NXUT, I_ITMF_P
	Integer*4	I_NXUT_P
	Integer*4	I_KZBY, I_NZBY
	Integer*4	I_TENS

	Common/I_CARL/	I_ATG, I_SUSO, I_SUSE
	Common/I_ESWE/	I_IPCO, I_IPCE, I_ICCO, I_ICCE,
     +			I_NRLO, I_NRLE, I_NRGR, I_NRCO, I_NRTJ
	Common/I_ESWC/	IB_NRLO, IB_NRLE, IB_NRTJ
	Common/I_ESWI/	I_IPTAI, I_NTAI
	Common/I_ESWA/	I_LHAC, I_NCHA, I_EHA
	Common/I_PTMA/	I_IVOL, I_IVEL, I_IVGR, I_IVCO, I_IVTM
	Common/I_GAIN/	I_LCHBG, I_LORD, I_IDEB, I_NONU, I_ISUI,
     +			I_INDL, I_ELM
	Common/I_ESWL/	I_IVAC, I_LITA, I_ITMF, I_NXUT, I_ITMF_P,
     +			I_NXUT_P
	Common/I_LRD/	I_KZBY, I_NZBY
	Common/I_STN/	I_TENS
! end include "indice.inc"

      COMMON/COMSOL/	SPGM(2),KER,IBT,BF4L,B4TROP
      COMMON/DBUG/	BS,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,
     +			B11,B12,B13,B14,B15
      COMMON/DMFX/	NXL,NL,NXP,NXJ,NT,NC,NG,NBY,NS,JOUR(5)
      COMMON/IPAC/	SIGA,SIGR,NSO,NTMA,NTMR,LBLC(7,2),NFA,NBFA,NFR,
     +			NBFR,NXU,BROB
      COMMON/NUFIC/	KFB,KFR,KCR,KS
      COMMON/PAREG/	ZMAXI,ZMAXF,PRECA,PRECR,SEUA,SEUR,CAET,CRET,
     +			UCAL,UDEP,UMIN,UMAX,CX,EPSDT,ZTMERO,FOURCH,
     +			SURCOA,SURCOR,ZCONT,PONMIN,PONMAX,PIVNUL,
     +			PONDNC,PONDAJ,DIVERM
      COMMON/TCAL/	VCAL,VCA2
C$$$
      COMMON/TMPR/  DSOM(2,NMS)
C$$$

      DIMENSION CARL(*), NA1(*), NA7(*), NDTM(*), NDOB(*), INTM(*)

C -----------------------------------------------------------
C			Debut de PREFIX			
C -----------------------------------------------------------

C      CALL MOVEK (SPGM, '*PREFIX*', 1, 1, 8)

      KER = 0
      IF (NS.GT.NMS) GOTO 4
      IF (NL.GT.NML) GOTO 5
      IF (NG.GT.NMG) GOTO 6
      IF (NC.GT.NMC) GOTO 7
      IF (NT.GT.NMT) GOTO 8
      IF (NBY.GT.NBY1) GOTO 10

      LON = 4*NS
      CALL NLIRE (DSOM, 'ASDSOM', LON, KER)
      IF (KER.NE.0) RETURN

C -----------------------------------------------------------
C LECTURE DES PARAMETRES DE REGLAGE ET INITIALISATION EVENTUELLE-
C -----------------------------------------------------------

      CALL NLIRE (ZMAXI, 'ASESWZ', 50, KER)
c      IF (KER.NE.0) RETURN

      IF ((ZMAXI.GT.0.AND.ZMAXI.LT.100)
     +    .AND.(UDEP.GT.150.AND.UDEP.LT.300)
     +    .AND.(UMIN.GT.100.AND.UMIN.LT.350) .AND. (UMIN.LT.UMAX)
     +    .AND.(UMAX.GT.100.AND.UMAX.LT.350.)
     +    .AND.(UCAL.GT.150.AND.UCAL.LT.300.)
     +    .AND.(PONMIN.GE.1.AND.PONMAX.LE.32000.)
     +	  .AND. (PONMIN.LE.PONMAX))
     +   GOTO 3

C      IF (BS) WRITE(108,1006)
      ZMAXI  =  20.
      ZMAXF  =   1.
      PRECA  =  10.
      PRECR  =  10.
      SEUA   =   5.
      SEUR   =   5.
      CAET   =    .8
      CRET   =  80.
      UCAL   = 225.
      UDEP   = 235.
      UMIN   = 150.
      UMAX   = 280.
      CX     =   1.2
      EPSDT  =    .01
      ZTMERO =   5.
      FOURCH =   3.
      SURCOA =   1.
      SURCOR =   1.
      ZCONT  =  10.
      PONMIN =   1.
      PONMAX = 225.
      PIVNUL =    .00001
      PONDNC =   1.
      PONDAJ = 225.
      DIVERM =   3.

      CALL NECRIR (ZMAXI, 'ASESWZ', 50, KER)
      IF (KER.NE.0) RETURN

    3 VCAL = UCAL
      VCA2 = VCAL*VCAL

C -----------------------------------------------------------
C	LECTURE DONNEES FIXES DANS LA BASE DU SYSDIC	
C -----------------------------------------------------------

      LON = 8*NL
      CALL NLIRE (CARL, 'ASLCAR', LON, KER)
      IF (KER.NE.0) RETURN

      LON = 2*NL + NG + NC + NT
      CALL NLIRE (NA1(I_NRLO), 'ASNROU', LON, KER)
      IF (KER.NE.0) RETURN

      LON = 2*NBY
      CALL NLIRE (NA7, 'ASNRSX', LON, KER)
      IF (KER.NE.0) RETURN

      LON = NL + 2*NS
      CALL NLIRE (INTM, 'ASINTM', LON, KER)
      IF (KER.NE.0) RETURN

      CALL NLIRE (NDTM, 'ASNDTM', NS, KER)
      IF (KER.NE.0) RETURN

C -----------------------------------------------------------
C PRISE EN COMPTE DES RESULTATS DE L'OBSERVABILITE (CONNEXITE) --
C -----------------------------------------------------------

      CALL NLIRE (NDOB, 'ASNDOB', NS, KER)
      IF (KER.NE.0) RETURN

      do I=1,NS
         IF (NDOB(I).NE.0) NDOB(I) = 3
      end do

      CALL PRXPLO (NA1(I_IPCO), NA1(I_IPCE), NA1(I_ICCO), NA1(I_ICCE),
     +		   NDOB,
     +             NA1(I_NRLO), NA1(I_NRLE), NA1(I_NRGR), NA1(I_NRCO),
     +		   NA1(I_NRTJ), NDTM)

      RETURN

C -----------------------------------------------------------
C	TRAITEMENT DES ERREURS SUR LES DIMENSIONS MAX	
C -----------------------------------------------------------

C    4 IF (B9) WRITE(KS,1000) NS, NMS
    4 KER = 11
      RETURN

C    5 IF (B9) WRITE(KS,1001) NL, NML
    5 KER = 12
      RETURN

C    6 IF (B9) WRITE(KS,1002) NG, NMG
    6 KER=13
      RETURN

C    7 IF (B9) WRITE(KS,1003) NC, NMC
    7 KER = 14
      RETURN

C    8 IF (B9) WRITE(KS,1004) NT, NMT
    8 KER = 15
      RETURN

C   10 IF (B9) WRITE(KS,1005) NBY, NBY1
   10 KER = 16
      RETURN

C 1000 FORMAT (3X,'NOMBRE DE SOMMETS ',I3,' SUPERIEUR AU MAX: ',I3)
C 1001 FORMAT (3X,'NOMBRE DE LIGNES ',I3,' SUPERIEUR AU MAX: ',I3)
C 1002 FORMAT (3X,'NOMBRE DE GROUPES ',I3,' SUPERIEUR AU MAX: ',I3)
C 1003 FORMAT(3X,'NOMBRE DE CONSOMMATIONS ',I3,' SUPERIEUR AU MAX: ',I3)
C 1004 FORMAT (3X,'NOMBRE DE TENSIONS ',I3,' SUPERIEUR AU MAX: ',I3)
C 1005 FORMAT (3X,'NOMBRE DE SAXBY ',I3,' SUPERIEUR AU MAX: ',I3)
C 1006 FORMAT (3X,'ATTENTION PARAMETRES DE REGLAGE REINITIALISES')
      END
