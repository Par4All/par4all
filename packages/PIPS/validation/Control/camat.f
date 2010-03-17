C -----------------------------------------------------------
C	PREPARATION DES DONNEES POUR L'ACTIF OU LE REACTIF
C -----------------------------------------------------------

C     KER=11: DEBORDEMENT NOMBRE DE TELEMESURES
C     KER=12: DEBORDEMENT NOMBRE DE TELEMESURES DANS LES IMJECTIONS
C     KER=13: DEBORDEMENT NOMBRE D'ELEMENTS DU JACOBIEN

      SUBROUTINE CAMAT (VCAL, NTM, ZIMP, NDTM, INTM, ADTR, ITMF_P,
     +			NXUT_P, NA1, Nx2, Nx3, Nx4, Nx5, Nx6,
     +			M_GAIN, NA7)

      IMPLICIT LOGICAL *1 (B)
c
c********************
C       character*40 sccs_id/'@(#)camat.f 1.1 94/05/13\n'/
c********************
c

C	en entree ;

C     VCAL	TENSION DE CALCUL
C     NTM	NOMBRE EFFECTIF DE T.M. POUR L EE
C     ZIMP	CARACTERISTIQUES DES LIAISONS
C     NDTM	TABLE DES NOEUDS ENTIEREMENT DEFINIS
C     INTM	INDICATION POUR ASSURER L'OBSERVABILITE
C     NA1	IPCO, IPCE, ICCO, ICCE, NRLO, NRLE, NRGR, NRCO, NRTJ,
C     NA7	KZBY, NZBY

C	en Sortie :

C     ADTR	ADMITANCES ENTRE LES SOMMETS ET LA TERRE
C     Nx2	IPTAI, NTAI
C     Nx3	LHAC, NCHA, EHA
C     Nx4	IVOL, IVEL, IVGR, IVCO, IVTM
C     N5BIS	ITMF_P, NXUT_P
C     Nx5	IVAC, LITA, ITMF, NXUT
C     Nx6	MVOL, MVEL, MVGR, MVCO, MVTV
C     M_GAIN	NS, NR, LF, LORD, IDEB, NONU, ISUI, INDL, ELM

	Include 'parametre.inc'
	Include	'indice.inc'

      COMMON/COMSOL/	SPGM(2),KER,IBT,BF4L,B4TROP
      COMMON/CTZRE/	TZEREA(2)
      COMMON/DBUG/	BS,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,
     +			B11,B12,B13,B14,B15
      COMMON/DIME/	IRDIN(4,3)
      COMMON/DMFX/	NXL,NL,NXP,NXJ,NT,NC,NG,NBY,NS,JOUR(5)
      COMMON/EVOL/	NVLOBS
      COMMON/IPAC/	SIGA,SIGR,NSO,NTMA,NTMR,LBLC(7,2),NFA,NBFA,NFR,
     +			NBFR,NXU,BROB
      COMMON/NUFIC/	KFB,KFR,KCR,KS

      Character*6	ITLITA(2), ITMTMA(2), ITIJAE(2), ITIATE(2)
      character*6       ITIGAE(2), ITVATM(2)

      DIMENSION ZIMP(*), INTM(*), NDTM(*), ADTR(*), NA1(*), NA7(*)
      DIMENSION Nx2(*), Nx3(*), Nx4(*), Nx5(*), Nx6(*), M_GAIN(*)
	Dimension	ITMF_P(*), NXUT_P(*)

      DATA ITLITA(1) /'ASESWL'/, ITLITA(2) /'ASESWM'/
      DATA ITMTMA(1) /'ASMTMA'/, ITMTMA(2) /'ASMTMR'/
      DATA ITIJAE(1) /'ASESWA'/, ITIJAE(2) /'ASESWB'/
      DATA ITIATE(1) /'ASESWI'/, ITIATE(2) /'ASESWJ'/
      DATA ITIGAE(1) /'ASESWG'/, ITIGAE(2) /'ASESWH'/
      DATA ITVATM(1) /'ASPTMA'/, ITVATM(2) /'ASPTMR'/

C -----------------------------------------------------------
C			Debut de CAMAT			
C -----------------------------------------------------------

      KER = 0
C      CALL MOVEK (SPGM, '*CAMAT *', 1, 1, 8)

C -----------------------------------------------------------
C	CONSTRUCTION DE LA TABLE DE DESCRIPTION DES T.M.
C -----------------------------------------------------------

C      IF (BS) WRITE(KS,1000) TZEREA(IBT)

      if (IBT.EQ.1) then
         LON = 2*NXL + NG + NC + NBY		
      else
         LON = 2*NXL + NG + NC + NT		
      end if

      CALL NLIRE (Nx4, ITVATM(IBT), LON, KER)	
      IF (KER.NE.0) GOTO 1

      CALL NLIRE (Nx6, ITMTMA(IBT), LON, KER)	
      IF (KER.NE.0) GOTO 1

C -----------------------------------------------------------
C		REJET DES TM FAUSSES DE FTMERR		
C -----------------------------------------------------------

      CALL CATMFA (ITMF_P, Nx4, Nx4(I_IVEL), Nx4(I_IVGR),
     +		   Nx4(I_IVCO), Nx4(I_IVTM), NDTM)
      IF (KER.NE.0) GOTO 1

      if (IBT.EQ.1) then
         MAXNTM     = IRDO*NMS/10
         IRDIN(1,3) = MAXNTM
         MAXMVA     = IRIN*NMS/10
         IRDIN(2,3) = MAXMVA
         MAXNH      = IRDH*NMS/10
         IRDIN(3,3) = MAXNH

         CALL CAISAX (Nx6(I_IVTM), Nx4(I_IVTM), NXUT_P,
     +		    NA7, NA7(I_NZBY), NA1(I_NRLO), NA1(I_NRLE), INTM)
      end if

      CALL CACONT (NTM, MVA, NH,
     +             NA1, NA1(I_NRLO), NA1(I_NRLE),
     +             NA1(I_NRGR), NA1(I_NRTJ),
     +             Nx4, Nx4(I_IVEL), Nx4(I_IVGR), Nx4(I_IVTM),
     +             Nx6, Nx6(I_IVEL), Nx6(I_IVGR), Nx6(I_IVTM),
     +		   Nx2(I_IPTAI), NDTM, INTM)

C      if (BS) then
C         WRITE(KS,1001) NTM, MAXNTM
C         WRITE(KS,1002) MVA, MAXMVA
C         WRITE(KS,1003) NH, MAXNH
C      end if

   10 IF (NTM.GT.MAXNTM) GOTO 900
      IF (MVA.GT.MAXMVA) GOTO 901
      IF (NH.GT.MAXNH)   GOTO 902

      Lon = 8
      CALL MOVEK (Nx2, I_IPTAI, 1, 1, Lon)

      I_LHAC = 7
      I_NCHA = I_LHAC + NTM + 1
      I_EHA  = I_NCHA + NH
      Lon    = 12
      CALL MOVEK (Nx3, I_LHAC, 1, 1, Lon)

      if (IBT.EQ.1) then	
	 LGR = NFA + NBFA
      else			
	 LGR = NFR + NBFR
      end if

      I_IVAC = 9
      I_LITA = I_IVAC + NTM
      I_ITMF = I_LITA + NTM
      I_NXUT = I_ITMF + LGR

      Lon = 16
      CALL MOVEK (Nx5, I_IVAC, 1, 1, Lon)

C Compactage de ITMF ---
C ------------------ ---

      if (LGR.GT.0) then
         do I = 1,LGR
C            Nx5(I_ITMF - 1 + I) = Nx5(I_ITMF_P -1 + I)
            Nx5(I_ITMF - 1 + I) = ITMF_P(I)
         end do
      end if

C Comptage de NXUT ---
C ---------------- ---

      if (NXU.GT.0) then
         do I = 1,NBY
C            Nx5(I_NXUT - 1 + I) = Nx5(I_NXUT_P -1 + I)
            Nx5(I_NXUT - 1 + I) = NXUT_P(I)
         end do
      end if

C     CALL CADSTM (NTM, MVA, NRLO, NRLE, NRGC, NRTJ,
C                  IVOL, IVEL, IVGC, IVTM
C                  NXUT, IVAC, LITA,
C 		   IPTAI, NTAI, ZIMP, ADTR)
      CALL CADSTM (NTM, MVA, NA1(I_NRLO), NA1(I_NRLE), NA1(I_NRGR),
     +		   NA1(I_NRTJ),
     +		   Nx4, Nx4(I_IVEL), Nx4(I_IVGR), Nx4(I_IVTM),
     + 		   Nx5(I_NXUT), Nx5(I_IVAC), Nx5(I_LITA),
     +		   Nx2(I_IPTAI), Nx2(I_NTAI), ZIMP, ADTR)
      IF (KER.NE.0) GOTO 1

C -----------------------------------------------------------
C	ECRITURE SUR DISQUE DES TABLEAUX D'ECRIVANT LES T.M.
C -----------------------------------------------------------

      LON         = I_NTAI + MVA - 1
      LBLC(2,IBT) = LON
cc      CALL NECRIR (Nx2, ITIATE(IBT), LON, KER)	
cc      IF (KER.NE.0) GOTO 1

      LON = I_NXUT - 1
      IF (NXU.GE.0) LON = LON + NBY
      LBLC(5,IBT) = LON
cc     CALL NECRIR (Nx5, ITLITA(IBT), LON, KER)	
cc      IF (KER.NE.0) GOTO 1

C -----------------------------------------------------------
C			ECRITURE DE LA TABLE ADTR	
C -----------------------------------------------------------

cc      if (IBT.EQ.2) then
cc         LON = 2*NSO
cc         CALL NECRIR (ADTR, 'ASESWC', LON, KER)
cc         IF (KER.NE.0) GOTO 1
cc      end if

C -----------------------------------------------------------
C			CONSTRUCTION DU JACOBIEN	
C -----------------------------------------------------------

C     CALL CAJAC (NTM, NH, VCAL,
C		  IPCO, IPCE, ICCO, ICCE,
C                 NRLO, NRLE, NRGR, NRCO,
C		  NRTJ, ZIMP,
C                 NXUT, IVAC, LITA
C		  LHAC, NCHA, EHA)

      CALL CAJAC (NTM, NH, VCAL,
     +		  NA1(I_IPCO), NA1(I_IPCE), NA1(I_ICCO), NA1(I_ICCE),
     +            NA1(I_NRLO), NA1(I_NRLE), NA1(I_NRGR), NA1(I_NRCO),
     +		  NA1(I_NRTJ), ZIMP,
     +		  Nx5(I_NXUT), Nx5(I_IVAC), Nx5(I_LITA),
     +            Nx3(I_LHAC), Nx3(I_NCHA), Nx3(I_EHA))

C -----------------------------------------------------------
C		ECRITURE DE LA MATRICE JACOBIEN		
C -----------------------------------------------------------

      LON = I_EHA + 2*NH - 1
      LBLC(3,IBT) = LON
cc      CALL NECRIR (Nx3, ITIJAE(IBT), LON, KER)
cc      IF (KER.NE.0) GOTO 1

C -----------------------------------------------------------
C		CONSTRUCTION DE LA MATRICE DE GAIN	
C -----------------------------------------------------------

C     CALL CAGAIN (NTM, INTM, LHAC, NCHA, EHA,
C		   IVAC, M_GAIN)

      CALL CAGAIN (NTM, INTM, Nx3(I_LHAC), Nx3(I_NCHA), Nx3(I_EHA),
     +             Nx5(I_IVAC), M_GAIN)
      IF (KER.NE.0) GOTO 1

C -----------------------------------------------------------
C		ECRITURE DE LA MATRICE DE GAIN SUR DISQUE
C -----------------------------------------------------------

      LON = LBLC(6,IBT)

C    1 IF (BS) WRITE(KS,1004) TZEREA(IBT), KER
    1 RETURN

C -----------------------------------------------------------
C			ERREURS DE DEBORDEMENT		
C -----------------------------------------------------------

C  900 IF (B9.AND..NOT.BS) WRITE(KS,1001) NTM, MAXNTM
  900 KER = 11
      GOTO 1

C  901 IF (B9.AND..NOT.BS) WRITE(KS,1002) MVA, MAXMVA
  901 KER = 12
      GOTO 1

C  902 IF(B9.AND..NOT.BS) WRITE(KS,1003) NH, MAXNH
  902 KER = 13
      GOTO 1

C 1000 FORMAT (///,1X,'* ENTREE DANS CAMAT , DEBUT DU TRAITEMENT ',
C     +        'DES DONNEES EN ',A4)
C 1001 FORMAT (3X,'NOMBRE DE TM: ',I5,' MAX: ',I5)
C 1002 FORMAT (3X,'NOMBRE DE TM ENTRANT DANS LES INJECTIONS: ',I5,
C     +	      ' MAX: ' ,I5)
C 1003 FORMAT(3X,'NOMBRE D''ELEMENTS DANS LE JACOBIEN: ',I5,' MAX: ',I5)
C 1004 FORMAT (1X,'* FIN DE CAMAT ',A4,20X,'KER=',I6)
      END
