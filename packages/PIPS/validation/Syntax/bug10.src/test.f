      SUBROUTINE COURC4
CC @(#)COURC4.f	4.7
C
C.......................................................................
C     PROGRAMME PRINCIPAL DE C O U R C I R C
C.......................................................................
C
CC #include "FICERR.H"
       CHARACTER  * 4   ANNEE
       CHARACTER  * 8   PGM , TITRERES
       CHARACTER  *80   TITRE 
C
       COMMON /PGMTIT/ ANNEE , PGM , TITRE , TITRERES
C
C.......................................................................
C    ALLOCATION DU FICHIER MENU
C.......................................................................
C
      CALL FIMENA
C
C.......................................................................
C VERIFICATION DES POINTEURS DU FUT ET RECHERCHE DE L'HYPOTHESE DEMARREE
C.......................................................................
C
      IER  = 0
      IKFF = 11
C
C.......................................................................
C DEFINITION DES FICHIERS NECESSAIRES AU PROGRAMME
C
C AFFECTATION DU NUMERO LOGIQUE DU FICHIER DICTIONNAIRE DE DICFUT
C AU NOM PHYSIQUE NOMDIC
C.......................................................................
C
      CALL AINOUT
      CALL INIPER
      CALL AFDEB
      CALL AFGKS
C
      CALL CDVERF(IKFF,NHYPD,ITOUCH,IER)
      IF(IER.NE.0.OR.ITOUCH.EQ.3)THEN
         GO TO 500
      ENDIF

C
C......................................................................
C     LECTURE DU FUT -PREPARATION DES BLOCS POUR SAISIE ET CALCUL
C......................................................................
C
      CALL SLDFUH(NHYPD)
C
C.....................................................................
C     APPEL DES MENUS GENERAUX
C.....................................................................
C
      CALL DIAG0(KCAL,ITOUCH)
C
C......................................................................
C     DESALLOCATION DE TOUS LES BLOCS EN MEMOIRE
C......................................................................
C
      IF(ITOUCH.EQ.3) THEN
         CALL SLDEDF
      ENDIF
C
 500  CONTINUE
C
C.....................................................................
C     DESTRUCTION DE L'APPLICATION PERICRAN
C.....................................................................
C
      CALL AFTERM
      CALL FEMPER
C
C.....................................................................
C     DESALLOCATION SYSTEMATIQUE DE LA MEMOIRE ALLOUEE.
C.....................................................................
C
      CALL BALAYE
C
      RETURN
C ......................................................................
C PROGICIEL: COURCIC
C AUTEUR: BONAMY
C PROVENANCE: 'VDCHHAB.COURCI1.FTN'
C FICHIER: COURC4.f
C VERSION: 4.7
C DATE D'ARCHIVAGE : 90/10/22 16:08:28
C DATE D'EXTRACTION: 94/07/27 17:17:56
C ......................................................................
      END
