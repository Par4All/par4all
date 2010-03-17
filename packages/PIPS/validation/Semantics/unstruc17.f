      subroutine unstruc17b

C     Bug in ridlev

      IERR = 0
      ITYP = 4
      IF(IUTI.LE.INMUTI) THEN
         IUTI = IUTI - 1
         GO TO 1
      ENDIF
3     IUTI = 0
      JCNUT = JCNUT + 1
      RETURN
 1    CONTINUE
      KTYP = 0
      PRINT *, IERR
      go to 3
      END

      subroutine unstruc17
     *  (AFLOT,NFIXE,ITYP,IERR,
     *  TEXT,IUTI,INMUTI,IMP,NUT,NCNUT,JCNUT,IPARNUT)
      INTEGER NFIXE,ITYP,IERR,IUTI,INMUTI,IMP,NUT,NCNUT,JCNUT,IPARNUT
      REAL AFLOT
      CHARACTER*8 TEXT
C
C     -VARIABLES LOCALES-
      CHARACTER*1   A(136)
      CHARACTER*136 AA
      CHARACTER*16 T
      LOGICAL TEST,NOMBRE,IEN,IED,DECI
      INTEGER NFIX1, IM, IC, IS, ICC, ICAR, KK, K
      REAL P(16),C1
C
      EQUIVALENCE (A(1),AA)
      SAVE AA
      DATA P/1.D01,1.D02,1.D03,1.D04,1.D05,1.D06,1.D07,1.D08,
     *                  1.D09,1.D10,1.D11,1.D12,1.D13,1.D14,1.D15,1.D16/
C
C-----------------------------------
C     --- DEBUT DES INSTRUCTIONS ---
C-----------------------------------
C     -PREPARATION DU DECODAGE-
      IERR = 0
      ITYP = 4
      IF(IUTI.LE.INMUTI) THEN
         IUTI = IUTI - 1
         GO TO 1
      ENDIF
3     IUTI = 0
      JCNUT = JCNUT + 1
C
C     -LECTURE D'UNE LIGNE DU FICHIER NUT-
      READ(NUT,101,END=7) AA(:NCNUT)
101   FORMAT(A)
C
C     -IMPRESSION EVENTUELLE DE LA LIGNE LUE-
      IF(IMP.EQ.1) WRITE(6,102) AA(:NCNUT)
102   FORMAT(1X,A)
      GO TO 1
8     IF(IUTI.LE.INMUTI) GO TO 4
      IF(ITYP.EQ.4) GO TO 3
4     IF(IERR.EQ.0) GO TO 5
      IF(IMP.EQ.0) RETURN
      WRITE(6,103) IUTI,JCNUT
103   FORMAT(//,' ERREUR DETECTEE PAR RIDLEV AU',I4,'EME CARACTERE DU ',
     *                                   I5,'EME ENREGISTREMENT LU.',//)
      RETURN
7     IERR = 2
      GO TO 6
5     CONTINUE
6     TEXT=T(:8)
      RETURN
1     AFLOT=0.
      C1 = 0.
      NFIXE = 0
      NFIX1 = 0
      IM = 0
      IC = 0
C
C     -DECODAGE DU CARACTERE COURANT-
9     IUTI = IUTI + 1
      IF(IUTI.GT.INMUTI) GO TO 10
C
C     -TRAITEMENT DES COMMENTAIRES-
      IF(IPARNUT.GT.0) THEN
         IF(A(IUTI).EQ.'(') IPARNUT = IPARNUT + 1
         IF(A(IUTI).EQ.')') IPARNUT = IPARNUT - 1
         GO TO 9
      ENDIF
      IF(A(IUTI).NE.' ') GO TO 11
C
C     -FIN DE LIGNE-
10    IF(TEST) THEN
         ITYP = 0
         GO TO 8
      ENDIF
 11   CONTINUE
      END
