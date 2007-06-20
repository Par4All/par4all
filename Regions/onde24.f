C     Modifications:
C      - function TIME added

      PROGRAM ONDE24

C      ---------------------------------------------------------------
C            MODELISATION PAR EQUATION D'ONDES D'UN PHENOMENE
C                  DE PROPAGATION EN DIMENSION 2.
C
C      HYPOTHESES  :   - ON CONSIDERE UN DOMAINE CARRE .
C      ----------      - LA SURFACE EST LIBRE.
C                      - ON CONSIDERE DES CONDITIONS ABSORBANTES DU SECOND
C                        ORDRE SUR LES BORDS  (cf. these CHALINDAR).
C                      - LE MILIEU EST HETEROGENE.
C
C
C      UTILISATION D'UN SCHEMA AUX DIFFERENCES FINIES D'ORDRE (2,4)
C      (ORDRE 2 EN TEMPS ET 4 EN ESPACE) COMME METHODE DE RESOLUTION :
C
C             - SCHEMA D'ORDRE (2,4) POUR LE DOMAINE INTERIEUR.
C             - SCHEMA D'ORDRE (2,2) POUR LE BORD INTERIEUR DU DOMAINE.
C             - CONDITIONS PARAXIALES D'ORDRE 2 POUR LE BORD DU DOMAINE.
C
C      ---------------------------------------------------------------



C      DECLARATION DE CONSTANTES
C      -------------------------
C      NPMAX      : NOMBRE DE POINTS MAXIMUM DU MAILLAGE SPATIAL
C      NTMAX      : NOMBRE MAXIMUM DE PAS DE TEMPS
C      NBTRAC   : NOMBRE DE TRACES SISMIQUES
C      TMAX        : TEMPS DE PRISE EN COMPTE DE LA FONCTION SOURCE (S)

      INTEGER    NPMAX 
      INTEGER    NTMAX 
      INTEGER    NBTRAC
      INTEGER    TMAX   
      REAL*8     PI     

      PARAMETER      (NPMAX     = 809)
      PARAMETER      (NTMAX     = 4000)
      PARAMETER      (NBTRAC  = 50)
      PARAMETER      (TMAX       = 1.0)
      PARAMETER      (PI         = 3.141592653589)


C      DECLARATION DE TABLEAUX
C      -----------------------
C      U      :    SOLUTION DE LA MODELISATION PAR EQUATION D'ONDE D'UN
C                  PHENOMENE DE PROPAGATION. UTILISATION D'UN SCHEMA AUX
C                  DIFFERENCES FINIES COMME METHODE DE RESOLUTION.
C
C                  U(.,.,KM) REPRESENTE U(t)
C                  U(.,.,KP) REPRESENTE U(t-1) et U(t+1)
C
C
C
C      V      :    CHAMP DE VITESSE CARACTERISTIQUE DU MILIEU.
C                  - LE TABLEAU MEMORISE LA CONSTANTE LIEE A LA VITESSE
C                         1/12*((V(i,j)*DELTAT/H)**2)
C                    POUR LES POINTS INTERIEURS DU DOMAINE.
C                  - LE TABLEAU MEMORISE LA CONSTANTE LIEE A LA VITESSE
C                         (V(i,j)*DELTAT/H)**2
C                    POUR LES POINTS CONSTITUANT LE BORD INTERNE DU DOMAINE
C                  - DANS LE CAS DES POINTS CONSTITUANT LES FRONTIERES
C                    E-W-S, LE TABLEAU CONTIENT LA MOYENNE HARMONIQUE
C                    DES VITESSES DES POINTS FRONTIERES ET DES POINTS
C                    CONNEXES A LA FRONTIERE.
C                  - DANS LE CAS DES POINTS CONSTITUANT LES COINS INFERIEURS
C                    DU DOMAINE, ET LES DEUX POINTS FRONTIERES CONNEXES
C                    AUX COINS,  LE TABLEAU CONTIENT LA MOYENNE HARMONIQUE
C                    DES VITESSES DES 4 POINTS CONNEXES AU POINT CONCERNE.
C                  - LES COINS SUPERIEURS SONT CONSIDERES COMME PARTIE
C                    DE LA SURFACE LIBRE.
C
C    B        :    TABLEAUX DE 3 VECTEURS CONTENANT LES CONSTANTES LIEES
C                  AUX EQUATIONS ASSOCIEES AUX BORDS.
C                       B1  =  1 - (1/(V'*DELTAT + H)) * (2*H + (V'*DELTAT)**2/H)
C                       B2  =  2*H / (V'*DELTAT + H)
C                       B3  =  (V'*DELTAT)*(V'*DELTAT) / 2*H*(V'*DELTAT + H)
C                  AVEC V': VITESSE HARMONIQUE SUR DEUX POINTS.
C
C                  DANS LE CAS PARTICULIER DES 3 POINTS COMPOSANT
C                  CHAQUE COIN E-S ET S-W, LES TABLEAUX B1 ET B2 CONTIENNENT 
C                  RESPECTIVEMENT LES CONSTANTES :
C                      B1   = 2*V'*DELTAT / ( 4*V'*DELTAT + 3*H)
C                      B2   = 3*H / (4*V'*DELTAT + 3*H)
C                  AVEC V': VITESSE HARMONIQUE SUR QUATRE POINTS.
C                  
C                  CODAGE :  B(.,1,1) <-  B1 EAST
C                            B(.,1,2) <-  B1 SOUTH
C                            B(.,1,3) <-  B1 WEST
C
C
C    UB       :    TABLEAU DE VECTEURS INTERMEDIAIRES UTILISES POUR LE CALCUL 
C                  DES BORDS E-S-W ; LE CODAGE EST LE MEME QUE POUR 
C                  LES VECTEURS B.
C
C    UINT    :     VECTEUR INTERMEDIAIRE UTILISE LORS DU CALCUL DE
C                  LA SOLUTION AUX BORDS
C
C

      REAL*8    U     (NPMAX+1,NPMAX,2)
      REAL*8    V     (NPMAX+1,NPMAX)
 
      REAL*8    B     (NPMAX+1,3,3)
      REAL*8    UB    (NPMAX+1,3)
      REAL*8    UINT  (NPMAX)

      REAL*8    SISMO (NBTRAC,NTMAX)

C     DECLARATION DE SCALAIRES
C     ------------------------
C     NP       : NOMBRE DE POINT SELON LA DIMENSION X DU MAILLAGE SPATIAL
C     NT       : NOMBRE DE PAS DE TEMPS DE LA SIMULATION
C     DELTAT   : INTERVALLE DE TEMPS               (0.0005 S)
C     H        : INTERVALLE D'ESPACE               (1.5    M)
C     F        : FREQUENCE DE LA SOURCE            (30     HERTZ)
C     IS,JS    : COORDONNEES DE LA SOURCE            
C
C     KM,KP    : POINTEURS PERMETTANT DE PERMUTER LES PLANS ESPACE.
C     I,J,K,N  : INDICES DE BOUCLES
C     Wi       : VARIABLES DE TRAVAIL
C     G0       : INTENSITE DE LA SOURCE A L'INSTANT t
C     TINIT,
C     TFIN,
C     TMESUR   : COMPTEURS DE GESTION DU TEMPS
C     TiCPU    : COMPTEURS DE GESTION DU TEMPS CPU
C     TiELAP   : COMPTEURS DE GESTION DU TEMPS ELAPSED
C     NBOPi    : COMPTEURS DU NOMBRE D'OPERATIONS FLOTTANTES
C     MFLOPS   : NOMBRE DE MFLOPS MESURES
C     UMIN     : VALEUR MIN DE LA SOLUTION U AU DERNIER PAS DE TEMPS
C     UMAX     : VALEUR MAX DE LA SOLUTION U AU DERNIER PAS DE TEMPS
C     VMIN     : VALEUR MIN DE LA SOLUTION DU CHAMP DE VITESSE
C     VMAX     : VALEUR MAX DE LA SOLUTION DU CHAMP DE VITESSE
C     SISMIN   : VALEUR MIN DE LA SOLUTION DU SISMOGRAMME
C     SISMAX   : VALEUR MAX DE LA SOLUTION DU SISMOGRAMME
C
      INTEGER   NP
      INTEGER   NT
      REAL*8    DELTAT 
      REAL*8    H      
      REAL*8    F      
      INTEGER   IS,JS
      INTEGER   KM,KP
      INTEGER   I,J,K,N
      REAL*8    W
      REAL*8    C1,C2,C3

      REAL*8    G0
      REAL*8    TMESUR
      REAL*8    TINIT
      REAL*8    TFIN
      REAL*8    T1CPU
      REAL*8    T2CPU
      REAL*8    T3CPU
      REAL*8    NBOP1
      REAL*8    NBOP2
      REAL*8    MFLOPS
      REAL*8    UMIN, UMAX
      REAL*8    VMIN, VMAX
      REAL*8    SISMIN, SISMAX


C
C             CORPS DU PROGRAMME
C             ------------------

      CALL OPENFILE

C
C     ACQUISITION DES PARAMETRES DE LA MODELISATION
C     ---------------------------------------------

C     WRITE (6,'('' ENTRER LE NB DE POINTS DU MAILLAGE CARRE (NP) ? '',$)')
      READ (4,*) NP
C     WRITE (6,'('' ENTRER LE NB DE PAS DE TEMPS TOTAL (NT) ? '',$)')
      READ (4,*) NT
C     WRITE (6,'('' ENTRER LE PAS DE DISCR. EN ESPACE (H) ? '',$)')
      READ (4,*) H
C     WRITE (6,'('' ENTRER LE PAS DE DISCR. EN TEMPS (DELTAT) ? '',$)')
      READ (4,*) DELTAT
C     WRITE (6,'('' ENTRER LA POSITION EN X DE LA SOURCE (JS) ? '',$)')
      READ (4,*) JS
C     WRITE (6,'('' ENTRER LA POSITION EN Z DE LA SOURCE (IS) ? '',$)')
      READ (4,*) IS
C     WRITE (6,'('' ENTRER LA FREQUENCE DE LA SOURCE (F) ? '',$)')
      READ (4,*) F


  

      WRITE(1,1100)
      WRITE(1,1110)
      WRITE(1,1120)
      WRITE(1,1130)NP,NP
      WRITE(1,1135)NT
      WRITE(1,1140)IS,JS 
      WRITE(1,1171)H
      WRITE(1,1170)DELTAT
      WRITE(1,1180)F

C////////////////BEGIN FUNCTION TIMER/////////////////
      CALL STIMER (W)
      TINIT = GETIME (W)
      TFIN  = GETIME (W)
      TMESUR = TFIN  - TINIT
C////////////////END   FUNCTION TIMER/////////////////
C
C     INITIALISATION DES VARIABLES SCALAIRES
C     --------------------------------------
      KM = 1
      KP = 2
      G0 = 0


C     INITIALISATION DU CHAMP VITESSE
C     V(I,J)     = READ (Vi) (CAS D'UN MILIEU HETEROGENE)
C     -------------------------------------------------------

C////////////////BEGIN FUNCTION TIMER/////////////////
      TINIT = GETIME (W)
C////////////////END   FUNCTION TIMER/////////////////

      
      READ (3,*) NP
      READ (3,*) NP
      DO 21 I = 1,NP
        DO 22 J = 1,NP
          READ (3,*) V(I,J)
22      CONTINUE
21    CONTINUE


C////////////////BEGIN FUNCTION TIMER/////////////////
      TFIN  = GETIME (W)
      T3CPU   = TFIN  - TINIT
C////////////////END   FUNCTION TIMER/////////////////

      VMAX = V(1,1)
      VMIN = V(1,1)
      DO 112 I = 1,NP
        DO 122 J = 1,NP
          IF (V(I,J).LT.VMIN) THEN
            VMIN = V(I,J)
          ENDIF
          IF (V(I,J).GT.VMAX) THEN
            VMAX = V(I,J)
          ENDIF
122     CONTINUE
112   CONTINUE
      WRITE (1,1051) VMIN,VMAX


C////////////////BEGIN FUNCTION TIMER/////////////////
      TINIT = GETIME (W)
C////////////////END   FUNCTION TIMER/////////////////


C     INITIALISATION DE LA SOLUTION U
C     U(.,.,t=0) = U(.,.,t=1) = 0
C     ET DES ENREGISTREMENTS 
C     SISMO (T=1) = 0.0
C     ------------------------------

      DO 10 K = 1,2
        DO 20 J = 1,NP
          DO 30 I = 1,NP
            U(I,J,K) = 0.0
30        CONTINUE
20      CONTINUE
10    CONTINUE


      DO 84 I = 1,NBTRAC 
        SISMO(I,1) = 0.0
84    CONTINUE




C     INITIALISATION DES CONSTANTES LIEES AUX COINS INFERIEURS
C            V(I,J)  <--  moyenne harmonique sur 4 points
C            B(I,.,.)<--  cf declarations
C     ET AUX BORDS E-S-W
C            V(I,J)  <--  moyenne harmonique sur 2 points
C            B(I,.,.)<--  cf declarations
C     --------------------------------------------------------


C     CAS DES TROIS POINTS DU COIN E-S
C     --------------------------------
          C1          = 4/((1/V(NP-1,NP))+(1/V(NP-1,NP-1)) +
     &                     (1/V(NP-2,NP))+(1/V(NP-2,NP-1)) )
          W           = C1*DELTAT
          B(NP-1,1,1) = 2*W/(4*W+3*H)
          B(NP-1,1,2) = 3*H/(4*W+3*H)

          C2          = 4/((1/V(NP,NP))+(1/V(NP,NP-1)) +
     &                     (1/V(NP-1,NP))+(1/V(NP-1,NP-1)) )
          W           = C2*DELTAT
          B(NP,2,1) = 2*W/(4*W+3*H)
          B(NP,2,2) = 3*H/(4*W+3*H)

          C3          = 4/((1/V(NP,NP))+(1/V(NP,NP-1)) +
     &                     (1/V(NP-1,NP))+(1/V(NP-1,NP-1)) )
          W           = C3*DELTAT
          B(NP-1,2,1) = 2*W/(4*W+3*H)
          B(NP-1,2,2) = 3*H/(4*W+3*H)

          V(NP-1,NP)  = C1
          V(NP,NP)    = C2
          V(NP,NP-1)  = C3

C     CAS DES 3 POINTS DU COIN W-S
C     ----------------------------
          C1          = 4/((1/V(NP,2))+(1/V(NP,3)) +
     &                     (1/V(NP-1,2))+(1/V(NP-1,3)) )
          W           = C1*DELTAT
          B(2,2,1) = 2*W/(4*W+3*H)
          B(2,2,2) = 3*H/(4*W+3*H)

          C2          = 4/((1/V(NP,1))+(1/V(NP,2)) +
     &                     (1/V(NP-1,1))+(1/V(NP-1,2)) )
          W           = C2*DELTAT
          B(1,2,1) = 2*W/(4*W+3*H)
          B(1,2,2) = 3*H/(4*W+3*H)

          C3          = 4/((1/V(NP-1,1))+(1/V(NP-1,2)) +
     &                     (1/V(NP-2,1))+(1/V(NP-2,2)) )
          W           = C3*DELTAT
          B(NP-1,3,1) = 2*W/(4*W+3*H)
          B(NP-1,3,2) = 3*H/(4*W+3*H)

          V(NP,2)    = C1
          V(NP,1)    = C2
          V(NP-1,1)  = C3


C     CAS DES BORDS   E - S - W
C     -------------------------
      DO 41 I = 2,NP-2
          V(I,NP)   = 2/((1/V(I,NP))+(1/V(I,NP-1)))
          W        = V(I,NP)*DELTAT
          B(I,1,1) = 1 - (1/(W+H)) * ( 2*H + W*W/H) 
          B(I,1,2) = 2*H/(W+H)
          B(I,1,3) = W*W/(2*H*(W+H))
41    CONTINUE

      DO 42 I = 3,NP-2
          V(NP,I)   = 2/((1/V(NP,I))+(1/V(NP-1,I)))
          W        = V(NP,I)*DELTAT
          B(I,2,1) = 1 - (1/(W+H)) * ( 2*H + W*W/H) 
          B(I,2,2) = 2*H/(W+H)
          B(I,2,3) = W*W/(2*H*(W+H))
42    CONTINUE

      DO 43 I = 2,NP-2
          V(I,1)   = 2/((1/V(I,1))+(1/V(I,2)))
          W        = V(I,1)*DELTAT
          B(I,3,1) = 1 - (1/(W+H)) * ( 2*H + W*W/H) 
          B(I,3,2) = 2*H/(W+H)
          B(I,3,3) = W*W/(2*H*(W+H))
43    CONTINUE

C
C     INITIALISATION DES CONSTANTES LIEES AUX BORDS INTERNES
C     ------------------------------------------------------
        DO 51 I = 3,NP-1
          V(I,2) = (V(I,2)*DELTAT/H)**2
51      CONTINUE

        DO 52 I = 3,NP-1
          V(I,NP-1) = (V(I,NP-1)*DELTAT/H)**2
52      CONTINUE

        DO 53 I = 3,NP-2
          V(NP-1,I) = (V(NP-1,I)*DELTAT/H)**2
53      CONTINUE


C     INITIALISATION DES CONSTANTES LIEES AU DOMAINE INTERIEUR
C            V(I,J) <--  1/12*((V(I,J)*DELTAT/H)**2)
C     --------------------------------------------------------
      DO 40 J = 3,NP-2
        DO 50 I = 3,NP-2
          V(I,J) = ((V(I,J)*DELTAT/H)**2)/12
50      CONTINUE
40    CONTINUE


C////////////////BEGIN FUNCTION TIMER/////////////////
      TFIN  = GETIME (W)
      T1CPU   = TFIN  - TINIT - TMESUR
      WRITE(1,1015) TMESUR
      WRITE(1,1010) T1CPU

      TINIT = GETIME (W)
C////////////////END   FUNCTION TIMER/////////////////

C     -----------------------------------------------------------
C     !     MODELISATION DE LA PROPAGATION DES ONDES            !
C     !               PAR EQUATION D'ONDE.                      !
C     !        RESOLUTION PAR DIFFERENCES FINIES.               !
C     -----------------------------------------------------------


C     ----------------------------
C     !     BOUCLE EN TEMPS      !
C     ----------------------------

      DO 70 N = 2,NT

C     PRISE EN COMPTE DE LA FONCTION SOURCE PENDANT UN TEMPS DE TMAX SEC.
C     Un = Un + SOURCE
C     ------------------------------------------------------------------
      W = F * (N * DELTAT - 1.0/F)
      IF (W**2.LT.TMAX) THEN
         W = (W*PI)**2
         G0 = (1.0 - 2.0*W) * EXP (-W)
         U(IS,JS,KM) = U(IS,JS,KM) + G0 * 12*V(IS,JS) * H**2
      ENDIF



C     --------------------------------------------------------
C     SAUVEGARDE DE LA FRANGE CONNEXE AUX BORDS POUR LE CALCUL
C     ULTERIEUR DE LA SOLUTION AUX BORDS
C     --------------------------------------------------------

      DO 71 I = 2,NP-1
         UB(I,1) = U(I,NP-1,KP)
         UB(I,2) = U(NP-1,I,KP)
         UB(I,3) = U(I,2,KP)
71    CONTINUE
      
C
C     ------------------------------------------------
C     SOLUTION SUR LE DOMAINE INTERIEUR
C     SCHEMA D'ORDRE 2 EN TEMPS ET D'ORDRE 4 EN ESPACE
C     ------------------------------------------------

      DO 80 J = 3,NP-2

C     levee des dependances pour forcer la vectorisation.
C     kp et km sont toujours different.
C DIR$ IVDEP

        DO 90 I = 3, NP-2

          U(I,J,KP) = 
     &       2 * U(I,J,KM) - U(I,J,KP)
     &   - V(I,J) * ( 60 * U(I,J,KM) 
     &   - 16 * ( U(I+1,J,KM) + U(I-1,J,KM) + U(I,J-1,KM) + U(I,J+1,KM))
     &        +   U(I+2,J,KM) + U(I-2,J,KM) + U(I,J-2,KM) + U(I,J+2,KM))

 90      CONTINUE
80    CONTINUE

C
C     ------------------------------------------------
C     SOLUTION SUR LES BORDS INTERIEURS
C     SCHEMA D'ORDRE 2 EN TEMPS ET D'ORDRE 2 EN ESPACE
C     ------------------------------------------------

C DIR$ IVDEP
      DO 140 I = 3,NP-1
          U(I,2,KP) = 
     &       2 * U(I,2,KM) - U(I,2,KP) - V(I,2) * 
     &           ( 4 * U(I,2,KM) - ( U(I+1,2,KM) + U(I-1,2,KM) 
     &               + U(I,1,KM) + U(I,3,KM)))
140    CONTINUE

C DIR$ IVDEP
      DO 141 I = 3,NP-1
          U(I,NP-1,KP) = 
     &       2 * U(I,NP-1,KM) - U(I,NP-1,KP) - V(I,NP-1) * 
     &           ( 4 * U(I,NP-1,KM) - ( U(I+1,NP-1,KM) + U(I-1,NP-1,KM) 
     &               + U(I,NP-2,KM) + U(I,NP,KM)))
141    CONTINUE


C DIR$ IVDEP
      DO 142 I = 3,NP-2
          U(NP-1,I,KP) = 
     &       2 * U(NP-1,I,KM) - U(NP-1,I,KP) - V(NP-1,I) * 
     &           ( 4 * U(NP-1,I,KM) - ( U(NP,I,KM) + U(NP-2,I,KM) 
     &               + U(NP-1,I-1,KM) + U(NP-1,I+1,KM)))
142    CONTINUE

C
C     --------------------------------------------------------
C     SOLUTION AU BORD EAST. CONDITIONS ABSORBANTES D'ORDRE 2.
C     --------------------------------------------------------

      DO 92 I = 3,NP-2
         UINT(I) =   B(I,1,1)*( U(I,NP,KP) + U(I,NP-1,KP) )
     &                - UB(I,1) 
     &              + B(I,1,2)*( U(I,NP,KM) + U(I,NP-1,KM) )
     &              + B(I,1,3)*( U(I+1,NP-1,KP) + U(I+1,NP,KP) +
     &                           U(I-1,NP-1,KP) + U(I-1,NP,KP) )
92    CONTINUE

      DO 93 I = 3,NP-2
        U(I,NP,KP) = UINT(I)
93    CONTINUE

C
C     --------------------------------------------------------
C     SOLUTION AU BORD SOUTH. CONDITIONS ABSORBANTES D'ORDRE 2.
C     --------------------------------------------------------

      DO 94 I = 3,NP-2
         UINT(I) =   B(I,2,1)*( U(NP,I,KP) + U(NP-1,I,KP) )
     &                - UB(I,2)
     &              + B(I,2,2)*( U(NP,I,KM) + U(NP-1,I,KM) )
     &              + B(I,2,3)*( U(NP-1,I+1,KP) + U(NP,I+1,KP) +
     &                           U(NP-1,I-1,KP) + U(NP,I-1,KP) )
94    CONTINUE

      DO 95 I = 3,NP-2
        U(NP,I,KP) = UINT(I)
95    CONTINUE
 
C
C     --------------------------------------------------------
C     SOLUTION AU BORD WEST. CONDITIONS ABSORBANTES D'ORDRE 2.
C     --------------------------------------------------------

      DO 96 I = 3,NP-2
         UINT(I) =   B(I,3,1)*( U(I,1,KP) + U(I,2,KP))
     &              - UB(I,3)
     &              + B(I,3,2)*( U(I,1,KM) + U(I,2,KM) )
     &              + B(I,3,3)*( U(I+1,2,KP) + U(I+1,1,KP) +
     &                           U(I-1,2,KP) + U(I-1,1,KP) )
96    CONTINUE

      DO 97 I = 3,NP-2
        U(I,1,KP) = UINT(I)
97    CONTINUE



C      -----------------------------------------------
C      CALCUL DU BORD DU COIN 
C      U(NP-1,NP,KP) ,  U(NP,NP-1,KP) ET U(NP-1,NP,KP)
C      -----------------------------------------------

      U(NP-1,NP,KP) = B(NP-1,1,1)*(U(NP-1,NP-1,KP)+U(NP-2,NP,KP))
     &              + B(NP-1,1,2)*U(NP-1,NP,KM)

      U(NP,NP-1,KP) = B(NP-1,2,1)*(U(NP,NP-2,KP)+U(NP-1,NP-1,KP))
     &              + B(NP-1,2,2)*U(NP,NP-1,KM)

      U(NP,NP,KP)   = B(NP,2,1)*(U(NP,NP-1,KP)+U(NP-1,NP,KP))
     &              + B(NP,2,2)*U(NP,NP,KM)


C      -----------------------------------------------
C      CALCUL DU BORD DU COIN 
C      U(NP,2,KP) ,  U(NP-1,1,KP) ET U(NP,1,KP)
C      -----------------------------------------------

      U(NP,2,KP)    = B(2,2,1)*(U(NP,3,KP)+U(NP-1,2,KP))
     &              + B(2,2,2)*U(NP,2,KM)

      U(NP-1,1,KP)  = B(NP-1,3,1)*(U(NP-1,2,KP)+U(NP-2,1,KP))
     &              + B(NP-1,3,2)*U(NP-1,1,KM)

      U(NP,1,KP)    = B(1,2,1)*(U(NP-1,1,KP)+U(NP,2,KP))
     &              + B(1,2,2)*U(NP,1,KM)


C
C     ENREGISTREMENT DU SISMOGRAMME
C     -----------------------------
      DO 81 I = 1,NBTRAC
        SISMO(I,N) = U(IS, NP - NBTRAC + I, KP)
81    CONTINUE


C     PERMUTATION A CHAQUE ITERATION DES PLANS ESPACE
C     -----------------------------------------------
C
      KM = KP
      KP = 3 - KP

70    CONTINUE

C////////////////BEGIN FUNCTION TIMER/////////////////
      TFIN  = GETIME (W)
      T2CPU   = TFIN  - TINIT - TMESUR
      WRITE(1,1000)T2CPU

      TINIT = GETIME (W)
C////////////////END   FUNCTION TIMER/////////////////

C
C     SORTIE FICHIER DU SISMOGRAMME
C     -----------------------------

      WRITE (2,*) NBTRAC
      WRITE (2,*) NT
      DO 82 K = 1,NBTRAC
        DO 83 J = 1,NT
          WRITE (2,*) SISMO(K,J)
83      CONTINUE
82    CONTINUE

C////////////////BEGIN FUNCTION TIMER/////////////////
      TFIN  = GETIME (W)
      T3CPU   = T3CPU   + TFIN  - TINIT -TMESUR
      WRITE(1,1020) T3CPU
      WRITE(1,1030) T1CPU+T2CPU+T3CPU
C////////////////END   FUNCTION TIMER/////////////////

C
C     CALCUL DU NOMBRE DE MFLOPS
C     --------------------------

C     INITIALISATIONS
C     ---------------
      NBOP1 = 114.0 + (NP-3)*72.0 + (NP-4)*(NP-4)*4.0

C     RESOLUTION
C     ----------
      NBOP2 = (NT-1) * ( (NP-4)*(NP-4)*14.0 + (NP-3)*27.0
     &                   +(NP-4)*33.0 + 26.0)
      

      MFLOPS = NBOP2/ (T2CPU*1.0E06)
      WRITE (1,1060) MFLOPS

      MFLOPS = (NBOP1 + NBOP2)/ ((T1CPU + T2CPU)*1.0E06)
      WRITE (1,1065 ) MFLOPS

      MFLOPS = (NBOP1 + NBOP2)/
     &            ((T1CPU+T2CPU+T3CPU)*1.0E06)
      WRITE (1,1070) MFLOPS


C
C     CALCUL DE STATISTIQUES SUR LE RESULTAT
C     --------------------------------------
      UMAX = U(1,1,KM)
      UMIN = U(1,1,KM)
      DO 110 J = 1,NP
        DO 120 I = 1,NP
          IF (U(I,J,KM).LT.UMIN) THEN
            UMIN = U(I,J,KM)
          ENDIF
          IF (U(I,J,KM).GT.UMAX) THEN
            UMAX = U(I,J,KM)
          ENDIF
120     CONTINUE
110   CONTINUE
      WRITE (1,1050) UMIN,UMAX


      SISMAX = SISMO(1,1)
      SISMIN = SISMO(1,1)
      DO 111 I = 1,NBTRAC 
        DO 121 J = 1,NT
          IF (SISMO(I,J).LT.SISMIN) THEN
            SISMIN = SISMO(I,J)
          ENDIF
          IF (SISMO(I,J).GT.SISMAX) THEN
            SISMAX = SISMO(I,J)
          ENDIF
121     CONTINUE
111   CONTINUE
      WRITE (1,1055) SISMIN,SISMAX



      WRITE(1,1200)

      CLOSE(UNIT=1)
      CLOSE(UNIT=2)
      CLOSE(UNIT=3)
      CLOSE(UNIT=4)
    
      STOP


C     DECLARATIONS DE FORMATS DE SORTIE
C     ---------------------------------

1100  FORMAT (1X,'DEBUT DU PROGRAMME'//)

1120  FORMAT (1X,'MODELISATION PAR EQUATION D"ONDES',/,
     &        1X,'SCHEMA D"ORDRE 4 EN ESPACE ET 2 EN TEMPS'//)
1130  FORMAT (1X,'TAILLE DU MAILLAGE NPXNP       : ',I4,'  X',I4)
1135  FORMAT (1X,'NOMBRE DE PAS DE TEMPS         : ',I4)
1140  FORMAT (1X,'POSITION DE LA SOURCE          : (',I3,',',I3,')')
1171  FORMAT (1X,'PAS DE DISCR. EN ESPACE (H)    : ',E12.5)
1170  FORMAT (1X,'PAS DE DISCR. EN TEMPS (DELTAT): ',E12.5)
1180  FORMAT (1X,'FREQUENCE DE LA SOURCE (F)     : ',E12.5)
1051  FORMAT (1X,'MIN DE V:',E11.5,3X,'MAX DE V:  ',E11.5//)
1200  FORMAT (1X,/,1X,'FIN DU PROGRAMME')

1000  FORMAT (1X,'TEMPS CPU DE LA RESOLUTION     (RESOL) : ',E15.5)
1010  FORMAT (1X,'TEMPS CPU DES INITIALISATIONS   (INIT) : ',E15.5)
1020  FORMAT (1X,'TEMPS CPU DES ENTREES/SORTIES    (E/S) : ',E15.5)
1030  FORMAT (1X,'TEMPS CPU TOTAL       (RESOL+INIT+E/S) : ',E15.5//)
1015  FORMAT (1X,'TEMPS CPU DE L"APPEL AU TIMER          : ',E15.5)
1050  FORMAT (1X,'MIN DE U     :',E15.5,3X,'MAX DE U     : ',E15.5)
1055  FORMAT (1X,'MIN DE SISMO :',E15.5,3X,'MAX DE SISMO : ',E15.5//)
1060  FORMAT (1X,'MFLOPS CPU PARTIE VECTORIELLE  (RESOL) : ',E15.5)
1065  FORMAT (1X,'MFLOPS CPU RESOLUTION     (RESOL+INIT) : ',E15.5)
1070  FORMAT (1X,'MFLOPS CPU GLOBAL     (RESOL+INIT+E/S) : ',E15.5//)




C/////////////////////////////////////////////////////////////////////
C
C   PARTIE SPECIFIQUE A CHAQUE MACHINE ET A CHAQUE MODELISATION
C 
C            - FORMATS DES COMMENTAIRES
C            - OUVERTURE DE FICHIERS
C            - FONCTIONS DE GESTION DES TIMERS
C
C/////////////////////////////////////////////////////////////////////


1110  FORMAT (1X,'*********************************************',/,
     &        1X,'       RESOLUTION SUR SUN 4                  ',/,
     &        1X,'     DOUBLE PRECISION (64 bits)              ',/,
     &        1X,'         VERSION STANDART                    ',/,
     &        1X,'                                             ',/,
     &        1X,'*********************************************'//)

      END

C
C     OUVERTURE DES FICHIERS
C     ----------------------

      SUBROUTINE OPENFILE

      OPEN(UNIT=4,FILE='MODELES/param2.dat')

      OPEN(UNIT=3,FILE='MODELES/modele2.dat')

      OPEN(UNIT=1,FILE='SORTIES/stat2')

      OPEN(UNIT=2,FILE='SORTIES/sismo2')

      RETURN
      END

C
C     GESTION DU TIMER
C     ----------------

      SUBROUTINE STIMER (X)
C     INITIALISE LE TIMER - INUTILE SUR DE NOMBREUSES MACHINES
      RETURN
      END


      FUNCTION GETIME (X)
C     REND LE TEMPS CPU INSTANTANE, EN SECONDES.
          EXTERNAL TIME
          INTEGER TIME
          GETIME = REAL(TIME())
      RETURN
      END

      integer function time
      time = 0.
      end
