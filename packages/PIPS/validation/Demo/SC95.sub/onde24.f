c onde24 in HPF
c
c     Adaptation hpf de onde24, Fabien Coelho et Ronan Keryell, 1993
c     
c     - taille NP passees en parame`tres et verifiee aux IO~;
c     - idem NT~;
c     - les surdeclarations inutiles ont ete corrigees (NPMAX+1->NP)
c     - tableaux B et UB coupes en 3~: E, S, W~;
c     - tableau temporaire UINT duplique en 3~: E, S, W~;
c     - reductions remplacees par des appels de fonctions~;
c     - directives de placement proposees par Ronan, et modifiees~;
c     - boucles paralle`les independantes,
c     - avec les variables privees~;
c     - mise en commentaire des mesures de performances~;
c     - & -> $ car marche mieux avec emacs~;
c     - fonctions REDMIN et REDMAX codees en 2D~;
c     - CONTINUE -> ENDDO
c     - boucle 71 distribuee
c     
c     + hint to the compiler (cfcd local/endlocal)
c
c     Il faudrait encore
c
c     - revoir les I/O...

      PROGRAM ONDE24

C  ------------------------------------------------------
C        MODELISATION PAR EQUATION D ONDES D UN PHENOMENE
C              DE PROPAGATION EN DIMENSION 2.
C
C  HYPOTHESES  :   - ON CONSIDERE UN DOMAINE CARRE .
C  ----------      - LA SURFACE EST LIBRE.
C                  - ON CONSIDERE DES CONDITIONS ABSORBANTES DU SECOND
C                    ORDRE SUR LES BORDS  (cf. these CHALINDAR).
C                  - LE MILIEU EST HETEROGENE.
C
C
C  UTILISATION DUN SCHEMA AUX DIFFERENCES FINIES DORDRE (2,4)
C  (ORDRE 2 EN TEMPS ET 4 EN ESPACE) COMME METHODE DE RESOLUTION :
C
C         - SCHEMA DORDRE (2,4) POUR LE DOMAINE INTERIEUR.
C         - SCHEMA DORDRE (2,2) POUR LE BORD INTERIEUR DU DOMAINE.
C         - CONDITIONS PARAXIALES DORDRE 2 POUR LE BORD DU DOMAINE.
C
C  -----------------------------------------------------



C  DECLARATION DE CONSTANTES
C  -------------------------
C  NPMAX      : NOMBRE DE POINTS MAXIMUM DU MAILLAGE SPATIAL
C  NTMAX      : NOMBRE MAXIMUM DE PAS DE TEMPS
C  NBTRAC   : NOMBRE DE TRACES SISMIQUES
C  TMAX        : TEMPS DE PRISE EN COMPTE DE LA FONCTION SOURCE (S)

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


C  DECLARATION DE TABLEAUX
C  -----------------------
C  U      :    SOLUTION DE LA MODELISATION PAR EQUATION DONDE DUN
C              PHENOMENE DE PROPAGATION. UTILISATION DUN SCHEMA AUX
C              DIFFERENCES FINIES COMME METHODE DE RESOLUTION.
C                U(.,.,KM) REPRESENTE U(t)
C              U(.,.,KP) REPRESENTE U(t-1) et U(t+1)
C
C  V      :    CHAMP DE VITESSE CARACTERISTIQUE DU MILIEU.
C              - LE TABLEAU MEMORISE LA CONSTANTE LIEE A LA VITESSE
C                     1/12*((V(i,j)*DELTAT/H)**2)
C                POUR LES POINTS INTERIEURS DU DOMAINE.
C              - LE TABLEAU MEMORISE LA CONSTANTE LIEE A LA VITESSE
C                     (V(i,j)*DELTAT/H)**2
C                POUR LES POINTS CONSTITUANT LE BORD INTERNE DU DOMAINE
C              - DANS LE CAS DES POINTS CONSTITUANT LES FRONTIERES
C                E-W-S, LE TABLEAU CONTIENT LA MOYENNE HARMONIQUE
C                DES VITESSES DES POINTS FRONTIERES ET DES POINTS
C                CONNEXES A LA FRONTIERE.
C              - DANS LE CAS DES POINTS CONSTITUANT LES COINS INF.
C                DU DOMAINE, ET LES DEUX POINTS FRONTIERES CONNEXES
C                AUX COINS,  LE TABLEAU CONTIENT LA MOYENNE HARMONIQUE
C                DES VITESSES DES 4 POINTS CONNEXES AU POINT CONCERNE.
C              - LES COINS SUPERIEURS SONT CONSIDERES COMME PARTIE
C                DE LA SURFACE LIBRE.
C
C  B      :    TABLEAUX DE 3 VECTEURS CONTENANT LES CONSTANTES LIEES
C              AUX EQUATIONS ASSOCIEES AUX BORDS.
C              B1 = 1 - (1/(V`*DELTAT + H)) * (2*H + (V`*DELTAT)**2/H)
C              B2 = 2*H / (V`*DELTAT + H)
C              B3 = (V`*DELTAT)*(V`*DELTAT) / 2*H*(V`*DELTAT + H)
C              AVEC V`: VITESSE HARMONIQUE SUR DEUX POINTS.
C
C              DANS LE CAS PARTICULIER DES 3 POINTS COMPOSANT
C              CHAQUE COIN E-S ET S-W, LES TABLEAUX B1 ET B2 
C              CONTIENNENT RESPECTIVEMENT LES CONSTANTES :
C                  B1   = 2*V`*DELTAT / ( 4*V`*DELTAT + 3*H)
C                  B2   = 3*H / (4*V`*DELTAT + 3*H)
C              AVEC V`: VITESSE HARMONIQUE SUR QUATRE POINTS.
C                  
C  UB     :    TABLEAU DE VECTEURS INTERMEDIAIRES UTILISES POUR LE 
C              CALCUL DES BORDS E-S-W ; LE CODAGE EST LE MEME QUE POUR 
C              LES VECTEURS B.
C
C  UINT   :    VECTEUR INTERMEDIAIRE UTILISE LORS DU CALCUL DE
C              LA SOLUTION AUX BORDS
C
C

c
c     nouvelles declarations avec NP et NT
c
c     pour BE, BW, UINT, (NP-1) est suffisant
c
      PARAMETER (NP = 800)
      PARAMETER (NT = 4000)

      REAL*8    U     (NP,NP,2)
      REAL*8    V     (NP,NP)
 
      REAL*8    BE   (NP,3)
      REAL*8    BS   (NP,3)
      REAL*8    BW   (NP,3)

      REAL*8    UBE  (NP)
      REAL*8    UBS  (NP)
      REAL*8    UBW  (NP)

      REAL*8    UINTE(NP)
      REAL*8    UINTS(NP)
      REAL*8    UINTW(NP)

      REAL*8    SISMO (NBTRAC,NT)

c
c     directives de placement HPF
c

CHPF$ TEMPLATE DOMAIN(NP,NP)
CHPF$ PROCESSORS PE(4,4)
CHPF$ DISTRIBUTE DOMAIN(BLOCK,BLOCK) ONTO PE

CHPF$ ALIGN V(I,J), U(I,J,*) WITH DOMAIN(I,J)

CHPF$ ALIGN UBE(I) WITH DOMAIN(I,799)
CHPF$ ALIGN UBS(I) WITH DOMAIN(799,I)
CHPF$ ALIGN UBW(I) WITH DOMAIN(I,2)

CHPF$ ALIGN BE(I,*), UINTE(I) WITH DOMAIN(I,800)
CHPF$ ALIGN BS(I,*), UINTS(I) WITH DOMAIN(800,I)
CHPF$ ALIGN BW(I,*), UINTW(I) WITH DOMAIN(I,1)

CHPF$ ALIGN SISMO(I,*) WITH DOMAIN(1,750+I)

C  DECLARATION DE SCALAIRES
C  ------------------------
C  NP       : NOMBRE DE POINT SELON LA DIMENSION X DU MAILLAGE SPATIAL
C  NT       : NOMBRE DE PAS DE TEMPS DE LA SIMULATION
C  DELTAT   : INTERVALLE DE TEMPS               (0.0005 S)
C  H        : INTERVALLE DESPACE               (1.5    M)
C  F        : FREQUENCE DE LA SOURCE            (30     HERTZ)
C  IS,JS    : COORDONNEES DE LA SOURCE            
C
C  KM,KP    : POINTEURS PERMETTANT DE PERMUTER LES PLANS ESPACE.
C  I,J,K,N  : INDICES DE BOUCLES
C  Wi       : VARIABLES DE TRAVAIL
C  G0       : INTENSITE DE LA SOURCE A LINSTANT t
C  TINIT,
C  TFIN,
C  TMESUR   : COMPTEURS DE GESTION DU TEMPS
C  TiCPU    : COMPTEURS DE GESTION DU TEMPS CPU
C  TiELAP   : COMPTEURS DE GESTION DU TEMPS ELAPSED
C  NBOPi    : COMPTEURS DU NOMBRE DOPERATIONS FLOTTANTES
C  MFLOPS   : NOMBRE DE MFLOPS MESURES
C  UMIN     : VALEUR MIN DE LA SOLUTION U AU DERNIER PAS DE TEMPS
C  UMAX     : VALEUR MAX DE LA SOLUTION U AU DERNIER PAS DE TEMPS
C  VMIN     : VALEUR MIN DE LA SOLUTION DU CHAMP DE VITESSE
C  VMAX     : VALEUR MAX DE LA SOLUTION DU CHAMP DE VITESSE
C  SISMIN   : VALEUR MIN DE LA SOLUTION DU SISMOGRAMME
C  SISMAX   : VALEUR MAX DE LA SOLUTION DU SISMOGRAMME
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

c      REAL*8    
c     $     TMESUR, TINIT, TFIN, T1CPU, T2CPU, T3CPU, 
c     $     NBOP1, NBOP2, MFLOPS

      REAL*8    UMIN, UMAX
      REAL*8    VMIN, VMAX
      REAL*8    SISMIN, SISMAX

      INTEGER   NPTMP, NTTMP
C
C             CORPS DU PROGRAMME
C             ------------------
     
      CALL OPENFILE

C
C ACQUISITION DES PARAMETRES DE LA MODELISATION
C -----------------------------------------

C WRITE (6,'('' ENTRER LE NB DE POINTS DU MAILLAGE CARRE (NP) ? '',$)')
      READ (4,*) NPTMP
      IF (NPTMP.NE.NP) STOP
C WRITE (6,'('' ENTRER LE NB DE PAS DE TEMPS TOTAL (NT) ? '',$)')
      READ (4,*) NTTMP
      IF (NTTMP.NE.NT) STOP
C WRITE (6,'('' ENTRER LE PAS DE DISCR. EN ESPACE (H) ? '',$)')
      READ (4,*) H
C WRITE (6,'('' ENTRER LE PAS DE DISCR. EN TEMPS (DELTAT) ? '',$)')
      READ (4,*) DELTAT
C WRITE (6,'('' ENTRER LA POSITION EN X DE LA SOURCE (JS) ? '',$)')
      READ (4,*) JS
C WRITE (6,'('' ENTRER LA POSITION EN Z DE LA SOURCE (IS) ? '',$)')
      READ (4,*) IS
C WRITE (6,'('' ENTRER LA FREQUENCE DE LA SOURCE (F) ? '',$)')
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

C
C INITIALISATION DES VARIABLES SCALAIRES
C ---------------------------------
      KM = 1
      KP = 2
      G0 = 0


C INITIALISATION DU CHAMP VITESSE
C V(I,J)     = READ (Vi) (CAS DUN MILIEU HETEROGENE)
C ----------------------------------------------

      READ (3,*) NPTMP
      IF (NPTMP.NE.NP) STOP
      READ (3,*) NPTMP
      IF (NPTMP.NE.NP) STOP
      DO I = 1,NP
         DO J = 1,NP
            READ (3,*) V(I,J)
         ENDDO
      ENDDO

c
c     reductions MIN et MAX
c
      VMIN = REDMIN(V(1,1), 1, NP, 1, NP)
      VMAX = REDMAX(V(1,1), 1, NP, 1, NP)
      
      WRITE (1,1051) VMIN,VMAX

C     INITIALISATION DE LA SOLUTION U
C     U(.,.,t=0) = U(.,.,t=1) = 0
C     ET DES ENREGISTREMENTS 
C     SISMO (T=1) = 0.0
C     ------------------------------

chpf$ independent(k,j,i)
      DO K = 1,2
         DO J = 1,NP
            DO I = 1,NP
               U(I,J,K) = 0.0
            ENDDO
         ENDDO
      ENDDO

chpf$ independent(i)
      DO I = 1,NBTRAC 
         SISMO(I,1) = 0.0
      ENDDO




C     INITIALISATION DES CONSTANTES LIEES AUX COINS INFERIEURS
C            V(I,J)  <--  moyenne harmonique sur 4 points
C            B(I,.,.)<--  cf declarations
C     ET AUX BORDS E-S-W
C            V(I,J)  <--  moyenne harmonique sur 2 points
C            B(I,.,.)<--  cf declarations
C     -------------------------------------------


C     CAS DES TROIS POINTS DU COIN E-S
C     --------------------------------

c
c how to help the compiler...
c
cfcd$ local
      C1          = 4/((1/V(NP-1,NP))+(1/V(NP-1,NP-1)) +
     $     (1/V(NP-2,NP))+(1/V(NP-2,NP-1)) )
      W           = C1*DELTAT
      BE(NP-1,1) = 2*W/(4*W+3*H)
      BE(NP-1,2) = 3*H/(4*W+3*H)
      V(NP-1,NP)  = C1
cfcd$ endlocal

cfcd$ local
      C2          = 4/((1/V(NP,NP))+(1/V(NP,NP-1)) +
     $     (1/V(NP-1,NP))+(1/V(NP-1,NP-1)) )
      W           = C2*DELTAT
      BS(NP,1) = 2*W/(4*W+3*H)
      BS(NP,2) = 3*H/(4*W+3*H)
      V(NP,NP)    = C2
cfcd$ endlocal

cfcd$ local
      C3          = 4/((1/V(NP,NP))+(1/V(NP,NP-1)) +
     $     (1/V(NP-1,NP))+(1/V(NP-1,NP-1)) )
      W           = C3*DELTAT
      BS(NP-1,1) = 2*W/(4*W+3*H)
      BS(NP-1,2) = 3*H/(4*W+3*H)
      V(NP,NP-1)  = C3
cfcd$ endlocal

C     CAS DES 3 POINTS DU COIN W-S
C     ----------------------------

cfcd$ local
      C1          = 4/((1/V(NP,2))+(1/V(NP,3)) +
     $     (1/V(NP-1,2))+(1/V(NP-1,3)) )
      W           = C1*DELTAT
      BS(2,1) = 2*W/(4*W+3*H)
      BS(2,2) = 3*H/(4*W+3*H)
      V(NP,2)    = C1
cfcd$ endlocal
      
cfcd$ local
      C2          = 4/((1/V(NP,1))+(1/V(NP,2)) +
     $     (1/V(NP-1,1))+(1/V(NP-1,2)) )
      W           = C2*DELTAT
      BS(1,1) = 2*W/(4*W+3*H)
      BS(1,2) = 3*H/(4*W+3*H)
      V(NP,1)    = C2
cfcd$ endlocal
      
cfcd$ local
      C3          = 4/((1/V(NP-1,1))+(1/V(NP-1,2)) +
     $     (1/V(NP-2,1))+(1/V(NP-2,2)) )
      W           = C3*DELTAT
      BW(NP-1,1) = 2*W/(4*W+3*H)
      BW(NP-1,2) = 3*H/(4*W+3*H)
      V(NP-1,1)  = C3
cfcd$ endlocal
      
C     CAS DES BORDS   E - S - W
C     -------------------------
      
c     
c     les variables reparties ecrites sont bien alignees
c     
      
chpf$ independent(i), new(W)
      DO I = 2,NP-2
         V(I,NP)   = 2/((1/V(I,NP))+(1/V(I,NP-1)))
         W        = V(I,NP)*DELTAT
         BE(I,1) = 1 - (1/(W+H)) * ( 2*H + W*W/H) 
         BE(I,2) = 2*H/(W+H)
         BE(I,3) = W*W/(2*H*(W+H))
      ENDDO
      
chpf$ independent(i), new(W)
      DO I = 3,NP-2
         V(NP,I)   = 2/((1/V(NP,I))+(1/V(NP-1,I)))
         W        = V(NP,I)*DELTAT
         BS(I,1) = 1 - (1/(W+H)) * ( 2*H + W*W/H) 
         BS(I,2) = 2*H/(W+H)
         BS(I,3) = W*W/(2*H*(W+H))
      ENDDO

chpf$ independent(i), new(W)
      DO I = 2,NP-2
         V(I,1)   = 2/((1/V(I,1))+(1/V(I,2)))
         W        = V(I,1)*DELTAT
         BW(I,1) = 1 - (1/(W+H)) * ( 2*H + W*W/H) 
         BW(I,2) = 2*H/(W+H)
         BW(I,3) = W*W/(2*H*(W+H))
      ENDDO

C
C     INITIALISATION DES CONSTANTES LIEES AUX BORDS INTERNES
C     ------------------------------------------------------

chpf$ independent(i)
      DO I = 3,NP-1
         V(I,2) = (V(I,2)*DELTAT/H)**2
      ENDDO

chpf$ independent(i)
      DO I = 3,NP-1
         V(I,NP-1) = (V(I,NP-1)*DELTAT/H)**2
      ENDDO

chpf$ independent(i)
      DO I = 3,NP-2
         V(NP-1,I) = (V(NP-1,I)*DELTAT/H)**2
      ENDDO


C     INITIALISATION DES CONSTANTES LIEES AU DOMAINE INTERIEUR
C            V(I,J) <--  1/12*((V(I,J)*DELTAT/H)**2)
C     -------------------------------------------

chpf$ independent(j,i)
      DO J = 3,NP-2
         DO I = 3,NP-2
            V(I,J) = ((V(I,J)*DELTAT/H)**2)/12
         ENDDO
      ENDDO

C     -----------------------------------------
C     !     MODELISATION DE LA PROPAGATION DES ONDES
C     !               PAR EQUATION DONDE.
C     !        RESOLUTION PAR DIFFERENCES FINIES.
C     -------------------------------------------


C     ----------------------------
C     !     BOUCLE EN TEMPS      !
C     ----------------------------

      DO N = 2,NT
         
C PRISE EN COMPTE DE LA FONCTION SOURCE PENDANT UN TEMPS DE TMAX SEC.
C Un = Un + SOURCE
C ---------------------------------------------------
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
C     boucle 71

chpf$ independent(i)
         DO I = 2,NP-1
            UBE(I) = U(I,NP-1,KP)
         ENDDO

chpf$ independent(i)
         DO I = 2,NP-1
            UBS(I) = U(NP-1,I,KP)
         ENDDO

chpf$ independent(i)
         DO I = 2,NP-1
            UBW(I) = U(I,2,KP)
         ENDDO
      
C
C     ------------------------------------------------
C     SOLUTION SUR LE DOMAINE INTERIEUR
C     SCHEMA DORDRE 2 EN TEMPS ET DORDRE 4 EN ESPACE
C     ------------------------------------------------

chpf$ independent(j,i)
         DO J = 3,NP-2
            DO I = 3, NP-2
               
               U(I,J,KP) = 
     $              2 * U(I,J,KM) - U(I,J,KP)
     $              - V(I,J) * ( 60 * U(I,J,KM) 
     $              - 16 * ( U(I+1,J,KM) + U(I-1,J,KM) 
     $                     + U(I,J-1,KM) + U(I,J+1,KM))
     $              +   U(I+2,J,KM) + U(I-2,J,KM) 
     $              + U(I,J-2,KM) + U(I,J+2,KM))
               
            ENDDO
         ENDDO

C
C     ------------------------------------------------
C     SOLUTION SUR LES BORDS INTERIEURS
C     SCHEMA DORDRE 2 EN TEMPS ET DORDRE 2 EN ESPACE
C     ------------------------------------------------

chpf$ independent(i)
         DO I = 3,NP-1
            U(I,2,KP) = 
     $           2 * U(I,2,KM) - U(I,2,KP) - V(I,2) * 
     $           ( 4 * U(I,2,KM) - ( U(I+1,2,KM) + U(I-1,2,KM) 
     $           + U(I,1,KM) + U(I,3,KM)))
         ENDDO
         
chpf$ independent(i)
         DO I = 3,NP-1
            U(I,NP-1,KP) = 
     $           2 * U(I,NP-1,KM) - U(I,NP-1,KP) - V(I,NP-1) * 
     $           ( 4 * U(I,NP-1,KM) - ( U(I+1,NP-1,KM) + U(I-1,NP-1,KM) 
     $           + U(I,NP-2,KM) + U(I,NP,KM)))
         ENDDO

chpf$ independent(i)
         DO I = 3,NP-2
            U(NP-1,I,KP) = 
     $           2 * U(NP-1,I,KM) - U(NP-1,I,KP) - V(NP-1,I) * 
     $           ( 4 * U(NP-1,I,KM) - ( U(NP,I,KM) + U(NP-2,I,KM) 
     $           + U(NP-1,I-1,KM) + U(NP-1,I+1,KM)))
         ENDDO

C
C     --------------------------------------------------------
C     SOLUTION AU BORD EAST. CONDITIONS ABSORBANTES DORDRE 2.
C     --------------------------------------------------------

chpf$ independent(i)
         DO I = 3,NP-2
            UINTE(I) = 
     $           BE(I,1)*( U(I,NP,KP) + U(I,NP-1,KP) ) - UBE(I) 
     $           + BE(I,2)*( U(I,NP,KM) + U(I,NP-1,KM) )
     $           + BE(I,3)*( U(I+1,NP-1,KP) + U(I+1,NP,KP) 
     $           + U(I-1,NP-1,KP) + U(I-1,NP,KP) )
         ENDDO
         
chpf$ independent(i)
         DO I = 3,NP-2
            U(I,NP,KP) = UINTE(I)
         ENDDO

C
C     --------------------------------------------------------
C     SOLUTION AU BORD SOUTH. CONDITIONS ABSORBANTES DORDRE 2.
C     --------------------------------------------------------

chpf$ independent(i)
         DO I = 3,NP-2
            UINTS(I) = 
     $           BS(I,1)*( U(NP,I,KP) + U(NP-1,I,KP) ) - UBS(I)
     $           + BS(I,2)*( U(NP,I,KM) + U(NP-1,I,KM) )
     $           + BS(I,3)*( U(NP-1,I+1,KP) + U(NP,I+1,KP) 
     $           + U(NP-1,I-1,KP) + U(NP,I-1,KP) )
         ENDDO

chpf$ independent(i)
         DO I = 3,NP-2
            U(NP,I,KP) = UINTS(I)
         ENDDO
 
C
C     --------------------------------------------------------
C     SOLUTION AU BORD WEST. CONDITIONS ABSORBANTES DORDRE 2.
C     --------------------------------------------------------

chpf$ independent(i)
         DO I = 3,NP-2
            UINTW(I) = 
     $           BW(I,1)*( U(I,1,KP) + U(I,2,KP)) - UBW(I)
     $           + BW(I,2)*( U(I,1,KM) + U(I,2,KM) )
     $           + BW(I,3)*( U(I+1,2,KP) + U(I+1,1,KP) 
     $           + U(I-1,2,KP) + U(I-1,1,KP) )
         ENDDO
      
chpf$ independent(i)
         DO I = 3,NP-2
            U(I,1,KP) = UINTW(I)
         ENDDO
      
C     -----------------------------------------------
C     CALCUL DU BORD DU COIN 
C     U(NP-1,NP,KP) ,  U(NP,NP-1,KP) ET U(NP-1,NP,KP)
C     -----------------------------------------------
   
cfcd$ local
         U(NP-1,NP,KP) = BE(NP-1,1)*(U(NP-1,NP-1,KP)+U(NP-2,NP,KP))
     $        + BE(NP-1,2)*U(NP-1,NP,KM)
cfcd$ endlocal

cfcd$ local
         U(NP,NP-1,KP) = BS(NP-1,1)*(U(NP,NP-2,KP)+U(NP-1,NP-1,KP))
     $        + BS(NP-1,2)*U(NP,NP-1,KM)
cfcd$ endlocal

cfcd$ local
         U(NP,NP,KP)   = BS(NP,1)*(U(NP,NP-1,KP)+U(NP-1,NP,KP))
     $        + BS(NP,2)*U(NP,NP,KM)
cfcd$ endlocal

C     -----------------------------------------------
C     CALCUL DU BORD DU COIN 
C     U(NP,2,KP) ,  U(NP-1,1,KP) ET U(NP,1,KP)
C     -----------------------------------------------

cfcd$ local
         U(NP-1,1,KP)  = BW(NP-1,1)*(U(NP-1,2,KP)+U(NP-2,1,KP))
     $        + BW(NP-1,2)*U(NP-1,1,KM)
cfcd$ endlocal
         
cfcd$ local
         U(NP,2,KP)    = BS(2,1)*(U(NP,3,KP)+U(NP-1,2,KP))
     $        + BS(2,2)*U(NP,2,KM)
cfcd$ endlocal

cfcd$ local
         U(NP,1,KP)    = BS(1,1)*(U(NP-1,1,KP)+U(NP,2,KP))
     $        + BS(1,2)*U(NP,1,KM)
cfcd$ endlocal

C
C     ENREGISTREMENT DU SISMOGRAMME
C     -----------------------------

chpf$ independent(i)
         DO I = 1,NBTRAC
            SISMO(I,N) = U(IS, NP - NBTRAC + I, KP)
         ENDDO


C     PERMUTATION A CHAQUE ITERATION DES PLANS ESPACE
C     -----------------------------------------------
C
         KM = KP
         KP = 3 - KP
         
      ENDDO
      
C     
C     SORTIE FICHIER DU SISMOGRAMME
C     -----------------------------
      
      WRITE (2,*) NBTRAC
      WRITE (2,*) NT
      DO K = 1,NBTRAC
         DO J = 1,NT
            WRITE (2,*) SISMO(K,J)
         ENDDO
      ENDDO


C
C     CALCUL DE STATISTIQUES SUR LE RESULTAT
C     --------------------------------------

c
c     reductions
c

      UMIN = REDMIN(U(1,1,KM), 1, NP, 1, NP)
      UMAX = REDMAX(U(1,1,KM), 1, NP, 1, NP)

      WRITE (1,1050) UMIN,UMAX


      SISMIN = REDMIN(SISMO(1,1), 1, NBTRAC, 1, NT)
      SISMAX = REDMAX(SISMO(1,1), 1, NBTRAC, 1, NT)

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
     $        1X,'SCHEMA D"ORDRE 4 EN ESPACE ET 2 EN TEMPS'//)
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
     $        1X,'       RESOLUTION SUR SUN 4                  ',/,
     $        1X,'     DOUBLE PRECISION (64 bits)              ',/,
     $        1X,'         VERSION STANDART                    ',/,
     $        1X,'                                             ',/,
     $        1X,'*********************************************'//)

      END

      SUBROUTINE OPENFILE
c
c     ouverture des fichiers
c

      OPEN(UNIT=4,FILE='MODELES/param2.dat')
      OPEN(UNIT=3,FILE='MODELES/modele2.dat')
      OPEN(UNIT=1,FILE='SORTIES/stat2')
      OPEN(UNIT=2,FILE='SORTIES/sismo2')

      RETURN

      END

c
c     calcul du min
c
      REAL*8 FUNCTION REDMIN(A, LO1, UP1, LO2, UP2)
      INTEGER LO1, UP1, LO2, UP2
      REAL*8 A(LO1:UP1,LO2:UP2), MINA
      INTEGER I,J
      MINA = A(LO1, LO2)
      DO J=LO2, UP2
         DO I=LO1, UP1
            IF (A(I,J).LT.MINA) MINA = A(I,J)
         ENDDO
      ENDDO
      REDMIN = MINA
      RETURN
      END
c
c     calcul du max
c
      REAL*8 FUNCTION REDMAX(A, LO1, UP1, LO2, UP2)
      INTEGER LO1, UP1, LO2, UP2
      REAL*8 A(LO1:UP1,LO2:UP2), MAXA
      INTEGER I,J
      MAXA = A(LO1, LO2)
      DO J=LO2, UP2
         DO I=LO1, UP1
            IF (A(I,J).GT.MAXA) MAXA = A(I,J)
         ENDDO
      ENDDO
      REDMAX = MAXA
      RETURN
      END

