c
CC*********************************************************************
CC
CC    PROGRAMME PRINCIPAL 
CC
CC---------------------------------------------------------------------
CC
CC    appel des sous-programmes :
CC          --> LEGO12	:	constitution de la topologie du modele
CC	    --> LEG334	:	resolution de l'etat permanent
CC	    --> LEGO45	:	resolution de l'etat transitoire
CC 
CC---------------------------------------------------------------------
CC
CC    MAIN     Appelle...    GARGZZ   LEGO12   LEG334   LEGO45
CC
CC*********************************************************************
c
c
      IMPLICIT NONE
      include 'edf.h'
c
c
      INTEGER ietape            ! numero de l'etape
      REAL etape                ! argument de GARGZZ (numero de l'etape)
      CHARACTER*8 noblc(n002)   ! tableau des noms des blocs
      INTEGER islb(n002)        ! index des modules des blocs (INDICE)
      INTEGER nusta(n002)       ! nombre variables etat par bloc (apres ORDTOP)

      INTEGER nusci(n002)       ! nombre variables algebriques par bloc (ORDTOP)

      INTEGER ningr(n002)       ! nombre total d'entrees

      CHARACTER*8 sivar(n003)   ! nom variables de sortie (algebrique puis etat)

      CHARACTER*40 liblsi(n003) ! libelle long des variables de sortie

      CHARACTER*10 unitsi(n003) ! unite des variables de sortie

      CHARACTER*8 vari(n004)    ! nom des variables d'entree

      CHARACTER*40 liblri(n004) ! libelle long des variables d'entree

      CHARACTER*10 unitri(n004) ! unite des variables d'entree

      REAL*8 xy(n003)           ! valeurs des variables de sortie

      REAL*8 uu(n004)           ! valeur des variables d'entree du modele

      REAL*8 xyu(n005)          ! valeurs des variables par bloc

      REAL*8 cnxyu(n005)        ! constantes de normalisation des XYU

      REAL*8 toll(n003)         ! tolerances pour chaque equation

      REAL*8 dati(n007)         ! donnees blocs (modifiables par  modules)

      INTEGER ipdati(m002)      ! pointeur des donnees des blocs dans DATI

      INTEGER ip(m002)          ! pointeur des blocs dans XYU 

      INTEGER ipvrs(n005)       ! pointeur des variables dans SIVAR et VARI

      INTEGER ips(m003)         ! [ips(i+1)-1]-ips(i)=nmbre connections sortie i

      INTEGER ipvrt(n005)       ! pointe les sorties dans le vecteur VAR

      INTEGER ipi(m004)         ! [ipi(i+1)-1]-ipi(i)=nmbre connections entree i

      INTEGER ipvri(m001)       ! pointe les entrees dans le vecteur VAR

c
c===========================
c     CORPS DU PROGRAMME
c===========================
c
c  on recupere le parametre de la ligne de commande
      CALL GARGZZ(1,etape)
      ietape = INT(etape)
c
c     appel etape 12
c
      IF (ietape.EQ.12) THEN
         CALL LEGO12
c
c     appel etape 334
c
      ELSEIF (ietape.EQ.334) THEN
         CALL LEG334(
     i               noblc,islb
     i              ,nusta,nusci,ningr
     i              ,sivar,liblsi,unitsi
     i              ,vari, liblri,unitri
     i              ,xy,uu,xyu
     i              ,cnxyu,toll
     i              ,dati,ipdati
     i              ,ip,ipvrs,ips,ipvrt,ipi,ipvri
     .              )
c
c     appel etape 45
c
      ELSEIF (ietape.EQ.45) THEN
         CALL LEGO45(
     i               noblc,islb
     i              ,nusta,nusci,ningr
     i              ,sivar,liblsi,unitsi
     i              ,vari, liblri,unitri
     i              ,xy,uu,xyu
     i              ,cnxyu,toll
     i              ,dati,ipdati
     i              ,ip,ipvrs,ips,ipvrt,ipi,ipvri
     .              )
      ENDIF
      END
