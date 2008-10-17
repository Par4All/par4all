c     PARAMETER.h
c
      INTEGER n000,n001,n002,n003,n004,n005,n006,n007,n008
      INTEGER nr00,maxper
      INTEGER l000,l001,l002,l022,l003,l005,l006,l007
      INTEGER m001,m002,m003,m004,m006,m007,m008,m009,m010
      INTEGER mxcol, mxrow
      INTEGER nzmax
c 
c     ----------------------------------
c     dimensions maximales du systeme  |
c     ----------------------------------
c
      PARAMETER ( n000 = 20)           ! facteur multiplicatif
      PARAMETER ( n001 = n000*10)      ! nombre max de modules
      PARAMETER ( n002 = n000*25)      ! nombre max de blocs
      PARAMETER ( n003 = n000*90)      ! nombre max etats+algebr.=ordre max syst.
                                       ! = nombre equations
      PARAMETER ( n004 = n000*25)      ! nombre max entrees vraies
      PARAMETER ( n005 = n000*100)     ! nombre max de variables
      PARAMETER ( n006 =  2000)          ! nombre max de variables dans un bloc
cmodeb loic
c      PARAMETER ( n007 = n000*500)     ! nombre max de donnees
      PARAMETER ( n007 = n000*500+1)   ! nombre max de donnees
cmofin loic
      PARAMETER ( n008 =  1800)          ! nombre maxi d'equations par module
c
      PARAMETER ( m001 = n005 - n003 ) ! nombre max total d'entrees
c                        m001 - n004   ! nombre max d'entrees connectees
c
      PARAMETER ( m002 = n002 + 1 )    ! 
      PARAMETER ( m003 = n003 + 1 )    ! 
      PARAMETER ( m004 = n004 + 1 )    ! 
c
c     ----------------------------------------------------
c     dimensions maximales pour la resolution du systeme |
c     ----------------------------------------------------
c
      PARAMETER ( nr00 = n006/2 )      ! nombre max termes <>0 ds chque equation
      PARAMETER ( m006 = n003 * nr00 ) ! nombre max termes <>0 du jacobien
cnag  PARAMETER ( m007 = 110000 )      ! nombre max termes <>0 jacobien factorise
cnag m007 mis a 110000 pour HARWELL NAG
      PARAMETER ( m007 = m006 * 2 )    ! nombre max termes <>0 jacobien factorise
      PARAMETER ( m008 = n003 * 5 )    ! dimension de ikma28,iwma28 (harwell)
      PARAMETER ( m009 = n003 * 8 )    ! dimension de iw (resolution harwell)
      PARAMETER (mxcol = m006)         ! pour ma28a (dimension LICN)
      PARAMETER (mxrow = m006)         ! pour ma28a (dimension LIRN)
      PARAMETER (nzmax = m006)         ! nombre de termes non nuls du vecteur jacobien
c
c     -------------------------------------------------------
c     dimensions maximales des regulations et perturbations |
c     -------------------------------------------------------
c
      PARAMETER ( l000 = 1 )           ! facteur multiplicatif
      PARAMETER ( l001 = l000 * 15)    ! nombre max de modules de regulation
      PARAMETER ( l002 = l001 * 3 )    ! nombre max de blocs de regulation
      PARAMETER ( l022 = l001 * 3+1)   ! 
      PARAMETER ( l003 = l002 * 10)    ! nombre max sort., sort. int.,entrees reg  
      PARAMETER ( l005 = l003)         ! nombre max var. regulation total 
                                       ! (entrees+sorties+sorties internes) 
      PARAMETER ( l006 = l002)         ! nombre max variables d'un bloc de regul. 
      PARAMETER ( l007 = 500)          ! nombre max de data par regulation
                                       ! (update PN.10:DARE08 rentre ds le rang)
      PARAMETER ( maxper = 5000 )      ! nombre max de perturbations
c
      PARAMETER ( m010 = n003 + n004 + l003 ) ! nombre var ds vecteur resultats
