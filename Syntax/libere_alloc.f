      subroutine LIBERE_ALLOC(IT_ALLOC,T_PTR)
C
C***********************************************************************
C
C  PROGICIEL : COCCINELLE      AU   9/9/99  PAR E.LEMOINE (SIMULOG)
C
C***********************************************************************
C
C  FONCTION : PROGRAMME DE LIBERATION DE L'ARCHIVAGE DES ALLOCATIONS
C
C-----------------------------------------------------------------------
C                         ARGUMENTS
C .________________.____.______________________________________________.
C ^    NOM         ^MODE^                     ROLE                     ^
C ^________________^____^______________________________________________^
C ^                ^    ^                                              ^
C ^ - IT_ALLOC     ^ -->^ TYPE D'ALLOCATION                            ^
C ^                ^    ^                                              ^
C ^ - T_PTR        ^ -->^ POINTEUR                                     ^
C ^                ^    ^                                              ^
C ^________________^____^______________________________________________^
C  MODE: -->(DONNEE NON MODIFIEE).<--(RESULTAT).<-->(DONNEE MODIFIEE)
C
C-----------------------------------------------------------------------
C%
C  SOUS-PROGRAMMES APPELES   :
C  SOUS-PROGRAMMES APPELANTS : LIBERE
C%
C***********************************************************************
C
! include "malloc.h"
c 
c.. Parameters .. 
      integer*4 NB_TYPE,NB_ALMAX
      parameter (NB_ALMAX = 300, NB_TYPE = 1)
c 
c.. Common Blocks .. 
c 
c.. Common Block of Size  3 (words)  ... 
c 
c... Variables in Common Block /tab1/ ... 
      integer*4 NB_ALLOC(NB_TYPE)
      common /tab1/ NB_ALLOC
c 
c.. Common Block of Size  900 (words)  ... 
c 
c... Variables in Common Block /tab2/ ... 
      integer*4 TAB_ALLOC(NB_ALMAX,NB_TYPE)
      common /tab2/ TAB_ALLOC
c 
c.. Common Block of Size  900 (words)  ... 
c 
c... Variables in Common Block /tab3/ ... 
      integer*4 SIZ_ALLOC(NB_ALMAX,NB_TYPE)
      common /tab3/ SIZ_ALLOC
c 
c.. Common Block of Size  1800 (words)  ... 
c 
c... Variables in Common Block /tab4/ ... 
      character NOM_ALLOC(NB_ALMAX,NB_TYPE)*10
      common /tab4/ NOM_ALLOC
c 
c.. Common Block of Size  9 (words)  ... 
c 
c... Variables in Common Block /tab5/ ... 
      integer*4 ITO_ALLOC(NB_TYPE),MAX_ALLOC(NB_TYPE),MAX_NBELT(NB_TYPE)
      common /tab5/ ITO_ALLOC,MAX_ALLOC,MAX_NBELT
! end include "malloc.h"
!       include 'malloc.h'
c 
c.. Formal Arguments .. 
      integer*4 IT_ALLOC
      pointer (T_PTR,A)
      real A(*)
c 
c.. Local Scalars .. 
      integer*4 I,J,NB
      logical*4 TROUVE
c 
c.. External Calls .. 
      external ARRET, FINALLOC
c 
c ... Executable Statements ...
c 
      TROUVE = .false.
c---
c 1. Liberation de l'archivage 
c---
c
c  allocation entiers     :  IT_ALLOC = 1
c  allocation reel simple :  IT_ALLOC = 2
c  allocation reel double :  IT_ALLOC = 3
c
c
c --> Si le pointeur est nul
c
      if (T_PTR .eq. 0) then
         write (6,1010)
         call FINALLOC
         call ARRET(198)
         stop
      end if
      if (IT_ALLOC.ge.1 .and. IT_ALLOC.le.3) then
         NB = NB_ALLOC(IT_ALLOC)
         do 10 I = 1,NB
c
c --------> Le pointeur est retrouve dans la table
c
            if (T_PTR .eq. TAB_ALLOC(I,IT_ALLOC)) then
               TROUVE = .true.
               ITO_ALLOC(IT_ALLOC) =
     *           ITO_ALLOC(IT_ALLOC) - SIZ_ALLOC(I,IT_ALLOC)
               do 20 J = I,NB-1
                  TAB_ALLOC(J,IT_ALLOC) = TAB_ALLOC(J+1,IT_ALLOC)
                  SIZ_ALLOC(J,IT_ALLOC) = SIZ_ALLOC(J+1,IT_ALLOC)
                  NOM_ALLOC(J,IT_ALLOC) = NOM_ALLOC(J+1,IT_ALLOC)
 20            continue
c               TAB_ALLOC(NB,IT_ALLOC) = 0
c               SIZ_ALLOC(NB,IT_ALLOC) = 0
c               NOM_ALLOC(NB,IT_ALLOC) = ''
               NB_ALLOC(IT_ALLOC) = NB_ALLOC(IT_ALLOC) - 1
            end if
 10      continue
      end if
c
c --> Le pointeur n'a pas ete retrouve dans la table
c
      if (.not. TROUVE) then
         write (6,1020) T_PTR
         call FINALLOC
         stop
      end if
      return
c 
c ... Format Declarations ...
c 
 1000 format (1x,'--> liberation : ',a8,3(' : ',i10))
c
c----%---+----=----+----=----+----=----+----=----+----=----+----=----+--
c
 1010 format (1x,'LE POINTEUR A LIBERER EST DEJA NUL !!!')
 1020 format (1x,'LE POINTEUR ',i10,' A LIBERER N''A PAS ETE TROUVE !!!'
     *)
      end
