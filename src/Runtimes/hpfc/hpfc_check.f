c
c Checks run-time library version
c
c $RCSfile: hpfc_check.f,v $ ($Date: 1994/05/13 14:48:29 $, )
c version $Revision$
c got on %D%, %T%
c $Id$
c
c
      subroutine hpfc_check()
      include 'parameters.h'
      include 'hpfc_commons.h'
      if  (REALNBOFARRAYS.GT.MAXNBOFARRAYS.OR.
     $     REALNBOFTEMPLATES.GT.MAXNBOFTEMPLATES.OR.
     $     REALNBOFPROCESSORS.GT.MAXNBOFPROCESSORS.OR.
     $     REALMAXSIZEOFPROCS.GT.MAXMAXSIZEOFPROCS) then
         print *, 
     $        'HPFC run-time library',
     $        '  must recompiled with larger parameters'
         stop
      endif
c
c initialize common hpfc_dynamic
c
      NBOFARRAYS = REALNBOFARRAYS
      NBOFTEMPLATES = REALNBOFTEMPLATES
      NBOFPROCESSORS = REALNBOFPROCESSORS
      MAXSIZEOFPROCS = REALMAXSIZEOFPROCS
      end
c
c that s all
c
