c
c Checks run-time library version
c
c $RCSfile: hpfc_check.f,v $ ($Date: 1995/09/15 15:57:02 $, )
c version $Revision$
c got on %D%, %T%
c $Id$
c
c
      subroutine hpfc_check()
      include 'real_parameters.h'
      include 'hpfc_commons.h'
      if  (REALNBOFARRAYS.GT.MAXNBOFARRAYS.OR.
     $     REALNBOFTEMPLATES.GT.MAXNBOFTEMPLATES.OR.
     $     REALNBOFPROCESSORS.GT.MAXNBOFPROCESSORS.OR.
     $     REALMAXSIZEOFPROCS.GT.MAXMAXSIZEOFPROCS.OR.
     $     REALMAXSIZEOFBUFFER.GT.MAXMAXSIZEOFBUFFER) then
         print *, 
     $        'HPFC run-time library',
     $        '  must be recompiled with larger parameters'
         stop
      endif
c
c initialize common hpfc_dynamic
c
      NBOFARRAYS = REALNBOFARRAYS
      NBOFTEMPLATES = REALNBOFTEMPLATES
      NBOFPROCESSORS = REALNBOFPROCESSORS
      MAXSIZEOFPROCS = REALMAXSIZEOFPROCS
c
c and hpfc_buffers
c
      SIZEOFBUFFER = REALMAXSIZEOFBUFFER
      end
c
c that s all
c
