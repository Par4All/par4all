c
c Checks run-time library version
c
c $RCSfile: hpfc_check.f,v $ version $Revision$
c ($Date: 1995/12/01 11:49:57 $, )
c
      subroutine hpfc_check()
      include 'real_parameters.h'
      include 'hpfc_commons.h'
      if  (REALNBOFARRAYS.GT.MAXNBOFARRAYS.OR.
     $     REALNBOFTEMPLATES.GT.MAXNBOFTEMPLATES.OR.
     $     REALNBOFPROCESSORS.GT.MAXNBOFPROCESSORS.OR.
     $     REALMAXSIZEOFPROCS.GT.MAXMAXSIZEOFPROCS.OR.
     $     REALMAXSIZEOFBUFFER.GT.MAXMAXSIZEOFBUFFER) then
         write (unit=0, fmt=*) 
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
      SIZEOFBUFFER = REALMAXSIZEOFBUFFER
      end
c
c that s all
c
