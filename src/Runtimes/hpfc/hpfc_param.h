c
c COMMON /HPFC_PARAM/
c
c $RCSfile: hpfc_param.h,v $ ($Date: 1994/04/11 10:23:54 $, )
c version $Revision$
c got on %D%, %T%
c $Id$
c
c the following files has to be included:
c     include 'parameters.h'
c
      common /HPFC_PARAM/
     $     ATOT(MAXNBOFARRAYS), 
     $     TTOP(MAXNBOFTEMPLATES), 
     $     NODIMA(MAXNBOFARRAYS),
     $     NODIMT(MAXNBOFTEMPLATES),
     $     NODIMP(MAXNBOFPROCESSORS), 
     $     RANGEA(MAXNBOFARRAYS, 7, 10),
     $     RANGET(MAXNBOFTEMPLATES, 7, 3),
     $     RANGEP(MAXNBOFPROCESSORS, 7, 3), 
     $     ALIGN(MAXNBOFARRAYS, 7, 3),
     $     DIST(MAXNBOFTEMPLATES, 7, 2)
c
      integer 
     $     ATOT, TTOP, NODIMA, NODIMT, NODIMP, 
     $     RANGEA, RANGET, RANGEP, ALIGN, DIST
c
