c
c inclusions for run-time support functions
c
c $RCSfile: hpfc_commons.h,v $ ($Date: 1994/04/11 10:23:52 $, )
c version $Revision$
c got on %D%, %T%
c $Id$
c
c
      include 'hpfc_parameters.h'
      include 'hpfc_param.h'
      include 'hpfc_globs.h'
      include 'hpfc_procs.h'
c
c parameters value to be define at the beginning
c
      integer
     $     NBOFARRAYS,
     $     NBOFTEMPLATES,
     $     NBOFPROCESSORS,
     $     MAXSIZEOFPROCS 
      common /hpfc_dynamic/ 
     $     NBOFARRAYS,
     $     NBOFTEMPLATES,
     $     NBOFPROCESSORS,
     $     MAXSIZEOFPROCS 
c
c that s all
c
