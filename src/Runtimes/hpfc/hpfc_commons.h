c
c inclusions for run-time support functions
c
c $RCSfile: hpfc_commons.h,v $ version $Revision$
c ($Date: 1995/09/27 18:03:33 $, )
c
c
      include 'hpfc_parameters.h'
      include 'hpfc_param.h'
      include 'hpfc_globs.h'
      include 'hpfc_procs.h'
      include 'hpfc_buffers.h'
      include 'hpfc_counts.h'
c
c parameters value to be defined when starting a program.
c
      integer
     $     NBOFARRAYS,
     $     NBOFTEMPLATES,
     $     NBOFPROCESSORS,
     $     MAXSIZEOFPROCS,
     $     SIZEOFBUFFER
      common /hpfc_dynamic/ 
     $     NBOFARRAYS,
     $     NBOFTEMPLATES,
     $     NBOFPROCESSORS,
     $     MAXSIZEOFPROCS,
     $     SIZEOFBUFFER
c
c that s all
c
