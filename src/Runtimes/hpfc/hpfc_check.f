!
! Checks run-time library version
!
! $RCSfile: hpfc_check.f,v $ version $Revision$
! ($Date: 1996/12/27 14:41:48 $, )
!
      subroutine hpfc check
      include 'global_parameters.h'
      include 'hpfc_commons.h'
      if  (REAL NB OF ARRAYS.GT.MAX NB OF ARRAYS.OR.
     $     REAL NB OF TEMPLATES.GT.MAX NB OF TEMPLATES.OR.
     $     REAL NB OF PROCESSORS.GT.MAX NB OF PROCESSORS.OR.
     $     REAL MAX SIZE OF PROCS.GT.MAX MAX SIZE OF PROCS.OR.
     $     REAL MAX SIZE OF BUFFER.GT.MAX MAX SIZE OF BUFFER) then
         write (unit=0, fmt=*) 
     $        'HPFC run-time library',
     $        '  must be recompiled with larger parameters'
         stop
      endif
!
! initialize common hpfc_dynamic
!
      NB OF ARRAYS = REAL NB OF ARRAYS
      NB OF TEMPLATES = REAL NB OF TEMPLATES
      NB OF PROCESSORS = REAL NB OF PROCESSORS
      MAX SIZE OF PROCS = REAL MAX SIZE OF PROCS
      SIZE OF BUFFER = REAL MAX SIZE OF BUFFER
      end
!
! that s all
!
