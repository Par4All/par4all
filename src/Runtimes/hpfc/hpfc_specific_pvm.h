!
! $Id$
!
! $Log: hpfc_specific_pvm.h,v $
! Revision 1.2  1997/07/03 11:10:51  zory
! pvm specific declarations
!
! Revision 1.1  1997/06/10 07:57:03  zory
! Initial revision
!
!

!
! PVM encoding to use.
!

      common /HPFC PVM ENCODING/
     $     HPFC BUFFER ENCODING(2)
      integer HPFC BUFFER ENCODING

!
! PVM type to use
!
     
      common /HPFC PVM COMMONS/
     $     HPFC TYPE PVM(8)
      
      integer HPFC TYPE PVM

!
! that's all 
!
