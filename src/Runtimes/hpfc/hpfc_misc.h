!
! $RCSfile: hpfc_misc.h,v $ (version $Revision$)
! $Date: 1997/01/02 18:42:19 $, 
!
      character*64 hpfc key
      common /HPFC MISC/ hpfc key
      integer 
     $     hpfc fake bufpck, 
     $     hpfc fake bufupk,
     $     hpfc fake bcasts
      common /hpfc counts/ 
     $     hpfc fake bufpck, 
     $     hpfc fake bufupk,
     $     hpfc fake bcasts
!
! that is all
!
