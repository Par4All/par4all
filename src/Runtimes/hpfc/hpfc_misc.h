c
c $RCSfile: hpfc_misc.h,v $ (version $Revision$)
c $Date: 1996/09/07 16:21:28 $, 
c
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
c
c that is all
c
