c
c $RCSfile: hpfc_misc.h,v $ (version $Revision$)
c $Date: 1996/03/14 11:10:31 $, 
c
      character*64 hpfc_key
      common /HPFC_MISC/ hpfc_key
      integer 
     $     hpfc_fake_bufpck, 
     $     hpfc_fake_bufupk,
     $     hpfc_fake_bcasts
      common /hpfc_counts/ 
     $     hpfc_fake_bufpck, 
     $     hpfc_fake_bufupk,
     $     hpfc_fake_bcasts
c
c that is all
c
