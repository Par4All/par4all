!
! COMMON /HPFC GLOBS/
!
! $Id$
! $Log: hpfc_globs.h,v $
! Revision 1.5  1997/06/03 15:16:02  zory
! *** empty log message ***
!
! Revision 1.4  1997/03/20 07:51:13  coelho
! better RCS headers.
!
!
      common /HPFC GLOBS/
     $     OWNER TID,
     $     OWNERLID,
     $     OPN,
     $     OINDP(7),
     $     OREPLICATED,
     $     SENDER TID,
     $     SENDERLID,
     $     COMPUTER TID,
     $     COMPUTERLID,
     $     CPN,
     $     CINDP(7),
     $     CPOS(7),
     $     CPOSPN,
     $     CPOS COMPUTED,
     $     NEIGHBORLID,
     $     NTID
!
      integer 
     $     OWNER TID,    OWNERLID, OPN, OINDP,
     $     SENDER TID,   SENDERLID,
     $     COMPUTER TID, COMPUTERLID, CPN, CINDP, CPOS, CPOSPN,
     $     NEIGHBORLID, N TID
!
      logical
     $     O REPLICATED, CPOS COMPUTED
!
!
! that s all
!
