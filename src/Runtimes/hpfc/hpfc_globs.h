!
! COMMON /HPFC GLOBS/
!
! $Id$
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
