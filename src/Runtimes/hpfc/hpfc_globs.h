!
! COMMON /HPFC GLOBS/
!
! $RCSfile: hpfc_globs.h,v $ ($Date: 1996/09/07 16:22:33 $, )
! version $Revision$
! got on %D%, %T%
! $Id$
!
!
      common /HPFC GLOBS/
     $     OWNER TID,
     $     OLID,
     $     OPN,
     $     OINDP(7),
     $     OREPLICATED,
     $     SENDER TID,
     $     SLID,
     $     COMPUTER TID,
     $     CLID,
     $     CPN,
     $     CINDP(7),
     $     CPOS(7),
     $     CPOSPN,
     $     CPOS COMPUTED,
     $     NLID,
     $     NTID
!
      integer 
     $     OWNER TID,    OLID, OPN, OINDP,
     $     SENDER TID,   SLID,
     $     COMPUTER TID, CLID, CPN, CINDP, CPOS, CPOSPN,
     $     N LID, N TID
!
      logical
     $     O REPLICATED, CPOS COMPUTED
!
!
! that s all
!
