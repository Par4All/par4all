!
! COMMON /HPFC GLOBS/
!
! $Id$
! $Log: hpfc_globs.h,v $
! Revision 1.4  1997/03/20 07:51:13  coelho
! better RCS headers.
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
