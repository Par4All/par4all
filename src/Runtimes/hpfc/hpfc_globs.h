c
c COMMON /HPFC_GLOBS/
c
c $RCSfile: hpfc_globs.h,v $ ($Date: 1994/09/21 17:23:03 $, )
c version $Revision$
c got on %D%, %T%
c $Id$
c
c
      common /HPFC_GLOBS/
     $     OWNERTID,
     $     OLID,
     $     OPN,
     $     OINDP(7),
     $     OREPLICATED,
     $     SENDERTID,
     $     SLID,
     $     COMPUTERTID,
     $     CLID,
     $     CPN,
     $     CINDP(7),
     $     CPOS(7),
     $     CPOSPN,
     $     CPOSCOMPUTED,
     $     NLID,
     $     NTID
c
      integer 
     $     OWNERTID,    OLID, OPN, OINDP,
     $     SENDERTID,   SLID,
     $     COMPUTERTID, CLID, CPN, CINDP, CPOS, CPOSPN,
     $     NLID, NTID
c
      logical
     $     OREPLICATED, CPOSCOMPUTED
c
c
c that s all
c
