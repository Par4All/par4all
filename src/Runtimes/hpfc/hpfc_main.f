! $RCSfile: hpfc_main.f,v $ (version $Revision$)
! $Date: 1996/09/07 14:49:29 $, 
!
! the main for the HPFC program (for both host and node)
!
      program MAIN
      include "hpfc_commons.h"
      call HPFC_INIT_MAIN
      if (HOSTTID.eq.MYTID) then
         call HOST
      else
         call NODE
      end if
      end
!
! end of $RCSfile: hpfc_main.f,v $
!
