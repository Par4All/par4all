! $RCSfile: hpfc_main.f,v $ (version $Revision$)
! $Date: 1996/09/07 15:50:52 $, 
!
! the main for the HPFC program (for both host and node)
!
      program MAIN
      include "hpfc_commons.h"
      call HPFC INIT MAIN
      if (HOST TID.eq.MY TID) then
         call HOST
      else
         call NODE
      end if
      end
!
! end of $RCSfile: hpfc_main.f,v $
!
