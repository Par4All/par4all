! $RCSfile: hpfc_main.f,v $ (version $Revision$)
! $Date: 1997/06/03 15:12:56 $, 
!
! the main for the HPFC program (for both host and node)
!
      program MAIN
      include "hpfc_commons.h"
      call HPFC INIT MAIN
      if (MY TID.eq.HOST TID) then
       call HOST
      else
       call NODE
      end if
      end
!
! end of $RCSfile: hpfc_main.f,v $
!
