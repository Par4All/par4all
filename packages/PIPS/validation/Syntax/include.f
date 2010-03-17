! some header
      program include

      implicit none

      include 'include_inc.h'
      include "include_inc.h"
  
      call sub

      end
! some space
      subroutine sub
      implicit none
      include 'include_inc.h'
! include 'in a comment'
      end
! some trailer
