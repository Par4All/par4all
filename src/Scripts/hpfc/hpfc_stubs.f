!
! Stubs for PIPS to deal with special FC directives. 
! functions with side effects, just in case. 
!
! (c) Fabien COELHO, 09/95
!
! $Id$
!
! synchronization 
      subroutine hpfc1
!!fcd$ fake
      print *, 'hpfc1: '
      end
! timer on
      subroutine hpfc2
!!fcd$ fake
      print *, 'hpfc2: '
      end
! timer off
      subroutine hpfc3(comment)
!!fcd$ fake
      character comment*(*)
      print *, 'hpfc3: ', comment
      end
      subroutine hpfc0(comment)
!!fcd$ fake
      character comment*(*)
      print *, 'hpfc0: ', comment
      end
! io/host section marker
      subroutine hpfc7
!!fcd$ fake
      print *, 'hpfc7: '
      end
! dead FC directive. one argument, why not...
      subroutine hpfc8(x)
!!fcd$ fake
      integer x
      print *, 'hpfc8: ', x
      end
!
! That is all
!
