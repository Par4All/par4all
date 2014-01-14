!
! $Id$
!
! Copyright 1989-2014 MINES ParisTech
!
! This file is part of PIPS.
!
! PIPS is free software: you can redistribute it and/or modify it
! under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! any later version.
!
! PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
! WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE.
!
! See the GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
!

!
! Stubs for PIPS to deal with special FC directives. 
! functions with side effects, just in case. 
!
! Fabien COELHO, 09/95
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
