!  $Id$
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
! HPFC Types definition 
!

      integer 
     $     HPFC INTEGER2,
     $     HPFC INTEGER4,
     $     HPFC REAL4,
     $     HPFC REAL8,
     $     HPFC STRING,
     $     HPFC BYTE1,
     $     HPFC COMPLEX8,
     $     HPFC COMPLEX16


      parameter(HPFC INTEGER2 = 1) 
      parameter(HPFC INTEGER4 = 2)
      parameter(HPFC REAL4 = 3)
      parameter(HPFC REAL8 = 4)
      parameter(HPFC STRING = 5)
      parameter(HPFC BYTE1 = 6)
      parameter(HPFC COMPLEX8 = 7)
      parameter(HPFC COMPLEX16 = 8)


!
! that's all
!
