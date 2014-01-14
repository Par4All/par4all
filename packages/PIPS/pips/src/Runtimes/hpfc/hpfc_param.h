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
! COMMON /HPFC PARAM/
!
! the following files has to be included:
!     include 'parameters.h'
!
      common /HPFC PARAM/
     $     ATOT(MAX NB OF ARRAYS), 
     $     TTOP(MAX NB OF TEMPLATES), 
     $     NODIMA(MAX NB OF ARRAYS),
     $     NODIMT(MAX NB OF TEMPLATES),
     $     NODIMP(MAX NB OF PROCESSORS), 
     $     RANGEA(MAX NB OF ARRAYS, 7, 10),
     $     RANGET(MAX NB OF TEMPLATES, 7, 3),
     $     RANGEP(MAX NB OF PROCESSORS, 7, 3), 
     $     ALIGN(MAX NB OF ARRAYS, 7, 3),
     $     DIST(MAX NB OF TEMPLATES, 7, 2),
     $     MSTATUS(MAX NB OF ARRAYS),
     $     LIVE MAPPING(MAX NB OF ARRAYS)
!
      integer 
     $     ATOT, TTOP, NODIMA, NODIMT, NODIMP, 
     $     RANGEA, RANGET, RANGEP, ALIGN, DIST, MSTATUS
      logical LIVE MAPPING
!
