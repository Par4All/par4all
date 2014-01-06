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
! Checks run-time library version
!
      subroutine hpfc check
      include 'global_parameters.h'
      include 'hpfc_commons.h'
      if  (REAL NB OF ARRAYS.GT.MAX NB OF ARRAYS.OR.
     $     REAL NB OF TEMPLATES.GT.MAX NB OF TEMPLATES.OR.
     $     REAL NB OF PROCESSORS.GT.MAX NB OF PROCESSORS.OR.
     $     REAL MAX SIZE OF PROCS.GT.MAX MAX SIZE OF PROCS.OR.
     $     REAL MAX SIZE OF BUFFER.GT.MAX MAX SIZE OF BUFFER) then
         write (unit=0, fmt=*) 
     $        'HPFC run-time library',
     $        '  must be recompiled with larger parameters'
         stop
      endif
!
! initialize common hpfc_dynamic
!
      NB OF ARRAYS = REAL NB OF ARRAYS
      NB OF TEMPLATES = REAL NB OF TEMPLATES
      NB OF PROCESSORS = REAL NB OF PROCESSORS
      MAX SIZE OF PROCS = REAL MAX SIZE OF PROCS
      SIZE OF BUFFER = REAL MAX SIZE OF BUFFER
      end
!
! that s all
!
