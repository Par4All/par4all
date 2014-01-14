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
! inclusions for run-time support functions
!
      include 'hpfc_parameters.h'
      include 'hpfc_param.h'
      include 'hpfc_globs.h'
      include 'hpfc_procs.h'
      include 'hpfc_buffers.h'
      include 'hpfc_misc.h'
      include 'hpfc_types.h'
!
! parameters value to be defined when starting a program.
!
      integer
     $     NBOFARRAYS,
     $     NBOFTEMPLATES,
     $     NBOFPROCESSORS,
     $     MAXSIZEOFPROCS,
     $     SIZEOFBUFFER
      common /hpfc dynamic/ 
     $     NBOFARRAYS,
     $     NBOFTEMPLATES,
     $     NBOFPROCESSORS,
     $     MAXSIZEOFPROCS,
     $     SIZEOFBUFFER
!
! that s all
!
