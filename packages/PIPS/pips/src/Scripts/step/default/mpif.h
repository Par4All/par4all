* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier
*
*This file is part of STEP.
*
*The program is distributed under the terms of the GNU General Public
*License.

***********************************************************************************************************
* Default mpif.h for re-entrance in PIPS                                                                  *
* This file is used as source for preprocessing include 'mpif.h' in "STEP.h" before source file splitting *
***********************************************************************************************************


      integer MPI_STATUS_SIZE
      parameter (MPI_STATUS_SIZE=5)

      integer MPI_MAX, MPI_MIN, MPI_SUM, MPI_PROD, MPI_LAND
      integer MPI_BAND, MPI_LOR, MPI_BOR, MPI_LXOR, MPI_BXOR
      
      parameter (MPI_MAX=1)
      parameter (MPI_MIN=2)
      parameter (MPI_SUM=3)
      parameter (MPI_PROD=4)
      parameter (MPI_LAND=5)
      parameter (MPI_BAND=6)
      parameter (MPI_LOR=7)
      parameter (MPI_BOR=8)
      parameter (MPI_LXOR=9)
      parameter (MPI_BXOR=10)
