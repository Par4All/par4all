#ifdef
* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier
*
*This file is part of STEP.
*
*The program is distributed under the terms of the GNU General Public
*License.
#endif
#if INTERLACED
#define MERGE _Merge
#else
#define MERGE
#endif

#define SUBROUTINES(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
#if INTERLACED                                                            !$\
      STEP_InitInterlaced(TYPE_SUFFIX,TYPE_FORTRAN)                       !$\
      MergeRegion(TYPE_SUFFIX,TYPE_FORTRAN)                               !$\
#else                                                                     !$\
      AlltoAll_nBlocking(TYPE_SUFFIX,TYPE_FORTRAN)                        !$\
      InitReduction(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)                    !$\
      Reducion(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)                         !$\
      STEP_MasterToAllScalar(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)           !$\
      STEP_MasterToAllRegion(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)           !$\
#endif                                                                    !$\
STEP_AlltoAllRegion(MERGE,TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)              !$\
AlltoAll_even(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)                             !$\
AlltoAll_even2(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)                            !$\
AlltoAll_odd(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)                              !$\
AlltoAll_odd2(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)

/*! SUBROUTINES(type_suffix, type_fortran,type_MPI)                       */

SUBROUTINES(_I1,integer*1,MPI_INTEGER1)
SUBROUTINES(_I2,integer*2,MPI_INTEGER2)
SUBROUTINES(_I4,integer*4,MPI_INTEGER4)    /*! Fortran : integer          */
SUBROUTINES(_I8,integer*8,MPI_INTEGER8)
SUBROUTINES(_R4,real*4,MPI_REAL4)          /*! Fortran : real             */
SUBROUTINES(_R8,real*8,MPI_REAL8)          /*! Fortran : double precision */
SUBROUTINES(_C8,complex*8,MPI_COMPLEX8)    /*! Fortran : complex          */
SUBROUTINES(_C16,complex*16,MPI_COMPLEX16)
