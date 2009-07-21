* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier
*
*This file is part of STEP.
*
*The program is distributed under the terms of the GNU General Public
*License.



      include 'mpif.h'

      integer STEP_MAX_NBNODE
      parameter (STEP_MAX_NBNODE = 16)

      integer MAX_NB_LOOPSLICES
      parameter (MAX_NB_LOOPSLICES = STEP_MAX_NBNODE)

      integer STEP_MAX_DIM
      parameter (STEP_MAX_DIM = 10)

      integer STEP_MAX_NBREQ
      parameter (STEP_MAX_NBREQ = 2*STEP_MAX_NBNODE*10)


      integer*4 STEP_size
      integer*4 STEP_rank
      COMMON /STEP/ STEP_size,STEP_rank

      integer STEP_SizeRegion

      integer*4 STEP_Status(1:MPI_STATUS_SIZE)
      integer*4 STEP_NbRequest

      integer IDX_SLICE_LOW,IDX_SLICE_UP,STEP_IDX
      parameter (IDX_SLICE_LOW = 1,IDX_SLICE_UP=2)

      integer STEP_NONBLOCKING,STEP_BLOCKING1,STEP_BLOCKING2
      parameter (STEP_NONBLOCKING=0,STEP_BLOCKING1=1,STEP_BLOCKING2=2)

      integer STEP_SUM,STEP_PROD
      integer STEP_AND,STEP_OR
      integer STEP_MIN,STEP_MAX
      integer STEP_IAND,STEP_IOR,STEP_IEOR
      parameter (STEP_SUM=MPI_SUM,STEP_PROD=MPI_PROD)
      parameter (STEP_AND=MPI_LAND,STEP_OR=MPI_LOR)
      parameter (STEP_MIN=MPI_MIN,STEP_MAX=MPI_MAX)
      parameter (STEP_IAND=MPI_BAND,STEP_IOR=MPI_BOR,STEP_IEOR=MPI_BXOR)
