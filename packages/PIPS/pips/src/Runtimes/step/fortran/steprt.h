      integer STEP_MAX_DIM
      parameter (STEP_MAX_DIM = 10)
      integer STEP_MAX_NBREQ
      parameter (STEP_MAX_NBREQ = 2*STEP_MAX_NBNODE*10)

      integer STEP_size
      integer STEP_rank
      COMMON /STEP/ STEP_size,STEP_rank
