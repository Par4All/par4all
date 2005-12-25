!
! $Id$
!

!
! MPI commons 
!
     
      common /HPFC MPI COMMONS/
     $     HPFC TYPE MPI(8),
     $     HPFC REDFUNC MPI(4),
     $     HPFC COMMUNICATOR,
     $     HPFC COMM NODES,
     $     RECEIVED BY BROADCAST,
     $     BCAST LENGTH, 
     $     BUFFER ATTACHED
      integer HPFC TYPE MPI,
     $     HPFC REDFUNC MPI,
     $     HPFC COMMUNICATOR,
     $     HPFC COMM NODES,
     $     BCAST LENGTH 
      logical RECEIVED BY BROADCAST,
     $     BUFFER ATTACHED



!
! explicit manipulation of buffers
!

      common /HPFC MPI BUFFER/
     $     PACKING BUFFER POSITION,
     $     UNPACKING BUFFER POSITION
      integer
     $     PACKING BUFFER POSITION,
     $     UNPACKING BUFFER POSITION


!
! that's all 
!
