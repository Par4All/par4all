!
! $Id$
!
! $Log: hpfc_specific_mpi.h,v $
! Revision 1.3  1997/07/21 14:30:58  zory
! new commons are useful with buffered communications
!
! Revision 1.2  1997/07/03 11:10:32  zory
! specific mpi commons
!
! Revision 1.1  1997/06/16 12:52:20  zory
! Initial revision
!
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
