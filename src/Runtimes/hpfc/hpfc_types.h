!
! $Id$
!
! $Log: hpfc_types.h,v $
! Revision 1.2  1997/07/03 11:11:54  zory
! type manipulation is pvm/mpi independent
!
! Revision 1.1  1997/06/10 07:55:57  zory
! Initial revision
!
!

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
