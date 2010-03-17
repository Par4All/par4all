!
! barrier program
!
! Example of a barrier directive
!
! 2008
! Creation: A. Muller, 2008
!
! deadlock if N not a multiple of nb processes

      program barrier
      implicit none
      integer N
      parameter (N=12)
      integer i,a(N)

!$omp parallel do
      do 10 i=1,N
         a(i)=i
         print *,i
!$omp barrier
 10   continue

      print *, a
      end
