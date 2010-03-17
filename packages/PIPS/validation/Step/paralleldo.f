!
! sum program_do
!
! Example of a parallel do directive
!
! 2008
! Creation: A. Muller, 2008

      program parallel_do
      implicit none
      integer N
      parameter (N=10)
      integer i,a(N)

!$omp PARALLEL DO
      do 10 i=1,N
         a(i)=i
 10   continue

      print *,a

      end
