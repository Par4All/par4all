!
! program_do
!
! Example of a parallel do directive with private clause
!
! 2008
! Creation: A. Muller, 2008

      program parallel_do
      implicit none
      integer N
      parameter (N=10)
      integer i,a(N),B(N)
      real f

!$omp PARALLEL DO private(i) private(b)
      do 10 i=1,N
         a(i)=i
         B(i)=i
         f=0
 10   continue

      print *,'A=',a
      print *,'B=',b

      end
