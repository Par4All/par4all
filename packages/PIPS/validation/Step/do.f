!
! do.f
!
! Example of a parallel and do directives
!
! 2008-2009
! Creation: A. Muller, FSC

      program my_do
      implicit none
      integer N
      parameter (N=10)
      integer i,a(N,2)


!$omp PARALLEL
!$omp DO
      do  i=1,N
         a(i,1)=i
      enddo

!$omp DO
      do 20 i=1,N
         a(i,2)=2*i
 20   continue
!$omp END PARALLEL

      print *,a

      end
